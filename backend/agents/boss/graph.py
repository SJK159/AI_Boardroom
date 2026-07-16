from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from backend.config import settings
from backend.db import DatabricksClient
from backend.schemas import AgentBriefing, BoardRecommendation

from .llm_outputs import AgentSelection, SynthesisOutput
from .prompts import (
    SELECTION_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_selection_prompt,
    build_synthesis_prompt,
)
from .registry import AVAILABLE_SPECIALISTS
from .state import BossState


def _format_briefings(briefings: list[AgentBriefing]) -> str:
    blocks = []
    for b in briefings:
        lines = [f"### {b.agent.value} Agent"]
        for f in b.findings:
            lines.append(f"- [{f.severity}] (confidence {f.confidence}) {f.claim} (source: {f.source})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class BossAgent:
    """Orchestrator: selects relevant specialists, runs them in parallel, synthesizes
    their findings into a validated BoardRecommendation. Has no domain tools of its own.
    """

    def __init__(self, db: DatabricksClient):
        self.db = db
        llm = ChatGroq(model=settings.boss_llm_model, api_key=settings.groq_api_key)
        self._selection_llm = llm.with_structured_output(AgentSelection)
        self._synthesis_llm = llm.with_structured_output(SynthesisOutput)
        self._graph = self._build_graph()

    def _select_specialists(self, state: BossState) -> dict:
        specialists = {a.value: e.description for a, e in AVAILABLE_SPECIALISTS.items()}
        prompt = build_selection_prompt(state["query"], specialists)
        result: AgentSelection = self._selection_llm.invoke([
            SystemMessage(content=SELECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return {"selected_agents": result.selected_agents, "selection_reasoning": result.reasoning}

    def _has_selected_agents(self, state: BossState) -> str:
        return "run_specialists" if state["selected_agents"] else "no_relevant_agents"

    def _run_specialists(self, state: BossState) -> dict:
        selected = [a for a in state["selected_agents"] if a in AVAILABLE_SPECIALISTS]
        briefings: list[AgentBriefing] = []

        with ThreadPoolExecutor(max_workers=max(len(selected), 1)) as pool:
            futures = {
                pool.submit(AVAILABLE_SPECIALISTS[agent_type].agent_class(self.db).run, state["query"]): agent_type
                for agent_type in selected
            }
            for future in as_completed(futures):
                briefings.append(future.result())

        return {"briefings": briefings}

    def _synthesize(self, state: BossState) -> dict:
        briefings_text = _format_briefings(state["briefings"])
        prompt = build_synthesis_prompt(state["query"], briefings_text)
        result: SynthesisOutput = self._synthesis_llm.invoke([
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return {
            "synthesis": result.synthesis,
            "dissents": result.dissents,
            "confidence_overall": result.confidence_overall,
            "action_items": result.action_items,
        }

    def _no_relevant_agents(self, state: BossState) -> dict:
        return {
            "briefings": [],
            "synthesis": "No specialist agent currently available is relevant to this query.",
            "dissents": [],
            "confidence_overall": 0.0,
            "action_items": [],
        }

    def _build_graph(self):
        graph = StateGraph(BossState)
        graph.add_node("select_specialists", self._select_specialists)
        graph.add_node("run_specialists", self._run_specialists)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("no_relevant_agents", self._no_relevant_agents)

        graph.set_entry_point("select_specialists")
        graph.add_conditional_edges(
            "select_specialists",
            self._has_selected_agents,
            {"run_specialists": "run_specialists", "no_relevant_agents": "no_relevant_agents"},
        )
        graph.add_edge("run_specialists", "synthesize")
        graph.add_edge("synthesize", END)
        graph.add_edge("no_relevant_agents", END)
        return graph.compile()

    def run(self, query: str) -> BoardRecommendation:
        final_state = self._graph.invoke({"query": query})
        return BoardRecommendation(
            query=query,
            agents_invoked=final_state["selected_agents"],
            briefings=final_state["briefings"],
            synthesis=final_state["synthesis"],
            dissents=final_state["dissents"],
            confidence_overall=final_state["confidence_overall"],
            action_items=final_state["action_items"],
        )
