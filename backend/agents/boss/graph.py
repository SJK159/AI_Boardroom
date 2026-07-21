from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from backend.config import settings
from backend.db import DatabricksClient
from backend.llm_utils import invoke_with_retry
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


def _emit(state: BossState, event_type: str, data: dict) -> None:
    """Fires the per-invocation progress callback if the caller supplied one (the API layer
    does, for live agent-status streaming); a no-op for direct/test callers that don't."""
    on_event = state.get("on_event")
    if on_event is not None:
        on_event(event_type, data)


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
        # temperature=0: a decision-support system should route/synthesize consistently for
        # the same query - the eval suite caught the default (non-zero) temperature causing
        # the same query to select different specialists across runs (see tests/test_synthesis_quality.py)
        llm = ChatGroq(model=settings.boss_llm_model, api_key=settings.groq_api_key, temperature=0)
        self._selection_llm = llm.with_structured_output(AgentSelection)
        self._synthesis_llm = llm.with_structured_output(SynthesisOutput)
        self._graph = self._build_graph()

    def _select_specialists(self, state: BossState) -> dict:
        specialists = {a.value: e.description for a, e in AVAILABLE_SPECIALISTS.items()}
        prompt = build_selection_prompt(state["query"], specialists)
        result: AgentSelection = invoke_with_retry(self._selection_llm, [
            SystemMessage(content=SELECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        _emit(state, "specialists_selected", {
            "agents": [a.value for a in result.selected_agents],
            "reasoning": result.reasoning,
        })
        return {"selected_agents": result.selected_agents, "selection_reasoning": result.reasoning}

    def _has_selected_agents(self, state: BossState) -> str:
        return "run_specialists" if state["selected_agents"] else "no_relevant_agents"

    def _run_specialists(self, state: BossState) -> dict:
        """Runs selected specialists in parallel. A single specialist's failure (e.g. the
        RAG vector index not being built yet) must not take down the whole session and
        discard other specialists' already-completed briefings - each future's exception is
        caught individually and surfaced to synthesis as an explicit gap instead."""
        selected = [a for a in state["selected_agents"] if a in AVAILABLE_SPECIALISTS]
        briefings: list[AgentBriefing] = []
        failed_specialists: list[str] = []

        for agent_type in selected:
            _emit(state, "specialist_started", {"agent": agent_type.value})

        with ThreadPoolExecutor(max_workers=max(len(selected), 1)) as pool:
            futures = {
                pool.submit(AVAILABLE_SPECIALISTS[agent_type].agent_class(self.db).run, state["query"]): agent_type
                for agent_type in selected
            }
            for future in as_completed(futures):
                agent_type = futures[future]
                try:
                    briefing = future.result()
                    briefings.append(briefing)
                    _emit(state, "specialist_completed", {
                        "agent": agent_type.value,
                        "finding_count": len(briefing.findings),
                    })
                except Exception as e:
                    failed_specialists.append(f"{agent_type.value}: {e}")
                    _emit(state, "specialist_failed", {"agent": agent_type.value, "error": str(e)})

        return {"briefings": briefings, "failed_specialists": failed_specialists}

    def _synthesize(self, state: BossState) -> dict:
        briefings_text = _format_briefings(state["briefings"])
        prompt = build_synthesis_prompt(state["query"], briefings_text, state.get("failed_specialists"))
        result: SynthesisOutput = invoke_with_retry(self._synthesis_llm, [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        _emit(state, "synthesis_ready", {"synthesis": result.synthesis})
        return {
            "synthesis": result.synthesis,
            "dissents": result.dissents,
            "confidence_overall": result.confidence_overall,
            "action_items": result.action_items,
        }

    def _no_relevant_agents(self, state: BossState) -> dict:
        _emit(state, "synthesis_ready", {
            "synthesis": "No specialist agent currently available is relevant to this query.",
        })
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

    def run(self, query: str, on_event: Callable[[str, dict], None] | None = None) -> BoardRecommendation:
        final_state = self._graph.invoke({"query": query, "on_event": on_event})
        briefings = final_state["briefings"]
        return BoardRecommendation(
            query=query,
            # reflects agents that actually SUCCEEDED, not just were selected - a specialist
            # that failed (see _run_specialists) has no briefing and shouldn't be listed as invoked
            agents_invoked=[b.agent for b in briefings],
            briefings=briefings,
            synthesis=final_state["synthesis"],
            dissents=final_state["dissents"],
            confidence_overall=final_state["confidence_overall"],
            action_items=final_state["action_items"],
        )
