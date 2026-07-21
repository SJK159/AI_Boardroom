from typing import TypedDict

from backend.schemas import AgentBriefing, AgentType, Dissent


class BossState(TypedDict, total=False):
    query: str
    selected_agents: list[AgentType]
    selection_reasoning: str
    briefings: list[AgentBriefing]
    failed_specialists: list[str]
    synthesis: str
    dissents: list[Dissent]
    confidence_overall: float
    action_items: list[str]
