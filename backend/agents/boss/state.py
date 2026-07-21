from typing import Callable, TypedDict

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
    # Not part of the recommendation - an optional per-invocation hook for progress events
    # (specialist_started/completed/failed, synthesis_ready). Threaded through state rather
    # than stored on self, since BossAgent instances are shared across concurrent requests
    # by the API layer and state is already the per-invocation-scoped container.
    on_event: Callable[[str, dict], None] | None
