"""Structured-output shapes the boss LLM must return at each graph step.

These are intermediate/internal — not part of the schemas package, which
defines the final validated contracts (AgentBriefing, BoardRecommendation, etc.)
that flow between agents and out to the UX layer.
"""
from pydantic import BaseModel, Field

from backend.schemas import AgentType, Dissent


class AgentSelection(BaseModel):
    selected_agents: list[AgentType] = Field(
        default_factory=list,
        description="Specialists relevant to answering the query. Empty if none apply.",
    )
    reasoning: str = Field(description="One or two sentences on why these agents were chosen.")


class SynthesisOutput(BaseModel):
    synthesis: str = Field(description="Board memo synthesizing all findings, citing agents/tools by name.")
    dissents: list[Dissent] = Field(
        default_factory=list,
        description="Cases where two or more agents' findings materially conflict. Empty if none.",
    )
    confidence_overall: float = Field(ge=0, le=1)
    action_items: list[str] = Field(default_factory=list)
