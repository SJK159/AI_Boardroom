from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .enums import AgentType
from .briefing import AgentBriefing


class Dissent(BaseModel):
    agents_involved: list[AgentType] = Field(min_length=2)
    topic: str
    summary: str
    resolution: Optional[str] = None


class BoardRecommendation(BaseModel):
    query: str
    agents_invoked: list[AgentType]
    briefings: list[AgentBriefing]
    synthesis: str
    dissents: list[Dissent] = Field(default_factory=list)
    confidence_overall: float = Field(ge=0, le=1)
    action_items: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True  # governance: recommendation, not auto-action
    timestamp: datetime = Field(default_factory=datetime.utcnow)
