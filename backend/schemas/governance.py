from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field

from .recommendation import BoardRecommendation


class GovernanceLog(BaseModel):
    session_id: str
    user_query: str
    recommendation: BoardRecommendation
    model_versions: dict = Field(default_factory=dict)
    human_decision: Optional[Literal["accepted", "rejected", "modified"]] = None
    human_notes: Optional[str] = None
    total_execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
