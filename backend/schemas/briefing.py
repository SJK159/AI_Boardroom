from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .enums import AgentType
from .findings import Finding, ToolCallRecord


class AgentBriefing(BaseModel):
    agent: AgentType
    findings: list[Finding] = Field(min_length=1)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[float] = None
