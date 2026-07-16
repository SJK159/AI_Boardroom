from typing import Literal, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    claim: str
    source: str  # tool name or doc citation
    confidence: float = Field(ge=0, le=1)
    supporting_data: dict = Field(default_factory=dict)
    severity: Optional[Literal["info", "warning", "critical"]] = "info"


class ToolCallRecord(BaseModel):
    tool_name: str
    input_params: dict = Field(default_factory=dict)
    output_summary: str
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
