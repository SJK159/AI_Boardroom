from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    query: str = Field(min_length=1)


class SessionCreateResponse(BaseModel):
    session_id: str


class DecisionRequest(BaseModel):
    decision: str = Field(pattern="^(accepted|rejected|modified)$")
    notes: str | None = None
