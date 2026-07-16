import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from backend.db import DatabricksClient
from backend.schemas import AgentBriefing, AgentType, Finding, ToolCallRecord


class SpecialistAgent(ABC):
    """Base class for the 7 board-persona agents.

    Subclasses register their tools (plain functions, one per domain capability)
    and implement `analyze()`, which calls those tools via `_call_tool()` so every
    invocation is captured as a `ToolCallRecord` — this is the logging middleware
    referenced in CLAUDE.md section 6, applied uniformly rather than per-agent.
    """

    agent_type: AgentType

    def __init__(self, db: DatabricksClient):
        self.db = db
        self._tool_calls: list[ToolCallRecord] = []

    def _call_tool(self, tool_name: str, func: Callable[..., Any], **kwargs) -> Any:
        # db is an internal dependency every tool takes, not a meaningful "input" for the
        # audit trail - and it isn't serializable (breaks MongoDB/JSON persistence of the
        # governance log), so it's excluded from what gets logged.
        loggable_params = {k: v for k, v in kwargs.items() if k != "db"}
        start = time.perf_counter()
        try:
            result = func(**kwargs)
            self._tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    input_params=loggable_params,
                    output_summary=str(result)[:500],
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    success=True,
                )
            )
            return result
        except Exception as e:
            self._tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    input_params=loggable_params,
                    output_summary="",
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                    success=False,
                    error_message=str(e),
                )
            )
            raise

    @abstractmethod
    def analyze(self, query: str) -> list[Finding]:
        """Run the agent's tools against `query` and return findings.

        Subclasses implement domain logic here, calling `self._call_tool(...)`
        for every tool invocation so it's captured in the briefing.
        """
        raise NotImplementedError

    def run(self, query: str) -> AgentBriefing:
        """Public entry point: produces a validated AgentBriefing."""
        self._tool_calls = []
        start = time.perf_counter()
        findings = self.analyze(query)
        return AgentBriefing(
            agent=self.agent_type,
            findings=findings,
            tool_calls=self._tool_calls,
            execution_time_ms=(time.perf_counter() - start) * 1000,
        )
