import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from backend.db import DatabricksClient
from backend.schemas import AgentBriefing, AgentType, Finding, ToolCallRecord


def _is_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


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
        # Any kwarg that isn't JSON/BSON-serializable (e.g. the `db` connection every tool
        # takes) breaks MongoDB persistence of the governance log if logged raw - checked by
        # actual serializability, not by parameter name, so a future tool taking some other
        # non-serializable object doesn't silently reintroduce the same failure mode. The key
        # is kept (with a type placeholder) rather than dropped, so the audit trail still shows
        # every argument the tool was called with.
        loggable_params = {
            k: (v if _is_serializable(v) else f"<{type(v).__name__} instance, not serializable>")
            for k, v in kwargs.items()
        }
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

    def _call_tools_parallel(self, calls: dict[str, tuple[Callable[..., Any], dict]]) -> dict[str, Any]:
        """Runs multiple independent tool calls concurrently instead of one at a time.

        Each tool call is a separate Databricks SQL round-trip with no dependency on any
        other tool's result within the same analyze() - serializing 7-8 of them cost roughly
        7-20s of pure network-wait per agent run for no reason. `calls` maps tool_name to
        (func, kwargs); every call still goes through `_call_tool` so logging is identical to
        the sequential path. list.append is atomic in CPython, so concurrent appends to
        self._tool_calls from multiple threads don't need an explicit lock - tool_calls may
        just end up logged in completion order rather than declaration order.

        If any call fails, its exception is re-raised only after every other call has finished
        (and been logged), rather than immediately - so one failing tool doesn't cut short the
        logging of the others.
        """
        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}

        def run_one(name: str, func: Callable[..., Any], kwargs: dict) -> None:
            try:
                results[name] = self._call_tool(name, func, **kwargs)
            except Exception as e:
                errors[name] = e

        with ThreadPoolExecutor(max_workers=max(len(calls), 1)) as pool:
            futures = [pool.submit(run_one, name, func, kwargs) for name, (func, kwargs) in calls.items()]
            for future in futures:
                future.result()

        if errors:
            raise next(iter(errors.values()))

        return results

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
