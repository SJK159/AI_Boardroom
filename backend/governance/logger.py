"""Session-level governance logging - wraps BossAgent from the outside.

Every specialist agent already logs its own tool calls automatically (SpecialistAgent's
_call_tool, base_agent.py) - that's per-tool traceability. This is the layer above it:
one GovernanceLog per boss-agent session, capturing the full audit trail CLAUDE.md section 6
calls for (query, recommendation, model versions, execution time, human decision) and
persisting it, rather than letting it evaporate once BossAgent.run() returns.

Deliberately NOT baked into BossAgent itself - same separation of concerns as DatabricksClient
being the only thing that knows SQL and BossAgent not knowing Mongo exists. BossAgent stays a
pure function of (query -> BoardRecommendation); this class adds persistence around it.
"""
import logging
import time
import uuid
from typing import Callable

from backend.agents.boss import BossAgent
from backend.config import settings
from backend.rag import EMBEDDING_MODEL_NAME
from backend.schemas import GovernanceLog

from .mongo_client import MongoClient

logger = logging.getLogger(__name__)


class GovernanceLogger:
    def __init__(self, boss_agent: BossAgent, mongo: MongoClient):
        self.boss_agent = boss_agent
        self.mongo = mongo

    def run_with_logging(
        self,
        query: str,
        on_event: Callable[[str, dict], None] | None = None,
        session_id: str | None = None,
    ) -> GovernanceLog:
        """Runs the boss agent and persists the full session as a GovernanceLog.

        Accepts a pre-generated session_id so callers that need the ID before the run
        completes (e.g. the API layer, to key a WebSocket connection to it) can supply one
        instead of only learning it after the fact.
        """
        start = time.perf_counter()
        recommendation = self.boss_agent.run(query, on_event=on_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        log = GovernanceLog(
            session_id=session_id or str(uuid.uuid4()),
            user_query=query,
            recommendation=recommendation,
            model_versions={
                "boss_llm": settings.boss_llm_model,
                "embedding_model": EMBEDDING_MODEL_NAME,
            },
            total_execution_time_ms=elapsed_ms,
        )
        # A recommendation that was successfully computed must still reach the caller even if
        # the governance write itself hits a transient failure (observed: intermittent Atlas
        # TLS handshake errors, already retried inside MongoClient) - losing a valid board
        # session over a logging-layer hiccup would be a worse outcome than one unlogged
        # session, and the failure is surfaced loudly here rather than swallowed silently.
        try:
            self.mongo.insert_governance_log(log.model_dump())
        except Exception:
            logger.exception(
                "Failed to persist governance log for session %s - recommendation still "
                "returned to caller, but this session has NO audit trail.", log.session_id,
            )
        return log

    def record_human_decision(self, session_id: str, decision: str, notes: str | None = None) -> bool:
        """Closes the human-oversight loop: accept/reject/modify a past recommendation."""
        if decision not in ("accepted", "rejected", "modified"):
            raise ValueError(f"decision must be accepted/rejected/modified, got: {decision}")
        return self.mongo.update_human_decision(session_id, decision, notes)

    def get_session(self, session_id: str) -> GovernanceLog | None:
        doc = self.mongo.find_by_session_id(session_id)
        return GovernanceLog.model_validate(doc) if doc else None

    def list_recent_sessions(self, limit: int = 10) -> list[GovernanceLog]:
        return [GovernanceLog.model_validate(doc) for doc in self.mongo.find_recent(limit)]
