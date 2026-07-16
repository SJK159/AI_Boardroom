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
import time
import uuid

from backend.agents.boss import BossAgent
from backend.config import settings
from backend.rag import EMBEDDING_MODEL_NAME
from backend.schemas import GovernanceLog

from .mongo_client import MongoClient


class GovernanceLogger:
    def __init__(self, boss_agent: BossAgent, mongo: MongoClient):
        self.boss_agent = boss_agent
        self.mongo = mongo

    def run_with_logging(self, query: str) -> GovernanceLog:
        """Runs the boss agent and persists the full session as a GovernanceLog."""
        start = time.perf_counter()
        recommendation = self.boss_agent.run(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        log = GovernanceLog(
            session_id=str(uuid.uuid4()),
            user_query=query,
            recommendation=recommendation,
            model_versions={
                "boss_llm": settings.boss_llm_model,
                "embedding_model": EMBEDDING_MODEL_NAME,
            },
            total_execution_time_ms=elapsed_ms,
        )
        self.mongo.insert_governance_log(log.model_dump())
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
