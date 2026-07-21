"""Thin wrapper around pymongo - the only file that talks to the MongoDB driver directly.

Same isolation pattern as backend/db/databricks_client.py: GovernanceLogger never imports
pymongo itself, so swapping the storage backend later touches exactly this file.
"""
import time
from typing import Any, Callable, TypeVar

import certifi
from pymongo import MongoClient as _PyMongoClient
from pymongo.collection import Collection
from pymongo.errors import AutoReconnect, ConnectionFailure

from backend.config import settings

T = TypeVar("T")


def _with_retry(fn: Callable[[], T], max_attempts: int = 3) -> T:
    """Atlas TLS handshakes have been observed to intermittently fail with
    TLSV1_ALERT_INTERNAL_ERROR even with the CA bundle pinned correctly (see __init__) -
    transient, not fixed by any client-side config found so far. A short retry absorbs it
    rather than failing an entire board session over one flaky handshake."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except (AutoReconnect, ConnectionFailure) as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(1.0 * (attempt + 1))
    raise last_error


class MongoClient:
    def __init__(self):
        # tlsCAFile is pinned to certifi's bundle rather than relying on the OS trust store -
        # on Windows, pymongo's default TLS context fails the Atlas handshake with
        # TLSV1_ALERT_INTERNAL_ERROR under some Python/OpenSSL combinations otherwise.
        self._client = _PyMongoClient(settings.mongodb_uri, tlsCAFile=certifi.where())
        self._db = self._client[settings.mongodb_db_name]

    @property
    def governance_logs(self) -> Collection:
        return self._db["governance_logs"]

    def insert_governance_log(self, doc: dict) -> None:
        _with_retry(lambda: self.governance_logs.insert_one(doc))

    def update_human_decision(self, session_id: str, decision: str, notes: str | None) -> bool:
        result = _with_retry(lambda: self.governance_logs.update_one(
            {"session_id": session_id},
            {"$set": {"human_decision": decision, "human_notes": notes}},
        ))
        return result.modified_count > 0

    def find_by_session_id(self, session_id: str) -> dict[str, Any] | None:
        doc = _with_retry(lambda: self.governance_logs.find_one({"session_id": session_id}))
        if doc:
            doc.pop("_id", None)
        return doc

    def find_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        docs = _with_retry(lambda: list(self.governance_logs.find().sort("timestamp", -1).limit(limit)))
        for doc in docs:
            doc.pop("_id", None)
        return docs

    def ping(self) -> bool:
        _with_retry(lambda: self._client.admin.command("ping"))
        return True
