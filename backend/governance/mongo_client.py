"""Thin wrapper around pymongo - the only file that talks to the MongoDB driver directly.

Same isolation pattern as backend/db/databricks_client.py: GovernanceLogger never imports
pymongo itself, so swapping the storage backend later touches exactly this file.
"""
from typing import Any

import certifi
from pymongo import MongoClient as _PyMongoClient
from pymongo.collection import Collection

from backend.config import settings


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
        self.governance_logs.insert_one(doc)

    def update_human_decision(self, session_id: str, decision: str, notes: str | None) -> bool:
        result = self.governance_logs.update_one(
            {"session_id": session_id},
            {"$set": {"human_decision": decision, "human_notes": notes}},
        )
        return result.modified_count > 0

    def find_by_session_id(self, session_id: str) -> dict[str, Any] | None:
        doc = self.governance_logs.find_one({"session_id": session_id})
        if doc:
            doc.pop("_id", None)
        return doc

    def find_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        docs = list(self.governance_logs.find().sort("timestamp", -1).limit(limit))
        for doc in docs:
            doc.pop("_id", None)
        return docs

    def ping(self) -> bool:
        self._client.admin.command("ping")
        return True
