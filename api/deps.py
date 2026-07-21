"""Process-wide singletons for the FastAPI service.

DatabricksClient and MongoClient are safe to share across concurrent requests (no per-request
mutable state - see their own docstrings); BossAgent likewise only holds LLM clients and a
DatabricksClient reference, with all per-call state threaded through BossState. Building fresh
ones per request would mean re-listing SQL warehouses (DatabricksClient) and re-opening a Mongo
connection pool (MongoClient) on every query, which is wasted work for no isolation benefit.
"""
from functools import lru_cache

from backend.agents.boss import BossAgent
from backend.db import DatabricksClient
from backend.governance import GovernanceLogger, MongoClient


@lru_cache
def get_databricks_client() -> DatabricksClient:
    return DatabricksClient()


@lru_cache
def get_mongo_client() -> MongoClient:
    return MongoClient()


@lru_cache
def get_boss_agent() -> BossAgent:
    return BossAgent(get_databricks_client())


@lru_cache
def get_governance_logger() -> GovernanceLogger:
    return GovernanceLogger(get_boss_agent(), get_mongo_client())
