"""Shared fixtures for the eval suite.

Session-scoped: one DatabricksClient/BossAgent/MongoClient for the whole test run, not one
per test - these are real connections to real services (Databricks SQL warehouse, Groq,
MongoDB Atlas), not mocks. This is a regression suite against the live system, per CLAUDE.md
section 9: "not a single demo query - a regression eval suite, run on every prompt/model/
schema change."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from backend.agents.boss import BossAgent
from backend.db import DatabricksClient
from backend.governance import GovernanceLogger, MongoClient


@pytest.fixture(scope="session")
def db() -> DatabricksClient:
    return DatabricksClient()


@pytest.fixture(scope="session")
def boss(db) -> BossAgent:
    return BossAgent(db)


@pytest.fixture(scope="session")
def mongo() -> MongoClient:
    return MongoClient()


@pytest.fixture(scope="session")
def governance_logger(boss, mongo) -> GovernanceLogger:
    return GovernanceLogger(boss, mongo)
