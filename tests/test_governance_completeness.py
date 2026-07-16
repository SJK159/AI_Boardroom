"""Governance completeness: does every session log have a full audit trail?

CLAUDE.md section 9: "every session log has a full audit trail (pass/fail, structural)."
Formalizes the manual check from notebooks/11_governance_demo.ipynb into a repeatable test.
Marked `llm` (calls the boss agent) and `mongo` (persists to Atlas).
"""
import pytest

pytestmark = [pytest.mark.llm, pytest.mark.mongo]

REQUIRED_FIELDS = ["session_id", "user_query", "recommendation", "model_versions", "timestamp"]


def test_session_persists_with_full_audit_trail(governance_logger):
    log = governance_logger.run_with_logging("How is our financial health looking?")

    fetched = governance_logger.get_session(log.session_id)
    assert fetched is not None, "session not found in MongoDB after logging"

    dumped = fetched.model_dump()
    missing = [f for f in REQUIRED_FIELDS if dumped.get(f) is None]
    assert not missing, f"audit trail incomplete, missing: {missing}"


def test_session_captures_model_versions(governance_logger):
    log = governance_logger.run_with_logging("How is our financial health looking?")

    assert "boss_llm" in log.model_versions
    assert "embedding_model" in log.model_versions
    assert log.model_versions["boss_llm"], "boss_llm version string is empty"


def test_human_decision_round_trips_correctly(governance_logger):
    log = governance_logger.run_with_logging("How is our financial health looking?")
    assert log.human_decision is None, "a fresh session should not have a decision yet"

    governance_logger.record_human_decision(log.session_id, "accepted", notes="test note")

    updated = governance_logger.get_session(log.session_id)
    assert updated.human_decision == "accepted"
    assert updated.human_notes == "test note"


def test_invalid_human_decision_is_rejected(governance_logger):
    log = governance_logger.run_with_logging("How is our financial health looking?")

    with pytest.raises(ValueError):
        governance_logger.record_human_decision(log.session_id, "maybe")


def test_recommendation_always_requires_human_approval(governance_logger):
    """Schema-level guarantee (BoardRecommendation.requires_human_approval defaults True) -
    verified here that persistence doesn't accidentally lose or override it."""
    log = governance_logger.run_with_logging("How is our financial health looking?")
    assert log.recommendation.requires_human_approval is True
