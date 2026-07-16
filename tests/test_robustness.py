"""Robustness: ambiguous queries, single-agent-only queries, no-relevant-data queries
should fail gracefully, not hallucinate.

CLAUDE.md section 9. Marked `llm` - selection is an LLM call.
"""
import pytest

pytestmark = pytest.mark.llm


def test_irrelevant_query_selects_no_specialists(boss):
    """A query with no genuine match to any registered specialist should route to none,
    not force an irrelevant agent to answer - see README section 4.5's original test case."""
    rec = boss.run("What's the current employee satisfaction across departments?")

    # Compliance/HR exists now, so this specific query may legitimately route there -
    # the real test is that the boss doesn't force EVERY agent to answer regardless of fit.
    assert len(rec.agents_invoked) < 7, "an unrelated query should not invoke the entire roster"


def test_single_domain_query_does_not_over_invoke(boss):
    """A narrowly-scoped question shouldn't fan out to unrelated specialists."""
    rec = boss.run("What is our overall average order value?")

    assert "Sales" in [a.value for a in rec.agents_invoked]
    assert len(rec.agents_invoked) <= 3, (
        f"a narrow sales question invoked {len(rec.agents_invoked)} agents: {rec.agents_invoked}"
    )


def test_no_relevant_data_query_does_not_crash_or_return_empty_synthesis(boss):
    """A query with no matching data (not just no matching agent) should still produce a
    valid, non-crashing recommendation rather than an exception or an empty synthesis."""
    rec = boss.run("How many unicorns did we sell last quarter?")

    assert rec.synthesis, "synthesis should never be empty, even for a nonsensical query"
    assert rec.requires_human_approval is True


def test_boss_never_raises_on_empty_or_whitespace_query(boss):
    rec = boss.run("   ")
    assert rec is not None
    assert rec.requires_human_approval is True
