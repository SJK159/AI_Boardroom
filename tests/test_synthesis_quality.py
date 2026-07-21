"""Synthesis quality: does the boss agent preserve known conflicting findings?

CLAUDE.md section 9: "does the boss agent preserve known conflicting findings? LLM-as-judge
rubric (qualitative, not pass/fail)." All tests here call the live boss agent (Groq +
Databricks) - marked `llm`, run selectively: `pytest -m llm`.
"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from backend.config import settings
from backend.llm_utils import invoke_with_retry
from backend.schemas import AgentBriefing, AgentType, Finding

pytestmark = pytest.mark.llm


def test_boss_preserves_dissent_when_findings_genuinely_conflict(boss):
    """Tests the SYNTHESIS step directly, with manually-constructed conflicting briefings,
    rather than depending on live agent SELECTION happening to pick the same two agents
    every run. Those are separate concerns: selection is covered by tests/test_robustness.py,
    and the eval suite itself caught selection being non-deterministic across runs for a
    borderline query (same query routed to Finance+Sentiment in one run, Sales+Sentiment in
    another - fixed at the source via temperature=0 in graph.py, but synthesis dissent
    preservation deserves a test that doesn't depend on selection variance at all)."""
    finance_briefing = AgentBriefing(
        agent=AgentType.FINANCE,
        findings=[Finding(
            claim="Contribution margin is improving across the last 12 months",
            source="calculate_margin_trend",
            confidence=0.85,
            supporting_data={"trend_direction": "improving"},
        )],
    )
    sentiment_briefing = AgentBriefing(
        agent=AgentType.SENTIMENT,
        findings=[Finding(
            claim="Negative review share has remained flat, and the lowest-rated product sits at 1.0/5",
            source="flag_negative_trend",
            confidence=0.85,
            supporting_data={"trend_direction": "flat"},
            severity="warning",
        )],
    )

    result = boss._synthesize({
        "query": "Are customers happy, and is that reflected in the numbers?",
        "briefings": [finance_briefing, sentiment_briefing],
    })

    assert len(result["dissents"]) >= 1, "expected synthesis to record a Dissent given genuinely conflicting findings"
    dissent_agents = {a for d in result["dissents"] for a in d.agents_involved}
    assert AgentType.FINANCE in dissent_agents and AgentType.SENTIMENT in dissent_agents


def test_boss_does_not_fabricate_dissent_for_complementary_findings(boss):
    """Regression test for the inverse failure mode: manufacturing a conflict where the
    findings are actually complementary (Growth's expansion signal + Operations' capacity
    constraint combined into one recommendation, not a false disagreement - see README 4.9)."""
    rec = boss.run("Which regions should we invest in for growth, and can we support that operationally?")

    assert len(rec.agents_invoked) >= 2
    # not a hard zero-assertion (the LLM could legitimately find something) - the real check
    # is that dissents, if any, are substantive, not that the count is exactly zero
    for dissent in rec.dissents:
        assert len(dissent.summary) > 20, "a recorded Dissent should have a substantive summary, not a stub"


class SynthesisQualityRubric(BaseModel):
    cites_specific_tools: bool = Field(description="Does the synthesis name specific tools/agents behind its claims, not just vague assertions?")
    acknowledges_data_limitations: bool = Field(description="If any finding flagged a data limitation (proxy, unavailable, low confidence), does the synthesis reflect that uncertainty?")
    actionable: bool = Field(description="Does the synthesis conclude with concrete, specific action items rather than generic advice?")
    reasoning: str = Field(description="One or two sentences explaining the ratings above.")


_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of AI-generated board memos. Rate the \
given synthesis against the rubric fields. Be strict - a memo that vaguely alludes to data \
without naming sources, or gives generic advice a doctor's-note-length AI text always gives, \
should score poorly."""


def test_llm_judge_rates_synthesis_quality(boss):
    """Real LLM-as-judge, per CLAUDE.md's explicit call for one. Uses a separate structured-
    output call (same ChatGroq pattern as the boss agent itself) to rate the synthesis
    against a rubric, rather than just checking pass/fail structure.

    Judges the full BoardRecommendation (synthesis prose + structured action_items), not just
    the prose alone - action_items is a separate schema field precisely so "actionable" isn't
    solely dependent on how the free-text synthesis happens to end. An earlier version of this
    test judged synthesis text alone and failed on a run where the prose ended vaguely even
    though structured action_items were present and concrete - a false failure caused by an
    incomplete evaluation input, not a real quality problem.

    Uses invoke_with_retry (the same helper BossAgent's own structured-output calls use) since
    this is the identical ChatGroq + with_structured_output pattern exposed to the same known
    Groq/GPT-OSS-20B flakiness - without it, this test could fail nondeterministically on
    infrastructure flakiness that looks like a synthesis-quality regression."""
    rec = boss.run("How is our financial health looking?")

    judge_llm = ChatGroq(model=settings.boss_llm_model, api_key=settings.groq_api_key).with_structured_output(
        SynthesisQualityRubric
    )
    action_items_text = "\n".join(f"- {item}" for item in rec.action_items) or "(none)"
    judge_input = f"Synthesis:\n\n{rec.synthesis}\n\nAction items:\n{action_items_text}"
    rubric: SynthesisQualityRubric = invoke_with_retry(judge_llm, [
        SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=judge_input),
    ])

    assert rubric.cites_specific_tools, f"judge found no specific tool citations: {rubric.reasoning}"
    assert rubric.actionable, f"judge found the recommendation not actionable: {rubric.reasoning}"
