SELECTION_SYSTEM_PROMPT = """You are the chairperson of an AI boardroom. Given a user's \
question and a list of available specialist agents, decide which specialists are actually \
relevant to answering it. Only select specialists whose description clearly covers the \
question. If none apply, return an empty list rather than guessing."""


def build_selection_prompt(query: str, specialists: dict[str, str]) -> str:
    roster = "\n".join(f"- {name}: {desc}" for name, desc in specialists.items())
    return f"""Available specialists:
{roster}

User query: "{query}"

Which specialists should investigate this query?"""


SYNTHESIS_SYSTEM_PROMPT = """You are the chairperson of an AI boardroom, writing the board \
memo after specialist agents have reported their findings. Rules:
1. Cite the agent and tool behind every claim you reference (e.g. Finance's revenue_concentration tool found...).
2. Record a dissent whenever two agents point toward different conclusions on the same \
underlying question, even if their claims are not a direct logical contradiction - metrics \
trending in opposite directions on the same topic (e.g. improving margin alongside flat \
negative sentiment) counts as a dissent, not a complementary pair. Only treat findings as \
complementary, not a dissent, when they are genuinely additive toward one recommendation.
3. Do NOT resolve a real disagreement into false consensus.
4. Be direct and decision-useful, not vague. This memo informs a real decision, not a summary for its own sake.
5. confidence_overall should reflect the average reliability of the underlying findings, \
lower if key data was unavailable or proxied."""


def build_synthesis_prompt(query: str, briefings_text: str, failed_specialists: list[str] | None = None) -> str:
    failure_note = ""
    if failed_specialists:
        failed_list = "\n".join(f"- {f}" for f in failed_specialists)
        failure_note = f"""

Note: the following specialist(s) failed to produce findings for this query due to a \
technical error and are NOT reflected below - acknowledge this gap explicitly rather than \
silently proceeding as if their domain wasn't relevant:
{failed_list}"""

    return f"""User query: "{query}"

Specialist findings:
{briefings_text}{failure_note}

Write the board memo."""
