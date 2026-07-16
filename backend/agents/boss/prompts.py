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
1. Cite the agent and tool behind every claim you reference (e.g. "Finance's revenue_concentration tool found...").
2. If two agents' findings conflict or point in different directions, do NOT resolve it into \
false consensus — call it out explicitly as a dissent.
3. Be direct and decision-useful, not vague. This memo informs a real decision, not a summary for its own sake.
4. confidence_overall should reflect the average reliability of the underlying findings, \
lower if key data was unavailable or proxied."""


def build_synthesis_prompt(query: str, briefings_text: str) -> str:
    return f"""User query: "{query}"

Specialist findings:
{briefings_text}

Write the board memo."""
