"""Shared LLM-call utilities - not agent-specific, used by BossAgent and by the eval suite's
LLM-as-judge test, which needs the same protection against the same known flakiness.
"""
import time


def invoke_with_retry(structured_llm, messages, max_attempts: int = 5):
    """Retry wrapper for structured-output LLM calls.

    GPT OSS 20B on Groq occasionally hallucinates a mismatched tool call during structured
    output (observed: 'attempted to call tool synthesisOutput which was not in request.tools')
    - a known reliability quirk of smaller open-weight models under strict function-calling,
    not a bug in the prompt or schema, and not fully suppressed by temperature=0 (routing/
    batching non-determinism on the provider side). Retrying the exact same call usually
    succeeds within a couple of attempts, but has been observed needing more on a bad day for
    the free-tier endpoint - hence 5 attempts, not 3. Any structured-output call against this
    model family should go through this wrapper, not just the boss agent's own calls.
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return structured_llm.invoke(messages)
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_error
