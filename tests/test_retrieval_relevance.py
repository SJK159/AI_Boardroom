"""Retrieval relevance: RAG search results vs. a hand-labeled relevance set.

CLAUDE.md section 9: "RAG results vs. hand-labeled relevance set (precision/recall)."
No LLM calls - the embedding model is local. Requires the vector indexes to already be built
(notebooks/10_rag_index_build.ipynb) - skipped with a clear reason if they aren't.
"""
import pytest

from backend.rag import get_policy_index, get_review_index


def _index_or_skip(get_fn, name: str):
    try:
        return get_fn()
    except RuntimeError:
        pytest.skip(f"{name} vector index not built - run notebooks/10_rag_index_build.ipynb first")


REVIEW_RELEVANCE_SET = [
    # (query, relevance_check) - relevance_check(match) -> bool
    ("product arrived broken", lambda m: int(m["review_score"]) <= 2),
    ("entrega atrasada", lambda m: "atras" in m["text"].lower()),  # Portuguese: "delay"
    ("terrible customer service", lambda m: int(m["review_score"]) <= 2),
]

POLICY_RELEVANCE_SET = [
    ("what is the on-time delivery requirement for sellers", "Service Level Agreement (SLA)"),
    ("how much annual leave do employees get", "Leave Policy"),
    ("when does the seller agreement renew", "Term and Renewal"),
    ("what happens if an employee is fired", "Termination Policy"),
]


@pytest.mark.parametrize("query,relevance_check", REVIEW_RELEVANCE_SET)
def test_review_search_precision_at_3(query, relevance_check):
    index = _index_or_skip(get_review_index, "review")
    results = index.search(query, top_k=3, min_score=0.0)  # no threshold - testing ranking quality itself

    assert len(results) > 0, f"no results at all for {query!r}"
    relevant_count = sum(1 for r in results if relevance_check(r))
    precision_at_3 = relevant_count / len(results)

    assert precision_at_3 >= 0.66, (
        f"query {query!r}: only {relevant_count}/{len(results)} top results relevant "
        f"(precision@3={precision_at_3:.2f}), expected >= 0.66"
    )


@pytest.mark.parametrize("query,expected_section", POLICY_RELEVANCE_SET)
def test_policy_search_top_result_matches_expected_section(query, expected_section):
    index = _index_or_skip(get_policy_index, "policy")
    results = index.search(query, top_k=1, min_score=0.0)

    assert len(results) > 0, f"no results at all for {query!r}"
    assert results[0]["section"] == expected_section, (
        f"query {query!r}: top match was {results[0]['section']!r}, expected {expected_section!r}"
    )
