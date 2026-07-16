"""The reviews Vector Search collection - built from order_reviews text.

Separate collection from policy docs per CLAUDE.md section 2: pattern retrieval over review
text (a different retrieval problem than the docs' precision fact-retrieval), and a different
metadata shape (doc_type is always "review" here, tagged with review_score/order_id instead
of the docs' section/document fields).
"""
from pathlib import Path

from backend.db import DatabricksClient

from .vector_index import VectorIndex

_CACHE_PATH = Path(__file__).parent / ".cache" / "review_index"
_cached_index: VectorIndex | None = None


def build_review_index(db: DatabricksClient, limit: int | None = None) -> VectorIndex:
    """Embeds every non-empty review comment and saves the index to disk. Run once (or after
    the underlying data changes) via notebooks/10_rag_index_build.ipynb - not called
    automatically on every query, since embedding tens of thousands of reviews takes minutes.
    """
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
        SELECT review_id, order_id, review_score, review_comment_message
        FROM {db.table('order_reviews')}
        WHERE review_comment_message IS NOT NULL AND length(review_comment_message) > 0
        {limit_clause}
    """
    rows = db.query(sql)
    texts = [r["review_comment_message"] for r in rows]
    metadata = [
        {
            "doc_type": "review",
            "review_id": r["review_id"],
            "order_id": r["order_id"],
            "review_score": r["review_score"],
            "text": r["review_comment_message"],
        }
        for r in rows
    ]

    index = VectorIndex()
    index.build(texts, metadata)
    index.save(_CACHE_PATH)
    return index


def get_review_index() -> VectorIndex:
    """Loads the pre-built index from disk. Raises with a clear next step if it hasn't been built."""
    global _cached_index
    if _cached_index is not None:
        return _cached_index
    if not VectorIndex.exists(_CACHE_PATH):
        raise RuntimeError(
            "Review vector index not built yet. Run notebooks/10_rag_index_build.ipynb first."
        )
    _cached_index = VectorIndex.load(_CACHE_PATH)
    return _cached_index
