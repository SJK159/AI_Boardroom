"""The institutional-documents Vector Search collection - built from the Compliance agent's
local markdown documents, chunked by section (clause-addressable, per CLAUDE.md section 2's
"smaller chunks, section/clause-based chunking" guidance for these docs).

Separate collection from reviews - different metadata shape (doc_type is
policy|registration|contract, tagged with document/section instead of review_id/order_id).
"""
from pathlib import Path

from backend.agents.compliance.document_loader import load_all_documents

from .vector_index import VectorIndex

_CACHE_PATH = Path(__file__).parent / ".cache" / "policy_index"
_cached_index: VectorIndex | None = None


def build_policy_index() -> VectorIndex:
    """Embeds every document section and saves the index to disk."""
    texts, metadata = [], []
    for doc in load_all_documents():
        for header, body in doc["sections"].items():
            if not body.strip():
                continue
            texts.append(f"{header}\n{body}")  # header included for retrieval context
            metadata.append({
                "doc_type": doc["doc_type"],
                "document": doc["title"],
                "section": header,
                "text": body,
            })

    index = VectorIndex()
    index.build(texts, metadata, show_progress=False)
    index.save(_CACHE_PATH)
    return index


def get_policy_index() -> VectorIndex:
    """Loads the pre-built index from disk. Raises with a clear next step if it hasn't been built."""
    global _cached_index
    if _cached_index is not None:
        return _cached_index
    if not VectorIndex.exists(_CACHE_PATH):
        raise RuntimeError(
            "Policy vector index not built yet. Run notebooks/10_rag_index_build.ipynb first."
        )
    _cached_index = VectorIndex.load(_CACHE_PATH)
    return _cached_index
