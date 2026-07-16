"""Thin wrapper around a local multilingual sentence-embedding model.

Model: paraphrase-multilingual-MiniLM-L12-v2. Chosen because it natively handles both
Portuguese (reviews) and English (institutional docs) in the same embedding space, so no
translation step is needed - simpler pipeline than translate-then-embed-with-English-model.
Runs on CPU: ~470MB model, no GPU required, embeds a few hundred short texts/second.

This is the local/free substitute for Databricks Vector Search's embedding endpoint - see
backend/rag/README.md (or notebooks/10_rag_index_build.ipynb) for the cost/architecture
tradeoff this was chosen over.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_batch(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Returns an (N, 384) array of unit-normalized embeddings."""
    model = _get_model()
    return model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=True
    )


def embed_one(text: str) -> np.ndarray:
    """Returns a (384,) unit-normalized embedding."""
    return embed_batch([text], show_progress=False)[0]
