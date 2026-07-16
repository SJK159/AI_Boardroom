"""A minimal local vector index: brute-force cosine similarity via normalized dot product.

No FAISS/Chroma dependency - at this dataset's scale (tens of thousands of vectors, not
millions), a single matrix multiply against an in-memory numpy array is fast enough (single-
digit milliseconds) and keeps the dependency surface small. This is the "collection" CLAUDE.md
describes for Vector Search - each VectorIndex instance is one collection (reviews and policy
docs get separate instances, never mixed), with per-item metadata carrying doc_type tagging.
"""
import pickle
from pathlib import Path

import numpy as np

from .embedder import embed_batch, embed_one


class VectorIndex:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []

    def build(self, texts: list[str], metadata: list[dict], show_progress: bool = True) -> None:
        assert len(texts) == len(metadata)
        self.vectors = embed_batch(texts, show_progress=show_progress)
        self.metadata = metadata

    def search(self, query: str, top_k: int = 5, min_score: float = 0.35) -> list[dict]:
        if self.vectors is None or len(self.vectors) == 0:
            return []
        query_vec = embed_one(query)
        scores = self.vectors @ query_vec  # cosine similarity, since vectors are unit-normalized
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {**self.metadata[i], "score": round(float(scores[i]), 4)}
            for i in top_idx
            if scores[i] >= min_score
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".vectors.npy"), self.vectors)
        with open(path.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path)
        index = cls()
        index.vectors = np.load(path.with_suffix(".vectors.npy"))
        with open(path.with_suffix(".meta.pkl"), "rb") as f:
            index.metadata = pickle.load(f)
        return index

    @staticmethod
    def exists(path: str | Path) -> bool:
        path = Path(path)
        return path.with_suffix(".vectors.npy").exists() and path.with_suffix(".meta.pkl").exists()
