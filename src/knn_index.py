"""Nearest-neighbor index over support-set embeddings (cosine similarity).

Uses FAISS when available; falls back to a numpy brute-force search so the
app and tests still run in minimal environments.
"""
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class KNNIndex:
    def __init__(self, embeddings: np.ndarray):
        """embeddings: (N, D) L2-normalized float32 matrix."""
        self.embeddings = embeddings.astype(np.float32)
        self.index = None
        if _HAS_FAISS:
            # Inner product on L2-normalized vectors == cosine similarity
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def search(self, query_emb: np.ndarray, k: int):
        """Returns (indices, similarities) with shapes (k,), (k,)."""
        q = np.atleast_2d(query_emb).astype(np.float32)
        if self.index is not None:
            sims, idx = self.index.search(q, k)
            return idx[0], sims[0]
        sims = self.embeddings @ q[0]
        idx = np.argsort(-sims)[:k]
        return idx.astype(np.int64), sims[idx]
