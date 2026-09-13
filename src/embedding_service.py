"""CPU-only embedding service with session-level caching."""
import functools
import numpy as np


@functools.lru_cache(maxsize=1)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode(texts: list[str], model_name: str) -> np.ndarray:
    """Encode texts into L2-normalized embeddings (cosine similarity ready)."""
    model = _load_model(model_name)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)
