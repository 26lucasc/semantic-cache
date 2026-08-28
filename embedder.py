"""Text -> vector. Swappable backend, local by default (no API key needed).

The local model runs on your machine and is free, which means you can run the
threshold sweep as many times as you want. Swap EMBED_BACKEND=openai later if
you want to compare -- part of the project is showing the threshold MOVES when
the embedding model changes.
"""
import config  # noqa: F401  -- loads .env, must come first
import functools
import os

import numpy as np

BACKEND = os.getenv("EMBED_BACKEND", "local")
LOCAL_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(LOCAL_MODEL)


def embed_many(texts: list[str]) -> np.ndarray:
    """Embed a batch. Returns shape (len(texts), dim), L2-normalized.

    Normalizing here means cosine similarity is just a dot product downstream.
    """
    if BACKEND == "local":
        vecs = _local_model().encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    if BACKEND == "openai":
        # TODO: pip install openai, set OPENAI_API_KEY
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            model=os.getenv("EMBED_MODEL", "text-embedding-3-small"), input=texts
        )
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    raise ValueError(f"unknown EMBED_BACKEND: {BACKEND}")


def embed(text: str) -> np.ndarray:
    return embed_many([text])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Inputs are already normalized, so this is a dot product."""
    return float(np.dot(a, b))
