"""Embedding backends.

* ``OpenRouterEmbeddings`` – any embedding model exposed by OpenRouter
  (default ``openai/text-embedding-3-small``, which handles Arabic well).
* ``HashingEmbeddings`` – dependency-free, deterministic character n-gram
  hashing. Not semantic, but lets the full pipeline and tests run offline.

Both return L2-normalised ``np.ndarray`` of shape (n, dim).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np

from ..arabic import normalize_arabic
from ..config import Settings, get_settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class OpenRouterEmbeddings:
    def __init__(self, settings: Settings | None = None, model: str | None = None, batch_size: int = 64):
        from openai import OpenAI

        self.settings = settings or get_settings()
        if not self.settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model or self.settings.embedding_model
        self.name = f"openrouter:{self.model}"
        self.batch_size = batch_size
        self.dim = -1  # discovered on first call
        self._client = OpenAI(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
            timeout=self.settings.request_timeout,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + self.batch_size]]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            # keep provider order stable
            out.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        arr = np.asarray(out, dtype=np.float32)
        self.dim = arr.shape[1]
        return _l2(arr)


class HashingEmbeddings:
    """Character 3-5-gram feature hashing over normalised text (offline fallback)."""

    name = "hashing-ngram"

    def __init__(self, dim: int = 4096, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _grams(self, text: str):
        t = f" {normalize_arabic(text)} "
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(t) - n + 1):
                yield t[i : i + n]

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for g in self._grams(text):
                h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
                arr[row, h % self.dim] += 1.0
        # sub-linear tf
        arr = np.log1p(arr)
        return _l2(arr)


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.use_openrouter:
        return OpenRouterEmbeddings(settings)
    log.warning("No OPENROUTER_API_KEY – using HashingEmbeddings (offline, non-semantic).")
    return HashingEmbeddings()
