"""Hybrid retrieval: BM25 (lexical, Arabic-normalised) + dense embeddings.

Why hybrid? Arabic queries often hinge on exact domain tokens (project names,
"سكني", "وافي", unit numbers) where lexical search is precise, while dense
embeddings handle paraphrase and cross-lingual queries (English question,
Arabic document). Reciprocal-rank fusion combines both without needing to
calibrate score scales.

The index is a plain JSON + .npy pair so it is transparent and versionable;
swapping in pgvector/Qdrant/Chroma is a one-class change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..arabic import tokenize
from ..config import DOCS_DIR, INDEX_DIR
from .chunking import Chunk, chunk_directory
from .embeddings import Embedder, get_embedder

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, embedder: Embedder):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks/embeddings length mismatch")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedder = embedder
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, docs_dir: Path = DOCS_DIR, embedder: Embedder | None = None) -> "HybridRetriever":
        embedder = embedder or get_embedder()
        chunks = chunk_directory(docs_dir)
        if not chunks:
            raise RuntimeError(f"no documents found in {docs_dir}")
        log.info("Embedding %d chunks with %s", len(chunks), embedder.name)
        emb = embedder.embed([c.text for c in chunks])
        return cls(chunks, emb, embedder)

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": self.embedder.name,
            "chunks": [dict(chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text, metadata=c.metadata) for c in self.chunks],
        }
        (index_dir / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        np.save(index_dir / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR, embedder: Embedder | None = None) -> "HybridRetriever":
        embedder = embedder or get_embedder()
        payload = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        if payload.get("embedder") != embedder.name:
            raise RuntimeError(
                f"index was built with {payload.get('embedder')!r} but current embedder is {embedder.name!r}; "
                "rebuild the index (python scripts/build_index.py)"
            )
        chunks = [Chunk(**c) for c in payload["chunks"]]
        emb = np.load(index_dir / "embeddings.npy")
        return cls(chunks, emb, embedder)

    @classmethod
    def load_or_build(cls, index_dir: Path = INDEX_DIR, docs_dir: Path = DOCS_DIR, embedder: Embedder | None = None):
        embedder = embedder or get_embedder()
        try:
            return cls.load(index_dir, embedder)
        except (FileNotFoundError, RuntimeError) as e:
            log.info("Building index (%s)", e)
            r = cls.build(docs_dir, embedder)
            r.save(index_dir)
            return r

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 5, *, lang: str | None = None, project_id: str | None = None,
               rrf_k: int = 60, candidates: int = 25) -> list[Hit]:
        """Return top-``k`` chunks by reciprocal-rank fusion of BM25 and dense scores.

        ``lang``/``project_id`` filter results by chunk metadata *after* scoring,
        with a soft preference (same-language chunks rank first, other language
        kept as fallback so cross-lingual questions still work).
        """
        q_tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.chunks))
        q_emb = self.embedder.embed([query])[0]
        dense_scores = self.embeddings @ q_emb

        bm25_order = np.argsort(-bm25_scores)[:candidates]
        dense_order = np.argsort(-dense_scores)[:candidates]

        fused: dict[int, float] = {}
        bm25_rank: dict[int, int] = {}
        dense_rank: dict[int, int] = {}
        for rank, idx in enumerate(bm25_order):
            if bm25_scores[idx] <= 0:
                continue
            bm25_rank[int(idx)] = rank + 1
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(dense_order):
            dense_rank[int(idx)] = rank + 1
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

        def boost(idx: int) -> float:
            meta = self.chunks[idx].metadata
            b = 1.0
            if lang and meta.get("lang") == lang:
                b *= 1.15
            if project_id and meta.get("project_id") in (project_id, "ALL"):
                b *= 1.10
            return b

        ranked = sorted(fused.items(), key=lambda kv: -kv[1] * boost(kv[0]))
        hits = [Hit(self.chunks[i], s * boost(i), bm25_rank.get(i), dense_rank.get(i)) for i, s in ranked[:k]]
        return hits
