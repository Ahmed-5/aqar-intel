"""Chunk the document corpus, embed it and save the hybrid index.

    python scripts/build_index.py

Uses OpenRouter embeddings when OPENROUTER_API_KEY is set, otherwise the
offline hashing embedder (BM25 still works either way).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aqar_intel.config import DOCS_DIR, INDEX_DIR  # noqa: E402
from aqar_intel.rag.embeddings import get_embedder  # noqa: E402
from aqar_intel.rag.retriever import HybridRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    embedder = get_embedder()
    retriever = HybridRetriever.build(DOCS_DIR, embedder)
    retriever.save(INDEX_DIR)
    print(f"Indexed {len(retriever.chunks)} chunks with {embedder.name} -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
