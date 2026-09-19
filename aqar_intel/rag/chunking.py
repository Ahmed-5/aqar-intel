"""Load Markdown/plain-text/PDF documents and split them into chunks.

Chunking is heading-aware: each Markdown ``##`` section becomes its own chunk
(split further only if it is long), and the document title is prepended to
every chunk so retrieval on a section like "Discounts" still knows which
project it belongs to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # optional dependency, only for PDFs

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # try to break on a paragraph/sentence boundary
        cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end), text.rfind("۔", start, end))
        if cut <= start + max_chars // 2:
            cut = end
        parts.append(text[start:cut].strip())
        if cut >= len(text):
            break
        start = max(cut - overlap, start + 1)
    return [p for p in parts if p]


def chunk_document(path: Path, *, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    raw = read_document(path)
    meta, body = _parse_front_matter(raw)
    doc_id = path.stem
    meta = {"source": path.name, **meta}

    # Title = first H1
    title_match = re.search(r"^#\s+(.*)$", body, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc_id

    # Split into sections at any heading; keep heading text as part of chunk.
    positions = [m.start() for m in _HEADING.finditer(body)] + [len(body)]
    sections = []
    if positions and positions[0] > 0:
        sections.append(body[: positions[0]])
    for a, b in zip(positions, positions[1:]):
        sections.append(body[a:b])

    chunks: list[Chunk] = []
    for sec in sections:
        sec = sec.strip()
        if not sec or sec.startswith("# "):  # skip bare title-only section
            lines = [ln for ln in sec.splitlines()[1:] if ln.strip()] if sec.startswith("# ") else []
            if not lines:
                continue
            sec = "\n".join(lines)
        for piece in _split_long(sec, max_chars, overlap):
            text = f"{title}\n{piece}" if not piece.startswith(title) else piece
            chunks.append(Chunk(chunk_id=f"{doc_id}#{len(chunks)}", doc_id=doc_id, text=text, metadata=dict(meta, title=title)))
    return chunks


def chunk_directory(docs_dir: Path, patterns: tuple[str, ...] = ("*.md", "*.txt", "*.pdf")) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pattern in patterns:
        for path in sorted(docs_dir.glob(pattern)):
            chunks.extend(chunk_document(path))
    return chunks
