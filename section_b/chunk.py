"""Optional preprocessing and chunking for the retrieval index."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

CHUNK_WORDS = 180
CHUNK_OVERLAP = 45
MAX_CONTENT_WORDS = 760
MAX_CHUNKS_PER_PAGE = 4
MIN_CHUNK_WORDS = 30


@dataclass
class Chunk:
    """A text view that maps back to one Wikipedia page."""

    page_id: int
    chunk_id: int
    text: str
    kind: str = "chunk"


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]


def _word_chunks(words: List[str]) -> List[str]:
    chunks: List[str] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for start in range(0, min(len(words), MAX_CONTENT_WORDS), step):
        part = words[start : start + CHUNK_WORDS]
        if len(part) < MIN_CHUNK_WORDS:
            break
        chunks.append(" ".join(part))
        if len(chunks) >= MAX_CHUNKS_PER_PAGE or start + CHUNK_WORDS >= len(words):
            break
    return chunks


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Every page gets title and lead views, plus overlapping title-prefixed
    content chunks. Short synthetic pages are also embedded as a full-page view.
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    paragraphs = _paragraphs(content)
    lead = paragraphs[0] if paragraphs else content
    full_text = entry_text(record)
    words = content.split()
    chunks: List[Chunk] = []
    if title:
        chunks.append(Chunk(page_id, len(chunks), title, "title"))
    if lead:
        chunks.append(Chunk(page_id, len(chunks), f"{title}\n\n{lead}".strip(), "lead"))
    if len(words) <= 1000:
        chunks.append(Chunk(page_id, len(chunks), full_text, "full"))
        for text in _word_chunks(words):
            chunks.append(Chunk(page_id, len(chunks), f"{title}\n\n{text}".strip(), "chunk"))
    else:
        first_words = " ".join(words[:CHUNK_WORDS])
        if first_words:
            chunks.append(Chunk(page_id, len(chunks), f"{title}\n\n{first_words}".strip(), "chunk"))
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
