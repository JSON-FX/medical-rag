"""Page-aware recursive text splitting.

Chunks never span a page boundary, so every chunk carries an exact page
number for citation (spec 6.1).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ChunkConfig

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    page_number: int
    text: str


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not seps:
        # No separators left: hard-slice so a pathological page still chunks.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else f"{buf}{sep}{part}"
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
            buf = ""
        if len(part) > size:
            out.extend(_split_recursive(part, size, rest))
        else:
            buf = part
    if buf:
        out.append(buf)
    return [c.strip() for c in out if c.strip()]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        out.append(previous[-overlap:] + current)
    return out


def chunk_pages(pages: list[PageText], cfg: ChunkConfig) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    index = 0
    for page in pages:
        pieces = _apply_overlap(_split_recursive(page.text, cfg.size, SEPARATORS), cfg.overlap)
        for piece in pieces:
            drafts.append(ChunkDraft(chunk_index=index, page_number=page.page_number, text=piece))
            index += 1
    return drafts
