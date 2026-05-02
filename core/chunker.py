"""Sliding-window chunker for long documents.

Walks the doc as a flat token stream and emits fixed-size windows with
overlap so context isn't lost at the seams. When a paragraph break
falls inside the last 20% of a window, snap the cut there instead of
the hard token limit so chunks read more naturally.
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    text: str
    token_start: int
    token_end: int


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _paragraph_token_positions(text: str) -> list[int]:
    cursor = 0
    breaks = []
    for para in _PARA_SPLIT_RE.split(text):
        cursor += len(_TOKEN_RE.findall(para))
        breaks.append(cursor)
    return breaks


def chunk_text(text: str, doc_name: str,
               chunk_size: int = 300,
               overlap: int = 50) -> list[Chunk]:
    """Split a document into overlapping fixed-size token chunks."""
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must exceed overlap ({overlap})")

    tokens = _tokenize(text)
    if not tokens:
        return []

    para_breaks = _paragraph_token_positions(text)
    chunks: list[Chunk] = []
    pos = 0
    chunk_idx = 0

    while pos < len(tokens):
        end = min(pos + chunk_size, len(tokens))

        soft_zone_start = pos + int(chunk_size * 0.8)
        for pb in para_breaks:
            if soft_zone_start <= pb <= end:
                end = pb
                break

        chunks.append(Chunk(
            chunk_id=f"{doc_name}::{chunk_idx:04d}",
            doc_name=doc_name,
            text=" ".join(tokens[pos:end]),
            token_start=pos,
            token_end=end,
        ))
        chunk_idx += 1

        if end >= len(tokens):
            break
        pos = end - overlap

    return chunks
