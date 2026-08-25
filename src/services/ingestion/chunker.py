"""
Structural chunker: converts PageContent objects into Chunk objects.

Strategy
--------
1. For each PageContent, split text into paragraphs (double-newline boundaries).
2. Accumulate paragraphs into a chunk until CHUNK_SIZE (in characters) is
   reached or exceeded.
3. When a chunk is finalised, carry the last CHUNK_OVERLAP characters into
   the next chunk as a prefix (overlap window).
4. A paragraph that is itself longer than CHUNK_SIZE is hard-split at the
   CHUNK_SIZE boundary to keep chunks bounded.

This approach:
  - Prefers paragraph boundaries (structural chunking).
  - Avoids splitting sentences unnecessarily within a paragraph.
  - Enforces a configurable maximum chunk size.
  - Applies configurable overlap.
  - Preserves all required citation metadata on every Chunk.

Responsibilities:
  - PageContent → Chunk objects ONLY
  - Does NOT generate embeddings, call APIs, or touch the vector store.
"""

from __future__ import annotations

import re
from typing import List
from uuid import UUID

from src.core.config import settings
from src.models.schemas import Chunk, PageContent


def chunk_pages(
    pages: List[PageContent],
    document_id: UUID,
    document_name: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """
    Convert a list of PageContent objects into a list of Chunk objects.

    Args:
        pages:          PageContent objects produced by the parser.
        document_id:    UUID of the parent document.
        document_name:  Human-readable filename of the parent document.
        chunk_size:     Max chunk length in characters. Defaults to
                        settings.CHUNK_SIZE.
        chunk_overlap:  Overlap length in characters. Defaults to
                        settings.CHUNK_OVERLAP.

    Returns:
        Non-empty list of Chunk objects, each carrying full citation metadata.

    Raises:
        ValueError: If pages list is empty.
    """
    if not pages:
        raise ValueError("Cannot chunk an empty list of pages.")

    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    # Clamp overlap to be strictly less than chunk size to avoid infinite loops.
    if overlap >= size:
        overlap = max(0, size - 1)

    chunks: List[Chunk] = []

    for page in pages:
        page_chunks = _chunk_page(
            page=page,
            document_id=document_id,
            document_name=document_name,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        chunks.extend(page_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Per-page chunking
# ---------------------------------------------------------------------------

def _chunk_page(
    page: PageContent,
    document_id: UUID,
    document_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Chunk]:
    """Split a single PageContent into one or more Chunk objects."""
    text = page.text.strip()
    if not text:
        return []

    paragraphs = _split_paragraphs(text)

    raw_texts = _build_raw_chunks(paragraphs, chunk_size, chunk_overlap)

    return [
        Chunk(
            document_id=document_id,
            document_name=document_name,
            page_number=page.page_number,
            text=raw,
            source_location=page.source_location,
        )
        for raw in raw_texts
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


def _split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs on two or more consecutive newlines.
    Single newlines within a paragraph are preserved.
    """
    parts = _PARAGRAPH_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _build_raw_chunks(
    paragraphs: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Accumulate paragraphs into chunks, respecting chunk_size and
    applying character-level overlap between consecutive chunks.

    A paragraph that is itself longer than chunk_size is hard-split.
    """
    # Expand any oversized paragraphs into hard-split sub-paragraphs first.
    expanded: List[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            expanded.append(para)
        else:
            expanded.extend(_hard_split(para, chunk_size))

    raw_chunks: List[str] = []
    current_parts: List[str] = []
    current_len: int = 0

    for para in expanded:
        # +1 accounts for the newline separator we'll join with
        added_len = len(para) + (1 if current_parts else 0)

        if current_parts and current_len + added_len > chunk_size:
            # Finalise the current chunk
            raw_chunks.append("\n".join(current_parts))
            # Start next chunk with overlap from the end of the current chunk
            overlap_text = _get_overlap_prefix(current_parts, chunk_overlap)
            current_parts = [overlap_text] if overlap_text else []
            current_len = len(overlap_text) if overlap_text else 0

        current_parts.append(para)
        current_len += added_len

    # Flush remaining content
    if current_parts:
        raw_chunks.append("\n".join(current_parts))

    return raw_chunks


def _hard_split(text: str, chunk_size: int) -> List[str]:
    """
    Naively split a single oversized block of text into chunks of at most
    chunk_size characters, splitting at word boundaries where possible.
    """
    result: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            result.append(text[start:].strip())
            break
        # Try to split at the last space within the window
        split_at = text.rfind(" ", start, end)
        if split_at <= start:
            split_at = end  # No space found; hard cut
        result.append(text[start:split_at].strip())
        start = split_at + 1
    return [r for r in result if r]


def _get_overlap_prefix(parts: List[str], overlap: int) -> str:
    """
    Return up to `overlap` characters taken from the *end* of the joined
    current-chunk text. We attempt to break on a word boundary.
    """
    if overlap <= 0 or not parts:
        return ""
    full = "\n".join(parts)
    if len(full) <= overlap:
        return full
    candidate = full[-overlap:]
    # Try to start the overlap at a word boundary
    space_idx = candidate.find(" ")
    if 0 < space_idx < len(candidate):
        candidate = candidate[space_idx + 1:]
    return candidate.strip()
