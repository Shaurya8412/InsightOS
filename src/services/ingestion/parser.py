"""
Document parser: converts raw file bytes into PageContent objects.

Responsibilities:
  - Read PDF, TXT, and Markdown files
  - Extract text per page (PDF) or as a single block (TXT/MD)
  - Return a list of PageContent objects
  - Raise DocumentExtractionError on any failure

This module does NOT chunk, embed, or call any external API.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from src.core.exceptions import DocumentExtractionError
from src.models.schemas import PageContent

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def parse_document(file_bytes: bytes, filename: str) -> List[PageContent]:
    """
    Parse a document from raw bytes and return a list of PageContent objects.

    Args:
        file_bytes: Raw file content.
        filename:   Original filename, used to determine file type and
                    as a source_location label for flat text files.

    Returns:
        Non-empty list of PageContent objects.

    Raises:
        DocumentExtractionError: On unsupported format, corrupt content,
                                 empty document, or any extraction failure.
    """
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise DocumentExtractionError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        return _parse_pdf(file_bytes, filename)
    else:
        return _parse_flat_text(file_bytes, filename)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _parse_pdf(file_bytes: bytes, filename: str) -> List[PageContent]:
    """Extract text from a PDF, one PageContent per page."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentExtractionError(
            f"Failed to open PDF '{filename}': {exc}"
        ) from exc

    if doc.page_count == 0:
        raise DocumentExtractionError(
            f"PDF '{filename}' contains no pages."
        )

    pages: List[PageContent] = []
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text")  # plain UTF-8 text
            # Always record the page even if it has no text, but only include
            # pages that carry meaningful content to avoid empty chunks later.
            if text.strip():
                human_page_number = page_index + 1  # 1-based, deterministic
                pages.append(
                    PageContent(
                        page_number=human_page_number,
                        text=text,
                        source_location=f"Page {human_page_number}",
                    )
                )
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(
            f"Error extracting text from PDF '{filename}': {exc}"
        ) from exc
    finally:
        doc.close()

    if not pages:
        raise DocumentExtractionError(
            f"PDF '{filename}' contains no extractable text content."
        )

    return pages


# ---------------------------------------------------------------------------
# TXT / Markdown extraction
# ---------------------------------------------------------------------------

def _parse_flat_text(file_bytes: bytes, filename: str) -> List[PageContent]:
    """
    Extract text from a TXT or Markdown file.

    These formats have no concept of pages, so page_number is set to None
    and source_location is set to the filename.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
        except Exception as exc:
            raise DocumentExtractionError(
                f"Could not decode '{filename}' as text: {exc}"
            ) from exc

    if not text.strip():
        raise DocumentExtractionError(
            f"File '{filename}' contains no text content."
        )

    return [
        PageContent(
            page_number=None,
            text=text,
            source_location=filename,
        )
    ]
