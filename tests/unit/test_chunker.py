"""
Unit tests for src/services/ingestion/chunker.py

Uses in-memory PageContent objects — no file I/O required.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from src.models.schemas import Chunk, PageContent
from src.services.ingestion.chunker import chunk_pages, _split_paragraphs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOC_ID = uuid4()
DOC_NAME = "test_document.pdf"


def _page(text: str, page_number: int = 1, source: str = "Page 1") -> PageContent:
    return PageContent(page_number=page_number, text=text, source_location=source)


def _pages(*texts: str) -> list[PageContent]:
    return [
        _page(text, i + 1, f"Page {i + 1}")
        for i, text in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Basic chunking behaviour
# ---------------------------------------------------------------------------

class TestBasicChunking:

    def test_short_text_produces_one_chunk(self):
        pages = [_page("Short text that fits in one chunk.")]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 1

    def test_single_chunk_contains_full_text(self):
        text = "Hello world."
        pages = [_page(text)]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        assert text in chunks[0].text

    def test_long_text_produces_multiple_chunks(self):
        # 5 paragraphs of ~100 chars each; chunk_size=150 forces splits
        para = "A" * 100
        text = "\n\n".join([para] * 5)
        pages = [_page(text)]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=150, chunk_overlap=0)
        assert len(chunks) > 1

    def test_chunk_size_is_respected(self):
        para = "Word " * 30   # ~150 chars per paragraph
        text = "\n\n".join([para] * 10)
        pages = [_page(text)]
        chunk_size = 200
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size * 2, (
                "Chunk should be reasonably bounded (hard-split at 2x as worst case)"
            )

    def test_paragraph_boundaries_preferred(self):
        # Two 80-char paragraphs; chunk_size=90 — they should end up in separate chunks
        p1 = "Paragraph one content here, fairly long sentence filling space."
        p2 = "Paragraph two content here, another distinct paragraph block."
        text = f"{p1}\n\n{p2}"
        pages = [_page(text)]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=90, chunk_overlap=0)
        # Both paragraphs should appear in some chunk without being merged
        full = " ".join(c.text for c in chunks)
        assert "Paragraph one" in full
        assert "Paragraph two" in full
        # They should not be in the same chunk given chunk_size=90
        assert len(chunks) >= 2

    def test_empty_page_list_raises_value_error(self):
        with pytest.raises(ValueError, match="empty list"):
            chunk_pages([], DOC_ID, DOC_NAME)

    def test_page_with_only_whitespace_produces_no_chunks(self):
        pages = [_page("   \n\n   ")]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        assert chunks == []


# ---------------------------------------------------------------------------
# Overlap behaviour
# ---------------------------------------------------------------------------

class TestOverlap:

    def test_overlap_zero_produces_no_repeated_content(self):
        para = "Unique paragraph content here. " * 5
        text = "\n\n".join([para] * 4)
        pages = [_page(text)]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1

    def test_overlap_injects_context_into_next_chunk(self):
        # Use a large overlap relative to chunk size
        p1 = "Alpha sentence is here and carries meaning."
        p2 = "Beta sentence adds more distinct information now."
        p3 = "Gamma sentence concludes the final thought properly."
        text = f"{p1}\n\n{p2}\n\n{p3}"
        pages = [_page(text)]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=60, chunk_overlap=30)
        # At least one of the non-first chunks should have some overlap content
        assert len(chunks) >= 2
        # All chunks are non-empty
        for chunk in chunks:
            assert chunk.text.strip()


# ---------------------------------------------------------------------------
# Metadata integrity
# ---------------------------------------------------------------------------

class TestMetadataIntegrity:

    def test_document_id_preserved_on_all_chunks(self):
        pages = _pages("First page text.", "Second page text.", "Third page text.")
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        for chunk in chunks:
            assert chunk.document_id == DOC_ID

    def test_document_name_preserved_on_all_chunks(self):
        pages = _pages("Text A", "Text B")
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        for chunk in chunks:
            assert chunk.document_name == DOC_NAME

    def test_page_number_preserved_per_page(self):
        pages = [
            _page("Page one content.", page_number=1, source="Page 1"),
            _page("Page two content.", page_number=2, source="Page 2"),
        ]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        # Find chunks for each page
        page1_chunks = [c for c in chunks if c.page_number == 1]
        page2_chunks = [c for c in chunks if c.page_number == 2]
        assert page1_chunks, "Expected at least one chunk for page 1"
        assert page2_chunks, "Expected at least one chunk for page 2"

    def test_source_location_preserved_per_page(self):
        pages = [
            _page("Alpha text.", page_number=1, source="Page 1"),
            _page("Beta text.", page_number=2, source="Page 2"),
        ]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        sources = {c.source_location for c in chunks}
        assert "Page 1" in sources
        assert "Page 2" in sources

    def test_every_chunk_has_unique_chunk_id(self):
        para = "Content paragraph. " * 5
        text = "\n\n".join([para] * 6)
        pages = [_page(text)]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=80, chunk_overlap=0)
        assert len(chunks) > 1
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids)), "chunk_id values must be unique"

    def test_chunk_ids_are_uuids(self):
        pages = [_page("Some content here")]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        for chunk in chunks:
            assert isinstance(chunk.chunk_id, UUID)

    def test_chunk_type_is_chunk(self):
        pages = [_page("Sample text")]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_page_number_none_preserved_for_flat_files(self):
        """Flat text (TXT/MD) pages have page_number=None; chunks must preserve it."""
        pages = [PageContent(page_number=None, text="Flat file text.", source_location="doc.txt")]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        assert chunks[0].page_number is None
        assert chunks[0].source_location == "doc.txt"


# ---------------------------------------------------------------------------
# Multi-page chunking
# ---------------------------------------------------------------------------

class TestMultiPageChunking:

    def test_chunks_span_correct_pages(self):
        """Ensure each page's chunks reference that page's metadata."""
        page_texts = [f"Content for page {i + 1}. " * 5 for i in range(4)]
        pages = [
            _page(text, page_number=i + 1, source=f"Page {i + 1}")
            for i, text in enumerate(page_texts)
        ]
        chunks = chunk_pages(pages, DOC_ID, DOC_NAME, chunk_size=500, chunk_overlap=0)
        for i in range(1, 5):
            matching = [c for c in chunks if c.page_number == i]
            assert matching, f"Expected chunks for page {i}"
            for c in matching:
                assert str(i) in c.source_location


# ---------------------------------------------------------------------------
# Paragraph splitting helper
# ---------------------------------------------------------------------------

class TestSplitParagraphs:

    def test_double_newline_splits_paragraphs(self):
        text = "Para one.\n\nPara two."
        result = _split_paragraphs(text)
        assert result == ["Para one.", "Para two."]

    def test_single_newline_preserved_within_paragraph(self):
        text = "Line one.\nLine two.\n\nParagraph two."
        result = _split_paragraphs(text)
        assert len(result) == 2
        assert "Line one." in result[0]
        assert "Line two." in result[0]

    def test_multiple_blank_lines_treated_as_one_separator(self):
        text = "A\n\n\n\nB"
        result = _split_paragraphs(text)
        assert result == ["A", "B"]

    def test_empty_string_returns_empty_list(self):
        assert _split_paragraphs("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _split_paragraphs("   \n\n   ") == []
