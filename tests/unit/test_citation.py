"""
Unit tests for the Citation Handler (src/services/rag/citation.py).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.core.exceptions import CitationError
from src.models.schemas import Chunk
from src.services.rag.citation import CitationHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chunks():
    doc_id = uuid4()
    c1 = Chunk(
        chunk_id=uuid4(),
        document_id=doc_id,
        document_name="doc_a.pdf",
        page_number=3,
        text="Python is a clean, readable language.",
        source_location="Page 3"
    )
    c2 = Chunk(
        chunk_id=uuid4(),
        document_id=doc_id,
        document_name="doc_b.pdf",
        page_number=5,
        text="SQL queries relational database tables.",
        source_location="Page 5"
    )
    return [c1, c2]


@pytest.fixture
def handler():
    return CitationHandler()


# ---------------------------------------------------------------------------
# Citation Parser Tests
# ---------------------------------------------------------------------------

class TestCitationHandler:

    def test_single_citation_mapping(self, handler, mock_chunks):
        text = "This is a statement [1]."
        citations = handler.parse_citations(text, mock_chunks)

        assert len(citations) == 1
        assert citations[0].chunk_id == mock_chunks[0].chunk_id
        assert citations[0].document_name == "doc_a.pdf"
        assert citations[0].page_number == 3
        assert citations[0].snippet == mock_chunks[0].text

    def test_multiple_citations_mapping(self, handler, mock_chunks):
        text = "Statements can reference [1] and another statement [2]."
        citations = handler.parse_citations(text, mock_chunks)

        assert len(citations) == 2
        assert citations[0].chunk_id == mock_chunks[0].chunk_id
        assert citations[0].document_name == "doc_a.pdf"
        assert citations[1].chunk_id == mock_chunks[1].chunk_id
        assert citations[1].document_name == "doc_b.pdf"

    def test_duplicate_citations_deduplicated(self, handler, mock_chunks):
        text = "First assertion [1]. Second assertion [2]. Third assertion [1]."
        citations = handler.parse_citations(text, mock_chunks)

        # Should only contain 2 unique citations
        assert len(citations) == 2
        assert citations[0].chunk_id == mock_chunks[0].chunk_id
        assert citations[1].chunk_id == mock_chunks[1].chunk_id

    def test_no_citations_returns_empty(self, handler, mock_chunks):
        text = "No citation references here."
        citations = handler.parse_citations(text, mock_chunks)
        assert citations == []

    def test_invalid_low_index_zero_raises_citation_error(self, handler, mock_chunks):
        text = "Invalid index zero [0]."
        with pytest.raises(CitationError, match="Invalid citation reference"):
            handler.parse_citations(text, mock_chunks)

    def test_invalid_high_index_raises_citation_error(self, handler, mock_chunks):
        text = "Index out of range [3]."
        with pytest.raises(CitationError, match="Invalid citation reference"):
            handler.parse_citations(text, mock_chunks)

    def test_malformed_citation_syntax_ignored(self, handler, mock_chunks):
        # Text with bracket formats that are not raw integers are ignored
        text = "Invalid formats [a] and [1.5] and [ ] and [1]."
        citations = handler.parse_citations(text, mock_chunks)

        # Only [1] should be parsed
        assert len(citations) == 1
        assert citations[0].chunk_id == mock_chunks[0].chunk_id

    def test_citation_metadata_comes_from_original_chunk_only(self, handler, mock_chunks):
        text = "Some answer [1]."
        citations = handler.parse_citations(text, mock_chunks)

        assert len(citations) == 1
        # Re-verify fields match chunk exactly (metadata integrity check)
        citation = citations[0]
        chunk = mock_chunks[0]
        assert citation.chunk_id == chunk.chunk_id
        assert citation.document_id == chunk.document_id
        assert citation.document_name == chunk.document_name
        assert citation.page_number == chunk.page_number
        assert citation.source_location == chunk.source_location
        assert citation.snippet == chunk.text

    def test_citation_snippet_truncation(self, handler):
        doc_id = uuid4()
        long_text = "A" * 200
        chunk = Chunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="long_doc.txt",
            page_number=None,
            text=long_text,
            source_location="long_doc.txt"
        )
        citations = handler.parse_citations("Statement [1].", [chunk])
        assert len(citations) == 1
        assert len(citations[0].snippet) == 153  # 150 chars + "..."
        assert citations[0].snippet.endswith("...")
