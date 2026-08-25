"""
Unit tests for src/services/ingestion/parser.py

All PDFs are generated in-memory using PyMuPDF so the test suite has no
dependency on external files.
"""

from __future__ import annotations

import io
from uuid import uuid4

import fitz  # PyMuPDF
import pytest

from src.core.exceptions import DocumentExtractionError
from src.models.schemas import PageContent
from src.services.ingestion.parser import parse_document


# ---------------------------------------------------------------------------
# Helpers to build in-memory PDFs
# ---------------------------------------------------------------------------

def _make_pdf(pages: list[str]) -> bytes:
    """Create a minimal valid PDF with the given list of page texts."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_empty_pdf() -> bytes:
    """Create a valid PDF with one page that has no text."""
    doc = fitz.open()
    doc.new_page()          # blank page
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF tests
# ---------------------------------------------------------------------------

class TestPDFParser:

    def test_single_page_pdf_returns_one_page_content(self):
        pdf_bytes = _make_pdf(["Hello, InsightOS!"])
        pages = parse_document(pdf_bytes, "test.pdf")
        assert len(pages) == 1
        assert isinstance(pages[0], PageContent)

    def test_single_page_text_is_extracted(self):
        pdf_bytes = _make_pdf(["Hello, InsightOS!"])
        pages = parse_document(pdf_bytes, "test.pdf")
        assert "Hello, InsightOS!" in pages[0].text

    def test_multi_page_pdf_returns_correct_count(self):
        pdf_bytes = _make_pdf(["Page one text", "Page two text", "Page three text"])
        pages = parse_document(pdf_bytes, "multi.pdf")
        assert len(pages) == 3

    def test_page_numbering_is_one_based_and_sequential(self):
        pdf_bytes = _make_pdf(["First", "Second", "Third"])
        pages = parse_document(pdf_bytes, "seq.pdf")
        assert [p.page_number for p in pages] == [1, 2, 3]

    def test_source_location_contains_page_number(self):
        pdf_bytes = _make_pdf(["Some content"])
        pages = parse_document(pdf_bytes, "loc.pdf")
        assert pages[0].source_location == "Page 1"

    def test_multi_page_source_locations(self):
        pdf_bytes = _make_pdf(["A", "B"])
        pages = parse_document(pdf_bytes, "multi_loc.pdf")
        assert pages[0].source_location == "Page 1"
        assert pages[1].source_location == "Page 2"

    def test_each_page_contains_its_text(self):
        texts = ["Alpha page", "Beta page", "Gamma page"]
        pdf_bytes = _make_pdf(texts)
        pages = parse_document(pdf_bytes, "content.pdf")
        for i, expected in enumerate(texts):
            assert expected in pages[i].text

    def test_empty_pdf_raises_extraction_error(self):
        empty_pdf = _make_empty_pdf()
        with pytest.raises(DocumentExtractionError, match="no extractable text"):
            parse_document(empty_pdf, "empty.pdf")

    def test_corrupt_bytes_raises_extraction_error(self):
        corrupt = b"not a real pdf at all %%EOF"
        with pytest.raises(DocumentExtractionError):
            parse_document(corrupt, "corrupt.pdf")

    def test_page_content_type_is_correct(self):
        pdf_bytes = _make_pdf(["type check"])
        pages = parse_document(pdf_bytes, "type.pdf")
        assert all(isinstance(p, PageContent) for p in pages)


# ---------------------------------------------------------------------------
# TXT tests
# ---------------------------------------------------------------------------

class TestTXTParser:

    def test_txt_returns_single_page_content(self):
        content = b"This is a plain text document."
        pages = parse_document(content, "notes.txt")
        assert len(pages) == 1

    def test_txt_text_is_preserved(self):
        content = b"Hello from TXT file."
        pages = parse_document(content, "notes.txt")
        assert "Hello from TXT file." in pages[0].text

    def test_txt_page_number_is_none(self):
        pages = parse_document(b"some text", "doc.txt")
        assert pages[0].page_number is None

    def test_txt_source_location_is_filename(self):
        pages = parse_document(b"some text", "report.txt")
        assert pages[0].source_location == "report.txt"

    def test_empty_txt_raises_extraction_error(self):
        with pytest.raises(DocumentExtractionError, match="no text content"):
            parse_document(b"   \n   ", "empty.txt")


# ---------------------------------------------------------------------------
# Markdown tests
# ---------------------------------------------------------------------------

class TestMarkdownParser:

    def test_md_returns_single_page_content(self):
        content = b"# Title\n\nSome markdown content."
        pages = parse_document(content, "readme.md")
        assert len(pages) == 1

    def test_md_text_is_preserved(self):
        content = b"# Title\n\nSome markdown content."
        pages = parse_document(content, "readme.md")
        assert "# Title" in pages[0].text
        assert "Some markdown content." in pages[0].text

    def test_md_page_number_is_none(self):
        pages = parse_document(b"# Heading", "doc.md")
        assert pages[0].page_number is None

    def test_md_source_location_is_filename(self):
        pages = parse_document(b"content", "notes.md")
        assert pages[0].source_location == "notes.md"

    def test_empty_md_raises_extraction_error(self):
        with pytest.raises(DocumentExtractionError):
            parse_document(b"", "empty.md")


# ---------------------------------------------------------------------------
# Unsupported format tests
# ---------------------------------------------------------------------------

class TestUnsupportedFormats:

    def test_unsupported_extension_raises_extraction_error(self):
        with pytest.raises(DocumentExtractionError, match="Unsupported file type"):
            parse_document(b"data", "data.xlsx")

    def test_docx_raises_extraction_error(self):
        with pytest.raises(DocumentExtractionError, match="Unsupported file type"):
            parse_document(b"data", "file.docx")
