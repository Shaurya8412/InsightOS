"""
Unit tests for the Gemini Generator (src/services/rag/generator.py).

All tests are fully mocked to avoid dependencies on real Gemini endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.core.config import settings
from src.core.exceptions import GenerationError
from src.models.schemas import Chunk
from src.services.rag.generator import Generator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_KEY = "fake-api-key-for-tests"
FAKE_MODEL = "gemini-2.0-flash"


@pytest.fixture
def mock_chunks():
    doc_id = uuid4()
    c1 = Chunk(
        chunk_id=uuid4(),
        document_id=doc_id,
        document_name="doc1.txt",
        page_number=1,
        text="This is text chunk 1.",
        source_location="Page 1"
    )
    c2 = Chunk(
        chunk_id=uuid4(),
        document_id=doc_id,
        document_name="doc2.txt",
        page_number=2,
        text="This is text chunk 2.",
        source_location="Page 2"
    )
    return [c1, c2]


# ---------------------------------------------------------------------------
# Generator Tests
# ---------------------------------------------------------------------------

class TestGenerator:

    def test_generator_valid_request(self, mock_chunks):
        with patch("src.services.rag.generator.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            
            # Setup mock generate_content response
            mock_response = MagicMock()
            mock_response.text = "This is the generated answer [1]."
            mock_client.models.generate_content.return_value = mock_response

            generator = Generator(api_key=FAKE_KEY, model=FAKE_MODEL)
            answer = generator.generate("What is the content?", mock_chunks)

            assert answer == "This is the generated answer [1]."
            mock_client.models.generate_content.assert_called_once()
            
            # Extract prompt passed to content generation
            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert call_kwargs["model"] == FAKE_MODEL
            assert "What is the content?" in call_kwargs["contents"]

    def test_generator_uses_config_model(self, mock_chunks):
        with patch("src.services.rag.generator.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "Answer"
            mock_client.models.generate_content.return_value = mock_response

            # Instantiate without specifying model to trigger default settings.LLM_MODEL
            generator = Generator(api_key=FAKE_KEY)
            generator.generate("Query", mock_chunks)

            call_kwargs = mock_client.models.generate_content.call_args[1]
            assert call_kwargs["model"] == settings.LLM_MODEL

    def test_missing_api_key_raises_generation_error(self):
        # Override settings.GEMINI_API_KEY with empty value and patch out environment
        with patch.dict("os.environ", {}, clear=True), \
             patch("src.services.rag.generator.settings.GEMINI_API_KEY", ""):
            with pytest.raises(GenerationError, match="GEMINI_API_KEY is not set"):
                Generator()

    def test_empty_query_rejected(self, mock_chunks):
        generator = Generator(api_key=FAKE_KEY)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generator.generate("", mock_chunks)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            generator.generate("   ", mock_chunks)

    def test_empty_chunks_rejected(self):
        generator = Generator(api_key=FAKE_KEY)
        with pytest.raises(ValueError, match="Context chunks cannot be empty"):
            generator.generate("valid query", [])

    def test_empty_response_from_api_raises_generation_error(self, mock_chunks):
        with patch("src.services.rag.generator.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = ""  # Empty response
            mock_client.models.generate_content.return_value = mock_response

            generator = Generator(api_key=FAKE_KEY)
            with pytest.raises(GenerationError, match="empty or invalid response"):
                generator.generate("query", mock_chunks)

    def test_api_exception_converted_to_generation_error(self, mock_chunks):
        with patch("src.services.rag.generator.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.models.generate_content.side_effect = RuntimeError("Connection timed out")

            generator = Generator(api_key=FAKE_KEY)
            with pytest.raises(GenerationError, match="Failed to generate content from Gemini API"):
                generator.generate("query", mock_chunks)


# ---------------------------------------------------------------------------
# Grounding Prompt Test
# ---------------------------------------------------------------------------

class TestGeneratorGroundingPrompt:

    def test_prompt_grounding_constraints(self, mock_chunks):
        with patch("src.services.rag.generator.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "Answer"
            mock_client.models.generate_content.return_value = mock_response

            generator = Generator(api_key=FAKE_KEY)
            generator.generate("What is python?", mock_chunks)

            # Get the exact prompt passed to models.generate_content
            contents = mock_client.models.generate_content.call_args[1]["contents"]

            # Grounding Instructions assertions
            assert "Answer the user query using ONLY the supplied context chunks" in contents
            assert "Do not use outside knowledge as evidence" in contents
            assert "provided documents do not contain enough information" in contents
            assert "Do not invent facts" in contents
            assert "Do not invent document names, page numbers, or source metadata" in contents
            assert "cite the sources using the chunk numbers in brackets, e.g. [1]" in contents
            
            # Format assertions (stable identifiers)
            assert "[1]" in contents
            assert "Document: doc1.txt" in contents
            assert "Page: 1" in contents
            assert "Text: This is text chunk 1." in contents

            assert "[2]" in contents
            assert "Document: doc2.txt" in contents
            assert "Page: 2" in contents
            assert "Text: This is text chunk 2." in contents

            # Query inclusion assertion
            assert "Query: What is python?" in contents
