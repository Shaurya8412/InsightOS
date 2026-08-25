"""
Unit tests for the RAG Orchestrator (src/services/rag/orchestrator.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.core.exceptions import CitationError, GenerationError, RetrievalError
from src.models.schemas import Chunk, Citation, RetrievalResult
from src.services.rag.citation import CitationHandler
from src.services.rag.generator import Generator
from src.services.rag.orchestrator import Orchestrator
from src.services.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_retriever():
    return MagicMock(spec=Retriever)


@pytest.fixture
def mock_generator():
    return MagicMock(spec=Generator)


@pytest.fixture
def mock_citation_handler():
    return MagicMock(spec=CitationHandler)


@pytest.fixture
def orchestrator(mock_retriever, mock_generator, mock_citation_handler):
    return Orchestrator(
        retriever=mock_retriever,
        generator=mock_generator,
        citation_handler=mock_citation_handler
    )


# ---------------------------------------------------------------------------
# Orchestrator Tests
# ---------------------------------------------------------------------------

class TestOrchestrator:

    def test_orchestrator_success_flow(
        self, orchestrator, mock_retriever, mock_generator, mock_citation_handler
    ):
        query = "What is RAG?"
        doc_id = uuid4()
        chunk = Chunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="test.txt",
            text="Context text",
            page_number=1,
            source_location="test.txt"
        )
        
        # 1. Setup mock retriever return
        mock_retriever.retrieve.return_value = [
            RetrievalResult(chunk=chunk, score=0.95)
        ]

        # 2. Setup mock generator return
        mock_generator.generate.return_value = "This is generated response [1]."

        # 3. Setup mock citation handler return
        mock_citation = Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_number=1,
            source_location="test.txt",
            snippet="Context text"
        )
        mock_citation_handler.parse_citations.return_value = [mock_citation]

        response = orchestrator.query(query, top_k=3, score_threshold=0.8)

        # Assert correct call ordering and arguments
        mock_retriever.retrieve.assert_called_once_with(query=query, top_k=3, score_threshold=0.8)
        mock_generator.generate.assert_called_once_with(query=query, context_chunks=[chunk])
        mock_citation_handler.parse_citations.assert_called_once_with(
            answer_text="This is generated response [1].", context_chunks=[chunk]
        )

        assert response.answer == "This is generated response [1]."
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == chunk.chunk_id

    def test_orchestrator_empty_query_rejected(self, orchestrator):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrator.query("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrator.query("   ")

    def test_orchestrator_empty_retrieval_short_circuits(
        self, orchestrator, mock_retriever, mock_generator
    ):
        query = "Unanswerable query"
        mock_retriever.retrieve.return_value = []

        response = orchestrator.query(query)

        # Generator must NOT be called on empty retrieval
        mock_generator.generate.assert_not_called()
        assert response.answer == "I cannot answer this question based on the provided documents."
        assert response.citations == []

    def test_orchestrator_retrieval_failure_propagates(self, orchestrator, mock_retriever):
        mock_retriever.retrieve.side_effect = RetrievalError("Qdrant unavailable")
        with pytest.raises(RetrievalError, match="Qdrant unavailable"):
            orchestrator.query("query")

    def test_orchestrator_generation_failure_propagates(
        self, orchestrator, mock_retriever, mock_generator
    ):
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk=Chunk(
                    document_id=uuid4(),
                    document_name="t.txt",
                    text="text"
                ),
                score=0.9
            )
        ]
        mock_generator.generate.side_effect = GenerationError("Gemini API key invalid")
        
        with pytest.raises(GenerationError, match="Gemini API key invalid"):
            orchestrator.query("query")

    def test_orchestrator_citation_failure_propagates(
        self, orchestrator, mock_retriever, mock_generator, mock_citation_handler
    ):
        mock_retriever.retrieve.return_value = [
            RetrievalResult(
                chunk=Chunk(
                    document_id=uuid4(),
                    document_name="t.txt",
                    text="text"
                ),
                score=0.9
            )
        ]
        mock_generator.generate.return_value = "Answer [999]"
        mock_citation_handler.parse_citations.side_effect = CitationError("Invalid index")

        with pytest.raises(CitationError, match="Invalid index"):
            orchestrator.query("query")
