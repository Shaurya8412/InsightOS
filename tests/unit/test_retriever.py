"""
Unit tests for the RAG Retriever (src/services/rag/retriever.py).

All tests are fully mocked to avoid dependencies on real Gemini or Qdrant endpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.core.exceptions import EmbeddingError, RetrievalError, VectorStoreError
from src.models.schemas import Chunk, VectorRecord
from src.services.embeddings.provider import EmbeddingProvider
from src.services.rag.retriever import Retriever
from src.services.vector_store.provider import VectorStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_provider():
    return MagicMock(spec=EmbeddingProvider)


@pytest.fixture
def mock_vector_store():
    return MagicMock(spec=VectorStore)


@pytest.fixture
def retriever(mock_embedding_provider, mock_vector_store):
    return Retriever(
        embedding_provider=mock_embedding_provider,
        vector_store=mock_vector_store
    )


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestRetrieverValidation:

    def test_empty_query_rejected(self, retriever):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("")

    def test_whitespace_query_rejected(self, retriever):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("   ")

    def test_non_string_query_rejected(self, retriever):
        with pytest.raises(ValueError, match="Query must be a string"):
            retriever.retrieve(123)  # type: ignore

    def test_invalid_top_k_rejected(self, retriever):
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retriever.retrieve("valid query", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retriever.retrieve("valid query", top_k=-5)

    def test_invalid_score_threshold_rejected(self, retriever):
        with pytest.raises(ValueError, match="score_threshold must be a float or integer"):
            retriever.retrieve("valid query", score_threshold="invalid")  # type: ignore


# ---------------------------------------------------------------------------
# Retrieval Flow Tests
# ---------------------------------------------------------------------------

class TestRetrieverFlow:

    def test_successful_retrieval(self, retriever, mock_embedding_provider, mock_vector_store):
        query = "What is RAG?"
        query_vector = [0.1, 0.2, 0.3]
        mock_embedding_provider.embed_query.return_value = query_vector

        chunk_id = uuid4()
        doc_id = uuid4()
        payload = {
            "document_id": str(doc_id),
            "document_name": "rag.txt",
            "page_number": None,
            "chunk_id": str(chunk_id),
            "text": "RAG stands for Retrieval-Augmented Generation.",
            "source_location": "rag.txt"
        }

        mock_record = VectorRecord(
            id=chunk_id,
            vector=[0.5, 0.5, 0.5],
            payload=payload,
            score=0.91
        )
        mock_vector_store.search.return_value = [mock_record]

        results = retriever.retrieve(query, top_k=3, score_threshold=0.8)

        # Assertions
        mock_embedding_provider.embed_query.assert_called_once_with(query)
        mock_vector_store.search.assert_called_once_with(
            query_vector=query_vector,
            top_k=3,
            score_threshold=0.8
        )

        assert len(results) == 1
        assert results[0].score == 0.91
        assert results[0].chunk.text == payload["text"]
        assert results[0].chunk.document_name == "rag.txt"
        assert results[0].chunk.chunk_id == chunk_id

    def test_retriever_preserves_score_ordering(self, retriever, mock_embedding_provider, mock_vector_store):
        mock_embedding_provider.embed_query.return_value = [0.1]
        id1, id2 = uuid4(), uuid4()
        payload = {
            "document_id": str(uuid4()),
            "document_name": "doc.pdf",
            "page_number": 1,
            "chunk_id": str(id1),
            "text": "text 1",
            "source_location": "Page 1"
        }
        
        r1 = VectorRecord(id=id1, vector=[0.1], payload=payload, score=0.98)
        
        payload2 = payload.copy()
        payload2["chunk_id"] = str(id2)
        payload2["text"] = "text 2"
        r2 = VectorRecord(id=id2, vector=[0.1], payload=payload2, score=0.82)

        # Simulating store search returning sorted list
        mock_vector_store.search.return_value = [r1, r2]

        results = retriever.retrieve("query")

        assert len(results) == 2
        assert results[0].score == 0.98
        assert results[1].score == 0.82
        assert results[0].chunk.chunk_id == id1
        assert results[1].chunk.chunk_id == id2

    def test_empty_search_returns_empty_list(self, retriever, mock_embedding_provider, mock_vector_store):
        mock_embedding_provider.embed_query.return_value = [0.1]
        mock_vector_store.search.return_value = []

        results = retriever.retrieve("query")

        assert results == []


# ---------------------------------------------------------------------------
# Error & Failure Tests
# ---------------------------------------------------------------------------

class TestRetrieverErrors:

    def test_embedding_failure_converted_to_retrieval_error(
        self, retriever, mock_embedding_provider
    ):
        mock_embedding_provider.embed_query.side_effect = EmbeddingError("API key limit exceeded")
        with pytest.raises(RetrievalError, match="Failed to generate query embedding"):
            retriever.retrieve("query")

    def test_vector_store_failure_converted_to_retrieval_error(
        self, retriever, mock_embedding_provider, mock_vector_store
    ):
        mock_embedding_provider.embed_query.return_value = [0.1]
        mock_vector_store.search.side_effect = VectorStoreError("Qdrant collection not found")

        with pytest.raises(RetrievalError, match="Vector store search failed"):
            retriever.retrieve("query")

    def test_malformed_vector_record_payload_raises_retrieval_error(
        self, retriever, mock_embedding_provider, mock_vector_store
    ):
        mock_embedding_provider.embed_query.return_value = [0.1]
        # Payload is missing required chunk_id field
        bad_payload = {
            "document_id": str(uuid4()),
            "document_name": "t.txt",
            "text": "text"
        }
        bad_record = VectorRecord(id=uuid4(), vector=[0.1], payload=bad_payload, score=0.9)
        mock_vector_store.search.return_value = [bad_record]

        with pytest.raises(RetrievalError, match="missing required field"):
            retriever.retrieve("query")

    def test_missing_score_in_vector_record_raises_retrieval_error(
        self, retriever, mock_embedding_provider, mock_vector_store
    ):
        mock_embedding_provider.embed_query.return_value = [0.1]
        chunk_id = uuid4()
        payload = {
            "document_id": str(uuid4()),
            "document_name": "doc.pdf",
            "page_number": 1,
            "chunk_id": str(chunk_id),
            "text": "text 1",
            "source_location": "Page 1"
        }
        bad_record = VectorRecord(id=chunk_id, vector=[0.1], payload=payload, score=None)
        mock_vector_store.search.return_value = [bad_record]

        with pytest.raises(RetrievalError, match="missing or invalid similarity score"):
            retriever.retrieve("query")
