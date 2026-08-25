"""
Unit and Integration tests for src/services/vector_store/qdrant.py

All unit tests mock QdrantClient to avoid external dependencies.
The integration test at the bottom runs only when a local Qdrant instance is reachable.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.models.schemas import Chunk, VectorRecord
from src.services.vector_store.provider import get_vector_store
from src.services.vector_store.qdrant import QdrantVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_DIMENSION = 128


def qdrant_available() -> bool:
    """Check if Qdrant server is reachable for integration tests."""
    try:
        url = urlparse(settings.QDRANT_URL)
        host = url.hostname or "localhost"
        port = url.port or 6333
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_client():
    """Create a mock QdrantClient."""
    mock_client = MagicMock()
    # Default to collection not existing
    mock_client.collection_exists.return_value = False
    return mock_client


@pytest.fixture
def mock_embed_provider():
    """Mock the embedding provider to return a deterministic dimension."""
    with patch("src.services.embeddings.provider.get_embedding_provider") as mock_get:
        mock_prov = MagicMock()
        mock_prov.embed_query.return_value = [0.1] * MOCK_DIMENSION
        mock_get.return_value = mock_prov
        yield mock_prov


@pytest.fixture
def store(mock_qdrant_client, mock_embed_provider):
    """Return QdrantVectorStore initialized with a mocked Qdrant client."""
    return QdrantVectorStore(client=mock_qdrant_client)


# ---------------------------------------------------------------------------
# Configuration & Collection Tests
# ---------------------------------------------------------------------------

class TestQdrantCollectionManagement:

    def test_factory_returns_qdrant_vector_store(self, mock_embed_provider):
        with patch("src.services.vector_store.qdrant.QdrantClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.collection_exists.return_value = False
            vs = get_vector_store()
            assert isinstance(vs, QdrantVectorStore)

    def test_collection_created_when_missing(self, mock_qdrant_client, mock_embed_provider):
        mock_qdrant_client.collection_exists.return_value = False

        store = QdrantVectorStore(client=mock_qdrant_client)

        mock_qdrant_client.collection_exists.assert_called_once_with(store.collection_name)
        mock_qdrant_client.create_collection.assert_called_once()
        assert store._dimension == MOCK_DIMENSION

    def test_existing_collection_reused_without_recreation(self, mock_qdrant_client, mock_embed_provider):
        mock_qdrant_client.collection_exists.return_value = True

        # Mock get_collection response to return matching size
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = MOCK_DIMENSION
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        store = QdrantVectorStore(client=mock_qdrant_client)

        mock_qdrant_client.collection_exists.assert_called_once_with(store.collection_name)
        mock_qdrant_client.create_collection.assert_not_called()
        mock_qdrant_client.get_collection.assert_called_once_with(store.collection_name)
        assert store._dimension == MOCK_DIMENSION

    def test_dimension_mismatch_raises_vector_store_error(self, mock_qdrant_client, mock_embed_provider):
        mock_qdrant_client.collection_exists.return_value = True

        # Mock get_collection response to return a different size (e.g. 768 instead of 128)
        mock_collection_info = MagicMock()
        mock_collection_info.config.params.vectors.size = 768
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        with pytest.raises(VectorStoreError, match="dimension 768, but configured embedding model expects 128"):
            QdrantVectorStore(client=mock_qdrant_client)

        # Ensure we never recreate or delete it on mismatch
        mock_qdrant_client.create_collection.assert_not_called()
        mock_qdrant_client.delete_collection.assert_not_called()


# ---------------------------------------------------------------------------
# Upsert Tests
# ---------------------------------------------------------------------------

class TestQdrantUpsert:

    def test_upsert_empty_batch_does_nothing(self, store, mock_qdrant_client):
        store.upsert([])
        mock_qdrant_client.upsert.assert_not_called()

    def test_upsert_single_point(self, store, mock_qdrant_client):
        doc_id = uuid4()
        chunk_id = uuid4()
        record = VectorRecord(
            id=chunk_id,
            vector=[0.5] * MOCK_DIMENSION,
            payload={
                "document_id": str(doc_id),
                "document_name": "doc.pdf",
                "page_number": 1,
                "chunk_id": str(chunk_id),
                "text": "sample text",
                "source_location": "Page 1"
            }
        )

        store.upsert([record])

        # Assert upsert called with PointStruct mapping ID and payload
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args[1]
        assert call_args["collection_name"] == store.collection_name
        points = call_args["points"]
        assert len(points) == 1
        assert points[0].id == str(chunk_id)
        assert points[0].vector == record.vector
        assert points[0].payload["text"] == "sample text"

    def test_upsert_dimension_mismatch_rejected(self, store):
        record = VectorRecord(
            id=uuid4(),
            vector=[0.1] * (MOCK_DIMENSION + 5),  # Mismatch size
            payload={}
        )

        with pytest.raises(VectorStoreError, match="Vector dimension mismatch"):
            store.upsert([record])

    def test_upsert_missing_payload_field_rejected(self, store):
        record = VectorRecord(
            id=uuid4(),
            vector=[0.1] * MOCK_DIMENSION,
            payload={
                "document_id": str(uuid4()),
                # document_name is missing
                "page_number": 1,
                "chunk_id": str(uuid4()),
                "text": "test",
                "source_location": "Page 1"
            }
        )

        with pytest.raises(VectorStoreError, match="Missing required payload field"):
            store.upsert([record])

    def test_upsert_invalid_payload_types_rejected(self, store):
        record = VectorRecord(
            id=uuid4(),
            vector=[0.1] * MOCK_DIMENSION,
            payload={
                "document_id": "not-a-uuid",  # invalid UUID
                "document_name": "doc.txt",
                "page_number": "not-an-int",  # invalid int
                "chunk_id": str(uuid4()),
                "text": "test",
                "source_location": "Page 1"
            }
        )

        with pytest.raises(VectorStoreError, match="Invalid payload format"):
            store.upsert([record])

    def test_upsert_chunks_helper_length_validation(self, store):
        chunks = [
            Chunk(
                document_id=uuid4(),
                document_name="t.txt",
                text="chunk 1"
            )
        ]
        embeddings = []  # length mismatch

        with pytest.raises(VectorStoreError, match="Mismatched input lengths"):
            store.upsert_chunks(chunks, embeddings)

    def test_upsert_chunks_helper_correct_mapping(self, store, mock_qdrant_client):
        doc_id = uuid4()
        chunk = Chunk(
            document_id=doc_id,
            document_name="t.txt",
            text="chunk 1",
            page_number=None,
            source_location="t.txt"
        )
        embeddings = [[0.1] * MOCK_DIMENSION]

        store.upsert_chunks([chunk], embeddings)

        mock_qdrant_client.upsert.assert_called_once()
        points = mock_qdrant_client.upsert.call_args[1]["points"]
        assert len(points) == 1
        assert points[0].id == str(chunk.chunk_id)
        assert points[0].vector == embeddings[0]
        assert points[0].payload["document_id"] == str(doc_id)
        assert points[0].payload["chunk_id"] == str(chunk.chunk_id)


# ---------------------------------------------------------------------------
# Get Tests
# ---------------------------------------------------------------------------

class TestQdrantGet:

    def test_get_existing_point(self, store, mock_qdrant_client):
        chunk_id = uuid4()
        doc_id = uuid4()
        payload = {
            "document_id": str(doc_id),
            "document_name": "t.txt",
            "page_number": None,
            "chunk_id": str(chunk_id),
            "text": "hello world",
            "source_location": "t.txt"
        }

        # Mock retrieve result
        mock_point = MagicMock()
        mock_point.id = str(chunk_id)
        mock_point.vector = [0.2] * MOCK_DIMENSION
        mock_point.payload = payload
        mock_qdrant_client.retrieve.return_value = [mock_point]

        res = store.get(chunk_id)

        assert res is not None
        assert res.id == chunk_id
        assert res.vector == [0.2] * MOCK_DIMENSION
        assert res.payload["text"] == "hello world"

    def test_get_missing_point_returns_none(self, store, mock_qdrant_client):
        mock_qdrant_client.retrieve.return_value = []
        res = store.get(uuid4())
        assert res is None

    def test_get_malformed_payload_raises_vector_store_error(self, store, mock_qdrant_client):
        chunk_id = uuid4()
        # Payload missing critical text field
        payload = {
            "document_id": str(uuid4()),
            "document_name": "t.txt",
            "chunk_id": str(chunk_id)
        }

        mock_point = MagicMock()
        mock_point.id = str(chunk_id)
        mock_point.vector = [0.2] * MOCK_DIMENSION
        mock_point.payload = payload
        mock_qdrant_client.retrieve.return_value = [mock_point]

        with pytest.raises(VectorStoreError, match="missing payload field"):
            store.get(chunk_id)


# ---------------------------------------------------------------------------
# Delete Tests
# ---------------------------------------------------------------------------

class TestQdrantDelete:

    def test_delete_document_calls_client_delete_with_filter(self, store, mock_qdrant_client):
        from qdrant_client import models as q_models
        document_id = uuid4()
        
        store.delete(document_id)
        
        mock_qdrant_client.delete.assert_called_once()
        call_args = mock_qdrant_client.delete.call_args[1]
        
        assert call_args["collection_name"] == store.collection_name
        
        # Verify Filter structure matches document_id MatchValue
        selector = call_args["points_selector"]
        assert isinstance(selector, q_models.Filter)
        assert len(selector.must) == 1
        condition = selector.must[0]
        assert isinstance(condition, q_models.FieldCondition)
        assert condition.key == "document_id"
        assert condition.match.value == str(document_id)

    def test_delete_exception_converted_to_vector_store_error(self, store, mock_qdrant_client):
        mock_qdrant_client.delete.side_effect = Exception("API error")
        
        with pytest.raises(VectorStoreError, match="Failed to delete points for document"):
            store.delete(uuid4())


# ---------------------------------------------------------------------------
# Failure & Error Handling Tests
# ---------------------------------------------------------------------------

class TestQdrantFailureHandling:

    def test_client_upsert_exception_converted_to_vector_store_error(self, store, mock_qdrant_client):
        mock_qdrant_client.upsert.side_effect = UnexpectedResponse(500, "Internal Server Error", b"error content", {})
        record = VectorRecord(
            id=uuid4(),
            vector=[0.1] * MOCK_DIMENSION,
            payload={
                "document_id": str(uuid4()),
                "document_name": "doc.pdf",
                "page_number": 1,
                "chunk_id": str(uuid4()),
                "text": "test",
                "source_location": "Page 1"
            }
        )

        with pytest.raises(VectorStoreError, match="Failed to upsert points"):
            store.upsert([record])

    def test_client_retrieve_exception_converted_to_vector_store_error(self, store, mock_qdrant_client):
        mock_qdrant_client.retrieve.side_effect = RuntimeError("Connection closed")
        with pytest.raises(VectorStoreError, match="Failed to retrieve point"):
            store.get(uuid4())


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------

class TestQdrantSearch:

    def test_search_successful(self, store, mock_qdrant_client):
        query_vector = [0.1] * MOCK_DIMENSION
        chunk_id = uuid4()
        doc_id = uuid4()
        payload = {
            "document_id": str(doc_id),
            "document_name": "test.txt",
            "page_number": None,
            "chunk_id": str(chunk_id),
            "text": "found it",
            "source_location": "test.txt"
        }

        # Mock query_points hit
        mock_hit = MagicMock()
        mock_hit.id = str(chunk_id)
        mock_hit.vector = [0.2] * MOCK_DIMENSION
        mock_hit.payload = payload
        mock_hit.score = 0.92
        
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_qdrant_client.query_points.return_value = mock_response

        results = store.search(query_vector, top_k=3)

        assert len(results) == 1
        assert results[0].id == chunk_id
        assert results[0].score == 0.92
        assert results[0].payload["text"] == "found it"

        mock_qdrant_client.query_points.assert_called_once_with(
            collection_name=store.collection_name,
            query=query_vector,
            limit=3,
            score_threshold=None,
            with_payload=True,
            with_vectors=True
        )

    def test_search_multiple_results_and_ordering(self, store, mock_qdrant_client):
        query_vector = [0.1] * MOCK_DIMENSION
        id1 = uuid4()
        id2 = uuid4()
        payload = {
            "document_id": str(uuid4()),
            "document_name": "test.txt",
            "page_number": None,
            "chunk_id": str(id1),
            "text": "t",
            "source_location": "test.txt"
        }

        # Create two hits
        h1 = MagicMock()
        h1.id = str(id1)
        h1.vector = [0.1] * MOCK_DIMENSION
        h1.payload = payload.copy()
        h1.score = 0.95

        h2 = MagicMock()
        h2.id = str(id2)
        h2.vector = [0.1] * MOCK_DIMENSION
        payload_h2 = payload.copy()
        payload_h2["chunk_id"] = str(id2)
        h2.payload = payload_h2
        h2.score = 0.85

        mock_response = MagicMock()
        mock_response.points = [h1, h2]
        mock_qdrant_client.query_points.return_value = mock_response

        results = store.search(query_vector, top_k=5)

        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.85

    def test_search_score_threshold_passed(self, store, mock_qdrant_client):
        query_vector = [0.1] * MOCK_DIMENSION
        mock_response = MagicMock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response

        results = store.search(query_vector, top_k=2, score_threshold=0.7)

        mock_qdrant_client.query_points.assert_called_once_with(
            collection_name=store.collection_name,
            query=query_vector,
            limit=2,
            score_threshold=0.7,
            with_payload=True,
            with_vectors=True
        )
        assert results == []

    def test_search_empty_results(self, store, mock_qdrant_client):
        mock_response = MagicMock()
        mock_response.points = []
        mock_qdrant_client.query_points.return_value = mock_response
        results = store.search([0.1] * MOCK_DIMENSION, top_k=2)
        assert results == []

    def test_search_invalid_vector_rejected(self, store):
        with pytest.raises(VectorStoreError, match="Query vector must be a non-empty list"):
            store.search([], top_k=5)
        
        with pytest.raises(VectorStoreError, match="Query vector must be a non-empty list"):
            store.search("invalid", top_k=5) # type: ignore

    def test_search_dimension_mismatch_rejected(self, store):
        with pytest.raises(VectorStoreError, match="Query vector dimension mismatch"):
            store.search([0.1] * (MOCK_DIMENSION + 3), top_k=5)

    def test_search_invalid_limit_rejected(self, store):
        with pytest.raises(VectorStoreError, match="Limit .* must be greater than zero"):
            store.search([0.1] * MOCK_DIMENSION, top_k=0)

    def test_search_client_exception_converted(self, store, mock_qdrant_client):
        mock_qdrant_client.query_points.side_effect = RuntimeError("Qdrant connection timeout")
        with pytest.raises(VectorStoreError, match="Failed to perform similarity search"):
            store.search([0.1] * MOCK_DIMENSION, top_k=3)

    def test_search_malformed_result_payload_rejected(self, store, mock_qdrant_client):
        # Payload missing required keys
        malformed_payload = {
            "text": "missing fields"
        }
        mock_hit = MagicMock()
        mock_hit.id = str(uuid4())
        mock_hit.vector = [0.1] * MOCK_DIMENSION
        mock_hit.payload = malformed_payload
        mock_hit.score = 0.8
        
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_qdrant_client.query_points.return_value = mock_response

        with pytest.raises(VectorStoreError, match="missing required field"):
            store.search([0.1] * MOCK_DIMENSION, top_k=3)

    def test_search_malformed_result_id_rejected(self, store, mock_qdrant_client):
        payload = {
            "document_id": str(uuid4()),
            "document_name": "t.txt",
            "page_number": None,
            "chunk_id": str(uuid4()),
            "text": "test",
            "source_location": "t.txt"
        }
        mock_hit = MagicMock()
        mock_hit.id = "invalid-uuid-string"
        mock_hit.vector = [0.1] * MOCK_DIMENSION
        mock_hit.payload = payload
        mock_hit.score = 0.8
        
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_qdrant_client.query_points.return_value = mock_response

        with pytest.raises(VectorStoreError, match="not a valid UUID"):
            store.search([0.1] * MOCK_DIMENSION, top_k=3)


# ---------------------------------------------------------------------------
# Conditional Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestQdrantIntegration:

    @pytest.fixture(autouse=True)
    def check_availability(self):
        if not qdrant_available():
            pytest.skip("Local Qdrant instance not reachable — skipping live integration tests")

    def test_live_qdrant_flow(self):
        # Temporarily override collection name to prevent dimension conflicts with production collection
        test_collection = "insightos_chunks_unit_integration_test"
        original_collection = settings.QDRANT_COLLECTION_NAME
        settings.QDRANT_COLLECTION_NAME = test_collection

        try:
            # We need a live embedding provider for dynamic dimension detection
            # But wait! If GEMINI_API_KEY is not set, GeminiEmbeddingProvider init will fail.
            # So we patch _detect_dimension to bypass embed call and return MOCK_DIMENSION
            # OR we check if a real API key is set.
            # To make it robust and run even without Gemini key (since Qdrant tests shouldn't require Gemini),
            # we patch the vector store's dimension detection to return 128.
            with patch.object(QdrantVectorStore, "_detect_dimension", return_value=MOCK_DIMENSION):
                store = QdrantVectorStore()

            doc_id = uuid4()
            chunk_id1 = uuid4()
            chunk_id2 = uuid4()

            c1 = Chunk(
                chunk_id=chunk_id1,
                document_id=doc_id,
                document_name="integration_test.pdf",
                text="First sentence.",
                page_number=1,
                source_location="Page 1"
            )
            c2 = Chunk(
                chunk_id=chunk_id2,
                document_id=doc_id,
                document_name="integration_test.pdf",
                text="Second sentence.",
                page_number=2,
                source_location="Page 2"
            )

            emb1 = [0.0] * MOCK_DIMENSION
            emb1[0] = 1.0
            emb2 = [0.0] * MOCK_DIMENSION
            emb2[1] = 1.0
            embs = [emb1, emb2]

            # 1. Upsert chunks
            store.upsert_chunks([c1, c2], embs)

            # 2. Get and Verify
            rec1 = store.get(chunk_id1)
            assert rec1 is not None
            assert rec1.id == chunk_id1
            assert rec1.vector == embs[0]
            assert rec1.payload["text"] == "First sentence."
            assert rec1.payload["document_name"] == "integration_test.pdf"

            # 3. Repeated upsert updates values instead of creating duplicates
            updated_text = "Updated first sentence."
            c1_updated = Chunk(
                chunk_id=chunk_id1,
                document_id=doc_id,
                document_name="integration_test.pdf",
                text=updated_text,
                page_number=1,
                source_location="Page 1"
            )
            emb_updated = [0.0] * MOCK_DIMENSION
            emb_updated[2] = 1.0
            store.upsert_chunks([c1_updated], [emb_updated])

            rec1_updated = store.get(chunk_id1)
            assert rec1_updated is not None
            assert rec1_updated.payload["text"] == updated_text
            assert rec1_updated.vector == emb_updated

            # 4. Delete document (purges both chunks)
            store.delete(doc_id)
            assert store.get(chunk_id1) is None
            assert store.get(chunk_id2) is None
        finally:
            # Cleanup the temporary collection
            try:
                if 'store' in locals():
                    store.client.delete_collection(test_collection)
            except Exception:
                pass
            settings.QDRANT_COLLECTION_NAME = original_collection
