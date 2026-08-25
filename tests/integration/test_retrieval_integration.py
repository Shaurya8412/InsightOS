"""
Real Retrieval Integration Tests: Verify the end-to-end embedding, storage,
and retrieval pipeline using real Gemini API and local Qdrant instance.

These tests are skipped automatically if GEMINI_API_KEY is not configured
or if local Qdrant is unreachable.
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from google.genai.errors import ClientError as GeminiClientError

from src.core.config import settings
from src.core.exceptions import EmbeddingError, RetrievalError
from src.models.schemas import Chunk, RetrievalResult
from src.services.embeddings.provider import GeminiEmbeddingProvider
from src.services.rag.retriever import Retriever
from src.services.vector_store.qdrant import QdrantVectorStore

# ---------------------------------------------------------------------------
# Helpers to probe external service availability
# ---------------------------------------------------------------------------

def qdrant_available() -> bool:
    """Return True if Qdrant is running and reachable."""
    try:
        url = urlparse(settings.QDRANT_URL)
        host = url.hostname or "localhost"
        port = url.port or 6333
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


def gemini_available() -> bool:
    """Return True if GEMINI_API_KEY env var is set."""
    # Check both environment variable and settings loaded config
    key = os.environ.get("GEMINI_API_KEY", "").strip() or settings.GEMINI_API_KEY.strip()
    return bool(key)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup_services():
    """
    Sets up real services, creating a temporary test collection.
    Cleans up (deletes) the collection afterwards.
    """
    if not gemini_available():
        pytest.skip("GEMINI_API_KEY not configured — skipping live integration tests")
    if not qdrant_available():
        pytest.skip("Local Qdrant not reachable on port 6333 — skipping live integration tests")

    # Use a dedicated, temporary collection name for integration tests
    test_collection = f"{settings.QDRANT_COLLECTION_NAME}_retrieval_integration_test"
    
    # Temporarily override QDRANT_COLLECTION_NAME in settings
    original_collection = settings.QDRANT_COLLECTION_NAME
    settings.QDRANT_COLLECTION_NAME = test_collection

    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or settings.GEMINI_API_KEY.strip()

    try:
        provider = GeminiEmbeddingProvider(api_key=api_key)
        store = QdrantVectorStore()
    except EmbeddingError as exc:
        # If it failed due to invalid credentials, skip test gracefully instead of failing
        msg = str(exc).lower()
        if "api key not valid" in msg or "invalid_argument" in msg or "client error" in msg:
            settings.QDRANT_COLLECTION_NAME = original_collection
            pytest.skip(f"GEMINI_API_KEY is configured but invalid/unauthorized: {exc}")
        raise
    except Exception as exc:
        settings.QDRANT_COLLECTION_NAME = original_collection
        raise

    yield provider, store

    # Cleanup: Delete the temporary collection in Qdrant
    try:
        store.client.delete_collection(test_collection)
    except Exception as e:
        print(f"Failed to clean up integration test collection '{test_collection}': {e}")
    finally:
        settings.QDRANT_COLLECTION_NAME = original_collection


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRetrievalIntegrationPipeline:

    def test_integration_flow_a_real_embedding(self, setup_services):
        """Test A: Verify real GeminiEmbeddingProvider can fetch vectors."""
        provider, _ = setup_services
        
        query = "Verify end-to-end integration of retrieval"
        vector = provider.embed_query(query)
        
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(val, float) for val in vector)
        # Verify dimension is positive and numeric
        assert len(vector) >= 256, f"Expected production embedding dimension >= 256, got {len(vector)}"

    def test_integration_flow_b_real_vector_storage(self, setup_services):
        """Test B: Verify records can be stored and retrieved by ID in Qdrant."""
        provider, store = setup_services

        doc_id = uuid4()
        c1 = Chunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="python_doc.txt",
            text="Python is a popular programming language known for its readability and clean syntax.",
            page_number=None,
            source_location="python_doc.txt"
        )
        c2 = Chunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="db_doc.txt",
            text="Databases are used to store and manage structured data efficiently, often using SQL.",
            page_number=None,
            source_location="db_doc.txt"
        )
        c3 = Chunk(
            chunk_id=uuid4(),
            document_id=doc_id,
            document_name="ml_doc.txt",
            text="Machine learning involves training algorithms on data to make predictions or decisions.",
            page_number=None,
            source_location="ml_doc.txt"
        )

        chunks = [c1, c2, c3]
        
        # Embed real text via real provider
        embeddings = provider.embed_texts([c.text for c in chunks])
        assert len(embeddings) == 3

        # Upsert chunks and embeddings into Qdrant
        store.upsert_chunks(chunks, embeddings)

        # Get point by ID and verify
        rec = store.get(c1.chunk_id)
        assert rec is not None
        assert rec.id == c1.chunk_id
        assert rec.payload["text"] == c1.text
        assert rec.payload["document_name"] == "python_doc.txt"

    def test_integration_flow_c_real_retrieval(self, setup_services):
        """Test C: Verify Retriever resolves queries correctly against Qdrant."""
        provider, store = setup_services
        retriever = Retriever(embedding_provider=provider, vector_store=store)

        query = "Tell me about coding in Python and writing software"
        results = retriever.retrieve(query, top_k=2)

        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].score is not None
        assert isinstance(results[0].score, float)
        
        # Confirm that the Python document ranks higher than others
        best_match = results[0].chunk
        assert best_match.document_name == "python_doc.txt"
        assert "readability" in best_match.text

    def test_integration_flow_d_no_result_threshold_behavior(self, setup_services):
        """Test D: Verify threshold filters out low-relevance results correctly."""
        provider, store = setup_services
        retriever = Retriever(embedding_provider=provider, vector_store=store)

        query = "Cooking pasta carbonara recipes"
        # Highly strict threshold should filter out unrelated docs (Python, database, ML)
        results = retriever.retrieve(query, top_k=5, score_threshold=0.95)
        
        assert results == [], f"Expected no results for unrelated query with high threshold, got: {results}"
