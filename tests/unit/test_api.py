"""
Unit tests for InsightOS FastAPI endpoints (src/api/routes.py).
Mocks all underlying services to run without external dependencies (Qdrant, Gemini, network).
Uses an isolated in-memory SQLite database session for endpoint calls.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, call, patch
from uuid import uuid4, UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.core.database import Base, get_db
from src.core.exceptions import (
    CitationError,
    DocumentExtractionError,
    EmbeddingError,
    GenerationError,
    RetrievalError,
    VectorStoreError,
)
from src.main import app
from src.models.db_models import Document
from src.models.schemas import Chunk, Citation, PageContent, QueryResponse

client = TestClient(app)


# ---------------------------------------------------------------------------
# Database Session Override Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(name="db_session", autouse=True)
def fixture_db_session():
    """
    Creates an isolated in-memory SQLite database, configures tables,
    and overrides the FastAPI get_db dependency.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    def _override_get_db():
        try:
            yield session
        finally:
            pass
            
    app.dependency_overrides[get_db] = _override_get_db
    
    yield session
    
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Health Endpoint Tests
# ---------------------------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Document Listing Endpoint Tests
# ---------------------------------------------------------------------------

class TestApiGetDocuments:

    def test_get_documents_empty(self, db_session):
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_documents_ordered_newest_first(self, db_session):
        from datetime import datetime, timedelta
        
        # Insert multiple documents with different uploaded_at stamps
        doc1 = Document(
            document_id=uuid4(),
            filename="older.pdf",
            status="indexed",
            chunk_count=3,
            file_size=1024,
            uploaded_at=datetime.utcnow() - timedelta(hours=2)
        )
        doc2 = Document(
            document_id=uuid4(),
            filename="newer.pdf",
            status="pending",
            chunk_count=0,
            file_size=2048,
            uploaded_at=datetime.utcnow()
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        # Call GET endpoint
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        
        res_json = response.json()
        assert len(res_json) == 2
        
        # Verify ordering (newest first)
        assert res_json[0]["filename"] == "newer.pdf"
        assert res_json[0]["status"] == "pending"
        assert res_json[1]["filename"] == "older.pdf"
        assert res_json[1]["status"] == "indexed"
        
        # Verify schema field presence
        for item in res_json:
            assert "document_id" in item
            assert "filename" in item
            assert "status" in item
            assert "chunk_count" in item
            assert "file_size" in item
            assert "uploaded_at" in item


# ---------------------------------------------------------------------------
# Upload Endpoint Tests
# ---------------------------------------------------------------------------

class TestApiUpload:

    @patch("src.api.routes.get_vector_store")
    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.chunk_pages")
    @patch("src.api.routes.parse_document")
    def test_upload_success(
        self, mock_parse, mock_chunk, mock_embed, mock_vector_store, db_session
    ):
        # 1. Mock service behaviors
        mock_pages = [PageContent(page_number=1, text="extracted text")]
        mock_parse.return_value = mock_pages

        chunk_id = uuid4()
        doc_id = uuid4()
        mock_chunk_obj = Chunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            document_name="test.pdf",
            text="extracted text",
            page_number=1,
            source_location="Page 1"
        )
        mock_chunk.return_value = [mock_chunk_obj]

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.embed_texts.return_value = [[0.1, 0.2]]
        mock_embed.return_value = mock_embedding_provider

        mock_store = MagicMock()
        mock_vector_store.return_value = mock_store

        # 2. Call upload endpoint
        file_content = b"PDF dummy content"
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", file_content, "application/pdf")},
        )

        # 3. HTTP Assertions
        assert response.status_code == 201
        res_json = response.json()
        assert "document_id" in res_json
        assert res_json["status"] == "success"
        assert res_json["chunks_indexed"] == 1

        # 4. Database Assertions (Lifecycle verification)
        resp_doc_id = UUID(res_json["document_id"])
        db_doc = db_session.query(Document).filter_by(document_id=resp_doc_id).first()
        assert db_doc is not None
        assert db_doc.filename == "test.pdf"
        assert db_doc.file_size == len(file_content)
        assert db_doc.status == "indexed"
        assert db_doc.chunk_count == 1

        # 5. Service call ordering verification (Task 5)
        assert mock_parse.call_count == 1
        assert mock_chunk.call_count == 1
        assert mock_embed.call_count == 1
        assert mock_vector_store.call_count == 1
        assert mock_store.upsert_chunks.call_count == 1

        # Verify sequential arguments passed down the pipeline
        mock_parse.assert_called_once_with(file_content, "test.pdf")
        
        mock_chunk.assert_called_once()
        assert mock_chunk.call_args[0][0] == mock_pages
        # The document ID generated by the API must be the one passed to chunk_pages
        assert mock_chunk.call_args[0][1] == resp_doc_id
        assert mock_chunk.call_args[0][2] == "test.pdf"
        
        mock_embedding_provider.embed_texts.assert_called_once_with(["extracted text"])
        mock_store.upsert_chunks.assert_called_once_with([mock_chunk_obj], [[0.1, 0.2]])

    def test_upload_unsupported_file_extension(self, db_session):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.png", b"png bytes", "image/png")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
        
        # Verify no database record was created
        assert db_session.query(Document).count() == 0

    def test_upload_empty_file_rejected(self, db_session):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400
        assert "Uploaded file is empty" in response.json()["detail"]
        
        # Verify no database record was created
        assert db_session.query(Document).count() == 0

    @patch("src.api.routes.parse_document")
    def test_upload_extraction_error_mapped_to_422(self, mock_parse, db_session):
        mock_parse.side_effect = DocumentExtractionError("Corrupted PDF format")
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", b"corrupted bytes", "application/pdf")},
        )
        assert response.status_code == 422
        assert "Corrupted PDF format" in response.json()["detail"]

        # Verify failed status persistence
        assert db_session.query(Document).count() == 1
        db_doc = db_session.query(Document).first()
        assert db_doc.status == "failed"
        assert db_doc.filename == "test.pdf"
        assert db_doc.file_size == len(b"corrupted bytes")

    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.chunk_pages")
    @patch("src.api.routes.parse_document")
    def test_upload_embedding_error_mapped_to_502(
        self, mock_parse, mock_chunk, mock_embed, db_session
    ):
        mock_parse.return_value = [PageContent(page_number=1, text="t")]
        mock_chunk.return_value = [
            Chunk(document_id=uuid4(), document_name="test.pdf", text="t")
        ]
        
        mock_provider = MagicMock()
        mock_provider.embed_texts.side_effect = EmbeddingError("Quota exceeded")
        mock_embed.return_value = mock_provider

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", b"pdf bytes", "application/pdf")},
        )
        assert response.status_code == 502
        assert "Quota exceeded" in response.json()["detail"]

        # Verify failed status persistence
        assert db_session.query(Document).count() == 1
        db_doc = db_session.query(Document).first()
        assert db_doc.status == "failed"
        assert db_doc.chunk_count == 0

    @patch("src.api.routes.get_vector_store")
    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.chunk_pages")
    @patch("src.api.routes.parse_document")
    def test_upload_vector_store_error_mapped_to_500(
        self, mock_parse, mock_chunk, mock_embed, mock_vector_store, db_session
    ):
        mock_parse.return_value = [PageContent(page_number=1, text="t")]
        mock_chunk.return_value = [
            Chunk(document_id=uuid4(), document_name="test.pdf", text="t")
        ]
        mock_provider = MagicMock()
        mock_provider.embed_texts.return_value = [[0.1]]
        mock_embed.return_value = mock_provider

        mock_store = MagicMock()
        mock_store.upsert_chunks.side_effect = VectorStoreError("Qdrant offline")
        mock_vector_store.return_value = mock_store

        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", b"pdf bytes", "application/pdf")},
        )
        assert response.status_code == 500
        assert "Qdrant offline" in response.json()["detail"]

        # Verify failed status persistence
        assert db_session.query(Document).count() == 1
        db_doc = db_session.query(Document).first()
        assert db_doc.status == "failed"


# ---------------------------------------------------------------------------
# Query Endpoint Tests
# ---------------------------------------------------------------------------

class TestApiQuery:

    @patch("src.api.routes.get_vector_store")
    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.Generator")
    @patch("src.api.routes.Orchestrator")
    def test_query_success(
        self, mock_orchestrator_cls, mock_gen_cls, mock_embed, mock_vector_store
    ):
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch

        expected_response = QueryResponse(
            answer="This is a test answer.",
            citations=[
                Citation(
                    chunk_id=uuid4(),
                    document_id=uuid4(),
                    document_name="test.pdf",
                    page_number=1,
                    source_location="Page 1",
                    snippet="Snippet text"
                )
            ]
        )
        mock_orch.query.return_value = expected_response

        # Execute POST /query
        response = client.post(
            "/api/v1/query",
            json={"query": "What is AI?", "top_k": 3}
        )

        assert response.status_code == 200
        res_json = response.json()
        assert res_json["answer"] == "This is a test answer."
        assert len(res_json["citations"]) == 1
        assert res_json["citations"][0]["document_name"] == "test.pdf"

        # Verify orchestrator interaction (Task 5)
        mock_orch.query.assert_called_once_with("What is AI?", top_k=3)

    def test_query_empty_query_rejected(self):
        # Empty query validation handled by Pydantic validation on QueryRequest
        response = client.post(
            "/api/v1/query",
            json={"query": "", "top_k": 3}
        )
        assert response.status_code == 422  # Pydantic validates before routing

    def test_query_whitespace_query_rejected(self):
        response = client.post(
            "/api/v1/query",
            json={"query": "   ", "top_k": 3}
        )
        assert response.status_code == 422

    @patch("src.api.routes.get_vector_store")
    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.Generator")
    @patch("src.api.routes.Orchestrator")
    def test_query_retrieval_error_mapped_to_500(
        self, mock_orchestrator_cls, mock_gen_cls, mock_embed, mock_vector_store
    ):
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.query.side_effect = RetrievalError("Qdrant collection index failure")

        response = client.post(
            "/api/v1/query",
            json={"query": "RAG query", "top_k": 5}
        )
        assert response.status_code == 500
        assert "Qdrant collection index failure" in response.json()["detail"]

    @patch("src.api.routes.get_vector_store")
    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.Generator")
    @patch("src.api.routes.Orchestrator")
    def test_query_generation_error_mapped_to_502(
        self, mock_orchestrator_cls, mock_gen_cls, mock_embed, mock_vector_store
    ):
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.query.side_effect = GenerationError("Gemini credentials expired")

        response = client.post(
            "/api/v1/query",
            json={"query": "RAG query", "top_k": 5}
        )
        assert response.status_code == 502
        assert "Gemini credentials expired" in response.json()["detail"]

    @patch("src.api.routes.get_vector_store")
    @patch("src.api.routes.get_embedding_provider")
    @patch("src.api.routes.Generator")
    @patch("src.api.routes.Orchestrator")
    def test_query_citation_error_mapped_to_500(
        self, mock_orchestrator_cls, mock_gen_cls, mock_embed, mock_vector_store
    ):
        mock_orch = MagicMock()
        mock_orchestrator_cls.return_value = mock_orch
        mock_orch.query.side_effect = CitationError("Invalid index mapped")

        response = client.post(
            "/api/v1/query",
            json={"query": "RAG query", "top_k": 5}
        )
        assert response.status_code == 500
        assert "Invalid index mapped" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Document Deletion Endpoint Tests
# ---------------------------------------------------------------------------

class TestApiDelete:

    @patch("src.api.routes.get_vector_store")
    def test_delete_document_success(self, mock_get_store, db_session):
        # 1. Insert a document into the SQLite database
        doc_id = uuid4()
        doc = Document(
            document_id=doc_id,
            filename="delete_test.pdf",
            status="indexed",
            chunk_count=3,
            file_size=1024,
        )
        db_session.add(doc)
        db_session.commit()

        # Mock vector store delete call
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # 2. Call DELETE endpoint
        response = client.delete(f"/api/v1/documents/{doc_id}")

        # 3. Assertions
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["status"] == "success"
        assert str(doc_id) in res_json["detail"]

        # Verify Qdrant deletion was called with the document ID
        mock_store.delete.assert_called_once_with(doc_id)

        # Verify document was deleted from SQLite
        db_doc = db_session.query(Document).filter_by(document_id=doc_id).first()
        assert db_doc is None

    @patch("src.api.routes.get_vector_store")
    def test_delete_non_existent_document_returns_404(self, mock_get_store, db_session):
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        doc_id = uuid4()
        response = client.delete(f"/api/v1/documents/{doc_id}")

        assert response.status_code == 404
        assert f"Document with ID '{doc_id}' not found" in response.json()["detail"]

        # Verify Qdrant delete was NOT called
        mock_store.delete.assert_not_called()

    @patch("src.api.routes.get_vector_store")
    def test_delete_qdrant_failure_rolls_back_sqlite_deletion(self, mock_get_store, db_session):
        # 1. Insert a document into SQLite
        doc_id = uuid4()
        doc = Document(
            document_id=doc_id,
            filename="failed_delete_test.pdf",
            status="indexed",
            chunk_count=5,
            file_size=5000,
        )
        db_session.add(doc)
        db_session.commit()

        # Mock vector store delete call to throw VectorStoreError
        mock_store = MagicMock()
        mock_store.delete.side_effect = VectorStoreError("Connection refused by Qdrant")
        mock_get_store.return_value = mock_store

        # 2. Call DELETE endpoint
        response = client.delete(f"/api/v1/documents/{doc_id}")

        # 3. Assertions
        assert response.status_code == 500
        assert "Connection refused by Qdrant" in response.json()["detail"]

        # Verify document still exists in SQLite (rollback occurred/not deleted)
        db_doc = db_session.query(Document).filter_by(document_id=doc_id).first()
        assert db_doc is not None
        assert db_doc.status == "indexed"

