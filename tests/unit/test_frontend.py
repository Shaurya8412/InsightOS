"""
Unit tests for the frontend API client (src/frontend/api_client.py).
Mocks backend HTTP servers using mock patches—no live endpoints or connections.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import httpx

from src.core.exceptions import CitationError
from src.frontend.api_client import APIClient, APIClientError
from src.models.schemas import Citation, QueryResponse

FAKE_API_URL = "http://localhost:9999"


@pytest.fixture
def api_client():
    return APIClient(FAKE_API_URL)


# ---------------------------------------------------------------------------
# API Client Tests
# ---------------------------------------------------------------------------

class TestAPIClient:

    @patch("src.frontend.api_client.httpx.Client")
    def test_upload_document_success(self, mock_client_cls, api_client):
        # Mock Client response
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 201
        doc_id = str(uuid4())
        mock_response.json.return_value = {
            "document_id": doc_id,
            "status": "success",
            "chunks_indexed": 3
        }
        mock_client.post.return_value = mock_response

        # Execute
        result = api_client.upload_document("hello.txt", b"file content bytes")

        # Verify
        assert result["document_id"] == doc_id
        assert result["status"] == "success"
        assert result["chunks_indexed"] == 3

        # Verify request parameters
        call_args = mock_client.post.call_args
        assert call_args[0][0] == f"{FAKE_API_URL}/api/v1/documents/upload"
        assert "file" in call_args[1]["files"]
        assert call_args[1]["files"]["file"][0] == "hello.txt"
        assert call_args[1]["files"]["file"][1] == b"file content bytes"

    @patch("src.frontend.api_client.httpx.Client")
    def test_upload_document_failure_raises_api_client_error(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.reason_phrase = "Bad Request"
        mock_response.json.return_value = {"detail": "Unsupported file type"}
        mock_client.post.return_value = mock_response

        # Execute & Assert
        with pytest.raises(APIClientError, match="Ingestion failed.*Unsupported file type"):
            api_client.upload_document("hello.png", b"png bytes")

    @patch("src.frontend.api_client.httpx.Client")
    def test_upload_document_timeout_raises_api_client_error(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.ConnectTimeout("Connection timed out")

        with pytest.raises(APIClientError, match="Failed to connect to the backend server"):
            api_client.upload_document("test.txt", b"bytes")

    @patch("src.frontend.api_client.httpx.Client")
    def test_query_rag_success(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        doc_id = uuid4()
        chunk_id = uuid4()
        
        mock_response.json.return_value = {
            "answer": "Grounded answer text",
            "citations": [
                {
                    "chunk_id": str(chunk_id),
                    "document_id": str(doc_id),
                    "document_name": "context.txt",
                    "page_number": 2,
                    "source_location": "Page 2",
                    "snippet": "Original document snippet text"
                }
            ]
        }
        mock_client.post.return_value = mock_response

        # Execute
        result = api_client.query_rag("How does RAG work?", top_k=5)

        # Verify output parsing
        assert isinstance(result, QueryResponse)
        assert result.answer == "Grounded answer text"
        assert len(result.citations) == 1
        
        citation = result.citations[0]
        assert citation.document_name == "context.txt"
        assert citation.page_number == 2
        assert citation.snippet == "Original document snippet text"

        # Verify request parameters
        call_args = mock_client.post.call_args
        assert call_args[0][0] == f"{FAKE_API_URL}/api/v1/query"
        assert call_args[1]["json"] == {"query": "How does RAG work?", "top_k": 5}

    @patch("src.frontend.api_client.httpx.Client")
    def test_query_rag_http_error_raises_api_client_error(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.reason_phrase = "Bad Gateway"
        mock_response.json.return_value = {"detail": "Gemini API failure"}
        mock_client.post.return_value = mock_response

        with pytest.raises(APIClientError, match="Query failed.*Gemini API failure"):
            api_client.query_rag("query")

    @patch("src.frontend.api_client.httpx.Client")
    def test_query_rag_malformed_json_raises_api_client_error(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"malformed": "body"}  # Missing expected field 'answer'
        mock_client.post.return_value = mock_response

        with pytest.raises(APIClientError, match="Failed to parse query response"):
            api_client.query_rag("query")


class TestAPIClientDocuments:

    @patch("src.frontend.api_client.httpx.Client")
    def test_get_documents_success(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        doc_id = str(uuid4())
        mock_response.json.return_value = [
            {
                "document_id": doc_id,
                "filename": "hello.txt",
                "status": "indexed",
                "chunk_count": 3,
                "file_size": 256,
                "uploaded_at": "2026-08-12T20:00:00"
            }
        ]
        mock_client.get.return_value = mock_response

        # Execute
        result = api_client.get_documents()

        # Verify
        assert len(result) == 1
        assert result[0]["filename"] == "hello.txt"
        assert result[0]["status"] == "indexed"
        
        # Verify request parameters
        call_args = mock_client.get.call_args
        assert call_args[0][0] == f"{FAKE_API_URL}/api/v1/documents"

    @patch("src.frontend.api_client.httpx.Client")
    def test_get_documents_failure_raises_api_client_error(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        mock_client.get.return_value = mock_response

        with pytest.raises(APIClientError, match="Failed to retrieve documents"):
            api_client.get_documents()

    @patch("src.frontend.api_client.httpx.Client")
    def test_delete_document_success(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "detail": "Deleted successfully"
        }
        mock_client.delete.return_value = mock_response

        # Execute
        doc_id = uuid4()
        result = api_client.delete_document(doc_id)

        # Verify
        assert result["status"] == "success"
        assert result["detail"] == "Deleted successfully"

        # Verify request parameters
        call_args = mock_client.delete.call_args
        assert call_args[0][0] == f"{FAKE_API_URL}/api/v1/documents/{doc_id}"

    @patch("src.frontend.api_client.httpx.Client")
    def test_delete_document_failure_raises_api_client_error(self, mock_client_cls, api_client):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Document not found"}
        mock_client.delete.return_value = mock_response

        with pytest.raises(APIClientError, match="Deletion failed.*Document not found"):
            api_client.delete_document(uuid4())

