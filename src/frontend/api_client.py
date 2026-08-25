"""
API client wrapper for communicating with the InsightOS FastAPI backend.
"""

from __future__ import annotations

import logging
import httpx

from src.models.schemas import QueryResponse

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Raised when API client request or parsing operations fail."""
    pass


class APIClient:
    """
    Client for interacting with the InsightOS HTTP REST API.
    Handles uploads and queries, mapping failures to clean APIClientErrors.
    """

    def __init__(self, api_url: str) -> None:
        """
        Initialise APIClient.

        Args:
            api_url: Base URL of the running FastAPI server.
        """
        self.api_url = api_url.rstrip("/")

    def upload_document(self, file_name: str, file_bytes: bytes) -> dict:
        """
        Upload a document to the backend for ingestion.

        Args:
            file_name: Name of the file being uploaded.
            file_bytes: Raw binary content of the file.

        Returns:
            Dict containing document_id, status, and chunks_indexed.

        Raises:
            APIClientError: On connection failures, server errors, or invalid status.
        """
        url = f"{self.api_url}/api/v1/documents/upload"
        files = {"file": (file_name, file_bytes)}

        try:
            # Set a long timeout for document ingestion since it does embedding/Qdrant operations
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, files=files)
        except httpx.RequestError as exc:
            raise APIClientError(
                f"Failed to connect to the backend server at {self.api_url}: {exc}"
            ) from exc

        # Handle unsuccessful status codes
        if not (200 <= response.status_code < 300):
            detail = response.reason_phrase
            try:
                # Try to parse detail from FastAPI default error structures
                body = response.json()
                if "detail" in body:
                    detail = body["detail"]
            except Exception:
                pass
            raise APIClientError(
                f"Ingestion failed (HTTP {response.status_code}): {detail}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError(
                f"Failed to parse ingestion response JSON: {exc}"
            ) from exc

    def query_rag(self, query: str, top_k: int | None = None) -> QueryResponse:
        """
        Send a natural language query to the RAG pipeline.

        Args:
            query: The user's search / question.
            top_k: Limit on context documents to retrieve.

        Returns:
            QueryResponse Pydantic model.

        Raises:
            APIClientError: On connection failures, LLM errors, or parsing issues.
        """
        url = f"{self.api_url}/api/v1/query"
        payload = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise APIClientError(
                f"Failed to connect to the backend server at {self.api_url}: {exc}"
            ) from exc

        # Handle unsuccessful status codes
        if not (200 <= response.status_code < 300):
            detail = response.reason_phrase
            try:
                body = response.json()
                if "detail" in body:
                    detail = body["detail"]
            except Exception:
                pass
            raise APIClientError(
                f"Query failed (HTTP {response.status_code}): {detail}"
            )

        try:
            return QueryResponse.model_validate(response.json())
        except Exception as exc:
            raise APIClientError(
                f"Failed to parse query response: {exc}"
            ) from exc

    def get_documents(self) -> list[dict]:
        """
        Fetch all persisted documents from the backend.

        Returns:
            List of dict containing document details.

        Raises:
            APIClientError: On connection or parsing failures.
        """
        url = f"{self.api_url}/api/v1/documents"
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
        except httpx.RequestError as exc:
            raise APIClientError(
                f"Failed to connect to the backend server at {self.api_url}: {exc}"
            ) from exc

        if not (200 <= response.status_code < 300):
            detail = response.reason_phrase
            try:
                body = response.json()
                if "detail" in body:
                    detail = body["detail"]
            except Exception:
                pass
            raise APIClientError(
                f"Failed to retrieve documents (HTTP {response.status_code}): {detail}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError(
                f"Failed to parse document list response: {exc}"
            ) from exc

    def delete_document(self, document_id: str | UUID) -> dict:
        """
        Request backend to delete the document and its associated vectors.

        Args:
            document_id: UUID of the document.

        Returns:
            Dict containing deletion status and detail.

        Raises:
            APIClientError: On connection or server deletion failures.
        """
        url = f"{self.api_url}/api/v1/documents/{document_id}"
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.delete(url)
        except httpx.RequestError as exc:
            raise APIClientError(
                f"Failed to connect to the backend server at {self.api_url}: {exc}"
            ) from exc

        if not (200 <= response.status_code < 300):
            detail = response.reason_phrase
            try:
                body = response.json()
                if "detail" in body:
                    detail = body["detail"]
            except Exception:
                pass
            raise APIClientError(
                f"Deletion failed (HTTP {response.status_code}): {detail}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise APIClientError(
                f"Failed to parse deletion response JSON: {exc}"
            ) from exc
