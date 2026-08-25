"""
Retriever: Orchestrates query embedding and vector database search to retrieve relevant text chunks.
"""

from __future__ import annotations

import logging
from typing import List

from src.core.config import settings
from src.core.exceptions import EmbeddingError, RetrievalError, VectorStoreError
from src.models.schemas import Chunk, RetrievalResult
from src.services.embeddings.provider import EmbeddingProvider
from src.services.vector_store.provider import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    RAG Retriever class.

    Responsible for:
    1. Validating queries.
    2. Generating query embeddings using an EmbeddingProvider.
    3. Performing similarity search against a VectorStore.
    4. Validating and returning ranked RetrievalResult objects.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        """
        Initialise Retriever with required backend services.

        Args:
            embedding_provider: Concrete implementation of EmbeddingProvider.
            vector_store: Concrete implementation of VectorStore.
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks relevant to the user query.

        Args:
            query: Natural-language query text.
            top_k: Maximum number of results to return. Fallbacks to config default.
            score_threshold: Optional minimum similarity threshold. Fallbacks to config.

        Returns:
            List of ranked RetrievalResult objects, highest score first.

        Raises:
            ValueError: On validation failure of inputs.
            RetrievalError: On embedding, storage, payload validation, or parsing failures.
        """
        # 1. Input Validation
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        resolved_top_k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        if not isinstance(resolved_top_k, int) or resolved_top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        resolved_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.RETRIEVAL_SCORE_THRESHOLD
        )
        if resolved_threshold is not None:
            if not isinstance(resolved_threshold, (int, float)):
                raise ValueError("score_threshold must be a float or integer.")

        # 2. Embedding Generation
        try:
            query_vector = self.embedding_provider.embed_query(query)
        except EmbeddingError as exc:
            raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(
                f"Unexpected error during query embedding: {exc}"
            ) from exc

        # 3. Similarity Search in Vector Store
        try:
            records = self.vector_store.search(
                query_vector=query_vector,
                top_k=resolved_top_k,
                score_threshold=resolved_threshold,
            )
        except VectorStoreError as exc:
            raise RetrievalError(f"Vector store search failed: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(
                f"Unexpected error during vector store search: {exc}"
            ) from exc

        # 4. Result Parsing & Mapping
        results = []
        for i, rec in enumerate(records):
            # Verify payload presence and keys
            payload = rec.payload
            if not payload or not isinstance(payload, dict):
                raise RetrievalError(
                    f"Vector record at index {i} is missing payload metadata."
                )

            # Ensure all required payload keys exist
            required_fields = [
                "document_id",
                "document_name",
                "page_number",
                "chunk_id",
                "text",
                "source_location",
            ]
            for field in required_fields:
                if field not in payload:
                    raise RetrievalError(
                        f"Vector record payload at index {i} is missing required field '{field}'."
                    )

            # Validate type correctness of the payload by building the Chunk model
            try:
                chunk = Chunk(**payload)
            except Exception as exc:
                raise RetrievalError(
                    f"Vector record payload at index {i} has invalid format: {exc}"
                ) from exc

            # Verify similarity score presence
            score = rec.score
            if score is None or not isinstance(score, (int, float)):
                raise RetrievalError(
                    f"Vector record at index {i} has missing or invalid similarity score."
                )

            results.append(RetrievalResult(chunk=chunk, score=float(score)))

        # Ordering is preserved as returned from VectorStore.search
        return results
