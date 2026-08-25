"""
Vector Store Provider: Generic interface and factory for vector storage backends.

Allows switching from Qdrant to another database without modifying orchestrators.
"""

from __future__ import annotations

import abc
import logging
from typing import List
from uuid import UUID

from src.models.schemas import VectorRecord

logger = logging.getLogger(__name__)


class VectorStore(abc.ABC):
    """Abstract base class representing a generic vector database store."""

    @abc.abstractmethod
    def upsert(self, records: List[VectorRecord]) -> None:
        """
        Insert or update a batch of vector records.

        Args:
            records: List of VectorRecord containing id, vector, and payload.

        Raises:
            VectorStoreError: If the operation fails.
            ValueError: If the input records list is empty or mismatching.
        """
        pass

    @abc.abstractmethod
    def get(self, point_id: UUID) -> VectorRecord | None:
        """
        Retrieve a single vector record by its ID.

        Args:
            point_id: Unique identifier for the stored vector point.

        Returns:
            VectorRecord if found, None if it doesn't exist.

        Raises:
            VectorStoreError: If the database operation fails.
        """
        pass

    @abc.abstractmethod
    def delete(self, document_id: UUID) -> None:
        """
        Delete all vector records associated with a document ID.

        Args:
            document_id: Unique identifier for the document parent.

        Raises:
            VectorStoreError: If the database operation fails.
        """
        pass

    @abc.abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[VectorRecord]:
        """
        Search for vectors similar to the query_vector.

        Args:
            query_vector: Embedding vector of the query.
            top_k: Maximum number of records to return.
            score_threshold: Optional similarity threshold.

        Returns:
            List of VectorRecord objects, sorted by score descending.

        Raises:
            VectorStoreError: If the database operation fails.
        """
        pass


def get_vector_store() -> VectorStore:
    """
    Factory function returning the configured VectorStore implementation.

    In the future, we can change the import here to swap backends.
    """
    from src.services.vector_store.qdrant import QdrantVectorStore
    return QdrantVectorStore()
