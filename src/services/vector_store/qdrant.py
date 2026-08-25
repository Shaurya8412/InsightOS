"""
Qdrant Vector Store: Concrete implementation of the VectorStore interface.
"""

from __future__ import annotations

import logging
from typing import List
from uuid import UUID

from qdrant_client import QdrantClient, models as q_models
from qdrant_client.http.exceptions import UnexpectedResponse
from src.core.config import settings
from src.core.exceptions import VectorStoreError
from src.models.schemas import Chunk, VectorRecord
from src.services.vector_store.provider import VectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """
    Qdrant implementation of the VectorStore.

    Uses the official qdrant-client package to interact with Qdrant.
    Handles connection, collection management (creation/validation),
    upsert, retrieval (get), and deletion.
    """

    def __init__(self, client: QdrantClient | None = None) -> None:
        """
        Initialise QdrantVectorStore.

        Connects to Qdrant and ensures the collection exists and is compatible.

        Args:
            client: Optional injected QdrantClient (useful for unit tests).
        """
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        if client is not None:
            self.client = client
        else:
            try:
                # API key is optional for local development
                api_key = settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
                self.client = QdrantClient(url=settings.QDRANT_URL, api_key=api_key)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to connect to Qdrant at {settings.QDRANT_URL}: {exc}"
                ) from exc

        self._dimension: int | None = None
        self._init_collection()

    # ------------------------------------------------------------------
    # VectorStore Interface Implementation
    # ------------------------------------------------------------------

    def upsert(self, records: List[VectorRecord]) -> None:
        """
        Upsert a batch of vector records.

        Args:
            records: List of VectorRecord objects.

        Raises:
            VectorStoreError: On vector validation, payload validation, or API failures.
        """
        if not records:
            return

        points = []
        for i, rec in enumerate(records):
            # 1. Validate vector presence
            if not rec.vector or not isinstance(rec.vector, list):
                raise VectorStoreError(
                    f"Vector record at index {i} has empty or invalid vector."
                )

            # 2. Validate vector dimension
            if len(rec.vector) != self._dimension:
                raise VectorStoreError(
                    f"Vector dimension mismatch at index {i}: expected {self._dimension}, "
                    f"got {len(rec.vector)}."
                )

            # 3. Validate payload existence
            payload = rec.payload
            if not payload or not isinstance(payload, dict):
                raise VectorStoreError(
                    f"Missing or invalid payload dictionary at index {i}."
                )

            # 4. Validate required payload keys
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
                    raise VectorStoreError(
                        f"Missing required payload field '{field}' at index {i}."
                    )

            # 5. Validate formatting by running through the Chunk model
            try:
                Chunk(**payload)
            except Exception as exc:
                raise VectorStoreError(
                    f"Invalid payload format at index {i}: {exc}"
                ) from exc

            points.append(
                q_models.PointStruct(
                    id=str(rec.id),
                    vector=rec.vector,
                    payload=payload,
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert points to collection '{self.collection_name}': {exc}"
            ) from exc

    def get(self, point_id: UUID) -> VectorRecord | None:
        """
        Retrieve a single vector record by its unique ID.

        Args:
            point_id: Unique UUID of the point.

        Returns:
            VectorRecord if found, None otherwise.

        Raises:
            VectorStoreError: On payload parsing or API failures.
        """
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(point_id)],
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve point '{point_id}' from collection '{self.collection_name}': {exc}"
            ) from exc

        if not results:
            return None

        point = results[0]
        payload = point.payload

        if not payload or not isinstance(payload, dict):
            raise VectorStoreError(
                f"Retrieved point '{point_id}' has missing or invalid payload."
            )

        # Validate required payload fields
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
                raise VectorStoreError(
                    f"Retrieved point '{point_id}' has missing payload field '{field}'."
                )

        # Validate type correctness using Chunk schema
        try:
            Chunk(**payload)
        except Exception as exc:
            raise VectorStoreError(
                f"Retrieved point '{point_id}' has invalid payload values: {exc}"
            ) from exc

        vector = point.vector
        if not vector or not isinstance(vector, list):
            raise VectorStoreError(
                f"Retrieved point '{point_id}' has missing or invalid vector."
            )

        return VectorRecord(
            id=point_id,
            vector=vector,
            payload=payload,
        )

    def delete(self, document_id: UUID) -> None:
        """
        Delete all vector points belonging to the given document_id.

        Args:
            document_id: Unique ID of the document whose points should be deleted.

        Raises:
            VectorStoreError: On API failures.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=q_models.Filter(
                    must=[
                        q_models.FieldCondition(
                            key="document_id",
                            match=q_models.MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete points for document '{document_id}' from collection '{self.collection_name}': {exc}"
            ) from exc

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> List[VectorRecord]:
        """
        Search for vectors similar to the query_vector in the Qdrant collection.

        Args:
            query_vector: Embedding vector of the query.
            top_k: Maximum number of records to return.
            score_threshold: Optional similarity threshold.

        Returns:
            List of VectorRecord objects, sorted by score descending.

        Raises:
            VectorStoreError: If validation or the database operation fails.
        """
        # 1. Input Validation
        if not query_vector or not isinstance(query_vector, list):
            raise VectorStoreError("Query vector must be a non-empty list of floats.")
        if len(query_vector) != self._dimension:
            raise VectorStoreError(
                f"Query vector dimension mismatch: expected {self._dimension}, got {len(query_vector)}."
            )
        if top_k <= 0:
            raise VectorStoreError(f"Limit (top_k) must be greater than zero, got {top_k}.")

        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=True,
            )
            hits = res.points
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to perform similarity search in collection '{self.collection_name}': {exc}"
            ) from exc

        records = []
        for i, hit in enumerate(hits):
            payload = hit.payload
            if not payload or not isinstance(payload, dict):
                raise VectorStoreError(
                    f"Search result at index {i} has missing or invalid payload."
                )

            # Validate required payload fields
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
                    raise VectorStoreError(
                        f"Search result payload at index {i} is missing required field '{field}'."
                    )

            # Validate type correctness using Chunk schema
            try:
                Chunk(**payload)
            except Exception as exc:
                raise VectorStoreError(
                    f"Search result payload at index {i} has invalid format: {exc}"
                ) from exc

            try:
                point_id = UUID(str(hit.id))
            except ValueError as exc:
                raise VectorStoreError(
                    f"Search result point ID '{hit.id}' at index {i} is not a valid UUID: {exc}"
                ) from exc

            records.append(
                VectorRecord(
                    id=point_id,
                    vector=hit.vector if hit.vector is not None else [],
                    payload=payload,
                    score=hit.score,
                )
            )

        return records

    # ------------------------------------------------------------------
    # Extra helper method for batch validation as per Q9
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Helper method to upsert chunks and embeddings.
        Enforces that chunks and embeddings match in count.

        Args:
            chunks: List of Chunk objects.
            embeddings: Parallel list of vector embeddings.

        Raises:
            VectorStoreError: If lengths mismatch or on storage failure.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatched input lengths: {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        records = []
        for chunk, emb in zip(chunks, embeddings):
            records.append(
                VectorRecord(
                    id=chunk.chunk_id,
                    vector=emb,
                    payload={
                        "document_id": str(chunk.document_id),
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "chunk_id": str(chunk.chunk_id),
                        "text": chunk.text,
                        "source_location": chunk.source_location,
                    },
                )
            )

        self.upsert(records)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _init_collection(self) -> None:
        """Ensure collection exists and has compatible vector configuration."""
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to verify existence of collection '{self.collection_name}': {exc}"
            ) from exc

        # Query embedding provider to detect configured dimension
        self._dimension = self._detect_dimension()

        if not exists:
            logger.info(
                "Creating Qdrant collection '%s' with dimension %d",
                self.collection_name,
                self._dimension,
            )
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=q_models.VectorParams(
                        size=self._dimension,
                        distance=q_models.Distance.COSINE,
                    ),
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to create collection '{self.collection_name}': {exc}"
                ) from exc
        else:
            try:
                info = self.client.get_collection(self.collection_name)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to fetch metadata for collection '{self.collection_name}': {exc}"
                ) from exc

            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                existing_size = next(iter(vectors.values())).size
            else:
                existing_size = vectors.size

            if existing_size != self._dimension:
                raise VectorStoreError(
                    f"Existing Qdrant collection '{self.collection_name}' has dimension {existing_size}, "
                    f"but configured embedding model expects {self._dimension}. "
                    "Initialization aborted to prevent data corruption."
                )

            logger.info(
                "Reusing existing Qdrant collection '%s' (dimension=%d)",
                self.collection_name,
                self._dimension,
            )

        # Ensure document_id payload index exists (required for filtering/deleting on Qdrant Cloud)
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=q_models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            logger.warning(
                "Could not ensure payload index on 'document_id' for collection '%s': %s",
                self.collection_name,
                exc,
            )

    def _detect_dimension(self) -> int:
        """Query embedding provider to dynamically determine vector dimension."""
        from src.services.embeddings.provider import get_embedding_provider

        try:
            provider = get_embedding_provider()
            dummy_vec = provider.embed_query("dimension check")
            if not dummy_vec or not isinstance(dummy_vec, list):
                raise ValueError("Embedding provider returned empty or invalid vector.")
            return len(dummy_vec)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to determine embedding dimension from provider: {exc}"
            ) from exc
