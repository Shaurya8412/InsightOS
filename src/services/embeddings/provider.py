"""
Embedding provider: clean interface between InsightOS and the Gemini embedding API.

Architecture
------------
- `EmbeddingProvider` is an abstract base class that defines the interface.
- `GeminiEmbeddingProvider` is the concrete Gemini implementation.
- The rest of the application (Retriever, Qdrant layer, Orchestrator) must
  depend on `EmbeddingProvider`, not on GeminiEmbeddingProvider directly.
- Swap the provider in `get_embedding_provider()` without touching callers.

Gemini SDK notes (as of google-genai 2.x)
-------------------------------------------
- Package:  google-genai
- Client:   google.genai.Client(api_key=...)
- Call:     client.models.embed_content(model=..., contents=...)
- Models:   gemini-embedding-001 (text, production default)
            gemini-embedding-2   (multimodal, future use)
- text-embedding-004 is deprecated and must NOT be used.

Service boundary
----------------
This module is ONLY responsible for text → vector conversion.
It does NOT know about chunks, pages, documents, Qdrant, or the LLM.
"""

from __future__ import annotations

import abc
import logging
from typing import List

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError as GeminiClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings
from src.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Maximum number of texts per API call.
# The Gemini embedding API accepts batches; we use a conservative limit.
_BATCH_LIMIT = 100


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class EmbeddingProvider(abc.ABC):
    """Abstract interface for all embedding providers."""

    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts and return their vector representations.

        Args:
            texts: Non-empty list of non-empty strings.

        Returns:
            List of float vectors in the same order as input texts.

        Raises:
            EmbeddingError: On any provider, network, or validation failure.
            ValueError: If `texts` is empty or contains an empty string.
        """

    def embed_query(self, text: str) -> List[float]:
        """
        Convenience method to embed a single query string.

        Args:
            text: Non-empty string.

        Returns:
            A single float vector.

        Raises:
            EmbeddingError: On any provider failure.
            ValueError: If `text` is empty.
        """
        results = self.embed_texts([text])
        return results[0]


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------

class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider backed by the Google Gemini API.

    Uses `client.models.embed_content` from the `google-genai` SDK.
    Model and API key are read from Settings; nothing is hardcoded.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        resolved_key = api_key or settings.GEMINI_API_KEY
        if not resolved_key:
            raise EmbeddingError(
                "GEMINI_API_KEY is not set. "
                "Provide it via environment variable or .env file."
            )
        self._model = model or settings.EMBEDDING_MODEL
        self._client = genai.Client(api_key=resolved_key)
        logger.info("GeminiEmbeddingProvider initialised with model=%s", self._model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.  Internally batched to respect API limits.

        Args:
            texts: Non-empty list of non-empty strings.

        Returns:
            Ordered list of float vectors matching the input order.

        Raises:
            ValueError:     If texts is empty or contains empty strings.
            EmbeddingError: On any API or validation failure.
        """
        _validate_inputs(texts)

        all_embeddings: List[List[float]] = []
        for batch_start in range(0, len(texts), _BATCH_LIMIT):
            batch = texts[batch_start : batch_start + _BATCH_LIMIT]
            batch_vectors = self._embed_batch(batch)
            all_embeddings.extend(batch_vectors)

        _validate_embeddings(all_embeddings)
        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        # Only retry on transient errors (network, 5xx).
        # 4xx ClientErrors (auth, bad request) are NOT transient — do not retry.
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Call the Gemini embedding API with automatic retry on transient errors."""
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
            )
        except GeminiClientError as exc:
            # 4xx errors (invalid key, bad request) are client-side faults;
            # retrying will never help — re-raise immediately as EmbeddingError.
            raise EmbeddingError(
                f"Gemini API client error for model '{self._model}': {exc}"
            ) from exc
        except Exception as exc:
            logger.warning("Gemini embedding API transient failure: %s", exc)
            raise  # tenacity will retry

        if not response or not response.embeddings:
            raise EmbeddingError(
                f"Gemini API returned empty embeddings for model '{self._model}'."
            )

        vectors = [list(emb.values) for emb in response.embeddings]
        return vectors

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Wrap retry helper and convert provider exceptions to EmbeddingError."""
        try:
            return self._embed_batch_with_retry(batch)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding request failed after retries for model "
                f"'{self._model}': {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Validation helpers (also used by tests)
# ---------------------------------------------------------------------------

def _validate_inputs(texts: List[str]) -> None:
    """Raise ValueError if the input list is empty or contains empty strings."""
    if not texts:
        raise ValueError("texts must be a non-empty list.")
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"Text at index {i} is empty or whitespace-only.")


def _validate_embeddings(embeddings: List[List[float]]) -> None:
    """
    Raise EmbeddingError if the returned vectors are malformed:
      - empty vector
      - non-numeric values
      - inconsistent dimensions across the batch
    """
    if not embeddings:
        raise EmbeddingError("Provider returned an empty embedding list.")

    first_dim = len(embeddings[0])
    if first_dim == 0:
        raise EmbeddingError("Provider returned a zero-dimension vector.")

    for i, vec in enumerate(embeddings):
        if len(vec) == 0:
            raise EmbeddingError(f"Embedding at index {i} is empty.")
        if len(vec) != first_dim:
            raise EmbeddingError(
                f"Inconsistent embedding dimensions: expected {first_dim}, "
                f"got {len(vec)} at index {i}."
            )
        if not all(isinstance(v, (int, float)) for v in vec):
            raise EmbeddingError(
                f"Embedding at index {i} contains non-numeric values."
            )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def get_embedding_provider() -> EmbeddingProvider:
    """
    Return the configured embedding provider.

    Swap the concrete class here to replace the entire embedding backend
    without touching any caller (Retriever, Orchestrator, Qdrant layer).
    """
    return GeminiEmbeddingProvider()
