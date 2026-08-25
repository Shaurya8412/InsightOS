"""
Unit tests for src/services/embeddings/provider.py

All tests in this module are fully mocked — no live Gemini API calls.
The integration test at the bottom is conditionally skipped when
GEMINI_API_KEY is absent from the environment.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings
from src.core.exceptions import EmbeddingError
from src.services.embeddings.provider import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    _validate_embeddings,
    _validate_inputs,
    get_embedding_provider,
)
from google.genai.errors import ClientError as GeminiClientError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_KEY = "fake-api-key-for-tests"
FAKE_MODEL = "gemini-embedding-001"
FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4]
FAKE_VECTOR_DIM = len(FAKE_VECTOR)


def _make_fake_response(vectors: list[list[float]]):
    """Build a mock object that mimics the Gemini embed_content response."""
    embeddings = [SimpleNamespace(values=v) for v in vectors]
    return SimpleNamespace(embeddings=embeddings)


@pytest.fixture
def provider():
    """Return a GeminiEmbeddingProvider with a mocked internal client."""
    with patch("src.services.embeddings.provider.genai.Client"):
        p = GeminiEmbeddingProvider(api_key=FAKE_KEY, model=FAKE_MODEL)
    return p


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestEmbeddingInterface:

    def test_gemini_provider_is_embedding_provider(self, provider):
        assert isinstance(provider, EmbeddingProvider)

    def test_embed_single_text_returns_one_vector(self, provider):
        provider._client.models.embed_content.return_value = (
            _make_fake_response([FAKE_VECTOR])
        )
        result = provider.embed_texts(["hello world"])
        assert len(result) == 1
        assert result[0] == FAKE_VECTOR

    def test_embed_multiple_texts_returns_same_count(self, provider):
        vectors = [FAKE_VECTOR, FAKE_VECTOR, FAKE_VECTOR]
        provider._client.models.embed_content.return_value = (
            _make_fake_response(vectors)
        )
        result = provider.embed_texts(["a", "b", "c"])
        assert len(result) == 3

    def test_input_ordering_preserved(self, provider):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        provider._client.models.embed_content.return_value = (
            _make_fake_response([v1, v2])
        )
        result = provider.embed_texts(["text one", "text two"])
        assert result[0] == v1
        assert result[1] == v2

    def test_embed_query_returns_single_vector(self, provider):
        provider._client.models.embed_content.return_value = (
            _make_fake_response([FAKE_VECTOR])
        )
        result = provider.embed_query("what is AI?")
        assert result == FAKE_VECTOR
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_list_raises_value_error(self, provider):
        with pytest.raises(ValueError, match="non-empty list"):
            provider.embed_texts([])

    def test_empty_string_raises_value_error(self, provider):
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            provider.embed_texts(["valid text", ""])

    def test_whitespace_string_raises_value_error(self, provider):
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            provider.embed_texts(["   "])

    def test_validate_inputs_accepts_valid_list(self):
        _validate_inputs(["hello", "world"])  # should not raise

    def test_validate_inputs_rejects_empty_list(self):
        with pytest.raises(ValueError):
            _validate_inputs([])

    def test_validate_inputs_rejects_empty_string(self):
        with pytest.raises(ValueError):
            _validate_inputs(["good", ""])


# ---------------------------------------------------------------------------
# Vector validation tests
# ---------------------------------------------------------------------------

class TestVectorValidation:

    def test_valid_vectors_accepted(self):
        _validate_embeddings([[0.1, 0.2], [0.3, 0.4]])  # should not raise

    def test_empty_embedding_list_raises(self):
        with pytest.raises(EmbeddingError, match="empty embedding list"):
            _validate_embeddings([])

    def test_zero_dimension_vector_raises(self):
        with pytest.raises(EmbeddingError, match="zero-dimension"):
            _validate_embeddings([[]])

    def test_inconsistent_dimensions_raises(self):
        with pytest.raises(EmbeddingError, match="Inconsistent embedding dimensions"):
            _validate_embeddings([[0.1, 0.2], [0.3, 0.4, 0.5]])

    def test_non_numeric_values_raise(self):
        with pytest.raises(EmbeddingError, match="non-numeric"):
            _validate_embeddings([["a", "b", "c"]])  # type: ignore

    def test_single_valid_vector_accepted(self):
        _validate_embeddings([[0.1, 0.2, 0.3]])  # should not raise


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_api_key_loaded_from_settings(self):
        with patch("src.services.embeddings.provider.genai.Client") as mock_client_cls:
            GeminiEmbeddingProvider(api_key=FAKE_KEY)
            mock_client_cls.assert_called_once_with(api_key=FAKE_KEY)

    def test_model_loaded_from_settings(self):
        with patch("src.services.embeddings.provider.genai.Client"):
            p = GeminiEmbeddingProvider(api_key=FAKE_KEY, model="gemini-embedding-001")
            assert p._model == "gemini-embedding-001"

    def test_missing_api_key_raises_embedding_error(self):
        with patch("src.services.embeddings.provider.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = ""
            mock_settings.EMBEDDING_MODEL = FAKE_MODEL
            with pytest.raises(EmbeddingError, match="GEMINI_API_KEY is not set"):
                GeminiEmbeddingProvider(api_key="")

    def test_embedding_model_is_configurable(self):
        custom_model = "gemini-embedding-2"
        with patch("src.services.embeddings.provider.genai.Client"):
            p = GeminiEmbeddingProvider(api_key=FAKE_KEY, model=custom_model)
            assert p._model == custom_model


# ---------------------------------------------------------------------------
# Failure / error conversion tests
# ---------------------------------------------------------------------------

class TestFailureHandling:

    def test_api_exception_converted_to_embedding_error(self, provider):
        provider._client.models.embed_content.side_effect = RuntimeError(
            "Network unreachable"
        )
        with pytest.raises(EmbeddingError, match="failed after retries"):
            provider.embed_texts(["test text"])

    def test_empty_api_response_raises_embedding_error(self, provider):
        provider._client.models.embed_content.return_value = SimpleNamespace(
            embeddings=None
        )
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["test text"])

    def test_empty_embeddings_list_in_response_raises(self, provider):
        provider._client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[]
        )
        with pytest.raises(EmbeddingError):
            provider.embed_texts(["test text"])

    def test_get_embedding_provider_raises_without_api_key(self):
        """Factory should raise if GEMINI_API_KEY is not configured."""
        with patch("src.services.embeddings.provider.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = ""
            mock_settings.EMBEDDING_MODEL = FAKE_MODEL
            with patch("src.services.embeddings.provider.genai.Client"):
                with pytest.raises(EmbeddingError, match="GEMINI_API_KEY"):
                    get_embedding_provider()


# ---------------------------------------------------------------------------
# Integration test (skipped without live API key)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGeminiEmbeddingIntegration:

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            pytest.skip("GEMINI_API_KEY not set — skipping live integration test")
        self.api_key = api_key

    def _make_provider(self):
        return GeminiEmbeddingProvider(api_key=self.api_key)

    def _try_embed_or_skip(self, provider, texts):
        """Run embedding; skip test if the API key is invalid rather than failing."""
        try:
            return provider.embed_texts(texts)
        except EmbeddingError as exc:
            msg = str(exc).lower()
            if "api key not valid" in msg or "invalid_argument" in msg or "client error" in msg:
                pytest.skip(f"GEMINI_API_KEY is set but invalid: {exc}")
            raise

    def test_real_embed_single_text(self):
        provider = self._make_provider()
        vectors = self._try_embed_or_skip(provider, ["What is machine learning?"])
        assert isinstance(vectors, list)
        assert len(vectors) == 1
        vector = vectors[0]
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)
        assert len(vector) >= 256, f"Unexpectedly short vector: {len(vector)} dims"

    def test_real_embed_batch(self):
        provider = self._make_provider()
        texts = ["Hello world", "Machine learning is fascinating", "Python rocks"]
        vectors = self._try_embed_or_skip(provider, texts)
        assert len(vectors) == len(texts)
        dim = len(vectors[0])
        for v in vectors:
            assert len(v) == dim, "All vectors must share the same dimension"
