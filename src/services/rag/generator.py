"""
Generator: Concrete service interface to invoke the Gemini LLM for grounded text generation.
"""

from __future__ import annotations

import logging
from typing import List

from google import genai
from google.genai.errors import ClientError as GeminiClientError

from src.core.config import settings
from src.core.exceptions import GenerationError
from src.models.schemas import Chunk

logger = logging.getLogger(__name__)


class Generator:
    """
    RAG Generator service.
    Handles grounded prompt construction and calls the Gemini LLM.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialise Generator.

        Args:
            api_key: Optional Gemini API key. Defaults to settings.GEMINI_API_KEY.
            model: Optional Gemini model name. Defaults to settings.LLM_MODEL.
        """
        resolved_key = api_key or settings.GEMINI_API_KEY
        if not resolved_key:
            raise GenerationError(
                "GEMINI_API_KEY is not set. Please provide it via environment variable or config."
            )
        self._model = model or settings.LLM_MODEL
        try:
            self._client = genai.Client(api_key=resolved_key)
        except Exception as exc:
            raise GenerationError(f"Failed to initialise Gemini client: {exc}") from exc

    def generate(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Construct prompt context and call Gemini to generate a grounded answer.

        Args:
            query: User's query string.
            context_chunks: List of retrieved Chunk models.

        Returns:
            The generated response string from the model.

        Raises:
            ValueError: If inputs are invalid.
            GenerationError: If Gemini API fails, returns empty, or encounters transient failures.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if not context_chunks:
            raise ValueError("Context chunks cannot be empty.")

        # Construct Grounding Prompt
        prompt = (
            "You are a helpful research assistant. Answer the user query using ONLY the supplied context chunks. "
            "Do not use outside knowledge as evidence. If the supplied context does not contain enough information "
            "to answer the question, state clearly that the provided documents do not contain enough information.\n\n"
            "Grounding Constraints:\n"
            "- Only answer from the supplied context.\n"
            "- Do not invent facts or extrapolate beyond the provided chunks.\n"
            "- Do not invent document names, page numbers, or source metadata.\n"
            "- You must cite the sources using the chunk numbers in brackets, e.g. [1], [2]. Only use identifiers from the list of chunks below.\n"
            "- Do not invent citation identifiers.\n\n"
            "Context Chunks:\n"
        )

        for i, chunk in enumerate(context_chunks, start=1):
            prompt += f"[{i}]\n"
            prompt += f"Document: {chunk.document_name}\n"
            if chunk.page_number is not None:
                prompt += f"Page: {chunk.page_number}\n"
            if chunk.source_location:
                prompt += f"Source Location: {chunk.source_location}\n"
            prompt += f"Text: {chunk.text}\n\n"

        prompt += f"Query: {query}\n"
        prompt += "Answer: "

        # Call Gemini SDK
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        except GeminiClientError as exc:
            raise GenerationError(
                f"Gemini API client error during generation with model '{self._model}': {exc}"
            ) from exc
        except Exception as exc:
            raise GenerationError(
                f"Failed to generate content from Gemini API with model '{self._model}': {exc}"
            ) from exc

        if not response or not response.text:
            raise GenerationError("Gemini API returned an empty or invalid response.")

        return response.text
