"""
Orchestrator: Glues together the retrieval, generation, and citation layers to execute a RAG query.
"""

from __future__ import annotations

import logging

from src.core.exceptions import CitationError, GenerationError, RetrievalError
from src.models.schemas import QueryResponse
from src.services.rag.citation import CitationHandler
from src.services.rag.generator import Generator
from src.services.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    RAG Orchestrator.
    Manages the RAG pipeline flow: Retrieval -> LLM Generation -> Citation Processing.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        citation_handler: CitationHandler,
    ) -> None:
        """
        Initialise Orchestrator.

        Args:
            retriever: The Retriever service.
            generator: The Generator service.
            citation_handler: The CitationHandler service.
        """
        self.retriever = retriever
        self.generator = generator
        self.citation_handler = citation_handler

    def query(
        self,
        user_query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> QueryResponse:
        """
        Execute RAG query query pipeline.

        Args:
            user_query: User's query string.
            top_k: Optional limit on retrieved documents.
            score_threshold: Optional similarity threshold.

        Returns:
            QueryResponse model with generated answer text and citations.

        Raises:
            ValueError: On validation failure of inputs.
            RetrievalError: On retrieval errors.
            GenerationError: On LLM generation errors.
            CitationError: On citation mapping errors.
        """
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty.")

        # 1. Retrieve Context Chunks
        try:
            retrieved_results = self.retriever.retrieve(
                query=user_query,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Unexpected error during retrieval inside Orchestrator: {exc}") from exc

        # 2. Short-Circuit if Empty Retrieval
        if not retrieved_results:
            logger.info("Retrieval returned no chunks. Short-circuiting query execution.")
            return QueryResponse(
                answer="I cannot answer this question based on the provided documents.",
                citations=[],
            )

        context_chunks = [res.chunk for res in retrieved_results]

        # 3. Generate Answer
        try:
            answer = self.generator.generate(
                query=user_query,
                context_chunks=context_chunks,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Unexpected error during generation inside Orchestrator: {exc}") from exc

        # 4. Parse & Map Citations
        try:
            citations = self.citation_handler.parse_citations(
                answer_text=answer,
                context_chunks=context_chunks,
            )
        except CitationError:
            raise
        except Exception as exc:
            raise CitationError(f"Unexpected error during citation mapping inside Orchestrator: {exc}") from exc

        # 5. Build final QueryResponse
        return QueryResponse(
            answer=answer,
            citations=citations,
        )
