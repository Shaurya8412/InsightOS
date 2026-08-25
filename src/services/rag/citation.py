"""
Citation Handler: Parses bracket citation markers from LLM output
and resolves them back to original chunk metadata.
"""

from __future__ import annotations

import logging
import re
from typing import List

from src.core.exceptions import CitationError
from src.models.schemas import Chunk, Citation

logger = logging.getLogger(__name__)


class CitationHandler:
    """
    RAG Citation Handler.
    Responsible for parsing, validating, and mapping LLM citations to original chunks.
    """

    def parse_citations(self, answer_text: str, context_chunks: List[Chunk]) -> List[Citation]:
        """
        Parse citation markers (e.g. [1], [2]) from the text, validate them,
        and construct the corresponding list of unique Citation schemas.

        Args:
            answer_text: Generated response text from the LLM.
            context_chunks: The exact list of Chunk objects retrieved for context.

        Returns:
            List of validated Citation schemas, deduplicated and ordered by first appearance.

        Raises:
            CitationError: If an invalid citation index (e.g. [0], out of bounds) is found.
        """
        if not answer_text:
            return []

        # Find all brackets containing digits, e.g. [1], [12]
        matches = re.findall(r"\[(\d+)\]", answer_text)
        
        seen_indices = set()
        citations = []

        for match in matches:
            val = int(match)
            index = val - 1  # Convert to 0-based list index
            
            # Validation: Check index range
            if index < 0 or index >= len(context_chunks):
                raise CitationError(
                    f"Invalid citation reference '[{match}]' found in response. "
                    f"Retrieved chunk count was {len(context_chunks)}."
                )

            # De-duplicate citations
            if index not in seen_indices:
                seen_indices.add(index)
                chunk = context_chunks[index]
                
                # Build snippet
                snippet = chunk.text
                if len(snippet) > 150:
                    snippet = snippet[:150] + "..."

                citations.append(
                    Citation(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        source_location=chunk.source_location,
                        snippet=snippet,
                    )
                )

        return citations
