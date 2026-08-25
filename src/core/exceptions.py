class InsightOSError(Exception):
    """Base exception for all InsightOS errors."""
    pass

class DocumentExtractionError(InsightOSError):
    """Raised when document extraction fails."""
    pass

class EmbeddingError(InsightOSError):
    """Raised when embedding generation fails."""
    pass

class VectorStoreError(InsightOSError):
    """Raised when vector store operations fail."""
    pass

class RetrievalError(InsightOSError):
    """Raised when retrieval operations fail."""
    pass

class GenerationError(InsightOSError):
    """Raised when LLM generation fails."""
    pass

class CitationError(InsightOSError):
    """Raised when citation extraction or mapping fails."""
    pass
