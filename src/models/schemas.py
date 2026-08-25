from datetime import datetime
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List
from uuid import UUID, uuid4

class PageContent(BaseModel):
    page_number: Optional[int]       # None for TXT/MD; 1-based int for PDF
    text: str
    source_location: Optional[str] = None  # e.g. "Page 3" or "test.txt"

class Document(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    document_name: str

class Chunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    document_name: str
    page_number: Optional[int] = None
    text: str
    source_location: Optional[str] = None

class EmbeddedChunk(Chunk):
    embedding: List[float]

class VectorRecord(BaseModel):
    id: UUID
    vector: List[float]
    payload: dict
    score: Optional[float] = None

class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float

class QdrantPoint(BaseModel):
    id: str  # Qdrant requires UUID string
    vector: List[float]
    payload: dict

class RetrievedChunk(Chunk):
    score: float

class Citation(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_name: str
    page_number: Optional[int] = None
    source_location: Optional[str] = None
    snippet: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

    @model_validator(mode='after')
    def validate_query(self) -> 'QueryRequest':
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return self

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]

class LLMStructuredOutput(BaseModel):
    answer: str
    citation_ids: List[str]


class DocumentResponse(BaseModel):
    document_id: UUID
    filename: str
    status: str
    chunk_count: int
    file_size: int
    uploaded_at: datetime
