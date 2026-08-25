import pytest
from pydantic import ValidationError
from uuid import uuid4
from src.models.schemas import Chunk, QueryRequest, RetrievalResult, VectorRecord

def test_query_request_valid():
    req = QueryRequest(query="What is AI?", top_k=3)
    assert req.query == "What is AI?"
    assert req.top_k == 3

def test_query_request_empty():
    with pytest.raises(ValidationError):
        QueryRequest(query="   ", top_k=5)

def test_query_request_invalid_top_k():
    with pytest.raises(ValidationError):
        QueryRequest(query="What is AI?", top_k=0)

def test_chunk_creation():
    doc_id = uuid4()
    chunk = Chunk(
        document_id=doc_id,
        document_name="test.pdf",
        text="This is a test",
        page_number=1,
        source_location="Page 1"
    )
    assert chunk.document_id == doc_id
    assert chunk.document_name == "test.pdf"
    assert chunk.text == "This is a test"
    assert chunk.chunk_id is not None

def test_vector_record_with_score():
    record_id = uuid4()
    record = VectorRecord(
        id=record_id,
        vector=[0.1, 0.2],
        payload={"key": "val"},
        score=0.95
    )
    assert record.id == record_id
    assert record.score == 0.95

def test_retrieval_result_creation():
    doc_id = uuid4()
    chunk = Chunk(
        document_id=doc_id,
        document_name="test.pdf",
        text="This is a test",
        page_number=1,
        source_location="Page 1"
    )
    res = RetrievalResult(
        chunk=chunk,
        score=0.88
    )
    assert res.chunk.text == "This is a test"
    assert res.score == 0.88
