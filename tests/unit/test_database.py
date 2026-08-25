"""
Unit tests for database setup and the Document SQL model.
Uses an isolated in-memory SQLite database connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.core.database import Base
from src.models.db_models import Document


@pytest.fixture(name="db_session")
def fixture_db_session():
    """
    Fixture creating an in-memory SQLite database, initializing tables,
    and yielding a transactional session.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Database Schema & Connection Tests
# ---------------------------------------------------------------------------

def test_database_table_creation(db_session):
    """Verify that tables are created successfully with correct columns."""
    engine = db_session.get_bind()
    inspector = inspect(engine)
    
    assert "documents" in inspector.get_table_names()
    
    columns = {col["name"]: col["type"].__class__.__name__ for col in inspector.get_columns("documents")}
    assert "document_id" in columns
    assert "filename" in columns
    assert "status" in columns
    assert "chunk_count" in columns
    assert "file_size" in columns
    assert "uploaded_at" in columns


# ---------------------------------------------------------------------------
# Document Model persistence & properties
# ---------------------------------------------------------------------------

def test_insert_and_query_document(db_session):
    """Verify inserting and querying a Document persists all properties."""
    doc_id = uuid.uuid4()
    filename = "test_document.pdf"
    file_size = 1048576  # 1MB
    chunk_count = 15

    new_doc = Document(
        document_id=doc_id,
        filename=filename,
        status="pending",
        chunk_count=chunk_count,
        file_size=file_size
    )

    db_session.add(new_doc)
    db_session.commit()

    # Query document back
    queried_doc = db_session.query(Document).filter_by(document_id=doc_id).first()
    assert queried_doc is not None
    assert queried_doc.document_id == doc_id
    assert queried_doc.filename == filename
    assert queried_doc.status == "pending"
    assert queried_doc.chunk_count == chunk_count
    assert queried_doc.file_size == file_size
    assert isinstance(queried_doc.uploaded_at, datetime)
    # Check that uploaded_at is reasonably recent/within seconds
    assert (datetime.now(timezone.utc) - queried_doc.uploaded_at.replace(tzinfo=timezone.utc)).total_seconds() < 10


def test_document_id_default_generation(db_session):
    """Verify that a UUID is automatically generated if document_id is omitted."""
    new_doc = Document(
        filename="no_uuid.txt",
        status="pending"
    )
    db_session.add(new_doc)
    db_session.commit()

    assert new_doc.document_id is not None
    assert isinstance(new_doc.document_id, uuid.UUID)


# ---------------------------------------------------------------------------
# Status Constraints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valid_status", ["pending", "indexed", "failed"])
def test_valid_statuses_accepted(db_session, valid_status):
    """Verify that 'pending', 'indexed', and 'failed' are accepted."""
    doc = Document(
        filename="test.txt",
        status=valid_status
    )
    db_session.add(doc)
    db_session.commit()

    queried = db_session.query(Document).filter_by(document_id=doc.document_id).first()
    assert queried.status == valid_status


def test_invalid_status_rejected_by_check_constraint(db_session):
    """Verify that status values outside allowed strings raise IntegrityError."""
    doc = Document(
        filename="invalid_status.txt",
        status="processing"  # 'processing' is not in ['pending', 'indexed', 'failed']
    )
    db_session.add(doc)
    
    # SQLite check constraint throws IntegrityError on commit
    with pytest.raises(IntegrityError):
        db_session.commit()
