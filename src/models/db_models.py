"""
Database models: defines SQLAlchemy models matching the SQL schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, String, Uuid

from src.core.database import Base


class Document(Base):
    """
    SQLAlchemy Document model representing the metadata and lifecycle status of ingested files.
    """

    __tablename__ = "documents"

    # UUID identifying the document (Primary Key)
    document_id = Column(Uuid, primary_key=True, default=uuid.uuid4, index=True)

    # Original uploaded filename
    filename = Column(String(255), nullable=False)

    # Supported states ONLY: pending, indexed, failed
    status = Column(String(50), nullable=False, default="pending")

    # Number of chunks associated with the document
    chunk_count = Column(Integer, nullable=False, default=0)

    # Original uploaded file size in bytes
    file_size = Column(Integer, nullable=False, default=0)

    # Timestamp representing when the document was registered
    uploaded_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Enforce status constraint on the database level
    __table_args__ = (
        CheckConstraint(
            status.in_(["pending", "indexed", "failed"]),
            name="check_valid_status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(document_id={self.document_id}, filename='{self.filename}', "
            f"status='{self.status}', chunk_count={self.chunk_count})>"
        )
