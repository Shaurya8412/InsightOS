import logging
import os
from uuid import uuid4, UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.core.exceptions import (
    CitationError,
    DocumentExtractionError,
    EmbeddingError,
    GenerationError,
    RetrievalError,
    VectorStoreError,
)
from src.models.db_models import Document
from src.models.schemas import DocumentResponse, QueryRequest, QueryResponse
from src.services.embeddings.provider import get_embedding_provider
from src.services.ingestion.chunker import chunk_pages
from src.services.ingestion.parser import parse_document
from src.services.rag.citation import CitationHandler
from src.services.rag.generator import Generator
from src.services.rag.orchestrator import Orchestrator
from src.services.rag.retriever import Retriever
from src.services.vector_store.provider import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.post("/documents/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload and ingest a document (.pdf, .txt, .md).
    Runs parsing, chunking, embedding, and vector database insertion.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # 1. Extension Validation
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    allowed_exts = {".pdf", ".txt", ".md"}
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed types: .pdf, .txt, .md",
        )

    # 2. Read bytes & check empty
    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read file content: {exc}"
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = uuid4()

    # Create and commit pending record
    doc_record = Document(
        document_id=document_id,
        filename=filename,
        status="pending",
        chunk_count=0,
        file_size=len(file_bytes),
    )
    try:
        db.add(doc_record)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(f"Database error registering document '{filename}': {exc}")
        raise HTTPException(
            status_code=500, detail=f"Database error during document registration: {exc}"
        )

    # Ingestion pipeline execution
    try:
        # 3. Parsing
        try:
            pages = parse_document(file_bytes, filename)
        except DocumentExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error during document parsing: {exc}"
            )

        # 4. Chunking
        try:
            chunks = chunk_pages(pages, document_id, filename)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error during document chunking: {exc}"
            )

        # 5. Embedding
        try:
            provider = get_embedding_provider()
            embeddings = provider.embed_texts([c.text for c in chunks])
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error during embedding generation: {exc}"
            )

        # 6. Storage Insertion
        try:
            store = get_vector_store()
            store.upsert_chunks(chunks, embeddings)
        except VectorStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Unexpected error during vector storage: {exc}"
            )

        # Update database record to indexed
        doc_record.status = "indexed"
        doc_record.chunk_count = len(chunks)
        db.commit()
    except Exception as exc:
        # Ingestion failed: update database status to failed
        db.rollback()
        try:
            doc_record.status = "failed"
            db.commit()
        except Exception as db_exc:
            db.rollback()
            logger.error(f"Failed to update document status to 'failed': {db_exc}")
        # Re-raise the exception
        raise exc

    return {
        "document_id": str(document_id),
        "status": "success",
        "chunks_indexed": len(chunks),
    }


@router.get("/documents", response_model=list[DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    """
    Retrieve all persisted document records, ordered by uploaded_at descending.
    """
    try:
        docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
        return [
            DocumentResponse(
                document_id=doc.document_id,
                filename=doc.filename,
                status=doc.status,
                chunk_count=doc.chunk_count,
                file_size=doc.file_size,
                uploaded_at=doc.uploaded_at,
            )
            for doc in docs
        ]
    except Exception as exc:
        logger.error(f"Failed to fetch documents: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error while retrieving documents: {exc}",
        )


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline.
    Embeds query, retrieves matching chunks, invokes Gemini, and resolves citations.
    """
    try:
        # Construct pipeline services using clean factory patterns
        retriever = Retriever(
            embedding_provider=get_embedding_provider(),
            vector_store=get_vector_store(),
        )
        generator = Generator()
        citation_handler = CitationHandler()
        
        orchestrator = Orchestrator(
            retriever=retriever,
            generator=generator,
            citation_handler=citation_handler,
        )

        return orchestrator.query(request.query, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RetrievalError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except CitationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected orchestrator error: {exc}"
        )


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Delete a document and all associated vector chunks.
    Verifies document existence in SQL, calls vector store purge,
    and removes SQL row metadata.
    """
    # 1. Verify existence in SQLite
    doc = db.query(Document).filter_by(document_id=document_id).first()
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # 2. Attempt Qdrant vector deletion
    try:
        store = get_vector_store()
        store.delete(document_id)
    except VectorStoreError as exc:
        # Do not delete SQL record on vector purge failure (Task 3 Case C)
        logger.error(f"Vector store deletion failed for document '{document_id}': {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error during vector store deletion for document '{document_id}': {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to delete vector points: {exc}")

    # 3. Purge SQLite document metadata row
    try:
        db.delete(doc)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(f"Database error during deletion of document '{document_id}': {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error during document metadata removal: {exc}",
        )

    return {
        "status": "success",
        "detail": f"Document '{document_id}' and all associated vector points successfully deleted.",
    }
