"""
Database: SQLite engine connection setup, session management, and Base models definitions.
"""

from __future__ import annotations

import logging
from typing import Generator as TypeGenerator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.core.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base class
Base = declarative_base()

# Engine creation (enable check_same_thread=False for SQLite multithreaded/concurrent route calls)
engine_args = {}
if settings.SQLITE_DB_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.SQLITE_DB_URL, **engine_args)

# Session factory setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize SQLite database and create all tables registered under Declarative Base.
    """
    logger.info("Initializing SQLite database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.error(f"Failed to create database tables: {exc}")
        raise exc


def get_db() -> TypeGenerator[Session, None, None]:
    """
    FastAPI dependency/session helper yielding clean SQL database sessions.
    Ensure sessions are closed immediately after operations complete.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
