"""
SQLAlchemy ORM models for the SQLite caching and run-log database.

Tables:
  - search_cache   : Cached web search results (keyed by query hash)
  - webpage_cache  : Cached webpage content (keyed by URL hash)
  - run_log        : Per-run state snapshots for traceability
"""

from datetime import datetime

from sqlalchemy import (
    Column, String, Text, Integer, DateTime, Boolean, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class SearchCache(Base):
    """Cached web search results."""
    __tablename__ = "search_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_hash = Column(String(16), unique=True, nullable=False, index=True)
    query = Column(Text, nullable=False)
    results_json = Column(Text, nullable=False)      # JSON-serialized list[SearchResult]
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)


class WebpageCache(Base):
    """Cached webpage content."""
    __tablename__ = "webpage_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url_hash = Column(String(16), unique=True, nullable=False, index=True)
    url = Column(Text, nullable=False)
    content_json = Column(Text, nullable=False)      # JSON: {title, content, status_code}
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_paywalled = Column(Boolean, default=False)


class RunLog(Base):
    """Per-run state log for traceability and debugging."""
    __tablename__ = "run_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), unique=True, nullable=False, index=True)
    document_id = Column(String(36), nullable=True)
    file_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(50), default="running")   # running, complete, failed
    current_step = Column(String(100), nullable=True)
    state_json = Column(Text, nullable=True)          # Full serialized EstimationState
    error_message = Column(Text, nullable=True)


def get_engine():
    """Create SQLAlchemy engine from configured database URL."""
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},  # Required for SQLite + threading
    )
    return engine


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Create a new database session."""
    from sqlalchemy.orm import sessionmaker
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()
