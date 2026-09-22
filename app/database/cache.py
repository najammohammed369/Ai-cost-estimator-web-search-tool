"""
Cache Manager — high-level interface for SQLite-backed caching.

Wraps the SQLAlchemy models with TTL-aware get/set methods for:
  - Search query results
  - Webpage content
  - Run state snapshots
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import structlog

from app.database.models import SearchCache, WebpageCache, RunLog, get_session, init_db
from app.config import get_settings

logger = structlog.get_logger(__name__)


def _hash_key(value: str) -> str:
    """Generate a short SHA-256 hash for use as a cache key."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class CacheManager:
    """
    Thread-safe cache manager backed by SQLite via SQLAlchemy.

    All methods create their own session and close it after the operation,
    making this safe for use in multi-threaded FastAPI handlers.
    """

    def __init__(self):
        # Ensure tables exist
        init_db()
        settings = get_settings()
        self.search_ttl = timedelta(seconds=settings.search_cache_ttl)
        self.webpage_ttl = timedelta(seconds=settings.webpage_cache_ttl)

    # -------------------------------------------------------------------------
    # Search Result Cache
    # -------------------------------------------------------------------------

    def get_search_results(self, query: str) -> Optional[list[dict]]:
        """
        Retrieve cached search results for a query.

        Returns:
            List of result dicts, or None if not cached / expired.
        """
        query_hash = _hash_key(query)
        session = get_session()
        try:
            row = session.query(SearchCache).filter_by(query_hash=query_hash).first()
            if row is None:
                return None
            if row.expires_at and row.expires_at < datetime.utcnow():
                session.delete(row)
                session.commit()
                logger.debug("search_cache_expired", query=query[:60])
                return None
            return json.loads(row.results_json)
        except Exception as e:
            logger.error("search_cache_get_failed", error=str(e))
            return None
        finally:
            session.close()

    def set_search_results(self, query: str, results: list[dict]) -> None:
        """
        Store search results in cache.

        Args:
            query: The search query string.
            results: List of serializable result dicts.
        """
        query_hash = _hash_key(query)
        expires_at = datetime.utcnow() + self.search_ttl
        session = get_session()
        try:
            existing = session.query(SearchCache).filter_by(query_hash=query_hash).first()
            results_json = json.dumps(results, ensure_ascii=False)
            if existing:
                existing.results_json = results_json
                existing.expires_at = expires_at
            else:
                session.add(SearchCache(
                    query_hash=query_hash,
                    query=query,
                    results_json=results_json,
                    expires_at=expires_at,
                ))
            session.commit()
            logger.debug("search_cache_set", query=query[:60], results=len(results))
        except Exception as e:
            logger.error("search_cache_set_failed", error=str(e))
            session.rollback()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Webpage Cache
    # -------------------------------------------------------------------------

    def get_webpage(self, url: str) -> Optional[dict]:
        """
        Retrieve cached webpage content.

        Returns:
            Dict with {title, content, status_code}, or None if not cached / expired.
        """
        url_hash = _hash_key(url)
        session = get_session()
        try:
            row = session.query(WebpageCache).filter_by(url_hash=url_hash).first()
            if row is None:
                return None
            if row.expires_at and row.expires_at < datetime.utcnow():
                session.delete(row)
                session.commit()
                return None
            return json.loads(row.content_json)
        except Exception as e:
            logger.error("webpage_cache_get_failed", url=url[:80], error=str(e))
            return None
        finally:
            session.close()

    def set_webpage(self, url: str, data: dict, is_paywalled: bool = False) -> None:
        """
        Store webpage content in cache.

        Args:
            url: The page URL.
            data: Dict with {title, content, status_code}.
            is_paywalled: Whether the page appeared to be paywalled.
        """
        url_hash = _hash_key(url)
        expires_at = datetime.utcnow() + self.webpage_ttl
        session = get_session()
        try:
            existing = session.query(WebpageCache).filter_by(url_hash=url_hash).first()
            content_json = json.dumps(data, ensure_ascii=False)
            if existing:
                existing.content_json = content_json
                existing.expires_at = expires_at
                existing.is_paywalled = is_paywalled
            else:
                session.add(WebpageCache(
                    url_hash=url_hash,
                    url=url,
                    content_json=content_json,
                    expires_at=expires_at,
                    is_paywalled=is_paywalled,
                ))
            session.commit()
            logger.debug("webpage_cache_set", url=url[:80])
        except Exception as e:
            logger.error("webpage_cache_set_failed", url=url[:80], error=str(e))
            session.rollback()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # Run Log
    # -------------------------------------------------------------------------

    def create_run(self, run_id: str, document_id: str, file_name: str) -> None:
        """Create a new run log entry."""
        session = get_session()
        try:
            session.add(RunLog(
                run_id=run_id,
                document_id=document_id,
                file_name=file_name,
                status="running",
            ))
            session.commit()
        except Exception as e:
            logger.error("run_log_create_failed", run_id=run_id, error=str(e))
            session.rollback()
        finally:
            session.close()

    def update_run(
        self,
        run_id: str,
        current_step: str,
        state: dict,
        status: str = "running",
        error_message: Optional[str] = None,
    ) -> None:
        """Update an existing run log with current state."""
        session = get_session()
        try:
            row = session.query(RunLog).filter_by(run_id=run_id).first()
            if row:
                row.current_step = current_step
                row.status = status
                row.state_json = json.dumps(state, ensure_ascii=False, default=str)
                row.error_message = error_message
                row.updated_at = datetime.utcnow()
                session.commit()
        except Exception as e:
            logger.error("run_log_update_failed", run_id=run_id, error=str(e))
            session.rollback()
        finally:
            session.close()

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get a run log entry by run_id."""
        session = get_session()
        try:
            row = session.query(RunLog).filter_by(run_id=run_id).first()
            if row is None:
                return None
            return {
                "run_id": row.run_id,
                "document_id": row.document_id,
                "file_name": row.file_name,
                "status": row.status,
                "current_step": row.current_step,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "error_message": row.error_message,
                "state": json.loads(row.state_json) if row.state_json else None,
            }
        except Exception as e:
            logger.error("run_log_get_failed", run_id=run_id, error=str(e))
            return None
        finally:
            session.close()
