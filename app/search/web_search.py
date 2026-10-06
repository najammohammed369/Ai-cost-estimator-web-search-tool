"""
Web Search Abstraction — pluggable search provider interface + SerpAPI implementation.

Providers:
  - SerpAPIProvider (default)
  - (Extensible: add GoogleCustomSearch, SerperProvider, BraveProvider)

All results are normalized to SearchResult schema and cached in SQLite.
"""

import hashlib
import time
from abc import ABC, abstractmethod

import structlog

from app.extraction.schemas import SearchResult
from app.config import get_settings

logger = structlog.get_logger(__name__)


def _classify_source_tier(url: str) -> int:
    """
    Classify URL into a source quality tier.

    Tier 1: Government, military, official procurement
    Tier 2: Major news, maritime publications, financial reports
    Tier 3: Ship databases, industry aggregators
    Tier 4: Blogs, forums, unverified sites
    """
    url_lower = url.lower()

    # Tier 1 indicators
    tier1_patterns = [
        ".gov", ".mil", ".navy", ".mod.uk", ".defense.gov",
        "mod.gov", "navy.mil", "dod.gov", "government.", "parliament.",
        "bundesmarine", "royalnavy.mod", "marine.defense",
        "shipbuilder.com/press", "shipbuilder.com/news",
    ]
    for pattern in tier1_patterns:
        if pattern in url_lower:
            return 1

    # Tier 2 indicators
    tier2_patterns = [
        "reuters.com", "bbc.com", "defensenews.com", "janes.com",
        "navaltoday.com", "navalnews.com", "navyrecognition.com",
        "marinelink.com", "maritimeexecutive.com", "lloydslist.com",
        "theloadstar.com", "tradewindsnews.com", "seatrade",
        "defenseindustrydaily.com", "breakingdefense.com",
        "shephard.co.uk", "monch.com", "bloomberg.com",
    ]
    for pattern in tier2_patterns:
        if pattern in url_lower:
            return 2

    # Tier 3 indicators
    tier3_patterns = [
        "shipspotting.com", "fleetmon.com", "marinetraffic.com",
        "globalsecurity.org", "military-today.com", "seaforces.org",
        "navsource.org", "hazegray.org", "shipbuildinghistory.com",
        "mdba.com", "mbda", "baesystems.com", "thalesgroup.com",
        "damen.com", "fincantieri.com", "navantia.es",
    ]
    for pattern in tier3_patterns:
        if pattern in url_lower:
            return 3

    return 4  # Default: unclassified / blog / forum


# =============================================================================
# Abstract Search Provider
# =============================================================================

class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """
        Execute a web search and return results.

        Args:
            query: Search query string.
            num_results: Number of results to return.

        Returns:
            List of SearchResult objects.
        """
        ...

    def search_batch(
        self,
        queries: list[str],
        num_results: int = 10,
        delay_seconds: float = 1.0,
    ) -> dict[str, list[SearchResult]]:
        """
        Execute multiple searches sequentially with a delay.

        Args:
            queries: List of query strings.
            num_results: Results per query.
            delay_seconds: Delay between requests.

        Returns:
            Dict mapping query -> list of SearchResult.
        """
        results: dict[str, list[SearchResult]] = {}
        for i, query in enumerate(queries):
            logger.info("batch_search_progress", query_num=i + 1, total=len(queries))
            results[query] = self.search(query, num_results=num_results)
            if i < len(queries) - 1:
                time.sleep(delay_seconds)
        return results


# =============================================================================
# SerpAPI Implementation
# =============================================================================

class SerpAPIProvider(SearchProvider):
    """
    Web search via SerpAPI (supports Google, Bing, and other engines).
    Requires SERPAPI_KEY in environment.
    """

    def __init__(self, api_key: str | None = None, engine: str = "google"):
        settings = get_settings()
        self.api_key = api_key or settings.serpapi_key
        self.engine = engine

        if not self.api_key:
            raise ValueError(
                "SERPAPI_KEY is not configured. Set it in .env or pass it directly."
            )

        logger.info("serpapi_provider_initialized", engine=engine)

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Execute a Google search via SerpAPI."""
        try:
            # Attempt to import the appropriate SerpAPI client.
            from serpapi import GoogleSearch  # older version
            SearchClient = GoogleSearch
        except ImportError:
            try:
                from serpapi import GoogleSearchResults  # newer google-search-results package
                SearchClient = GoogleSearchResults
            except ImportError as e:
                raise ImportError(
                    "SerpAPI client not found. Install 'google-search-results' package."
                ) from e
        except ImportError:
            # Fallback for newer serpapi layout
            from serpapi.google_search import GoogleSearch

        logger.info("serpapi_search", query=query[:80], num_results=num_results)

        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": self.engine,
                "num": min(num_results, 10),  # SerpAPI max is 10 per call
                "gl": "us",   # Country for results
                "hl": "en",   # Language
            }

            search = SearchClient(params)
            raw = search.get_dict()

            results: list[SearchResult] = []

            # Organic results
            for i, item in enumerate(raw.get("organic_results", [])[:num_results]):
                url = item.get("link", "")
                results.append(
                    SearchResult(
                        query=query,
                        title=item.get("title", ""),
                        url=url,
                        snippet=item.get("snippet", ""),
                        source=item.get("source", ""),
                        published_date=item.get("date"),
                        position=i + 1,
                        source_tier=_classify_source_tier(url),
                    )
                )

            logger.info(
                "serpapi_search_complete",
                query=query[:80],
                results_returned=len(results),
            )
            return results

        except Exception as e:
            logger.error("serpapi_search_failed", query=query[:80], error=str(e))
            return []


# =============================================================================
# Provider Factory
# =============================================================================

def get_search_provider(provider_name: str | None = None) -> SearchProvider:
    """
    Factory function to get the configured search provider.

    Args:
        provider_name: Provider name (overrides config). Options: "serpapi".

    Returns:
        SearchProvider instance.
    """
    settings = get_settings()
    name = (provider_name or settings.search_provider).lower()

    if name == "serpapi":
        return SerpAPIProvider()
    else:
        raise ValueError(
            f"Unknown search provider: '{name}'. Supported: 'serpapi'. "
            f"Extend web_search.py to add more providers."
        )


# =============================================================================
# Query hash utility
# =============================================================================

def hash_query(query: str) -> str:
    """Generate a short hash of a query string for caching."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]
