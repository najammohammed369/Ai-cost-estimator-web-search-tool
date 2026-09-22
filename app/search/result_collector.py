"""
Result Collector — orchestrates multiple search queries, deduplicates results,
and organizes them by relevance and source quality.
"""

import structlog
from collections import defaultdict

from app.extraction.schemas import SearchResult, SearchQuery
from app.search.web_search import SearchProvider, get_search_provider
from app.database.cache import CacheManager

logger = structlog.get_logger(__name__)


class ResultCollector:
    """
    Manages the execution of multiple search queries and collection of results.

    Features:
    - Deduplication by URL
    - SQLite caching to avoid repeat searches
    - Source tier tracking
    - Organized output by query category
    """

    def __init__(
        self,
        provider: SearchProvider | None = None,
        cache: CacheManager | None = None,
        num_results_per_query: int = 10,
        delay_between_requests: float = 1.0,
    ):
        self.provider = provider or get_search_provider()
        self.cache = cache or CacheManager()
        self.num_results_per_query = num_results_per_query
        self.delay = delay_between_requests

    def collect(
        self,
        queries: list[dict],
        run_id: str,
    ) -> list[SearchResult]:
        """
        Execute a list of search queries and collect all results.

        Results are:
        1. Checked in cache first (skip API call if cached)
        2. Deduplicated by URL across all queries
        3. Sorted by source tier (1=best) then position

        Args:
            queries: List of query dicts with 'query', 'category', 'priority'.
            run_id: Run identifier for logging.

        Returns:
            Deduplicated, sorted list of SearchResult objects.
        """
        seen_urls: set[str] = set()
        all_results: list[SearchResult] = []
        results_by_category: dict[str, list[SearchResult]] = defaultdict(list)

        # Sort by priority (highest first)
        sorted_queries = sorted(
            queries, key=lambda q: q.get("priority", 5), reverse=True
        )

        total = len(sorted_queries)
        for i, query_dict in enumerate(sorted_queries):
            query_str = query_dict.get("query", "")
            category = query_dict.get("category", "general")

            if not query_str:
                continue

            logger.info(
                "executing_search",
                run_id=run_id,
                query_num=i + 1,
                total=total,
                category=category,
                query=query_str[:80],
            )

            # Check cache first
            cached = self.cache.get_search_results(query_str)
            if cached is not None:
                results = [SearchResult(**r) for r in cached]
                logger.info(
                    "search_cache_hit",
                    query=query_str[:60],
                    cached_results=len(results),
                )
            else:
                results = self.provider.search(
                    query=query_str,
                    num_results=self.num_results_per_query,
                )
                # Cache the results
                if results:
                    self.cache.set_search_results(
                        query_str,
                        [r.model_dump() for r in results],
                    )

                # Rate limiting delay (not needed after last query)
                if i < total - 1 and results:
                    import time
                    time.sleep(self.delay)

            # Deduplicate by URL and collect
            for result in results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
                    results_by_category[category].append(result)

        # Sort: Tier 1 first, then by position within tier
        all_results.sort(key=lambda r: (r.source_tier, r.position))

        logger.info(
            "result_collection_complete",
            run_id=run_id,
            queries_executed=total,
            unique_results=len(all_results),
            tier1_results=sum(1 for r in all_results if r.source_tier == 1),
            tier2_results=sum(1 for r in all_results if r.source_tier == 2),
        )

        return all_results

    def add_vessel_specific_queries(
        self,
        vessel_names: list[str],
        run_id: str,
    ) -> list[SearchResult]:
        """
        Generate and execute cost-specific queries for discovered vessel names.

        Used after initial search to drill into specific candidate vessels.

        Args:
            vessel_names: List of vessel names to search for.
            run_id: Run identifier.

        Returns:
            Additional search results for the specific vessels.
        """
        vessel_queries = []
        for name in vessel_names:
            for suffix in [
                "contract value",
                "procurement cost",
                "construction cost",
                "shipbuilding contract",
                "acquisition cost",
                "delivered cost",
            ]:
                vessel_queries.append({
                    "query": f'"{name}" {suffix}',
                    "category": "vessel_specific",
                    "rationale": f"Cost evidence search for {name}",
                    "priority": 9,  # High priority
                })

        logger.info(
            "vessel_specific_queries_generated",
            run_id=run_id,
            vessel_count=len(vessel_names),
            query_count=len(vessel_queries),
        )

        return self.collect(vessel_queries, run_id=run_id)
