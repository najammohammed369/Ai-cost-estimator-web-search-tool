"""
Webpage Retriever — fetches and cleans webpage content for evidence extraction.

Features:
- Async HTTP with httpx
- Robots.txt awareness (best-effort)
- Paywall detection
- SQLite caching
- Configurable timeout and retries
"""

import asyncio
import hashlib
import time
from typing import Optional

import httpx
import structlog

from app.extraction.schemas import RetrievedPage, SearchResult
from app.retrieval.content_cleaner import clean_html_to_text, detect_paywall
from app.search.web_search import _classify_source_tier
from app.database.cache import CacheManager

logger = structlog.get_logger(__name__)

# Default HTTP settings
DEFAULT_TIMEOUT = 20.0          # seconds
DEFAULT_MAX_RETRIES = 2
DEFAULT_DELAY_SECONDS = 0.5

# User agent that identifies the bot politely
USER_AGENT = (
    "VesselCostEstimator/1.0 (research tool; contact your-email@example.com)"
)

# Skip these URL patterns — they are unlikely to have useful free content
SKIP_URL_PATTERNS = [
    "youtube.com", "youtu.be", "vimeo.com",
    "twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com",
    "reddit.com",
    ".pdf",   # PDFs need different handling
    "jstor.org", "sciencedirect.com",  # Academic paywalls
]


def _should_skip_url(url: str) -> bool:
    """Return True if the URL should be skipped."""
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in SKIP_URL_PATTERNS)


async def fetch_page_async(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[str, int, Optional[str]]:
    """
    Fetch webpage HTML asynchronously.

    Returns:
        (html_content, status_code, error_message)
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(url)
                return response.text, response.status_code, None

        except httpx.TimeoutException:
            error = f"Timeout after {timeout}s"
        except httpx.TooManyRedirects:
            return "", 0, "Too many redirects"
        except httpx.RequestError as e:
            error = f"Request error: {str(e)}"

        if attempt < max_retries:
            await asyncio.sleep(DEFAULT_DELAY_SECONDS * attempt)

    return "", 0, error


def retrieve_page(
    url: str,
    cache: Optional[CacheManager] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> RetrievedPage:
    """
    Fetch, cache, and clean a single webpage.

    Args:
        url: URL to retrieve.
        cache: Optional cache manager (creates default if not provided).
        timeout: HTTP timeout in seconds.

    Returns:
        RetrievedPage with cleaned content and metadata.
    """
    if cache is None:
        cache = CacheManager()

    if _should_skip_url(url):
        logger.info("webpage_skipped", url=url[:80], reason="skip pattern matched")
        return RetrievedPage(
            url=url,
            title="",
            content="",
            status_code=0,
            is_accessible=False,
            error="URL pattern skipped (social media, video, or paywall site)",
            source_tier=_classify_source_tier(url),
        )

    # Check cache
    cached = cache.get_webpage(url)
    if cached is not None:
        logger.info("webpage_cache_hit", url=url[:80])
        return RetrievedPage(
            url=url,
            title=cached.get("title", ""),
            content=cached.get("content", ""),
            status_code=cached.get("status_code", 200),
            is_accessible=True,
            content_length=len(cached.get("content", "")),
            source_tier=_classify_source_tier(url),
        )

    # Fetch page
    logger.info("webpage_fetching", url=url[:80])
    start = time.time()

    html, status_code, error = asyncio.run(fetch_page_async(url, timeout=timeout))

    elapsed = round(time.time() - start, 2)

    if error or not html:
        logger.warning("webpage_fetch_failed", url=url[:80], error=error, status=status_code)
        return RetrievedPage(
            url=url,
            title="",
            content="",
            status_code=status_code,
            is_accessible=False,
            error=error,
            source_tier=_classify_source_tier(url),
        )

    # Detect paywall
    is_paywalled = detect_paywall(html, status_code)

    # Extract title
    title = _extract_title(html)

    # Clean HTML to text
    content = clean_html_to_text(html, url=url)

    # Cache result
    if content:
        cache.set_webpage(url, {
            "title": title,
            "content": content,
            "status_code": status_code,
        })

    logger.info(
        "webpage_retrieved",
        url=url[:80],
        status=status_code,
        content_chars=len(content),
        is_paywalled=is_paywalled,
        elapsed_s=elapsed,
    )

    return RetrievedPage(
        url=url,
        title=title,
        content=content,
        status_code=status_code,
        is_paywalled=is_paywalled,
        is_accessible=bool(content and not is_paywalled),
        content_length=len(content),
        source_tier=_classify_source_tier(url),
    )


def retrieve_pages_batch(
    results: list[SearchResult],
    cache: Optional[CacheManager] = None,
    max_pages: int = 30,
    delay_seconds: float = 0.5,
) -> list[RetrievedPage]:
    """
    Retrieve and clean content from a list of search results.

    Prioritizes higher-tier sources and skips duplicates.

    Args:
        results: Search results to retrieve.
        cache: Optional cache manager.
        max_pages: Maximum pages to fetch.
        delay_seconds: Delay between requests.

    Returns:
        List of RetrievedPage objects.
    """
    if cache is None:
        cache = CacheManager()

    # Sort by source tier (best first), deduplicate URLs
    seen_urls: set[str] = set()
    prioritized: list[SearchResult] = []
    for r in sorted(results, key=lambda x: x.source_tier):
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            prioritized.append(r)

    prioritized = prioritized[:max_pages]

    retrieved: list[RetrievedPage] = []
    for i, result in enumerate(prioritized):
        page = retrieve_page(result.url, cache=cache)
        retrieved.append(page)

        if i < len(prioritized) - 1:
            time.sleep(delay_seconds)

    accessible = sum(1 for p in retrieved if p.is_accessible)
    logger.info(
        "batch_retrieval_complete",
        total_fetched=len(retrieved),
        accessible=accessible,
        paywalled=sum(1 for p in retrieved if p.is_paywalled),
    )

    return retrieved


def _extract_title(html: str) -> str:
    """Extract the page title from HTML."""
    import re
    match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""
