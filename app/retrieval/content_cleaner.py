"""
Webpage Content Cleaner — converts raw HTML to clean, LLM-ready text.

Uses trafilatura as primary extractor with BeautifulSoup as fallback.
Preserves tables and key content while removing navigation, ads, boilerplate.
"""

import re
import structlog

logger = structlog.get_logger(__name__)

# Maximum characters to retain per page for LLM context
MAX_PAGE_CHARS = 15_000


def clean_html_to_text(
    html: str,
    url: str = "",
    max_chars: int = MAX_PAGE_CHARS,
) -> str:
    """
    Extract clean main content from HTML.

    Strategy:
      1. Try trafilatura (best for news/articles)
      2. Fall back to BeautifulSoup for structured pages

    Args:
        html: Raw HTML string.
        url: Source URL (used for context in logging).
        max_chars: Maximum characters to return.

    Returns:
        Clean text string.
    """
    text = _trafilatura_extract(html, url)

    if not text or len(text.strip()) < 100:
        text = _beautifulsoup_extract(html)

    if not text:
        return ""

    # Normalize
    text = _normalize_text(text)

    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated ...]"

    return text.strip()


def _trafilatura_extract(html: str, url: str = "") -> str:
    """Use trafilatura to extract main content."""
    try:
        import trafilatura
        result = trafilatura.extract(
            html,
            url=url or None,
            include_tables=True,
            include_links=False,
            include_images=False,
            no_fallback=False,
            favor_precision=False,
            favor_recall=True,
        )
        return result or ""
    except Exception as e:
        logger.debug("trafilatura_extraction_failed", error=str(e))
        return ""


def _beautifulsoup_extract(html: str) -> str:
    """Fallback: use BeautifulSoup to extract text from HTML."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "advertisement", "iframe", "form"]):
            tag.decompose()

        # Extract tables as text
        tables_text: list[str] = []
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                tables_text.append("\n".join(rows))

        # Get main body text
        body = soup.find("body")
        if body:
            text = body.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")

        # Append tables
        if tables_text:
            text += "\n\n" + "\n\n".join(tables_text)

        return text
    except Exception as e:
        logger.debug("beautifulsoup_extraction_failed", error=str(e))
        return ""


def _normalize_text(text: str) -> str:
    """Normalize whitespace and clean extracted text."""
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace tabs with space
    text = text.replace("\t", " ")
    # Collapse multiple spaces
    text = re.sub(r" {3,}", "  ", text)
    # Collapse more than 2 blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_paywall(html: str, status_code: int = 200) -> bool:
    """
    Detect common paywall patterns in HTML.

    Returns:
        True if the content appears to be paywalled.
    """
    if status_code in (401, 403, 402):
        return True

    paywall_indicators = [
        "subscribe to read",
        "subscription required",
        "sign in to read",
        "create a free account",
        "this article is for subscribers",
        "this content is for subscribers",
        "subscribers only",
        "for subscribers",
        "unlock this article",
        "register to continue reading",
        "paywall",
        "premium content",
    ]

    html_lower = html.lower()
    matches = sum(1 for phrase in paywall_indicators if phrase in html_lower)
    return matches >= 2
