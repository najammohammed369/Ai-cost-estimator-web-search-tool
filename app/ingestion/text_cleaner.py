"""
Text Cleaner — removes noise from extracted document text.

Handles:
- Repeated headers and footers (detected via cross-page duplication)
- Excessive whitespace and control characters
- Logical section detection
"""

import re
from collections import Counter
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CleanedPage:
    """A cleaned page with normalized text."""
    page_number: int
    raw_text: str
    cleaned_text: str
    removed_lines: list[str]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces and blank lines."""
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Replace tabs with space
    text = text.replace("\t", " ")
    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)
    # Collapse more than 2 consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_repeated_lines(pages_text: list[str], threshold: float = 0.6) -> set[str]:
    """
    Detect lines that appear on many pages (likely headers/footers).

    A line is flagged as repeated if it appears on more than `threshold`
    proportion of pages.

    Args:
        pages_text: List of raw text strings, one per page.
        threshold: Proportion of pages a line must appear on to be flagged.

    Returns:
        Set of normalized line strings that are likely headers/footers.
    """
    if len(pages_text) < 3:
        # Not enough pages to reliably detect headers/footers
        return set()

    # Count how many pages each line appears on
    line_page_counts: Counter = Counter()
    for text in pages_text:
        # Get unique lines per page (avoid counting repeated content within a page)
        page_lines = set(
            line.strip()
            for line in text.split("\n")
            if len(line.strip()) > 3  # Skip very short lines
        )
        for line in page_lines:
            line_page_counts[line] += 1

    num_pages = len(pages_text)
    repeated = {
        line
        for line, count in line_page_counts.items()
        if count / num_pages >= threshold
    }

    if repeated:
        logger.debug(
            "detected_repeated_lines",
            count=len(repeated),
            examples=list(repeated)[:3],
        )

    return repeated


def clean_page_text(
    page_text: str,
    page_number: int,
    repeated_lines: set[str],
) -> CleanedPage:
    """
    Clean a single page's text by removing noise.

    Args:
        page_text: Raw extracted text.
        page_number: 1-indexed page number.
        repeated_lines: Set of lines known to be headers/footers.

    Returns:
        CleanedPage with cleaned text and a log of removed lines.
    """
    raw_text = page_text
    lines = page_text.split("\n")
    cleaned_lines: list[str] = []
    removed_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip lines flagged as repeated headers/footers
        if stripped in repeated_lines:
            removed_lines.append(stripped)
            continue

        # Skip lines that are just page numbers (e.g., "- 1 -", "Page 1", "1")
        if re.match(r"^[-–—]?\s*\d+\s*[-–—]?$", stripped):
            removed_lines.append(stripped)
            continue
        if re.match(r"^[Pp]age\s+\d+(\s+of\s+\d+)?$", stripped):
            removed_lines.append(stripped)
            continue

        cleaned_lines.append(line)

    cleaned_text = normalize_whitespace("\n".join(cleaned_lines))

    return CleanedPage(
        page_number=page_number,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        removed_lines=removed_lines,
    )


def clean_document_pages(
    pages: list[tuple[int, str]],
    remove_headers_footers: bool = True,
) -> list[CleanedPage]:
    """
    Clean all pages of a document.

    Args:
        pages: List of (page_number, text) tuples.
        remove_headers_footers: Whether to detect and remove repeated lines.

    Returns:
        List of CleanedPage objects.
    """
    pages_text = [text for _, text in pages]

    # Detect repeated lines (headers/footers) across all pages
    repeated_lines: set[str] = set()
    if remove_headers_footers and len(pages) >= 3:
        repeated_lines = detect_repeated_lines(pages_text)

    cleaned: list[CleanedPage] = []
    for page_number, text in pages:
        cleaned.append(
            clean_page_text(text, page_number, repeated_lines)
        )

    total_removed = sum(len(p.removed_lines) for p in cleaned)
    logger.info(
        "document_cleaning_complete",
        total_pages=len(cleaned),
        total_removed_lines=total_removed,
        repeated_lines_detected=len(repeated_lines),
    )

    return cleaned


def split_into_sections(text: str) -> list[dict]:
    """
    Split document text into logical sections based on heading patterns.

    Returns:
        List of dicts: {"title": str, "content": str}
    """
    # Patterns that look like section headings
    heading_pattern = re.compile(
        r"^(?:"
        r"\d+[\.\)]\s+[A-Z][^\n]{3,60}"   # "1. Title" or "1) Title"
        r"|[A-Z][A-Z\s]{4,50}[A-Z]"         # ALL CAPS heading (4+ chars)
        r"|#{1,3}\s+.+"                      # Markdown headings
        r")$",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"title": "Document", "content": text}]

    sections: list[dict] = []
    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append({"title": title, "content": content})

    # Prepend any text before the first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, {"title": "Preamble", "content": preamble})

    return sections
