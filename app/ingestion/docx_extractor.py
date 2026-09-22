"""
DOCX Extractor — uses python-docx to extract text and tables from Word documents.

Extracts paragraphs, headings, and tables while approximating page boundaries
via section breaks.
"""

from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DocxPageContent:
    """Content extracted from an approximate 'page' of a DOCX document."""
    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    char_count: int = 0


@dataclass
class DocxExtractionResult:
    """Result of extracting a full DOCX document."""
    file_path: str
    total_pages: int
    pages: list[DocxPageContent]
    full_text: str            # Complete document text for LLM context
    warnings: list[str] = field(default_factory=list)


def extract_docx(file_path: str | Path) -> DocxExtractionResult:
    """
    Extract text and tables from a DOCX file using python-docx.

    DOCX files don't have explicit page boundaries, so we approximate pages
    by splitting on page breaks embedded in paragraphs. If no page breaks
    are found, the full document is treated as a single page.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        DocxExtractionResult with per-page content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid DOCX.
    """
    from docx import Document
    from docx.oxml.ns import qn

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")
    if file_path.suffix.lower() not in (".docx", ".doc"):
        raise ValueError(f"Not a DOCX file: {file_path}")

    logger.info("docx_extraction_start", file=str(file_path))

    try:
        doc = Document(str(file_path))
    except Exception as e:
        raise ValueError(f"Could not open DOCX '{file_path}': {e}") from e

    warnings: list[str] = []

    # Extract all paragraphs, splitting on page breaks
    pages: list[DocxPageContent] = []
    current_page_lines: list[str] = []
    page_number = 1

    def has_page_break(paragraph) -> bool:
        """Check if a paragraph contains a page break."""
        for run in paragraph.runs:
            if run._element.find(qn("w:lastRenderedPageBreak")) is not None:
                return True
            if run._element.find(qn("w:br")) is not None:
                br_elem = run._element.find(qn("w:br"))
                if br_elem is not None and br_elem.get(qn("w:type")) == "page":
                    return True
        return False

    def flush_page():
        """Save current lines as a page."""
        nonlocal page_number, current_page_lines
        text = "\n".join(current_page_lines).strip()
        if text:
            pages.append(DocxPageContent(
                page_number=page_number,
                text=text,
                char_count=len(text),
            ))
            page_number += 1
        current_page_lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            current_page_lines.append(text)
        if has_page_break(para):
            flush_page()

    # Flush remaining content
    flush_page()

    # If no page breaks found (common), treat as single page
    if not pages:
        all_text = "\n".join(
            p.text.strip() for p in doc.paragraphs if p.text.strip()
        )
        pages.append(DocxPageContent(
            page_number=1,
            text=all_text,
            char_count=len(all_text),
        ))
        warnings.append(
            "No page breaks detected in DOCX — document treated as a single page."
        )

    # Extract tables and assign to nearest page (simplified: add to last page)
    for table in doc.tables:
        rows: list[list[str]] = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(cells)
        if rows and pages:
            pages[-1].tables.append(rows)

    # Build full text for LLM context (join all pages)
    full_text = "\n\n".join(
        f"[Page {p.page_number}]\n{p.text}" for p in pages
    )

    logger.info(
        "docx_extraction_complete",
        file=str(file_path),
        total_pages=len(pages),
        total_chars=len(full_text),
        table_count=sum(len(p.tables) for p in pages),
    )

    return DocxExtractionResult(
        file_path=str(file_path),
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
        warnings=warnings,
    )
