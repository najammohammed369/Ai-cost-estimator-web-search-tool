"""
Document Processor — unified entry point for ingesting PDF, DOCX, and TXT files.

Generates a document_id, routes to the appropriate extractor, cleans the text,
and produces a standardized DocumentContent object that is cached to disk.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import structlog

from app.ingestion.pdf_extractor import extract_pdf
from app.ingestion.docx_extractor import extract_docx
from app.ingestion.text_cleaner import (
    clean_document_pages,
    split_into_sections,
)
from app.config import get_settings

logger = structlog.get_logger(__name__)

SUPPORTED_FORMATS = {".pdf", ".docx", ".doc", ".txt"}


@dataclass
class DocumentPage:
    """A single page of a processed document."""
    page_number: int
    raw_text: str
    cleaned_text: str
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass
class DocumentContent:
    """
    Standardized representation of an extracted document.

    This is the primary output of the ingestion pipeline and the input
    to the specification extraction stage.
    """
    document_id: str
    file_name: str
    file_path: str
    file_format: str
    total_pages: int
    pages: list[DocumentPage]
    full_text: str              # Complete cleaned text for LLM (with page markers)
    sections: list[dict]        # Logically segmented sections
    is_scanned: bool = False
    warnings: list[str] = field(default_factory=list)


def _build_full_text(pages: list[DocumentPage]) -> str:
    """Build annotated full text with page markers for LLM context."""
    parts: list[str] = []
    for page in pages:
        text = page.cleaned_text.strip()
        if text:
            parts.append(f"[PAGE {page.page_number}]\n{text}")
    return "\n\n".join(parts)


def process_document(file_path: str | Path) -> DocumentContent:
    """
    Process a vessel specification document and return structured content.

    Supports: PDF, DOCX, TXT
    Steps:
      1. Detect format
      2. Extract raw text + tables
      3. Clean text (remove headers/footers, normalize whitespace)
      4. Build full annotated text
      5. Split into logical sections
      6. Save to disk for traceability

    Args:
        file_path: Path to the document.

    Returns:
        DocumentContent with per-page cleaned text.

    Raises:
        ValueError: If the format is unsupported or the document is invalid.
        FileNotFoundError: If the file does not exist.
    """
    settings = get_settings()
    file_path = Path(file_path)
    document_id = str(uuid.uuid4())[:8]  # Short readable ID
    file_format = file_path.suffix.lower()

    if file_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: '{file_format}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    logger.info(
        "document_processing_start",
        document_id=document_id,
        file=str(file_path),
        format=file_format,
    )

    warnings: list[str] = []
    raw_pages: list[tuple[int, str]] = []
    all_tables: dict[int, list] = {}

    # --- Extract based on format ---
    if file_format == ".pdf":
        result = extract_pdf(file_path)
        warnings.extend(result.warnings)
        is_scanned = result.is_scanned
        for page in result.pages:
            raw_pages.append((page.page_number, page.text))
            if page.tables:
                all_tables[page.page_number] = page.tables

    elif file_format in (".docx", ".doc"):
        result = extract_docx(file_path)
        warnings.extend(result.warnings)
        is_scanned = False
        for page in result.pages:
            raw_pages.append((page.page_number, page.text))
            if page.tables:
                all_tables[page.page_number] = page.tables

    elif file_format == ".txt":
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        raw_pages = [(1, raw_text)]
        is_scanned = False

    # --- Clean pages ---
    cleaned = clean_document_pages(raw_pages, remove_headers_footers=True)

    # --- Build DocumentPage list ---
    pages: list[DocumentPage] = [
        DocumentPage(
            page_number=cp.page_number,
            raw_text=cp.raw_text,
            cleaned_text=cp.cleaned_text,
            tables=all_tables.get(cp.page_number, []),
        )
        for cp in cleaned
    ]

    # --- Build annotated full text ---
    full_text = _build_full_text(pages)

    # --- Detect sections ---
    sections = split_into_sections(full_text)

    doc_content = DocumentContent(
        document_id=document_id,
        file_name=file_path.name,
        file_path=str(file_path),
        file_format=file_format,
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
        sections=sections,
        is_scanned=is_scanned,
        warnings=warnings,
    )

    # --- Save extracted content for traceability ---
    _save_extraction(doc_content, settings.upload_dir)

    logger.info(
        "document_processing_complete",
        document_id=document_id,
        total_pages=len(pages),
        total_chars=len(full_text),
        sections=len(sections),
        is_scanned=is_scanned,
    )

    return doc_content


def _save_extraction(doc: DocumentContent, upload_dir: str) -> None:
    """Persist the extracted document to disk as JSON for traceability."""
    output_dir = Path(upload_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{doc.document_id}_{doc.file_name}.json"

    # Convert to dict — keep tables but truncate very long texts for readability
    data = {
        "document_id": doc.document_id,
        "file_name": doc.file_name,
        "file_path": doc.file_path,
        "file_format": doc.file_format,
        "total_pages": doc.total_pages,
        "is_scanned": doc.is_scanned,
        "warnings": doc.warnings,
        "sections": doc.sections,
        "pages": [
            {
                "page_number": p.page_number,
                "char_count": len(p.cleaned_text),
                "table_count": len(p.tables),
                "cleaned_text": p.cleaned_text[:2000] + "…"
                if len(p.cleaned_text) > 2000
                else p.cleaned_text,
            }
            for p in doc.pages
        ],
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("extraction_saved", path=str(output_path))
