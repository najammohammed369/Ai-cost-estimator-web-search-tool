"""
PDF Extractor — uses PyMuPDF (pymupdf) to extract text and tables from PDF files.

Imports pymupdf directly (NOT the deprecated 'fitz' import).
Preserves page numbers, detects scanned-only PDFs, and extracts tables.
"""

from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""
    page_number: int          # 1-indexed
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)  # [table][row][cell]
    has_images: bool = False
    char_count: int = 0


@dataclass
class PDFExtractionResult:
    """Result of extracting a full PDF document."""
    file_path: str
    total_pages: int
    pages: list[PageContent]
    is_scanned: bool = False       # True if most pages have no text
    warnings: list[str] = field(default_factory=list)


def extract_pdf(file_path: str | Path) -> PDFExtractionResult:
    """
    Extract text and tables from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        PDFExtractionResult with per-page content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or is password-protected.
    """
    import pymupdf  # NOT fitz — use pymupdf directly

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    logger.info("pdf_extraction_start", file=str(file_path))

    try:
        doc = pymupdf.open(str(file_path))
    except Exception as e:
        raise ValueError(f"Could not open PDF '{file_path}': {e}") from e

    if doc.is_encrypted:
        raise ValueError(f"PDF is password-protected: {file_path}")

    pages: list[PageContent] = []
    total_text_chars = 0
    warnings: list[str] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        # Extract plain text
        text = page.get_text("text")
        char_count = len(text.strip())
        total_text_chars += char_count

        # Check for images (indicator of scanned content)
        image_list = page.get_images(full=False)
        has_images = len(image_list) > 0

        # Extract tables if available (PyMuPDF >= 1.23)
        tables: list[list[list[str]]] = []
        try:
            found_tables = page.find_tables()
            for table in found_tables.tables:
                extracted = table.extract()
                if extracted:
                    # Normalize: convert None cells to empty string
                    cleaned = [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in extracted
                    ]
                    tables.append(cleaned)
        except AttributeError:
            # find_tables() not available in this PyMuPDF version
            if page_idx == 0:
                warnings.append(
                    "Table extraction not available — update PyMuPDF to >= 1.23.0"
                )

        pages.append(
            PageContent(
                page_number=page_number,
                text=text,
                tables=tables,
                has_images=has_images,
                char_count=char_count,
            )
        )

    doc.close()

    # Detect scanned PDF: if >50% of pages have <50 characters of text
    total_pages = len(pages)
    scanned_pages = sum(1 for p in pages if p.char_count < 50)
    is_scanned = total_pages > 0 and (scanned_pages / total_pages) > 0.5

    if is_scanned:
        warnings.append(
            f"PDF appears to be scanned or image-based: {scanned_pages}/{total_pages} "
            f"pages have minimal text. OCR may be required for accurate extraction."
        )

    logger.info(
        "pdf_extraction_complete",
        file=str(file_path),
        total_pages=total_pages,
        total_chars=total_text_chars,
        is_scanned=is_scanned,
        warnings=len(warnings),
    )

    return PDFExtractionResult(
        file_path=str(file_path),
        total_pages=total_pages,
        pages=pages,
        is_scanned=is_scanned,
        warnings=warnings,
    )
