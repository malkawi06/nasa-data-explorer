"""PDF documents / research papers: text (pymupdf), tables (pdfplumber), OCR fallback."""

from __future__ import annotations

import shutil
from pathlib import Path

from ..core import ReadOptions, ReadResult
from ..registry import reader
from ._doctools import doc_metadata

MAX_TABLE_PAGES = 40
MAX_TABLES = 15
MIN_CHARS_PER_PAGE = 25


def _ocr(doc) -> tuple[list[str], str | None]:
    if shutil.which("tesseract") is None:
        return [], "scanned PDF but the tesseract binary is not installed (OCR skipped)"
    try:
        import io

        import pytesseract
        from PIL import Image
    except ImportError:
        return [], "scanned PDF: pip install 'nasa-data-explorer[docs]' for OCR"
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        pages.append(pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png")))))
    return pages, None


def _tables(path: Path) -> list[dict]:
    try:
        import pdfplumber
    except ImportError:
        return []
    out = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages[:MAX_TABLE_PAGES], 1):
            for t in page.extract_tables():
                if t and len(t) > 1:
                    out.append({"page": pno, "rows": [[c or "" for c in r] for r in t[:30]]})
                if len(out) >= MAX_TABLES:
                    return out
    return out


@reader(
    "pdf",
    category="Documents",
    extensions=(".pdf",),
    magic=(b"%PDF",),
    requires=("pymupdf",),
    extra="docs",
)
def read_pdf(path: Path, opts: ReadOptions) -> ReadResult:
    import pymupdf

    notes = []
    with pymupdf.open(path) as doc:
        pages = [p.get_text() for p in doc]
        info = doc.metadata or {}
        toc = [t[1] for t in doc.get_toc()]
        ocr = sum(len(p.strip()) for p in pages) < MIN_CHARS_PER_PAGE * max(1, len(pages))
        if ocr:
            ocr_pages, note = _ocr(doc)
            if ocr_pages:
                pages, notes = ocr_pages, ["text extracted with OCR (no text layer)"]
            elif note:
                notes.append(note)
    authors = [
        a.strip() for a in (info.get("author") or "").replace(";", ",").split(",") if a.strip()
    ]
    meta = doc_metadata(
        pages,
        title=(info.get("title") or "").strip(),
        authors=authors,
        sections=toc,
        tables=_tables(path),
    )
    meta["ocr"] = ocr
    meta["notes"] = notes
    meta["pdf_info"] = {k: v for k, v in info.items() if v}
    return ReadResult("document", None, meta, pages)
