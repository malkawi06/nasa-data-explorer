"""Lightweight PDF fallback via pure-Python pypdf (used where pymupdf is unavailable,
e.g. the Pyodide browser build). No table extraction or OCR."""

from __future__ import annotations

from pathlib import Path

from ..core import ReadOptions, ReadResult
from ..registry import reader
from ._doctools import doc_metadata


@reader(
    "pdf-lite",
    category="Documents",
    extensions=(".pdf",),
    magic=(b"%PDF",),
    requires=("pypdf",),
    priority=20,
)
def read_pdf_lite(path: Path, opts: ReadOptions) -> ReadResult:
    from pypdf import PdfReader

    pdf = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in pdf.pages]
    info = pdf.metadata or {}
    title = str(info.get("/Title") or "").strip()
    authors = [
        a.strip() for a in str(info.get("/Author") or "").replace(";", ",").split(",") if a.strip()
    ]
    meta = doc_metadata(pages, title=title, authors=authors)
    meta["ocr"] = False
    meta["notes"] = ["read with pypdf (no table extraction or OCR)"]
    if sum(len(p.strip()) for p in pages) < 25 * max(1, len(pages)):
        meta["notes"].append(
            "little or no text layer: this looks like a scanned PDF (OCR needs the desktop install)"
        )
    return ReadResult("document", None, meta, pages)
