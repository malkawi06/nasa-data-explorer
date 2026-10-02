"""HTML pages and Markdown documents."""

from __future__ import annotations

import re
from pathlib import Path

from ..core import ReadOptions, ReadResult
from ..registry import looks_like_text, reader, text_head
from ._doctools import chunk_paragraphs, doc_metadata


def _is_html(head: bytes) -> bool:
    return looks_like_text(head) and bool(
        re.match(r"(<!doctype html|<html)", text_head(head)[:200], re.I)
    )


@reader(
    "html",
    category="Documents",
    extensions=(".html", ".htm", ".xhtml"),
    sniff=_is_html,
    requires=("bs4",),
    extra="docs",
    priority=40,
)
def read_html(path: Path, opts: ReadOptions) -> ReadResult:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    author = soup.find("meta", attrs={"name": re.compile("author", re.I)})
    headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    tables = []
    for t in soup.find_all("table")[:15]:
        rows = [
            [c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])]
            for r in t.find_all("tr")[:30]
        ]
        if rows:
            tables.append({"page": None, "rows": rows})
    figs = [f.get_text(" ", strip=True) for f in soup.find_all("figcaption")]
    paras = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    pages = chunk_paragraphs(paras)
    meta = doc_metadata(
        pages,
        title=(h1.get_text(strip=True) if h1 else "") or title,
        authors=[author["content"]] if author and author.get("content") else [],
        sections=headings,
        tables=tables,
    )
    meta["figure_captions"] += [{"page": None, "caption": f[:300]} for f in figs]
    return ReadResult("document", None, meta, pages)


@reader("markdown", category="Documents", extensions=(".md", ".markdown", ".rst"))
def read_markdown(path: Path, opts: ReadOptions) -> ReadResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    headings = [ln.lstrip("#").strip() for ln in lines if re.match(r"^#{1,4}\s", ln)]
    title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), "")
    rows = [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in lines
        if ln.strip().startswith("|") and not re.match(r"^\s*\|[\s:|-]+\|\s*$", ln)
    ]
    tables = [{"page": None, "rows": rows[:30]}] if rows else []
    pages = chunk_paragraphs([ln for ln in lines if ln.strip()])
    return ReadResult("document", None, doc_metadata(pages, title, None, headings, tables), pages)


@reader(
    "text",
    category="Documents",
    extensions=(".txt", ".text", ".log"),
    sniff=looks_like_text,
    priority=90,
)
def read_text(path: Path, opts: ReadOptions) -> ReadResult:
    text = path.read_text(encoding="utf-8", errors="replace")
    pages = chunk_paragraphs([ln for ln in text.splitlines() if ln.strip()])
    return ReadResult("document", None, doc_metadata(pages), pages)
