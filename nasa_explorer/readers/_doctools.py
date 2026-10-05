"""Heuristics shared by document readers: sections, captions, authors, NASA mentions."""

from __future__ import annotations

import re

SECTION_WORDS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "data",
    "datasets",
    "data and methods",
    "methods",
    "methodology",
    "materials and methods",
    "study area",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "summary",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
    "limitations",
    "future work",
)
_NUMBERED = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{2,80})$")
# template and publisher lines printed above the title on the first page
_BANNER = re.compile(
    r"\btemplate\b|\bpreprint\b|submitted to|accepted (for|in|to)|manuscript|copyright|©"
    r"|journal of|proceedings of|this is a .*version",
    re.I,
)
_JOINER = re.compile(
    r"\b(a|an|and|at|by|for|from|in|into|of|on|over|the|to|under|using|via|with)$|[:,\-–]$", re.I
)
_CAPTION = re.compile(r"^\s*((Fig\.|Figure|FIGURE|Fig)\s*\d+[a-z]?[.:]?\s.*)$")
_TABLE_CAPTION = re.compile(r"^\s*((Table|TABLE)\s*\d+[.:]?\s.*)$")

# Missions, instruments and data systems frequently cited in NASA-related papers
NASA_TERMS = (
    "MODIS",
    "VIIRS",
    "Landsat",
    "Terra",
    "Aqua",
    "Suomi NPP",
    "NOAA-20",
    "GPM",
    "IMERG",
    "TRMM",
    "GRACE",
    "GRACE-FO",
    "SMAP",
    "ICESat",
    "ICESat-2",
    "GEDI",
    "ECOSTRESS",
    "OCO-2",
    "OCO-3",
    "SWOT",
    "NISAR",
    "TEMPO",
    "PACE",
    "CALIPSO",
    "CloudSat",
    "CERES",
    "AIRS",
    "MLS",
    "OMI",
    "Aura",
    "ASTER",
    "SRTM",
    "MERRA-2",
    "GEOS",
    "GLDAS",
    "FLDAS",
    "NLDAS",
    "FIRMS",
    "Earthdata",
    "GIBS",
    "Worldview",
    "Giovanni",
    "AppEEARS",
    "NASA POWER",
    "SEDAC",
    "HLS",
    "EMIT",
    "TROPICS",
    "Sentinel-1",
    "Sentinel-2",
    "Sentinel-5P",
    "GOES",
    "Hubble",
    "JWST",
    "James Webb",
    "TESS",
    "Kepler",
    "Chandra",
    "Spitzer",
    "Fermi",
    "Swift",
    "WISE",
    "NEOWISE",
    "Juno",
    "Perseverance",
    "Curiosity",
    "MRO",
    "LRO",
    "Artemis",
    "ISS",
    "OSIRIS-REx",
    "Cassini",
    "Voyager",
    "SOHO",
    "SDO",
    "Parker Solar Probe",
    "DSCOVR",
    "ACE",
    "MAVEN",
    "Europa Clipper",
    "Psyche",
    "DART",
)
_TERM_RE = re.compile(
    r"(?<![\w-])("
    + "|".join(re.escape(t) for t in sorted(NASA_TERMS, key=len, reverse=True))
    + r")(?![\w-])"
)


def sections_from_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or len(line) > 90:
            continue
        low = re.sub(r"^[\dIVX]+[.)]?\s*", "", line).strip(" :.").lower()
        if (low in SECTION_WORDS or _NUMBERED.match(line)) and line not in out:
            out.append(line)
    return out[:80]


def captions(pages: list[str]) -> tuple[list[dict], list[dict]]:
    figs, tabs = [], []
    for p, text in enumerate(pages, 1):
        for line in text.splitlines():
            if m := _CAPTION.match(line):
                figs.append({"page": p, "caption": m.group(1).strip()[:300]})
            elif m := _TABLE_CAPTION.match(line):
                tabs.append({"page": p, "caption": m.group(1).strip()[:300]})
    return figs[:100], tabs[:100]


def nasa_mentions(pages: list[str]) -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for p, text in enumerate(pages, 1):
        for m in _TERM_RE.finditer(text):
            pg = found.setdefault(m.group(1), [])
            if p not in pg:
                pg.append(p)
    return dict(sorted(found.items(), key=lambda kv: -len(kv[1])))


def guess_title_authors(first_page: str) -> tuple[str, list[str]]:
    lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
    start = next(
        (
            i
            for i, ln in enumerate(lines[:15])
            if 15 <= len(ln) <= 250
            and not ln.lower().startswith(("doi", "http", "vol", "journal", "arxiv"))
            and not _BANNER.search(ln)
        ),
        0,
    )
    title, end = (lines[start], start) if lines else ("", -1)
    # a long title wraps: the line ends on a joining word, or the next one goes on in lower case
    while end + 1 < min(len(lines), start + 3) and (
        _JOINER.search(title) or lines[end + 1][:1].islower()
    ):
        end += 1
        title += " " + lines[end]
    authors: list[str] = []
    if lines:
        for ln in lines[end + 1 : end + 6]:
            if ln.lower().startswith(("abstract", "1 ", "1.", "introduction")):
                break
            if (
                ("," in ln or " and " in ln)
                and len(ln) < 300
                and not re.search(r"\d{4}|@|university|institute|department", ln, re.I)
            ):
                authors = [
                    a.strip(" *0123456789†‡") for a in re.split(r",|\band\b", ln) if a.strip()
                ]
                break
    return title, authors[:30]


def chunk_paragraphs(paragraphs: list[str], words_per_page: int = 500) -> list[str]:
    """Pseudo-pages for formats without pagination (DOCX, HTML, Markdown, TXT)."""
    pages, cur, n = [], [], 0
    for para in paragraphs:
        cur.append(para)
        n += len(para.split())
        if n >= words_per_page:
            pages.append("\n".join(cur))
            cur, n = [], 0
    if cur:
        pages.append("\n".join(cur))
    return pages or [""]


def doc_metadata(
    pages: list[str],
    title: str = "",
    authors: list[str] | None = None,
    sections: list[str] | None = None,
    tables: list | None = None,
) -> dict:
    g_title, g_authors = guess_title_authors(pages[0] if pages else "")
    figs, tab_caps = captions(pages)
    return {
        "title": title or g_title,
        "authors": authors or g_authors,
        "sections": sections if sections else sections_from_lines("\n".join(pages).splitlines()),
        "figure_captions": figs,
        "table_captions": tab_caps,
        "tables": tables or [],
        "nasa_mentions": nasa_mentions(pages),
        "n_pages": len(pages),
        "n_words": sum(len(p.split()) for p in pages),
    }
