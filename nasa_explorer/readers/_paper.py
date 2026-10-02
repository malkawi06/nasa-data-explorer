"""What a reader of a research paper looks for first: DOI, abstract, the numbers it reports
(with units and pages), study area coordinates and period, references and data availability.
Pure regular expressions, so it works for every document reader (PDF, DOCX, HTML, text)."""

from __future__ import annotations

import re

DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+[A-Za-z0-9])")
NUM = r"[-+−]?\d+(?:[.,]\d+)?"
UNITS = (
    r"mm(?:\s*/\s*(?:yr|year|day|d|decade))?|cm|km(?:2|²)?|m(?:2|²|/s|\s*s-1)?|K(?:/decade)?|"
    r"°\s?C(?:/decade)?|°C|°|deg(?:rees)?|%|W\s*m-2|W\s*/\s*m(?:2|²)|kWh(?:\s*/\s*m(?:2|²))?(?:\s*/\s*day)?|"
    r"ppm|ppb|Gt|Mt|ha|hPa|kPa|Pa|μm|nm|days?|years?|yr|months?|per\s+(?:year|decade)"
)
QUANTITY = re.compile(rf"(?<![\w.]){NUM}\s?(?:{UNITS})(?![A-Za-z])")
STAT = re.compile(rf"\b(?:p|r|R2|R²|r2)\s*(?:=|<|>|≤|≥)\s*{NUM}")
COORD = re.compile(rf"({NUM})\s*°?\s*([NS])\s*[,;/ ]\s*({NUM})\s*°?\s*([EW])\b", re.I)
YEARS = re.compile(r"\b((?:19|20)\d{2})\s*(?:-|–|to|through|until)\s*((?:19|20)\d{2})\b")
AVAILABLE = re.compile(
    r"data (?:are|is|were) (?:freely |publicly |openly )?available|available (?:at|from|online)|"
    r"can be (?:downloaded|accessed)|data availability|code (?:is|are) available",
    re.I,
)
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
HEADING = re.compile(r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?\s+)?[A-Z][A-Za-z ,&-]{2,60}$")
MAX_NUMBERS = 40


def _sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in SENTENCE.split(flat) if s.strip()]


def doi(pages: list[str]) -> str | None:
    for text in pages[:2]:  # the paper's own DOI is on its first page(s)
        if m := DOI.search(text):
            return m.group(1).rstrip(".")
    return None


def abstract(pages: list[str]) -> str | None:
    text = "\n".join(pages[:2])
    m = re.search(r"(?im)^\s*abstract[.:]?\s*$|\babstract[.:]\s", text)
    if not m:
        return None
    lines = []
    for line in text[m.end() :].splitlines():
        if lines and (
            HEADING.match(line)
            and len(line.split()) <= 8
            or re.match(r"(?i)^\s*(1\.?\s+)?introduction\b", line)
        ):
            break
        if line.strip():
            lines.append(line.strip())
    body = " ".join(lines)
    return body[:2500] if len(body) > 80 else None


def key_numbers(pages: list[str]) -> list[dict]:
    """Sentences that report quantities, with their page: what the paper actually measured."""
    out, seen = [], set()
    for page, text in enumerate(pages, 1):
        for sent in _sentences(text):
            if re.match(r"(?i)^\[?\d+\]|^references\b", sent) or len(sent) > 400:
                continue  # reference list entries
            values = [m.group(0) for m in QUANTITY.finditer(sent)] + [
                m.group(0) for m in STAT.finditer(sent)
            ]
            if not values or sent in seen:
                continue
            seen.add(sent)
            out.append({"page": page, "text": sent[:300], "values": values[:8]})
            if len(out) >= MAX_NUMBERS:
                return out
    return out


def coordinates(pages: list[str]) -> list[dict]:
    out = []
    for page, text in enumerate(pages, 1):
        for m in COORD.finditer(text):
            lat = float(m.group(1).replace("−", "-").replace(",", ".")) * (
                -1 if m.group(2).upper() == "S" else 1
            )
            lon = float(m.group(3).replace("−", "-").replace(",", ".")) * (
                -1 if m.group(4).upper() == "W" else 1
            )
            if abs(lat) <= 90 and abs(lon) <= 180:
                out.append(
                    {"page": page, "lat": round(lat, 5), "lon": round(lon, 5), "text": m.group(0)}
                )
    return out[:20]


def study_period(pages: list[str]) -> dict | None:
    spans = [(int(a), int(b)) for text in pages for a, b in YEARS.findall(text) if int(a) <= int(b)]
    if not spans:
        return None
    return {
        "start": min(a for a, _ in spans),
        "end": max(b for _, b in spans),
        "mentions": len(spans),
    }


def references(pages: list[str]) -> dict:
    text = "\n".join(pages)
    heads = list(re.finditer(r"(?im)^\s*(references|bibliography|literature cited)\s*$", text))
    if not heads:
        return {"count": 0}
    m = heads[-1]  # the last heading wins (tables of contents mention it too)
    tail = text[m.end() :]
    entries = re.findall(r"(?m)^\s*(?:\[\d+\]|\d{1,3}\.\s)", tail)
    if not entries:  # author-year style: one entry per line that starts with a surname and a year
        entries = re.findall(r"(?m)^[A-Z][A-Za-z'\-]+,\s.*\(?(?:19|20)\d{2}", tail)
    return {"count": len(entries), "dois": len(DOI.findall(tail))}


def data_availability(pages: list[str]) -> list[dict]:
    out = []
    for page, text in enumerate(pages, 1):
        for sent in _sentences(text):
            if AVAILABLE.search(sent):
                out.append({"page": page, "text": sent[:300]})
    return out[:10]


def paper_facts(pages: list[str]) -> dict:
    return {
        "doi": doi(pages),
        "abstract": abstract(pages),
        "key_numbers": key_numbers(pages),
        "coordinates": coordinates(pages),
        "study_period": study_period(pages),
        "references": references(pages),
        "data_availability": data_availability(pages),
    }
