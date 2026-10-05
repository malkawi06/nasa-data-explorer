"""Paper extraction on real research papers (open arXiv PDFs that use NASA data).

The truth is arXiv's own metadata (title, abstract, DOI), fetched at test time. Skipped when
arXiv is unreachable. Each paper is scored, then the whole set must pass thresholds, so one
unusual layout does not fail the run but a general regression does."""

import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import pytest
import requests

from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file

pytest.importorskip("pypdf")

PAPERS = ("2512.15222", "2404.10135", "2307.10843", "2605.14426")  # GPM IMERG studies
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _truth(ids) -> dict:
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
    root = ET.fromstring(requests.get(url, timeout=30).content)
    out = {}
    for e in root.findall("a:entry", ATOM):
        pid = re.sub(r"v\d+$", "", e.find("a:id", ATOM).text.rsplit("/", 1)[-1])
        doi = e.find("arxiv:doi", ATOM)
        out[pid] = {
            "title": e.find("a:title", ATOM).text,
            "abstract": e.find("a:summary", ATOM).text,
            "doi": doi.text.strip().lower() if doi is not None else None,
        }
    return out


@pytest.mark.network
def test_title_abstract_and_doi_from_real_papers(tmp_path):
    try:
        truth = _truth(PAPERS)
        pdfs = {}
        for pid in PAPERS:
            r = requests.get(f"https://arxiv.org/pdf/{pid}", timeout=60)
            r.raise_for_status()
            pdfs[pid] = r.content
    except (requests.RequestException, ET.ParseError) as exc:
        pytest.skip(f"arXiv unreachable: {type(exc).__name__}")

    rows = []
    for pid, data in pdfs.items():
        path = tmp_path / f"{pid}.pdf"
        path.write_bytes(data)
        rep = process_file(path, ReadOptions(plots=False), tmp_path / "r")
        s = rep.analysis.get("summary", {})
        t = truth[pid]
        abstract = s.get("abstract") or ""
        doi_in_pdf = t["doi"] and t["doi"] in data.decode("latin-1").lower()
        rows.append(
            {
                "paper": pid,
                "title": _similar(s.get("title", ""), t["title"]) >= 0.8,
                # the opening of the abstract is enough: PDFs hyphenate and wrap lines
                "abstract": _similar(abstract[:300], t["abstract"][: len(abstract[:300])]) >= 0.8,
                "doi": (s.get("doi") or "").lower() == t["doi"] if doi_in_pdf else None,
                "got_title": s.get("title", "")[:120],
                "want_title": " ".join(t["title"].split())[:120],
                "got_abstract": abstract[:120],
            }
        )
    report = "\n".join(str(r) for r in rows)
    titles = sum(r["title"] for r in rows)
    abstracts = sum(r["abstract"] for r in rows)
    dois = [r["doi"] for r in rows if r["doi"] is not None]
    assert titles >= len(rows) - 1, report
    assert abstracts >= len(rows) - 1, report
    assert all(dois), report
