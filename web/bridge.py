"""Glue between the browser page (Pyodide) and nasa_explorer.

Runs unchanged in CPython too, which is how tests/test_web_bridge.py checks it.
Every function returns a JSON string so the JavaScript side never touches Python objects.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process
from nasa_explorer.registry import readers

WORK = Path("/tmp/nasa_explorer_web")
UNAVAILABLE_IN_BROWSER = {"cfgrib", "pyhdf", "pymupdf"}  # need native libraries Pyodide lacks


def formats() -> str:
    rows = []
    for r in sorted(readers(), key=lambda r: (r.category, r.name)):
        missing = r.missing()
        rows.append(
            {
                "category": r.category,
                "name": r.name,
                "extensions": list(r.extensions),
                "missing": missing,
                "browser": not (set(missing) & UNAVAILABLE_IN_BROWSER),
            }
        )
    return json.dumps(rows)


def missing_modules(problems: list[str]) -> list[str]:
    """Module names from registry messages like 'vector reader disabled: missing geopandas.'"""
    found: list[str] = []
    for p in problems:
        if m := re.search(r"disabled: missing ([\w., ]+?)\.", p):
            found += [x.strip() for x in m.group(1).split(",") if x.strip() not in found]
    return found


def new_run(run: str) -> str:
    """Create an empty input folder; the page writes the dropped files into it."""
    src = WORK / "in" / re.sub(r"\W+", "_", run)
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    return str(src)


def analyse_dir(src_dir: str, options: str = "{}") -> str:
    """Process every dropped file together (so .shp sidecars and archives work) and
    return one payload per report: metadata plus the full HTML and JSON."""
    opts = json.loads(options)
    bbox = opts.get("bbox")
    read_opts = ReadOptions(
        var=opts.get("var") or None,
        bbox=tuple(float(v) for v in bbox) if bbox else None,
        start=opts.get("start") or None,
        end=opts.get("end") or None,
        lang=opts.get("lang", "en"),
    )
    out_dir = WORK / "out" / Path(src_dir).name
    shutil.rmtree(out_dir, ignore_errors=True)
    results = []
    for rep in process(src_dir, read_opts, out_dir):
        payload = json.loads(rep.json_path.read_text(encoding="utf-8"))
        results.append(
            {
                "file": rep.file,
                "reader": rep.reader,
                "kind": rep.kind,
                "error": rep.error,
                "headline": payload.get("headline", ""),
                "problems": payload.get("problems", []),
                "missing": missing_modules(payload.get("problems", [])),
                "html": rep.html_path.read_text(encoding="utf-8"),
                "json": rep.json_path.read_text(encoding="utf-8"),
            }
        )
    return json.dumps(results)
