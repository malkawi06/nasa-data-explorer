"""Glue between the browser page (Pyodide) and nasa_explorer.

Runs unchanged in CPython too, which is how tests/test_web_bridge.py checks it.
Every function returns a JSON string so the JavaScript side never touches Python objects.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from nasa_explorer import ai, analog, power, report
from nasa_explorer.ai_view import render_ai
from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import FileReport, analog_report, process
from nasa_explorer.registry import readers

WORK = Path("/tmp/nasa_explorer_web")
UNAVAILABLE_IN_BROWSER = {"cfgrib", "pyhdf", "pymupdf"}  # need native libraries Pyodide lacks
_REPORTS: dict[str, tuple[FileReport, dict, ReadOptions]] = {}  # key -> (report, payload, options)
_JOBS: dict[str, tuple[ai.Workflow, dict]] = {}  # job id -> (generator, context)


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


def analyse_dir(src_dir: str, options: str = "{}", progress=None) -> str:
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

    def on_file(label: str, i: int, n: int) -> None:
        if progress is not None:
            progress(label, i, n)

    for rep in process(src_dir, read_opts, out_dir, on_file=on_file):
        payload = json.loads(rep.json_path.read_text(encoding="utf-8"))
        key = f"{Path(src_dir).name}/{rep.file}"
        _REPORTS[key] = (rep, payload, read_opts)
        results.append(_entry(key, rep, payload))
    return json.dumps(results)


def _entry(key: str, rep: FileReport, payload: dict) -> dict:
    return {
        "key": key,
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


# --- NASA POWER and the Moon / Mars analog finder (the browser does the HTTP requests) ---


def power_urls(lat: float, lon: float, start: str = "", end: str = "") -> str:
    s, e = (start, end) if start and end else power.default_period()
    return json.dumps(
        {
            "daily": power.daily_url(lat, lon, s, e),
            "climatology": power.climatology_url(lat, lon),
            "start": s,
            "end": e,
        }
    )


def analog_sites() -> str:
    targets = [
        {"key": t.key, "name": t.name, "description": t.description}
        for t in analog.TARGETS.values()
    ]
    return json.dumps({"targets": targets, "sites": list(analog.SITES)})


def _inside(bbox, lat: float, lon: float) -> bool:
    return bool(bbox) and len(bbox) == 4 and bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _local_features(lat: float, lon: float) -> dict:
    """Terrain and NDVI from files already analysed in this session that cover the site."""
    out: dict = {}
    for rep, _, _ in _REPORTS.values():
        a = rep.analysis or {}
        bbox = (a.get("coverage", {}).get("space") or {}).get("bbox")
        if a.get("body", "Earth") != "Earth" or not _inside(bbox, lat, lon):
            continue
        if a.get("terrain"):
            out.update(
                {k: v for k, v in analog.terrain_features(a["terrain"]).items() if v is not None}
            )
        main = str(a.get("main_variable") or "")
        if "ndvi" in (rep.file + main).lower() and main in a.get("statistics", {}):
            mean = a["statistics"][main].get("mean")
            if mean is not None:
                out["ndvi"] = mean / 10000 if mean > 1.5 else mean  # MODIS stores NDVI × 10000
    return out


def analog_run(run: str, target: str, sites_json: str, clims_json: str, lang: str = "en") -> str:
    """Rank sites with the climatologies the browser fetched (None where a fetch failed)."""
    sites, clims = json.loads(sites_json), json.loads(clims_json)
    features = []
    for site, clim in zip(sites, clims, strict=True):
        feats = analog.climate_features(clim) if clim else {}
        feats.update(_local_features(float(site["lat"]), float(site["lon"])))
        features.append(feats or None)
    out_dir = WORK / "out" / run
    rep = analog_report(target, sites, features, out_dir, lang)
    payload = json.loads(rep.json_path.read_text(encoding="utf-8"))
    key = f"{run}/{rep.file}"
    _REPORTS[key] = (rep, payload, ReadOptions(lang=lang))
    return json.dumps([_entry(key, rep, payload)])


# --- AI (bring your own key): Python builds and verifies prompts, JavaScript calls the provider ---


def _pages(rep: FileReport) -> list[str]:
    chunks = rep.json_path.with_name(rep.json_path.name[: -len(".json")] + ".chunks.json")
    if rep.kind != "document" or not chunks.exists():
        return []
    return [c["text"] for c in json.loads(chunks.read_text(encoding="utf-8"))]


def _step(job: str, step: ai.Step) -> str:
    return json.dumps(
        {
            "job": job,
            "step": {
                "label": step.label,
                "system": step.system,
                "prompt": step.prompt,
                "json": step.json_mode,
                "tokens": step.tokens,
                "image": step.image,
            },
        }
    )


def ai_start(key: str, lang: str, provider: str, model: str, send_image: bool = False) -> str:
    """Begin the verified analysis of one report; returns the first prompt to send.
    `send_image` (opt-in) lets a vision-capable provider see a downscaled copy of an image."""
    rep, payload, _ = _REPORTS[key]
    image = ai.vision_image(rep) if send_image and provider in ai.VISION else None
    if rep.kind == "document":
        flow = ai.paper_workflow(
            _pages(rep), payload["analysis"]["summary"].get("title") or rep.file, lang
        )
    else:
        flow = ai.data_workflow(rep.analysis, rep.file, rep.reader, lang, image)
    job = f"ai:{key}"
    _JOBS[job] = (
        flow,
        {
            "key": key,
            "lang": lang,
            "provider": provider,
            "model": model,
            "kind": "analysis",
            "source": rep.source,
        },
    )
    return _step(job, next(flow))


def chat_start(key: str, question: str, lang: str) -> str:
    rep, _, _ = _REPORTS[key]
    context = ai.chat_context(rep.analysis, rep.file, rep.reader, _pages(rep), question)
    flow = ai.chat_workflow(question, context, lang)
    job = f"chat:{key}:{len(_JOBS)}"
    _JOBS[job] = (flow, {"key": key, "lang": lang, "kind": "chat"})
    return _step(job, next(flow))


def ai_next(job: str, reply: str) -> str:
    """Feed the provider's reply; returns the next prompt, or the finished (verified) result."""
    flow, ctx = _JOBS[job]
    try:
        return _step(job, flow.send(reply))
    except StopIteration as done:
        result = done.value
    del _JOBS[job]
    if ctx["kind"] == "chat":
        return json.dumps({"job": job, "done": True, "answer": result["answer"]})
    rep, payload, opts = _REPORTS[ctx["key"]]
    if result.get("saw_image"):
        from nasa_explorer.analysis.image_metrics import jpeg_b64

        result["preview"] = jpeg_b64(ctx["source"], ai.PREVIEW_SIDE)
    payload["ai"] = {"provider": ctx["provider"], "model": ctx["model"], **result}
    rep.json_path.write_text(report.dumps(payload), encoding="utf-8")
    page = report.page(
        f"{rep.file} - report", report.render_body(payload, rep.plots, ctx["lang"]), ctx["lang"]
    )
    rep.html_path.write_text(page, encoding="utf-8")
    panel = report.page("AI analysis", render_ai(payload["ai"], ctx["lang"]), ctx["lang"])
    return json.dumps(
        {
            "job": job,
            "done": True,
            "verification": result.get("verification", {}),
            "panel": panel,
            "html": page,
            "json": rep.json_path.read_text(encoding="utf-8"),
        }
    )
