"""Process a file, an archive or a whole folder into reports/."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import archives, products, report
from .analysis import analyze
from .analysis.quality import unreadable
from .core import ReadOptions, ReadResult
from .readers.planetary import DATA_EXTS, label_for
from .readers.zarr_store import is_zarr_dir
from .registry import read_file, readers

log = logging.getLogger("nasa_explorer")

SIDECARS = {
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".sbn",
    ".sbx",
    ".qix",
    ".aux.xml",
    ".ovr",
    ".idx",
    ".ncx",
    ".ncx4",
}
SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


@dataclass
class FileReport:
    file: str
    source: Path
    reader: str
    category: str
    kind: str
    analysis: dict
    html_path: Path | None = None
    json_path: Path | None = None
    plots: list[tuple[str, str, bytes]] = field(default_factory=list)
    error: str | None = None

    def _repr_html_(self) -> str:  # Jupyter
        return (
            self.html_path.read_text(encoding="utf-8")
            if self.html_path
            else f"<pre>{self.analysis}</pre>"
        )


def human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _safe_name(label: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", label).strip("_")[:150] or "file"


def iter_inputs(folder: Path, out_dir: Path | None = None) -> list[Path]:
    """Files to process in a folder: Zarr stores count as one item; shapefile sidecars skipped."""
    found: list[Path] = []
    for path in sorted(folder.iterdir()):
        if path.name in SKIP_NAMES or path.name.startswith("."):
            continue
        if out_dir is not None and path.resolve() == out_dir.resolve():
            continue
        if path.is_dir():
            found += (
                [path]
                if (is_zarr_dir(path) or path.suffix.lower() == ".zarr")
                else iter_inputs(path, out_dir)
            )
            continue
        low = path.name.lower()
        if any(low.endswith(s) for s in SIDECARS) and any(
            path.with_name(path.name[: -len(s)] + ".shp").exists()
            or path.with_suffix(".shp").exists()
            for s in SIDECARS
            if low.endswith(s)
        ):
            continue
        if path.suffix.lower() in DATA_EXTS and label_for(path) != path:
            continue  # PDS data file: its label is processed (and opens the data)
        found.append(path)
    return found


def _category(reader_name: str) -> str:
    return next((r.category for r in readers() if r.name == reader_name), "Unknown")


def _headline(kind: str, a: dict) -> str:
    s = a.get("summary", {})
    cov = a.get("coverage", {})
    parts: list[str] = []
    if kind == "grid":
        parts.append(
            f"{len(s.get('variables', []))} variables, dims "
            + " × ".join(f"{k}={v}" for k, v in list(s.get("dimensions", {}).items())[:4])
        )
    elif kind == "table":
        parts.append(f"{s.get('n_rows', 0):,} rows × {s.get('n_columns', 0)} columns")
    elif kind == "document":
        parts.append(f"{s.get('title', '')[:80]} ({s.get('pages', 0)} pages)")
    elif kind == "image":
        parts.append(f"{s.get('width')}×{s.get('height')} {s.get('mode', '')}")
    elif kind == "tree":
        parts.append(f"{s.get('n_datasets', 0)} HDF5 datasets")
    else:
        errors = [q["message"] for q in a.get("quality", []) if q["level"] == "error"]
        parts.append(errors[0][:120] if errors else f"signature: {s.get('signature', '?')}")
    if t := cov.get("time"):
        parts.append(f"{str(t.get('start', ''))[:10]} → {str(t.get('end', ''))[:10]}")
    if e := a.get("events"):
        parts.append(
            f"{e['total']:,} events, peak {e['peak_month']} ({e['peak_month_share_pct']:g}%)"
        )
    elif tr := a.get("terrain"):
        res, dep = tr.get("at_analysis_resolution", {}), tr.get("depressions", {})
        bits = [f"{tr.get('body', 'Earth')} DEM"]
        if res:
            bits.append(f"{res['flat_lt10_pct']:g}% flat (<10°) at {res['pixel_m']:,.0f} m")
        if dep.get("count") is not None:
            bits.append(f"{dep['count']} crater-like depressions")
        parts.append(", ".join(bits))
    elif veg := a.get("vegetation"):
        parts.append(
            f"NDVI mean {veg['mean']:g}, bare ground (<0.1) {veg['share_pct']['bare_0_0.1']:g}%"
        )
    elif a.get("trends"):
        t0 = a["trends"][0]
        parts.append(f"{t0['variable']}: {t0['text'].split(';')[0]}")
    return " · ".join(parts)


def chunks_for(rep: FileReport, res: ReadResult | None) -> list[dict]:
    """Text chunks used by the Q&A index: document pages, or a data summary."""
    if res is not None and res.kind == "document":
        return [
            {"file": rep.file, "page": i, "text": t}
            for i, t in enumerate(res.pages, 1)
            if t.strip()
        ]
    a = rep.analysis
    brief = {k: a.get(k) for k in ("summary", "coverage", "statistics", "trends")}
    text = (
        f"Data file {rep.file} ({rep.reader}, {rep.kind}). " + json.dumps(brief, default=str)[:6000]
    )
    return [{"file": rep.file, "page": None, "text": text}]


def _cached(path: Path, out_dir: Path, stem: str) -> FileReport | None:
    js, page = out_dir / f"{stem}.json", out_dir / f"{stem}.html"
    if not (js.exists() and page.exists()) or js.stat().st_mtime < path.stat().st_mtime:
        return None
    d = json.loads(js.read_text(encoding="utf-8"))
    return FileReport(
        d["file"],
        path,
        d["reader"],
        d["category"],
        d["kind"],
        d["analysis"],
        page,
        js,
        error=d.get("error"),
    )


def process_file(
    path: Path,
    opts: ReadOptions,
    out_dir: Path,
    label: str | None = None,
    ai: bool = False,
    reuse: bool = False,
) -> FileReport:
    label = label or path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(label)
    if reuse and (cached := _cached(path, out_dir, stem)):
        log.info("reusing existing report for %s", label)
        return cached
    res: ReadResult | None = None
    problems: list[str] = []
    error = None
    reader_name = "unknown"
    try:
        res, reader_name, problems = read_file(path, opts)
        res.metadata.setdefault("file_name", label)
        analysis, figs = analyze(res, opts)
        analysis["products"] = products.identify(label, analysis)
        if res.kind == "binary":
            analysis["quality"] = unreadable(
                path.stat().st_size, str(res.metadata.get("signature", "")), problems
            ) + analysis.get("quality", [])
    except Exception as exc:  # report the failure instead of crashing a folder run
        log.exception("failed on %s", path)
        analysis, figs, error = {"summary": {}, "notes": []}, [], f"{type(exc).__name__}: {exc}"
    rep = FileReport(
        label,
        path,
        reader_name,
        _category(reader_name),
        res.kind if res else "binary",
        analysis,
        error=error,
    )

    plot_dir = out_dir / f"{stem}_plots"
    for i, (title, png) in enumerate(figs, 1):
        plot_dir.mkdir(exist_ok=True)
        fname = f"{i:02d}_{_safe_name(title)[:60]}.png"
        (plot_dir / fname).write_bytes(png)
        rep.plots.append((title, f"{plot_dir.name}/{fname}", png))

    ai_block = None
    if ai and res is not None and error is None:
        from . import ai as ai_mod

        ai_block = ai_mod.interpret(rep, res, opts.lang)

    size = _size(path)
    payload = {
        "file": label,
        "source": str(path),
        "reader": reader_name,
        "category": rep.category,
        "kind": rep.kind,
        "size_bytes": size,
        "size_human": human_size(size),
        "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
        "problems": problems,
        "error": error,
        "analysis": analysis,
        "plots": [{"title": t, "file": f} for t, f, _ in rep.plots],
    }
    if ai_block:
        payload["ai"] = ai_block
    payload["headline"] = _headline(rep.kind, analysis)
    rep.json_path = out_dir / f"{stem}.json"
    rep.json_path.write_text(report.dumps(payload), encoding="utf-8")
    (out_dir / f"{stem}.chunks.json").write_text(
        report.dumps(chunks_for(rep, res)), encoding="utf-8"
    )
    rep.html_path = out_dir / f"{stem}.html"
    body = report.render_body(payload, rep.plots, opts.lang)
    rep.html_path.write_text(report.page(f"{label} - report", body, opts.lang), encoding="utf-8")
    return rep


def process(
    path: str | Path,
    opts: ReadOptions | None = None,
    out_dir: str | Path = "reports",
    ai: bool = False,
    reuse: bool = False,
    on_file: Callable[[str, int, int], None] | None = None,
) -> list[FileReport]:
    """Process a file, archive or folder. Writes one report per file plus index.html.

    reuse=True skips files whose report is newer than the file (used by --ask).
    on_file(label, index, total) is called before each file (progress for UIs)."""
    opts = opts or ReadOptions()
    path, out_dir = Path(path), Path(out_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[FileReport] = []

    def handle(p: Path, label: str, depth: int = 0) -> None:
        if archives.is_archive(p) and depth < 3:
            extracted = out_dir / ".extracted"
            extracted.mkdir(exist_ok=True)
            dest = Path(tempfile.mkdtemp(prefix=f"{_safe_name(p.name)}_", dir=extracted))
            try:
                members = archives.extract(p, dest)
            except Exception as exc:
                reports.append(process_file(p, opts, out_dir, label, ai))
                reports[-1].analysis.setdefault("notes", []).append(
                    f"archive extraction failed: {exc}"
                )
                return
            items = iter_inputs(dest) if members else []
            log.info("%s: %d files inside archive", label, len(items))
            nonlocal total
            total += len(items) - 1
            for m in items:
                handle(m, f"{label}/{m.relative_to(dest)}", depth + 1)
            return
        log.info("processing %s", label)
        if on_file:
            on_file(label, len(reports), total)
        reports.append(process_file(p, opts, out_dir, label, ai, reuse))

    if path.is_dir() and not (is_zarr_dir(path) or path.suffix.lower() == ".zarr"):
        inputs = [(p, str(p.relative_to(path))) for p in iter_inputs(path, out_dir)]
    else:
        inputs = [(path, path.name)]
    total = len(inputs)  # archives may add more; the count grows as they are opened
    for p, label in inputs:
        handle(p, label)

    # index every report in out_dir, so repeated runs accumulate instead of overwriting
    entries = []
    for js in sorted(out_dir.glob("*.json")):
        if js.name.endswith(".chunks.json") or not js.with_suffix(".html").exists():
            continue
        data = json.loads(js.read_text(encoding="utf-8"))
        entries.append({**data, "html": js.with_suffix(".html").name})
    report.write_index(entries, out_dir, opts.lang)
    return reports
