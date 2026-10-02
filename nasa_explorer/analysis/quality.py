"""Rule-based data-quality checks run on every analysis (no AI involved).

Each issue: {"level": "error"|"warning"|"info", "code", "variable", "message"}.
The same list is shown in the report and handed to the AI as trusted context.
"""

from __future__ import annotations

import re

from ..core import IN_BROWSER
from .stats import span_text

FILL_LIKE = (-9999.0, -999.0, -999.9, -99999.0, -32767.0, -32768.0, 65535.0, 32767.0)
TEMP_NAME = re.compile(r"(^t2m$|temp|^t$|^ta$|^tas|lst|skt|sst|surface_temperature|^t2m)", re.I)
PRECIP_NAME = re.compile(r"(precip|rain|prcp|prectot|^pr$|^tp$|snowfall)", re.I)
NONNEG_NAME = re.compile(
    r"(precip|rain|prcp|frp|count|area|depth|height|aod|chlor|concentration|xco2)", re.I
)
INDEX_NAME = re.compile(r"(ndvi|evi|ndwi|nbr)", re.I)
PCT_UNITS = {"%", "percent", "pct"}


def _issue(level: str, code: str, variable: str | None, message: str) -> dict:
    return {"level": level, "code": code, "variable": variable, "message": message}


def _units_by_var(analysis: dict) -> dict[str, str]:
    s = analysis.get("summary", {})
    return {
        v["name"]: str(v.get("units", ""))
        for v in s.get("variables", [])
        if isinstance(v, dict) and "name" in v
    }


def _value_checks(name: str, st: dict, units: str, kind: str = "") -> list[dict]:
    out: list[dict] = []
    if "min" not in st:
        if st.get("count", 0) == 0:
            out.append(
                _issue("error", "all_missing", name, f"{name}: every value is missing or fill")
            )
        return out
    lo, hi = st["min"], st["max"]
    u = units.strip().lower()
    if kind != "image" and (  # saturated pixels (255, 65535) are real in photos
        abs(lo) >= 9.9e19 or abs(hi) >= 9.9e19 or lo in FILL_LIKE or hi in FILL_LIKE
    ):
        bad = lo if (abs(lo) >= 9.9e19 or lo in FILL_LIKE) else hi
        out.append(
            _issue(
                "error",
                "undeclared_fill",
                name,
                f"{name}: value {bad:g} looks like an undeclared fill value - statistics are distorted; mask it",
            )
        )
    if st.get("missing_pct", 0) > 50:
        out.append(
            _issue(
                "warning",
                "mostly_missing",
                name,
                f"{name}: {st['missing_pct']:.0f}% of values are missing",
            )
        )
    elif st.get("missing_pct", 0) > 20:
        out.append(
            _issue(
                "info", "missing", name, f"{name}: {st['missing_pct']:.0f}% of values are missing"
            )
        )
    if st.get("std") == 0 and st.get("count", 0) > 1:
        out.append(
            _issue("info", "constant", name, f"{name}: constant value {lo:g} (no information)")
        )
    kelvin = u in ("k", "kelvin") or (u == "" and TEMP_NAME.search(name) and lo > 100)
    if kelvin and (lo < 150 or hi > 350):
        out.append(
            _issue(
                "warning",
                "implausible_range",
                name,
                f"{name}: {lo:g}-{hi:g} K is outside plausible Earth temperatures (150-350 K)",
            )
        )
    if u in ("degc", "c", "°c", "celsius", "deg c", "degrees c") and (lo < -90 or hi > 60):
        out.append(
            _issue(
                "warning",
                "implausible_range",
                name,
                f"{name}: {lo:g}-{hi:g} °C is outside plausible near-surface temperatures",
            )
        )
    if u in PCT_UNITS and (lo < 0 or hi > 100):
        out.append(
            _issue(
                "warning",
                "implausible_range",
                name,
                f"{name}: percentage outside 0-100 ({lo:g}-{hi:g})",
            )
        )
    if (PRECIP_NAME.search(name) or NONNEG_NAME.search(name)) and lo < 0:
        out.append(
            _issue(
                "warning",
                "negative_values",
                name,
                f"{name}: negative values ({lo:g}) for a quantity that cannot be negative",
            )
        )
    if INDEX_NAME.search(name) and (lo < -1.0001 or hi > 1.0001):
        out.append(
            _issue(
                "warning",
                "unscaled",
                name,
                f"{name}: range {lo:g}-{hi:g} is outside -1..1 - a scale factor (often 0.0001) was probably not applied",
            )
        )
    iqr = st.get("p75", 0) - st.get("p25", 0)
    zero_inflated = abs(st.get("p25", 0)) <= 1e-6 * max(1.0, abs(hi))  # e.g. rain: mostly dry
    if iqr > 0 and not zero_inflated and (hi > st["p75"] + 20 * iqr or lo < st["p25"] - 20 * iqr):
        out.append(
            _issue(
                "info",
                "outliers",
                name,
                f"{name}: extreme outliers (min {lo:g}, max {hi:g} vs interquartile range {st['p25']:g}-{st['p75']:g})",
            )
        )
    if st.get("sampled"):
        out.append(
            _issue(
                "info",
                "sampled",
                name,
                f"{name}: statistics from a regular sample ({st['sampled']})",
            )
        )
    return out


def _coverage_checks(kind: str, analysis: dict) -> list[dict]:
    out: list[dict] = []
    cov = analysis.get("coverage", {})
    t = cov.get("time") or {}
    # many rows per time step are normal for point data and for several stations side by side
    point_data = kind == "table" and ("space" in cov or "series_column" in t)
    if t.get("duplicates") and not point_data:
        out.append(
            _issue(
                "warning",
                "duplicate_times",
                None,
                f"{t['duplicates']} duplicate timestamps - averaging or de-duplication needed",
            )
        )
    if t.get("gaps"):
        out.append(
            _issue(
                "info",
                "time_gaps",
                None,
                f"{t['gaps']} gaps in the {t.get('resolution', '')} time series (largest {t.get('largest_gap', '?')})",
            )
        )
    bbox = (cov.get("space") or {}).get("bbox")
    if bbox and len(bbox) == 4:
        w, s, e, n = bbox
        if s < -90.001 or n > 90.001:
            out.append(
                _issue(
                    "error",
                    "bad_latitude",
                    None,
                    f"latitude outside -90..90 ({s:g}..{n:g}) - columns swapped or projected?",
                )
            )
        if w < -180.001 or e > 360.001:
            out.append(
                _issue(
                    "error", "bad_longitude", None, f"longitude outside -180..360 ({w:g}..{e:g})"
                )
            )
        if e > 180:
            out.append(
                _issue(
                    "info",
                    "lon_0_360",
                    None,
                    "longitudes use 0-360; convert to -180..180 before mapping with other data",
                )
            )
    short = [t for t in analysis.get("trends", []) if "short_record_years" in t]
    if short:
        names = ", ".join(t["variable"] for t in short)
        out.append(
            _issue(
                "warning",
                "short_record",
                None,
                f"only {span_text(short[0]['short_record_years'])} of data ({names}) - too short to "
                "tell a long-term trend from the seasonal cycle",
            )
        )
    trends = analysis.get("trends", [])
    if seasonal := [t for t in trends if t.get("seasonal")]:  # one line, not one per variable
        names = ", ".join(f"{t['variable']} ({_pct(t['seasonal_strength'])})" for t in seasonal)
        out.append(
            _issue(
                "info",
                "seasonal",
                None,
                f"strong annual cycle, so trends use deseasonalized values (share of variance): {names}",
            )
        )
    if acf := [t for t in trends if "Hamed-Rao" in t.get("method", "")]:
        names = ", ".join(f"{t['variable']} (r={t['lag1_autocorr']:.2f})" for t in acf)
        out.append(
            _issue(
                "info",
                "autocorrelated",
                None,
                f"autocorrelated series, p-values corrected with Hamed-Rao (lag-1 r): {names}",
            )
        )
    for tr in trends:
        if tr.get("extremes"):
            ex = ", ".join(
                f"{e['date']} ({e['anomaly']:+.3g} {tr.get('units', '')}, {e['z']:+.1f}σ)".replace(
                    " ,", ","
                )
                for e in tr["extremes"][:5]
            )
            out.append(
                _issue(
                    "info",
                    "extremes",
                    tr["variable"],
                    f"{tr['variable']}: unusual periods vs the normal for that time of year: {ex}",
                )
            )
    return out


def _pct(share: float) -> str:
    return ">99%" if share > 0.99 else f"{100 * share:.0f}%"


def _document_checks(analysis: dict) -> list[dict]:
    s = analysis.get("summary", {})
    out = []
    if s.get("words", 0) < 50 and s.get("ocr") is not None:  # only page formats can be scans
        out.append(
            _issue(
                "warning",
                "little_text",
                None,
                "almost no text extracted - scanned or image-only document?",
            )
        )
    if s.get("ocr"):
        out.append(
            _issue(
                "info",
                "ocr",
                None,
                "text comes from OCR: numbers and names may contain recognition errors",
            )
        )
    if not s.get("sections"):
        out.append(_issue("info", "no_sections", None, "no section headings detected"))
    return out


def unreadable(size: int, signature: str, problems: list[str]) -> list[dict]:
    """Why a file ended up as 'unknown': empty, or a known format that failed to open."""
    if size == 0:
        return [_issue("error", "empty_file", None, "the file is empty (0 bytes)")]
    if signature == "JPEG 2000" and IN_BROWSER:  # neither GDAL nor Pillow here decodes it
        return [
            _issue(
                "error",
                "unsupported_in_browser",
                None,
                "JPEG 2000 (HiRISE, LROC, Sentinel-2 ...) cannot be decoded in the browser. Run "
                "the command-line tool on it, or convert it first: "
                "gdal_translate -of GTiff -co COMPRESS=DEFLATE in.jp2 out.tif",
            )
        ]
    if problems and signature not in ("", "unrecognised"):
        first = problems[0].split("\n")[0][:220]
        return [
            _issue(
                "error",
                "unreadable",
                None,
                f"has a {signature} signature but could not be opened - truncated or corrupted "
                f"download? ({first})",
            )
        ]
    return []


def check(kind: str, analysis: dict) -> list[dict]:
    if kind == "document":
        issues = _document_checks(analysis)
    elif kind in ("grid", "table", "tree", "image"):
        units = _units_by_var(analysis)
        dem = (analysis.get("terrain") or {}).get("variable")
        issues = [
            i
            for name, st in analysis.get("statistics", {}).items()
            for i in _value_checks(name, st, units.get(name, ""), kind)
            if not (name == dem and i["code"] == "outliers")  # craters and peaks are real
        ]
        issues += _coverage_checks(kind, analysis)
        if kind == "table" and analysis.get("summary", {}).get("n_rows") == 0:
            issues.append(
                _issue("error", "no_rows", None, "the table has a header but no data rows")
            )
    else:
        issues = []
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda i: order[i["level"]])
