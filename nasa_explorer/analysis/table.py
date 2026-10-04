"""Analysis of tabular / point data (pandas or geopandas DataFrame)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .. import plots
from ..core import ReadOptions, ReadResult
from . import events as ev
from .stats import numeric_stats, seasonal_cycle, time_coverage, trend

SENTINELS = (-9999, -9999.0, -999, -999.0, -999.9, -99999, -99.99, -8888, 9.96921e36, 1e20)
LAT_RE = re.compile(r"^(lat|latitude|lat_dd|lat_deg|decimallatitude|y_lat)$", re.I)
LON_RE = re.compile(r"^(lon|long|longitude|lng|lon_dd|lon_deg|decimallongitude|x_lon)$", re.I)
DATE_RE = re.compile(
    r"(date|time|datetime|timestamp|day|key|period|datum|fecha|epoch|تاريخ|التاريخ|وقت|زمن)", re.I
)
# values that are unmistakably dates, so the column name does not matter
DATE_VALUE_RE = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
)
DAY_FIRST_RE = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.]\d{4}")
MAX_TREND_COLS = 20
MAX_TS_COLS = 4


def _clean_sentinels(df: pd.DataFrame) -> dict[str, int]:
    replaced = {}
    for col in df.select_dtypes(include="number").columns:
        mask = df[col].isin(SENTINELS) | np.isinf(df[col].astype("float64"))
        if mask.any():
            df.loc[mask, col] = np.nan
            replaced[str(col)] = int(mask.sum())
    return replaced


def _parse_dates(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(200)
        if sample.empty or not sample.str.contains(r"\d").all():
            return None
        fmt = "%Y%m%d" if sample.str.fullmatch(r"\d{8}").all() else None
        parts = sample.str.extract(DAY_FIRST_RE).dropna().astype(int)
        dayfirst = bool(len(parts)) and (parts[0] > 12).any() and not (parts[1] > 12).any()
        parsed = pd.to_datetime(
            s.astype(str), errors="coerce", format=fmt, dayfirst=dayfirst, utc=_has_tz(sample)
        )
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        return parsed if parsed.notna().mean() > 0.9 else None
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        v = s.dropna()
        for unit, lo, hi in (("ms", 1e11, 4.2e12), ("s", 1e8, 4.2e9)):  # epoch, 1973-2100
            if len(v) and v.between(lo, hi).all():
                return pd.to_datetime(s, unit=unit, errors="coerce")
    return None


def _has_tz(sample: pd.Series) -> bool:
    return bool(sample.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").any())


def _date_like_values(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).str.strip().head(200)
    return len(sample) > 0 and sample.str.fullmatch(DATE_VALUE_RE).mean() > 0.9


def find_time(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    cols = {c.lower(): c for c in map(str, df.columns)}
    # NASA POWER style split columns
    if {"year", "mo", "dy"} <= set(cols) or {"year", "month", "day"} <= set(cols):
        y, m, d = (cols["year"], cols.get("mo", cols.get("month")), cols.get("dy", cols.get("day")))
        t = pd.to_datetime(dict(year=df[y], month=df[m], day=df[d]), errors="coerce")
        if t.notna().mean() > 0.9:
            return t, f"{y}+{m}+{d}"
    if {"year", "doy"} <= set(cols):
        t = pd.to_datetime(
            df[cols["year"]].astype("Int64").astype(str)
            + df[cols["doy"]].astype("Int64").astype(str).str.zfill(3),
            format="%Y%j",
            errors="coerce",
        )
        if t.notna().mean() > 0.9:
            return t, f"{cols['year']}+{cols['doy']}"
    ordered = sorted(df.columns, key=lambda c: 0 if DATE_RE.search(str(c)) else 1)
    for c in ordered:
        if (t := _parse_dates(df[c])) is not None:
            named = DATE_RE.search(str(c))
            numeric = pd.api.types.is_numeric_dtype(df[c])
            if numeric and not named:
                continue  # epoch numbers only for time-like names
            if not (
                named or _date_like_values(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])
            ):
                continue  # value-parsing alone is trusted only for unmistakable date strings
            return t, str(c)
    if "year" in cols and pd.api.types.is_numeric_dtype(df[cols["year"]]):
        y = df[cols["year"]]
        if y.between(1000, 3000).mean() > 0.95:
            return pd.to_datetime(
                y.astype("Int64").astype(str), format="%Y", errors="coerce"
            ), cols["year"]
    return None, None


def find_latlon(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat = next(
        (c for c in df.columns if LAT_RE.match(str(c)) and pd.api.types.is_numeric_dtype(df[c])),
        None,
    )
    lon = next(
        (c for c in df.columns if LON_RE.match(str(c)) and pd.api.types.is_numeric_dtype(df[c])),
        None,
    )
    if (
        lat is not None
        and lon is not None
        and df[lat].abs().max() <= 90
        and df[lon].abs().max() <= 360
    ):
        return str(lat), str(lon)
    return None, None


def _reader_notes(m: dict) -> list[str]:
    out = []
    if len(m.get("sheets", {})) > 1:
        others = ", ".join(f"'{k}'" for k in m["sheets"] if k != m.get("sheet_used"))
        out.append(
            f"workbook has {len(m['sheets'])} sheets; analysed '{m['sheet_used']}' (the largest), not {others}"
        )
    if m.get("units"):
        out.append(
            "units row under the header: " + ", ".join(f"{k} [{v}]" for k, v in m["units"].items())
        )
    if m.get("decimal") == "comma":
        out.append("decimal commas read as decimal points")
    if m.get("numbers_from_text"):
        out.append("numbers written as text converted: " + ", ".join(m["numbers_from_text"]))
    if m.get("dropped_empty_columns"):
        out.append(f"{m['dropped_empty_columns']} empty unnamed column(s) dropped")
    return out


def analyze_table(res: ReadResult, opts: ReadOptions) -> tuple[dict, list[tuple[str, bytes]]]:
    df = res.data
    is_geo = hasattr(df, "geometry") and getattr(df, "_geometry_column_name", None) in df.columns
    gdf = df if is_geo else None
    df = pd.DataFrame(df.drop(columns=df.geometry.name)) if is_geo else df.copy()
    for c in df.select_dtypes(
        include="object"
    ).columns:  # lists/dicts cannot be counted or compared
        if df[c].map(lambda v: isinstance(v, list | dict)).any():
            df[c] = df[c].map(
                lambda v: json.dumps(v, default=str) if isinstance(v, list | dict) else v
            )
    notes = _reader_notes(res.metadata)
    replaced = _clean_sentinels(df)
    if replaced:
        notes.append(
            "fill values (-9999/-999/..., ±inf) treated as missing: "
            + ", ".join(f"{k}: {v}" for k, v in replaced.items())
        )

    times, tcol = find_time(df)
    lat, lon = find_latlon(df)
    if is_geo and lat is None and len(gdf):
        g4326 = gdf.to_crs(4326) if gdf.crs and not gdf.crs.is_geographic else gdf
        cent = g4326.geometry.representative_point()
        df["_lon"], df["_lat"] = cent.x.to_numpy(), cent.y.to_numpy()
        lat, lon = "_lat", "_lon"

    if opts.var and opts.var not in df.columns:
        raise KeyError(f"--var {opts.var!r} not found. Columns: {', '.join(map(str, df.columns))}")
    if opts.bbox and lat:
        w, s, e, n = opts.bbox
        keep = df[lon].between(w, e) & df[lat].between(s, n)
        df, times = df[keep], times[keep] if times is not None else None
        notes.append(f"subset to bbox: {int(keep.sum())} rows kept")
    if (opts.start or opts.end) and times is not None:
        keep = times.between(
            pd.Timestamp(opts.start or "1000-01-01"), pd.Timestamp(opts.end or "2999-12-31")
        )
        df, times = df[keep], times[keep]
        notes.append(f"subset to time range: {int(keep.sum())} rows kept")

    coord_cols = {c for c in (lat, lon, tcol) if c}
    numeric = [
        c
        for c in df.select_dtypes(include="number").columns
        if str(c) not in coord_cols
        and not (
            str(tcol or "").count("+")
            and str(c).lower() in ("year", "mo", "dy", "doy", "month", "day")
        )
    ]
    if opts.var:
        numeric = [opts.var] + [c for c in numeric if c != opts.var]

    columns = []
    for c in df.columns:
        if c in ("_lat", "_lon"):
            continue
        col = df[c]
        columns.append(
            {
                "name": str(c),
                "dtype": str(col.dtype),
                "missing_pct": round(100 * col.isna().mean(), 2) if len(col) else 0.0,
                "unique": int(col.nunique(dropna=True)),
                "example": str(col.dropna().iloc[0])[:60] if col.notna().any() else "",
            }
        )
    summary = {"n_rows": int(len(df)), "n_columns": int(len(columns)), "columns": columns}
    if res.metadata.get("header_text"):
        summary["header_text"] = res.metadata["header_text"][:1500]
    stats = {
        str(c): numeric_stats(df[c].to_numpy(dtype="float64", na_value=np.nan)) for c in numeric
    }

    coverage: dict = {}
    if times is not None:
        coverage["time"] = {**(time_coverage(times) or {}), "column": tcol}
    if lat:
        coverage["space"] = {
            "bbox": [
                float(df[lon].min()),
                float(df[lat].min()),
                float(df[lon].max()),
                float(df[lat].max()),
            ],
            "bbox_order": "W, S, E, N (degrees)",
            "columns": [lon, lat],
        }
    if is_geo:
        coverage.setdefault("space", {})["crs"] = res.metadata.get("crs")
        coverage["space"]["bbox_native"] = [float(v) for v in gdf.total_bounds]

    figs: list[tuple[str, bytes]] = []
    trends: list[dict] = []
    event_info: dict | None = None
    if series_col := ev.series_column(times, df, lat):
        coverage.setdefault("time", {})["series_column"] = series_col
        notes.append(
            f"{df[series_col].nunique()} separate series in '{series_col}'; time series and "
            "trends use the mean across them"
        )
    elif ev.is_event_table(times, bool(lat)):
        # rows are events (e.g. fire detections): analyse how many happen when and where
        event_info, counts = ev.summarize(times, df, lat, lon)
        tc = coverage.get("time", {})
        if tc.pop("gaps", None):  # days without events are not missing data
            tc.pop("largest_gap", None)
        if t := trend(counts, "events per day", "events/day"):
            trends.append(t)
        if opts.plots:
            monthly = counts.resample("MS").sum()
            figs.append(
                (
                    "Events over time",
                    plots.time_series(
                        {"events per month": monthly}, "Number of events per month", "events"
                    ),
                )
            )
            if (cl := seasonal_cycle(counts)) is not None:
                figs.append(
                    (
                        "Seasonal cycle of events",
                        plots.seasonal_cycle(
                            {"events per day": cl}, "Average events per day, by month"
                        ),
                    )
                )
    if times is not None and numeric:
        ts = df[numeric].set_index(pd.DatetimeIndex(times)).sort_index()
        ts = ts[ts.index.notna()]
        per_date = ts.groupby(
            ts.index.normalize() if ts.index.normalize().nunique() > 1 else ts.index
        ).mean()
        for c in [c for c in numeric if df[c].nunique() > 2][:MAX_TREND_COLS]:  # not 0/1 flags
            if t := trend(per_date[c], str(c)):
                if event_info is not None:  # a daily mean of a few events is noise, not news
                    t["extremes"] = []
                trends.append(t)
        if opts.plots and len(per_date) > 1 and event_info is None:
            lines = {t["variable"]: (t["slope_per_year"], t["intercept"]) for t in trends}
            title = "Time series" + (" (mean per date)" if len(per_date) < len(ts) else "")
            figs.append(
                (
                    title,
                    plots.time_series(
                        {str(c): per_date[c] for c in numeric[:MAX_TS_COLS]},
                        title,
                        trend_lines=lines,
                    ),
                )
            )
            clims = {
                str(c): cl
                for c in numeric[:MAX_TS_COLS]
                if (cl := seasonal_cycle(per_date[c])) is not None
            }
            if clims:
                figs.append(("Seasonal cycle", plots.seasonal_cycle(clims, "Mean annual cycle")))
    if opts.plots:
        if lat:
            color_col = numeric[0] if numeric else None
            color = df[color_col].to_numpy(dtype="float64", na_value=np.nan) if color_col else None
            figs.insert(
                0,
                (
                    "Scatter map",
                    plots.scatter_map(
                        df[lon].to_numpy(float),
                        df[lat].to_numpy(float),
                        color,
                        f"Point locations{f' coloured by {color_col}' if color_col else ''}",
                        str(color_col or ""),
                    ),
                ),
            )
        if is_geo and gdf.geom_type.iloc[0] not in ("Point", "MultiPoint"):
            figs.insert(0, ("Geometries", plots.geometries(gdf, "Geometries")))
        if numeric:
            figs.append(("Histograms", plots.histograms(df[numeric], "Distributions")))
        if len(numeric) >= 2:
            figs.append(
                ("Correlation heatmap", plots.correlation(df[numeric], "Correlation (Pearson)"))
            )
    out = {
        "summary": summary,
        "coverage": coverage,
        "statistics": stats,
        "trends": trends,
        "notes": notes,
        "categories": ev.categories(df, {str(c) for c in (lat, lon, tcol) if c}),
        "correlations": ev.top_correlations(df[numeric]) if len(numeric) >= 2 else {},
    }
    if event_info:
        out["events"] = event_info
    return out, figs
