"""Event / point data (fire detections, earthquakes, observations): what matters is how many
events happen when and where, not the average of each column per day."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

EVENTS_PER_DATE = 1.5  # more rows than this per date on average => treat rows as events
HOTSPOT_DEG = 0.5
MAX_CATEGORIES = 20
MIN_CLUSTER_SHARE = 0.002  # a cell must hold >=0.2% of events to seed or join a cluster
MIN_CORRELATION = 0.3
MAX_SERIES = 200  # more distinct values than this is an ID, not a station list


def is_event_table(times: pd.Series | None, has_coords: bool) -> bool:
    if times is None:
        return False
    days = pd.DatetimeIndex(times.dropna()).normalize()
    if len(days) < 10:
        return False
    per_date = len(days) / max(1, days.nunique())
    return per_date >= EVENTS_PER_DATE or (has_coords and per_date > 1.0)


def series_column(times: pd.Series | None, df: pd.DataFrame, lat: str | None = None) -> str | None:
    """A text column that splits repeated dates into separate series (station, site, city):
    such a table is several time series side by side, not a list of events. A station keeps
    its position; events that merely share a region name (earthquakes) do not."""
    if times is None:
        return None
    days = pd.DatetimeIndex(times).normalize()
    if days.nunique() == len(days):
        return None
    for col in df.columns:
        values = df[col]
        n = values.nunique()
        if pd.api.types.is_numeric_dtype(values) or not 1 < n <= min(MAX_SERIES, len(df) / 3):
            continue  # a series needs several rows; one value per row is an ID
        if lat is not None and df.groupby(values.astype(str))[lat].nunique().median() > 1:
            continue
        pairs = pd.Series(list(zip(days, values.astype(str), strict=True)))
        if pairs.duplicated().mean() < 0.05:
            return str(col)
    return None


def daily_counts(times: pd.Series) -> pd.Series:
    """Events per calendar day, with zero-event days filled in."""
    days = pd.DatetimeIndex(times.dropna()).normalize()
    counts = pd.Series(1, index=days).groupby(level=0).sum()
    full = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full, fill_value=0).astype(float)


def summarize(
    times: pd.Series, df: pd.DataFrame, lat: str | None, lon: str | None
) -> tuple[dict, pd.Series]:
    counts = daily_counts(times)
    by_month = counts.groupby(counts.index.month).sum()
    share = (100 * by_month / by_month.sum()).round(1)
    peak = int(share.idxmax())
    out: dict = {
        "total": int(counts.sum()),
        "days": int(len(counts)),
        "days_with_events_pct": round(100 * float((counts > 0).mean()), 1),
        "per_day_mean": round(float(counts.mean()), 2),
        "busiest_day": counts.idxmax().strftime("%Y-%m-%d"),
        "busiest_day_count": int(counts.max()),
        "peak_month": calendar.month_name[peak],
        "peak_month_share_pct": float(share[peak]),
        "month_share_pct": {calendar.month_abbr[m]: float(v) for m, v in share.items()},
    }
    if lat and lon:
        out["hotspots"] = hotspots(df[lat], df[lon])
    return out, counts


def hotspots(lat: pd.Series, lon: pd.Series, top: int = 5) -> dict:
    """Where events concentrate: neighbouring busy HOTSPOT_DEG cells are merged into clusters
    (8-connected), so one fire region is reported once instead of split across cells."""
    pts = pd.DataFrame({"lat": lat, "lon": lon}).dropna()
    if pts.empty:
        return {}
    iy = np.floor(pts["lat"].to_numpy() / HOTSPOT_DEG).astype(int)
    ix = np.floor(pts["lon"].to_numpy() / HOTSPOT_DEG).astype(int)
    cell_counts = (
        pd.Series(1, index=pd.MultiIndex.from_arrays([iy, ix])).groupby(level=[0, 1]).sum()
    )
    busy = cell_counts[cell_counts >= max(2, MIN_CLUSTER_SHARE * len(pts))]
    label: dict[tuple[int, int], int] = {}
    for start in busy.index:  # flood fill over busy cells
        if start in label:
            continue
        label[start] = len(set(label.values()))
        stack = [start]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (y + dy, x + dx)
                    if nb in busy.index and nb not in label:
                        label[nb] = label[start]
                        stack.append(nb)
    keys = list(zip(iy, ix, strict=True))
    pts["cluster"] = [label.get(k, -1) for k in keys]
    clustered = pts[pts["cluster"] >= 0].groupby("cluster")
    summary = clustered.agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        count=("lat", "size"),
        south=("lat", "min"),
        north=("lat", "max"),
        west=("lon", "min"),
        east=("lon", "max"),
    )
    summary = summary.sort_values("count", ascending=False).head(top)
    return {
        str(i + 1): {
            "lat": round(float(r.lat), 2),
            "lon": round(float(r.lon), 2),
            "count": int(r.count),
            "share_pct": round(100 * r.count / len(pts), 1),
            "bbox": [
                round(float(r.west), 2),
                round(float(r.south), 2),
                round(float(r.east), 2),
                round(float(r.north), 2),
            ],
        }
        for i, r in enumerate(summary.itertuples())
    }


def categories(df: pd.DataFrame, skip: set[str]) -> dict:
    """Value shares of low-cardinality text columns (e.g. satellite, day/night flag)."""
    out = {}
    for col in df.columns:
        if str(col) in skip or pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna().astype(str)
        if 1 < values.nunique() <= MAX_CATEGORIES:
            share = (100 * values.value_counts(normalize=True)).round(1).head(8)
            out[str(col)] = {str(k): float(v) for k, v in share.items()}
    return out


def top_correlations(df: pd.DataFrame, top: int = 5) -> dict:
    if df.shape[1] < 2:
        return {}
    corr = df.corr(numeric_only=True)
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.at[a, b]
            if np.isfinite(r) and abs(r) >= MIN_CORRELATION:
                pairs.append((abs(r), f"{a} ~ {b}", round(float(r), 2)))
    return {name: r for _, name, r in sorted(pairs, reverse=True)[:top]}
