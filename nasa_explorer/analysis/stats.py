"""Numeric statistics, time resolution and trend tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERCENTILES = (5, 25, 50, 75, 95)
MAX_SAMPLE = 5_000_000
MIN_TREND_POINTS = 6


def numeric_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype="float64").ravel()
    total = arr.size
    finite = arr[np.isfinite(arr)]
    out: dict = {
        "count": int(finite.size),
        "missing_pct": round(100 * (1 - finite.size / total), 2) if total else 0.0,
    }
    if finite.size == 0:
        return out
    pct = np.percentile(finite, PERCENTILES)
    out.update(
        min=float(finite.min()),
        max=float(finite.max()),
        mean=float(finite.mean()),
        std=float(finite.std()),
        **{f"p{p}": float(v) for p, v in zip(PERCENTILES, pct, strict=True)},
    )
    return out


def human_delta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds <= 0:
        return "irregular"
    days = seconds / 86400
    for label, size in (("year", 365.25), ("month", 30.44), ("week", 7), ("day", 1)):
        n = days / size
        if n >= 0.9 and abs(n - round(n)) < 0.12 * max(1, round(n)):
            n = round(n)
            return f"{n} {label}{'s' if n > 1 else ''}"
    for label, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        if seconds >= size:
            n = seconds / size
            return f"{n:g} {label}{'s' if n != 1 else ''}" if n == round(n) else f"{n:.2f} {label}s"
    return f"{seconds:.3g} s"


def time_coverage(times) -> dict | None:
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(times), errors="coerce")).dropna()
    if idx.empty:
        return None
    uniq = idx.unique().sort_values()
    diffs = pd.Series(uniq).diff().dt.total_seconds().to_numpy()[1:]
    return {
        "start": str(uniq[0]),
        "end": str(uniq[-1]),
        "n_steps": int(len(uniq)),
        "resolution": human_delta(float(np.median(diffs))) if diffs.size else "single time",
    }


def _decimal_years(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year + (idx.dayofyear - 1 + idx.hour / 24) / np.where(idx.is_leap_year, 366, 365)


def trend(series: pd.Series, name: str, units: str = "") -> dict | None:
    """Linear slope per year + Mann-Kendall test, described in plain words."""
    s = pd.Series(series).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
    s = s.groupby(level=0).mean().sort_index()
    if len(s) < MIN_TREND_POINTS or s.nunique() < 2:
        return None
    from scipy.stats import linregress

    x = _decimal_years(s.index)
    lr = linregress(x, s.to_numpy(dtype="float64"))
    out = {
        "variable": name,
        "n": int(len(s)),
        "slope_per_year": float(lr.slope),
        "slope_per_decade": float(lr.slope * 10),
        "intercept": float(lr.intercept),
        "units": units,
        "r2": float(lr.rvalue**2),
        "regression_p": float(lr.pvalue),
    }
    try:
        import pymannkendall as mk

        res = mk.original_test(s.to_numpy(dtype="float64"))
        out.update(mk_trend=res.trend, mk_p=float(res.p), sens_slope_per_step=float(res.slope))
    except ImportError:
        out.update(mk_trend="n/a", mk_p=float("nan"))
    out["text"] = describe_trend(out)
    return out


def describe_trend(t: dict) -> str:
    p = t.get("mk_p", float("nan"))
    unit = f" {t['units']}" if t.get("units") else ""
    slope = f"{t['slope_per_year']:+.4g}{unit}/year"
    ptxt = "p<0.001" if p < 0.001 else f"p={p:.3g}"
    direction = "increasing" if t["slope_per_year"] > 0 else "decreasing"
    if not np.isfinite(p):
        return f"slope {slope} (Mann-Kendall unavailable)"
    if p < 0.05:
        level = "99%" if p < 0.01 else "95%"
        return f"{direction}, significant at {level} ({ptxt}); slope {slope}"
    if p < 0.1:
        return f"weakly {direction} (significant at 90% only, {ptxt}); slope {slope}"
    return f"no significant trend ({ptxt}); slope {slope}"
