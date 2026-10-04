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
    out = {
        "start": str(uniq[0]),
        "end": str(uniq[-1]),
        "n_steps": int(len(uniq)),
        "resolution": human_delta(float(np.median(diffs))) if diffs.size else "single time",
        "duplicates": int(len(idx) - len(uniq)),
    }
    if diffs.size:
        med = float(np.median(diffs))
        # months/years vary in length, so only steps clearly longer than usual count as gaps
        gaps = diffs[diffs > 1.6 * med] if med > 0 else diffs[:0]
        out["gaps"] = int(gaps.size)
        if gaps.size:
            out["largest_gap"] = human_delta(float(gaps.max()))
    return out


def _decimal_years(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year + (idx.dayofyear - 1 + idx.hour / 24) / np.where(idx.is_leap_year, 366, 365)


SEASONAL_STRENGTH = 0.2  # share of variance explained by the mean annual cycle


def _clean_series(series: pd.Series) -> pd.Series:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
    return s.groupby(level=0).mean().sort_index().astype("float64")


def seasonal_cycle(series: pd.Series) -> pd.Series | None:
    """Mean value per calendar month, when the series spans at least two years."""
    s = _clean_series(series)
    if len(s) < 24 or (s.index[-1] - s.index[0]).days < 700:
        return None
    clim = s.groupby(s.index.month).mean()
    return clim if len(clim) == 12 else None


def _lag1(x: np.ndarray) -> float:
    if x.size < 4 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


SEN_SAMPLE = 400_000  # random pairs that bracket the median slope of long series
SEN_BLOCK = 65_536  # pairwise slopes per numpy call: few calls, still cache-sized


def _pair_slopes(x: np.ndarray):
    """Every (x[i + k] - x[i]) / k, a block of lags k at a time (few numpy calls even in
    the browser). The ragged ends of a block are NaN, which no comparison selects."""
    n = x.size
    padded = np.concatenate([x, np.full(n, np.nan)])
    k = 1
    while k < n:
        b = max(1, min(n - k, SEN_BLOCK // (n - k)))
        win = np.lib.stride_tricks.sliding_window_view(padded[k : n + b - 1], n - k)
        yield (win - x[: n - k]) / np.arange(k, k + b)[:, None]
        k += b


def _sen_slope(x: np.ndarray) -> float:
    """Theil-Sen slope per step: the median of all (x[j] - x[i]) / (j - i), exactly.

    A daily series of 40 years has 128 million pairs, too many to hold. A random sample
    of pairs brackets the median; one pass over the pairs counts the slopes on each side
    of the bracket and keeps only those inside it. Ties at the bracket ends (dry days give
    many slopes of exactly 0) are counted, not stored."""
    n = x.size
    total = n * (n - 1) // 2
    ranks = ((total - 1) // 2, total // 2)  # the two middle slopes; equal when total is odd
    if total <= 4 * SEN_SAMPLE:
        d = np.concatenate([s[~np.isnan(s)] for s in _pair_slopes(x)])
        return float(np.mean(np.partition(d, ranks)[list(ranks)]))
    rng = np.random.default_rng(0)
    # sample size that minimises sorting the sample plus the slopes inside the bracket
    i, j = rng.integers(0, n, (2, int(min(SEN_SAMPLE, (3 * total) ** (2 / 3)))))
    i, j = np.minimum(i, j)[i != j], np.maximum(i, j)[i != j]
    sample = (x[j] - x[i]) / (j - i)
    m, q = sample.size, ranks[1] / total
    pad = 6 * np.sqrt(m * q * (1 - q)) + 2
    while True:
        a, b = int(m * q - pad), int(m * q + pad)
        part = np.partition(sample, [max(a, 0), min(b, m - 1)])
        lo, hi = (part[a] if a >= 0 else -np.inf), (part[b] if b < m else np.inf)
        below = at_lo = at_hi = 0
        inside = []
        for s in _pair_slopes(x):
            below += int(np.count_nonzero(s < lo))
            near = s[(s >= lo) & (s <= hi)]
            at_lo += int(np.count_nonzero(near == lo))
            at_hi += int(np.count_nonzero(near == hi)) if hi != lo else 0
            inside.append(near[(near > lo) & (near < hi)])
        mid = np.concatenate(inside)
        edges = np.cumsum([below, at_lo, mid.size, at_hi], dtype=np.int64)
        if below <= ranks[0] and ranks[1] < edges[3]:
            pos = [int(r - edges[1]) for r in ranks if edges[1] <= r < edges[2]]
            mid = np.partition(mid, pos) if pos else mid
            vals = [
                lo if r < edges[1] else mid[r - edges[1]] if r < edges[2] else hi for r in ranks
            ]
            return float(np.mean(vals))
        pad *= 4  # the sample missed the median (practically never): widen and recount


def _mann_kendall(x: np.ndarray, hamed_rao: bool) -> tuple[str, float, float, str]:
    """Mann-Kendall test (optionally with the Hamed-Rao 1998 variance correction for
    autocorrelation) and the Sen slope, as pymannkendall computes them, in O(n log n)
    time apart from the slope pass. Returns (trend, p, slope per step, method)."""
    from scipy.stats import kendalltau, norm, rankdata

    n = x.size
    pairs = n * (n - 1) / 2
    t = np.unique(x, return_counts=True)[1].astype("float64")  # int32 in the browser: overflows
    ties = float(np.sum(t * (t - 1) / 2))
    tau = kendalltau(np.arange(n), x).statistic  # tau-b: (P - Q) / sqrt(pairs (pairs - ties))
    s = float(round(tau * np.sqrt(pairs * (pairs - ties)))) if pairs > ties else 0.0
    var_s = (n * (n - 1) * (2.0 * n + 5) - float(np.sum(t * (t - 1) * (2 * t + 5)))) / 18
    slope = _sen_slope(x)
    method = "Mann-Kendall"
    if hamed_rao:
        r = rankdata(x - np.arange(1, n + 1) * slope)
        r -= r.mean()
        f = np.fft.rfft(r, 2 * n)
        acov = np.fft.irfft(f * np.conj(f), 2 * n)[:n]
        if acov[0] > 0:  # an exact straight line leaves no residual to correct
            acf = acov[1:] / acov[0]
            lag = np.arange(1, n, dtype="float64")
            strong = np.abs(acf) > norm.ppf(0.975) / np.sqrt(n)
            sni = np.sum(((n - lag) * (n - lag - 1) * (n - lag - 2) * acf)[strong])
            corrected = var_s * (1 + 2 / (n * (n - 1) * (n - 2)) * sni)
            if corrected > 0:
                var_s, method = corrected, "Hamed-Rao modified Mann-Kendall"
    z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0.0
    p = float(2 * norm.sf(abs(z)))
    strong = abs(z) > norm.ppf(0.975)
    return ("increasing" if z > 0 else "decreasing") if strong else "no trend", p, slope, method


def trend(series: pd.Series, name: str, units: str = "") -> dict | None:
    """Linear slope per year + a Mann-Kendall-family test, described in plain words.

    - A strong annual cycle is removed first (monthly climatology), otherwise it inflates
      the noise and can bias the slope when the record starts/ends mid-season.
    - Monthly data with a complete calendar uses the Seasonal Mann-Kendall test.
    - Significant lag-1 autocorrelation (common in climate series) switches to the
      Hamed-Rao modified test, which corrects the otherwise over-confident p-value.
    """
    s = _clean_series(series)
    if len(s) < MIN_TREND_POINTS or s.nunique() < 2:
        return None
    from scipy.stats import linregress

    x = np.asarray(_decimal_years(s.index), dtype="float64")
    y = s.to_numpy()
    clim = seasonal_cycle(s)
    strength = 0.0
    anomalies = y
    if clim is not None:
        fitted = clim.reindex(s.index.month).to_numpy()
        strength = float(np.var(fitted) / np.var(y)) if np.var(y) else 0.0
        if strength > SEASONAL_STRENGTH:
            anomalies = y - fitted
    seasonal = anomalies is not y
    lr = linregress(x, anomalies)
    residuals = anomalies - (lr.slope * x + lr.intercept)
    r1 = _lag1(residuals)
    out = {
        "variable": name,
        "n": int(len(s)),
        "slope_per_year": float(lr.slope),
        "slope_per_decade": float(lr.slope * 10),
        "intercept": float(y.mean() - lr.slope * x.mean()),  # line through the raw series
        "units": units,
        "r2": float(lr.rvalue**2),
        "regression_p": float(lr.pvalue),
        "seasonal": seasonal,
        "seasonal_strength": round(strength, 3),
        "lag1_autocorr": round(r1, 3),
    }
    try:
        step_days = float(pd.Series(s.index).diff().dt.total_seconds().median()) / 86400
        monthly = 27 <= step_days <= 32 and clim is not None
        # only positive autocorrelation inflates significance
        autocorrelated = r1 > 2 / np.sqrt(len(s))
        if seasonal and monthly and _complete_months(s):
            import pymannkendall as mk

            res = mk.seasonal_test(y, period=12)
            mk_trend, p, slope, method = res.trend, res.p, res.slope, "Seasonal Mann-Kendall"
        else:
            mk_trend, p, slope, method = _mann_kendall(
                anomalies, hamed_rao=autocorrelated and len(s) >= 10
            )
            if seasonal:
                method += " on deseasonalized anomalies"
        out.update(
            mk_trend=mk_trend,
            mk_p=float(p),
            sens_slope_per_step=float(slope),
            method=method,
        )
    except ImportError:
        out.update(mk_trend="n/a", mk_p=float("nan"), method="linear regression only")
    span = float(x[-1] - x[0])
    if span < MIN_SPAN_YEARS - 0.05 and float(pd.Series(s.index).diff().dt.days.median()) < 60:
        out["short_record_years"] = round(span, 2)  # one or two seasons: slope ~ the season
    out["extremes"] = _extremes(s.index, residuals)
    out["text"] = describe_trend(out)
    return out


MIN_SPAN_YEARS = 2.0  # below this, sub-monthly data cannot separate a trend from the seasons
SKEWED = 3.0  # upper tail this many times longer than the lower one
EXTREME_Z = 5.0  # robust z; lower values flag ordinary heavy-tailed noise


def _extremes(idx: pd.DatetimeIndex, residuals: np.ndarray, top: int = 5) -> list[dict]:
    """Periods far from normal after removing the annual cycle and the trend (robust z-score).
    Strongly right-skewed series (daily rain, fire power) are skipped: their big days are the
    normal shape of the data, and a z-score would call every storm a 10σ anomaly."""
    q05, q50, q95 = np.percentile(residuals, [5, 50, 95])
    if q95 - q50 > SKEWED * max(q50 - q05, 1e-12):  # quantile skew: a few outliers do not move it
        return []
    med = float(np.median(residuals))
    # MAD alone collapses for zero-inflated series (daily rain), turning every shower into 200σ
    scale = max(1.4826 * float(np.median(np.abs(residuals - med))), 0.5 * float(np.std(residuals)))
    if not np.isfinite(scale) or scale == 0:
        return []
    z = (residuals - med) / scale
    order = np.argsort(-np.abs(z))
    fmt = "%Y-%m" if len(idx) > 1 and (idx[1] - idx[0]).days >= 27 else "%Y-%m-%d"
    return [
        {"date": idx[i].strftime(fmt), "anomaly": float(residuals[i] - med), "z": float(z[i])}
        for i in order[:top]
        if abs(z[i]) >= EXTREME_Z
    ]


def anomalies(series: pd.Series) -> pd.Series | None:
    """Series minus its mean annual cycle (None when there is no strong cycle)."""
    s = _clean_series(series)
    clim = seasonal_cycle(s)
    if clim is None:
        return None
    return s - clim.reindex(s.index.month).to_numpy()


def _complete_months(s: pd.Series) -> bool:
    """Seasonal MK needs a regular monthly series starting in January-ish with no gaps."""
    periods = s.index.to_period("M")
    return periods.is_unique and len(periods) == (periods[-1] - periods[0]).n + 1


def describe_trend(t: dict) -> str:
    p = t.get("mk_p", float("nan"))
    unit = f" {t['units']}" if t.get("units") else ""
    slope = f"{t['slope_per_year']:+.4g}{unit}/year"
    ptxt = "p<0.001" if p < 0.001 else f"p={p:.3g}"
    direction = "increasing" if t["slope_per_year"] > 0 else "decreasing"
    if not np.isfinite(p):
        return f"slope {slope} (Mann-Kendall unavailable)"
    if "short_record_years" in t:
        return (
            f"record too short for a trend ({span_text(t['short_record_years'])}): the {direction} slope {slope} "
            f"mostly reflects the season ({ptxt})"
        )
    if p < 0.05:
        level = "99%" if p < 0.01 else "95%"
        return f"{direction}, significant at {level} ({ptxt}); slope {slope}{_how(t)}"
    if p < 0.1:
        return f"weakly {direction} (significant at 90% only, {ptxt}); slope {slope}{_how(t)}"
    return f"no significant trend ({ptxt}); slope {slope}{_how(t)}"


def span_text(years: float) -> str:
    return f"{round(years * 12)} months" if years < 2 else f"{years:.1f} years"


def _how(t: dict) -> str:
    method = t.get("method", "Mann-Kendall")
    return "" if method == "Mann-Kendall" else f" [{method}]"
