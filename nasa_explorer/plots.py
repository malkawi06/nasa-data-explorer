"""Quick-look plots (matplotlib, Agg). Every function returns PNG bytes or None."""

from __future__ import annotations

import io
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# matplotlib before 3.11 draws text left to right without Arabic letter joining, so "درجة" comes out as
# separate, reversed letters. Shape the letters and put right-to-left runs in visual order.
RTL_RUN = re.compile(
    r"[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]+(?:[\s_\-]+[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]+)*"
)


def rtl(text: str) -> str:
    """Visual-order text for labels that contain Arabic (unchanged otherwise)."""
    if not RTL_RUN.search(text):
        return text
    try:
        import arabic_reshaper

        text = arabic_reshaper.reshape(text)
    except ImportError:
        pass
    runs = re.split(f"({RTL_RUN.pattern})", text)  # odd items are right-to-left runs
    flipped = [r[::-1] if i % 2 else r for i, r in enumerate(runs)]
    mostly_rtl = sum(len(r) for r in runs[1::2]) * 2 >= len(text.strip())
    return "".join(reversed(flipped) if mostly_rtl else flipped)


if tuple(int(v) for v in re.findall(r"\d+", matplotlib.__version__)[:2]) < (3, 11):
    # 3.11+ lays out Arabic itself (libraqm); older builds, as in Pyodide, need the help
    _set_text = Text.set_text

    def _set_rtl_text(self, s):
        return _set_text(self, rtl(s) if isinstance(s, str) else s)

    Text.set_text = _set_rtl_text

# Validated categorical order (CVD-safe); assigned in order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, BASE, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#5598e7", "#f0efec", "#e66767", "#a32b2b"]
)
SEQ, DIV = SEQ.with_extremes(bad="#f0efec"), DIV.with_extremes(bad="#f0efec")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9,
        "lines.linewidth": 2,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    }
)

MAX_POINTS = 50_000


def _png(fig) -> bytes:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _cmap_for(arr: np.ndarray):
    """Diverging when the data straddles zero roughly symmetrically, else sequential."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return SEQ, None, None
    lo, hi = np.percentile(finite, [2, 98])
    if lo < 0 < hi and min(-lo, hi) / max(-lo, hi) > 0.3:
        m = max(-lo, hi)
        return DIV, -m, m
    return SEQ, lo, hi


def grid_map(
    arr: np.ndarray,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    title: str,
    label: str,
    use_cartopy: bool = True,
    stipple: np.ndarray | None = None,
    symmetric: bool = False,
) -> bytes:
    """Map of a 2-D field. `stipple` marks cells (e.g. significant trends) with dots;
    `symmetric` forces a zero-centred diverging scale (for slopes and anomalies)."""
    arr = np.asarray(arr, dtype="float64")
    cmap, vmin, vmax = _cmap_for(arr)
    if symmetric:
        finite = np.abs(arr[np.isfinite(arr)])
        m = float(np.percentile(finite, 98)) if finite.size else 1.0
        cmap, vmin, vmax = DIV, -m, m
    extent = None
    if (
        lat is not None
        and lon is not None
        and lat.ndim == 1
        and lon.ndim == 1
        and lat.size > 1
        and lon.size > 1
    ):
        dlat, dlon = abs(lat[1] - lat[0]) / 2, abs(lon[1] - lon[0]) / 2
        extent = [lon.min() - dlon, lon.max() + dlon, lat.min() - dlat, lat.max() + dlat]
        if lat[0] < lat[-1]:
            arr = arr[::-1]
            stipple = stipple[::-1] if stipple is not None else None
        if lon[0] > lon[-1]:
            arr = arr[:, ::-1]
            stipple = stipple[:, ::-1] if stipple is not None else None
    pts = _stipple_points(stipple, arr.shape, extent, lat, lon)
    if use_cartopy and extent is not None:
        try:
            return _cartopy_map(arr, extent, cmap, vmin, vmax, title, label, pts)
        except Exception:
            pass  # cartopy missing or Natural Earth data unavailable offline
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.grid(False)
    if lat is not None and lon is not None and lat.ndim == 2:
        im = ax.pcolormesh(lon, lat, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    else:
        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper",
            aspect="auto" if extent is None else "equal",
            interpolation="nearest",
        )
    if pts is not None:
        ax.scatter(*pts, s=2, color=INK, linewidths=0)  # the title explains the dots
    if extent is not None or (lat is not None and lat.ndim == 2):
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set_title(title)
    return _png(fig)


def _stipple_points(stipple, shape, extent, lat, lon):
    if stipple is None or not np.any(stipple):
        return None
    ys, xs = np.nonzero(stipple)
    if ys.size > 20_000:
        keep = np.random.default_rng(0).choice(ys.size, 20_000, replace=False)
        ys, xs = ys[keep], xs[keep]
    ny, nx = shape
    if extent is not None:
        return (
            extent[0] + (xs + 0.5) * (extent[1] - extent[0]) / nx,
            extent[3] - (ys + 0.5) * (extent[3] - extent[2]) / ny,
        )
    if lat is not None and lon is not None and lat.ndim == 2:
        return lon[ys, xs], lat[ys, xs]
    return xs.astype(float), ys.astype(float)


def _cartopy_map(arr, extent, cmap, vmin, vmax, title, label, pts=None) -> bytes:
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
    )
    if pts is not None:
        ax.scatter(
            *pts,
            s=2,
            color=INK,
            linewidths=0,
            transform=ccrs.PlateCarree(),
        )
    ax.coastlines(linewidth=0.6, color=INK_2)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color=GRID)
    gl.top_labels = gl.right_labels = False
    fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax.set_title(title)
    return _png(fig)  # raises here if coastline data cannot be fetched


def image2d(arr: np.ndarray, title: str, label: str = "") -> bytes:
    return grid_map(arr, None, None, title, label, use_cartopy=False)


def _runs(dates: list, max_gap_days: int = 62) -> list[list]:
    """Group dates that follow each other (e.g. three months of one heatwave) for one label."""
    runs: list[list] = []
    for d in dates:
        if runs and (d - runs[-1][-1]).days <= max_gap_days:
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


def time_series(
    series: dict[str, pd.Series],
    title: str,
    ylabel: str = "",
    trend_lines: dict[str, tuple[float, float]] | None = None,
    marks: dict[str, list[str]] | None = None,
) -> bytes:
    """Up to 8 series; a single series needs no legend (the title names it).
    `marks` labels dates (e.g. unusual months) on the named series."""
    items = list(series.items())[: len(SERIES)]
    fig, axes = plt.subplots(
        len(items), 1, figsize=(8, 2.4 * len(items) + 0.4), sharex=True, squeeze=False
    )
    for ax, (color, (name, s)) in zip(axes[:, 0], zip(SERIES, items, strict=False), strict=False):
        s = s.dropna()
        ax.plot(
            s.index,
            s.to_numpy(),
            color=color,
            linewidth=1.6 if len(s) > 200 else 2,
            marker="o" if len(s) <= 40 else None,
            markersize=4,
        )
        for group in _runs(sorted(pd.Timestamp(d) for d in (marks or {}).get(name, []))):
            idx = s.index[s.index.get_indexer(group, method="nearest")]
            ax.scatter(
                idx, s.loc[idx], s=36, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=3
            )
            peak = s.loc[idx].abs().idxmax()
            text = group[0].strftime("%Y-%m") + (f" – {group[-1]:%Y-%m}" if len(group) > 1 else "")
            ax.annotate(
                text,
                (peak, s.loc[peak]),
                xytext=(6, -2),
                textcoords="offset points",
                fontsize=7.5,
                color=INK_2,
            )
        if trend_lines and name in trend_lines:
            slope, intercept = trend_lines[name]
            x = s.index.year + (s.index.dayofyear - 1) / 365.25
            ax.plot(
                s.index,
                slope * x + intercept,
                color=INK_2,
                linewidth=1,
                linestyle="--",
                label="linear trend",
            )
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
        ax.set_ylabel(name if len(items) > 1 else ylabel, fontsize=8)
    axes[0, 0].set_title(title)
    return _png(fig)


def scatter_map(
    lon: np.ndarray, lat: np.ndarray, color: np.ndarray | None, title: str, label: str = ""
) -> bytes:
    if lon.size > MAX_POINTS:
        idx = np.random.default_rng(0).choice(lon.size, MAX_POINTS, replace=False)
        lon, lat = lon[idx], lat[idx]
        color = color[idx] if color is not None else None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    kw = dict(s=12, linewidths=0.4, edgecolors=SURFACE)
    if color is not None:
        cmap, vmin, vmax = _cmap_for(np.asarray(color, dtype="float64"))
        sc = ax.scatter(lon, lat, c=color, cmap=cmap, vmin=vmin, vmax=vmax, **kw)
        fig.colorbar(sc, ax=ax, label=label, shrink=0.85)
    else:
        ax.scatter(lon, lat, color=SERIES[0], **kw)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    return _png(fig)


def geometries(gdf, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    gdf.plot(ax=ax, color=SERIES[0], edgecolor=SURFACE, linewidth=0.4, markersize=12)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    return _png(fig)


def histograms(df: pd.DataFrame, title: str) -> bytes:
    cols = list(df.columns)[:12]
    n = len(cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), cols, strict=False):
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if vals.size:
            ax.hist(
                vals,
                bins=min(40, max(5, int(np.sqrt(vals.size)))),
                color=SERIES[0],
                edgecolor=SURFACE,
                linewidth=0.8,
            )
        ax.set_title(str(col)[:30], fontsize=9)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)


def correlation(df: pd.DataFrame, title: str) -> bytes:
    corr = df.corr(numeric_only=True)
    cols = corr.columns[:20]
    corr = corr.loc[cols, cols]
    size = max(4, 0.45 * len(cols) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    ax.grid(False)
    im = ax.imshow(corr.to_numpy(), cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [str(c)[:18] for c in cols], rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), [str(c)[:18] for c in cols])
    if len(cols) <= 10:
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = corr.iat[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#ffffff" if abs(v) > 0.6 else INK,
                    )
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title(title)
    return _png(fig)


def color_histogram(arr: np.ndarray, bands: list[str], title: str) -> bytes:
    named = {"R": "#e34948", "G": "#008300", "B": "#2a78d6", "A": MUTED, "L": INK_2}
    fig, ax = plt.subplots(figsize=(8, 3.2))
    flat = arr.reshape(-1, arr.shape[-1])
    if flat.shape[0] > 2_000_000:
        flat = flat[:: flat.shape[0] // 2_000_000 + 1]
    if arr.dtype == np.uint8:
        lo, hi = 0.0, 255.0
    else:  # 16-bit / float: a few saturated pixels would squash everything into one bin
        lo, hi = (float(v) for v in np.nanpercentile(flat[:, : len(bands)], [0.1, 99.9]))
        hi = hi if hi > lo else lo + 1.0
    for i, band in enumerate(bands):
        counts, edges = np.histogram(flat[:, i], bins=64, range=(lo, hi))
        ax.plot(edges[:-1], counts, color=named.get(band, SERIES[i % len(SERIES)]), label=band)
    ax.set_yscale("log" if counts.max() > 50 * max(1, np.median(counts)) else "linear")
    if len(bands) > 1:
        ax.legend(frameon=False, labelcolor=INK_2)
    ax.set_xlabel("pixel value" + ("" if arr.dtype == np.uint8 else " (0.1-99.9 percentile range)"))
    ax.set_ylabel("pixels")
    ax.set_title(title)
    return _png(fig)


def relief(z: np.ndarray, shade: np.ndarray, title: str, marks=()) -> bytes:
    """Elevation coloured on top of a hillshade; `marks` = (row, col, radius_px) circles."""
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.grid(False)
    ax.axis("off")
    finite = z[np.isfinite(z)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
    # low ground dark, high ground light: craters read as holes, not domes
    im = ax.imshow(z, cmap=SEQ.reversed(), vmin=lo, vmax=hi, interpolation="nearest")
    ax.imshow(shade, cmap="gray", vmin=0, vmax=1, alpha=0.5, interpolation="nearest")
    for r, c, rad in marks:
        ax.add_patch(Circle((c, r), max(rad, 3), fill=False, edgecolor="#eb6834", linewidth=1.4))
    fig.colorbar(im, ax=ax, shrink=0.75, label="elevation (m)")
    ax.set_title(title, fontsize=10)
    return _png(fig)


def slope_histogram(slope: np.ndarray, limits, pixel_m: float) -> bytes:
    s = slope[np.isfinite(slope)]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    if s.size:
        ax.hist(
            np.minimum(s, 60),
            bins=np.arange(0, 61, 1.0),
            weights=np.full(s.size, 100.0 / s.size),
            color=SERIES[0],
        )
        top = ax.get_ylim()[1]
        for i, lim in enumerate(limits):  # staggered so neighbouring labels never overlap
            ax.axvline(lim, color=INK_2, linewidth=1, linestyle="--")
            ax.text(
                lim + 0.4,
                top * (0.93 - 0.1 * (i % 2)),
                f"{100 * (s < lim).mean():.0f}% < {lim}°",
                fontsize=7.5,
                color=INK_2,
            )
        ax.set_xlim(0, min(60, max(30, float(np.percentile(s, 99.5)) + 5)))
    ax.set_xlabel(f"slope (°) at {pixel_m:,.0f} m baseline")
    ax.set_ylabel("% of area")
    ax.set_title("Slope distribution")
    return _png(fig)


def analog_bars(rows: list[dict], title: str) -> bytes:
    """Horizontal bars: each site's score split into its weighted factor contributions."""
    rows = rows[:20][::-1]
    keys = list(
        dict.fromkeys(f["key"] for r in rows for f in r["factors"] if f.get("score") is not None)
    )
    labels = {f["key"]: f["label"] for r in rows for f in r["factors"]}
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rows) + 1.6))
    left = np.zeros(len(rows))
    for i, key in enumerate(keys):
        part = []
        for r in rows:
            got = sum(f["weight"] for f in r["factors"] if f.get("score") is not None)
            f = next(
                (f for f in r["factors"] if f["key"] == key and f.get("score") is not None), None
            )
            part.append(0.0 if f is None or not got else f["score"] * f["weight"] / got)
        ax.barh(
            range(len(rows)),
            part,
            left=left,
            color=SERIES[i % len(SERIES)],
            label=labels[key],
            height=0.7,
        )
        left += np.asarray(part)
    for y, r in enumerate(rows):
        lo, hi = r.get("score_range") or (r["score"], r["score"])
        if hi > lo:  # how far the score moves when weights and levels vary
            ax.plot([lo, hi], [y, y], color=INK, lw=1.2, solid_capstyle="butt")
        ax.text(max(left[y], hi) + 1, y, f"{r['score']}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r["name"][:42] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("analog score (0-100), split into factor contributions; line = 90% range")
    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12 - 1.2 / (len(rows) + 3)),
    )
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    return _png(fig)


def site_map(rows: list[dict], title: str) -> bytes:
    """World scatter of sites coloured by score; the five best are labelled."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    lon = np.array([r["lon"] for r in rows], dtype="float64")
    lat = np.array([r["lat"] for r in rows], dtype="float64")
    val = np.array([r["score"] for r in rows], dtype="float64")
    sc = ax.scatter(
        lon, lat, c=val, cmap=SEQ, vmin=0, vmax=100, s=46, edgecolors=INK, linewidths=0.4
    )
    for r in sorted(rows, key=lambda r: -r["score"])[:5]:
        ax.annotate(
            r["name"].split(",")[0][:28],
            (r["lon"], r["lat"]),
            fontsize=7.5,
            xytext=(4, 4),
            textcoords="offset points",
        )
    fig.colorbar(sc, ax=ax, label="analog score", shrink=0.8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    return _png(fig)


def thumbnail(arr: np.ndarray, title: str) -> bytes:
    fig, ax = plt.subplots(figsize=(6, min(9, max(2, 6 * arr.shape[0] / max(arr.shape[1], 1)))))
    ax.grid(False)
    ax.axis("off")
    show = arr[..., 0] if arr.shape[-1] == 1 else arr[..., :3]
    ax.imshow(show, cmap="gray" if arr.shape[-1] == 1 else None)
    ax.set_title(title)
    return _png(fig)


MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def seasonal_cycle(clims: dict[str, pd.Series], title: str) -> bytes:
    """Mean annual cycle (12 monthly means) per variable, as small multiples."""
    items = list(clims.items())[: len(SERIES)]
    fig, axes = plt.subplots(
        1, len(items), figsize=(min(12, 3.4 * len(items) + 1), 2.8), squeeze=False
    )
    for ax, color, (name, clim) in zip(axes[0], SERIES, items, strict=False):
        ax.plot(
            range(1, 13),
            clim.reindex(range(1, 13)).to_numpy(),
            color=color,
            marker="o",
            markersize=4,
        )
        ax.set_xticks(range(1, 13), MONTHS)
        ax.set_title(str(name)[:30], fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)
