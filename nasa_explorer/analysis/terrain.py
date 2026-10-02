"""Terrain of elevation models (Earth, Moon, Mars): slope, roughness, flat ground and
crater-like closed depressions, measured in metres on the right planetary body.

Slope and roughness depend on the pixel size they are measured at (a 1 km grid hides the
boulders a lander cares about), so every number carries its baseline, and a per-baseline
profile lets two DEMs be compared at the same scale.
"""

from __future__ import annotations

import math
import re

import numpy as np
import xarray as xr

from .. import bodies, plots

ELEVATION = re.compile(
    r"(?<![a-z])(dem|dtm|dsm|srtm|nasadem|gdem|copdem|cop.?dem|glo.?30|glo.?90|gmted|etopo|"
    r"gebco|elev\w*|height|altitude|topo\w*|terrain|ldem|lola|mola|megdr|hrsc|dtee\w*|astgtm|"
    r"bathy\w*)(?![a-z])"
    r"|(?<![a-z0-9])[ns]\d{2}[ew]\d{3}(?![0-9])"  # SRTM / ASTER / Copernicus tile names
    r"|\.(hgt|dem)$",
    re.I,
)
MAX_SIDE = 2048  # analysis grid; larger DEMs are block-averaged (not strided: striding aliases)
FILL_SIDE = 768  # depression search grid
SLOPE_LIMITS = (5, 10, 15, 25)  # degrees
SLOPE_BINS = np.arange(0, 62, 2.0)  # 0-60° in 2° bins; the last bin collects everything steeper
BASELINES_M = (10, 30, 100, 300, 1000, 3000, 10000)
BASELINE_TOLERANCE = 0.25
MIN_DEPRESSION_PX = 9
MAX_FILL_ITER = 800


def is_elevation(label: str, name: str, da: xr.DataArray) -> bool:
    """A single 2-D field whose file or variable name says it is elevation."""
    if sum(1 for d in da.dims if da.sizes[d] > 1) != 2:
        return False
    text = " ".join(
        [
            label,
            name,
            *(str(da.attrs.get(k, "")) for k in ("long_name", "standard_name", "description")),
        ]
    )
    return bool(ELEVATION.search(text)) or da.attrs.get("standard_name") in (
        "surface_altitude",
        "height_above_reference_ellipsoid",
        "height_above_mean_sea_level",
    )


def _crs(ds: xr.Dataset):
    try:
        return ds.rio.crs
    except Exception:
        return None


def pixel_size(ds: xr.Dataset, da: xr.DataArray, body: str) -> tuple[float, np.ndarray] | None:
    """(dy, dx for each row) in metres, or None without georeferencing."""
    ydim, xdim = da.dims[-2], da.dims[-1]
    crs = _crs(ds)
    if crs is not None and crs.is_projected:
        unit = crs.linear_units_factor[1]
        xres, yres = ds.rio.resolution()
        return abs(yres) * unit, np.full(da.sizes[ydim], abs(xres) * unit)
    ys = ds[ydim].values if ydim in ds.coords else None
    xs = ds[xdim].values if xdim in ds.coords else None
    geographic = (crs is not None and crs.is_geographic) or str(ydim).lower().startswith("lat")
    if crs is None and not geographic and ys is not None and xs is not None and ys.size > 1:
        # no CRS (e.g. ESRI ASCII grid without .prj): accept degrees only if they look like it
        span_ok = np.all(np.abs(ys) <= 90) and np.all((xs >= -180) & (xs <= 360))
        geographic = bool(span_ok and abs(float(ys[1] - ys[0])) < 1)
    if not geographic or ys is None or xs is None or ys.size < 2 or xs.size < 2:
        return None
    radius = bodies.radius_m(body)
    dlat = float(np.median(np.abs(np.diff(ys.astype("float64")))))
    dlon = float(np.median(np.abs(np.diff(xs.astype("float64")))))
    dx = np.radians(dlon) * radius * np.cos(np.radians(np.clip(ys.astype("float64"), -90, 90)))
    return np.radians(dlat) * radius, np.maximum(dx, 1e-6)


def _block_mean(z: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return z
    ny, nx = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    blocks = z[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(all="ignore"):
        valid = np.isfinite(blocks).sum(axis=(1, 3))
        mean = np.nansum(blocks, axis=(1, 3)) / np.maximum(valid, 1)
    return np.where(valid > f * f / 2, mean, np.nan)  # mostly-void blocks stay void


def _rows(dx: np.ndarray, f: int, n: int) -> np.ndarray:
    return dx[: n * f].reshape(n, f).mean(axis=1) * f if f > 1 else dx[:n]


def slope_deg(z: np.ndarray, dy: float, dx: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        gy = np.gradient(z, axis=0) / dy
        gx = np.gradient(z, axis=1) / dx[:, None]
        return np.degrees(np.arctan(np.hypot(gx, gy)))


def tri(z: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index: mean absolute height difference to the 8 neighbours."""
    acc, cnt = np.zeros_like(z), np.zeros_like(z)
    p = np.pad(z, 1, constant_values=np.nan)
    ny, nx = z.shape
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if oy or ox:
                d = np.abs(p[1 + oy : 1 + oy + ny, 1 + ox : 1 + ox + nx] - z)
                ok = np.isfinite(d)
                acc[ok] += d[ok]
                cnt += ok
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def hillshade(z: np.ndarray, dy: float, dx: np.ndarray, azimuth=315.0, altitude=45.0) -> np.ndarray:
    with np.errstate(all="ignore"):
        gy = np.gradient(z, axis=0) / dy
        gx = np.gradient(z, axis=1) / dx[:, None]
        slope = np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(gy, -gx)  # ESRI convention: rows run north to south
        az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
        shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def _filled(z: np.ndarray) -> np.ndarray:
    """Depressions filled to their spill level (morphological reconstruction by erosion).
    Valleys drain over the edge and stay open; only closed basins (craters, pits) fill."""
    from scipy import ndimage

    valid = np.isfinite(z)
    if not valid.all():  # voids take their nearest valid height so they neither drain nor pool
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    try:
        from skimage.morphology import reconstruction

        seed = np.full_like(z, z.max())
        seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = z[0, :], z[-1, :], z[:, 0], z[:, -1]
        return reconstruction(seed, z, method="erosion")
    except ImportError:
        pass
    marker = np.full_like(z, z.max())
    marker[0, :], marker[-1, :], marker[:, 0], marker[:, -1] = z[0, :], z[-1, :], z[:, 0], z[:, -1]
    for _ in range(MAX_FILL_ITER):  # geodesic erosion until stable (capped: huge basins underfill)
        new = np.maximum(ndimage.grey_erosion(marker, size=(3, 3)), z)
        if np.array_equal(new, marker):
            break
        marker = new
    return marker


def depressions(
    z: np.ndarray, dy: float, dx: np.ndarray, rough_m: float, coords: tuple | None
) -> dict:
    """Closed depressions deeper than the local roughness: craters and pits, not valleys."""
    from scipy import ndimage

    f = max(1, math.ceil(max(z.shape) / FILL_SIDE))
    zz, dyy, dxx = _block_mean(z, f), dy * f, _rows(dx, f, z.shape[0] // f)
    valid = np.isfinite(zz)
    if valid.mean() < 0.5 or min(zz.shape) < 16:
        return {}
    depth = _filled(zz) - np.where(valid, zz, np.nan)
    threshold = max(1.0, 3.0 * rough_m)
    mask = np.nan_to_num(depth) > threshold
    labels, n = ndimage.label(mask)
    if n == 0:
        return {
            "count": 0,
            "per_1000_km2": 0.0,
            "threshold_depth_m": round(threshold, 2),
            "pixel_m": round(float(dyy), 1),
        }
    index = np.arange(1, n + 1)
    area_px = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    max_depth = np.asarray(ndimage.maximum(np.nan_to_num(depth), labels, index))
    centres = np.asarray(ndimage.center_of_mass(mask, labels, index)).reshape(-1, 2)
    edge = np.zeros(n + 1, bool)
    for border in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        edge[np.unique(border)] = True
    keep = (area_px >= MIN_DEPRESSION_PX) & ~edge[1:]
    if not keep.any():
        return {
            "count": 0,
            "per_1000_km2": 0.0,
            "threshold_depth_m": round(threshold, 2),
            "pixel_m": round(float(dyy), 1),
        }
    rows = np.clip(centres[keep, 0].round().astype(int), 0, zz.shape[0] - 1)
    cell = dyy * dxx[rows]
    diameter = 2 * np.sqrt(area_px[keep] * cell / math.pi)
    depth_m = max_depth[keep]
    area_km2 = float(np.nansum(valid * (dyy * dxx[:, None])) / 1e6)
    order = np.argsort(-diameter)[:5]
    radius_px = diameter / 2 / math.sqrt(dyy * float(np.median(dxx))) * f
    largest = []
    for i in order:
        r, c = centres[keep][i]
        item = {
            "diameter_m": round(float(diameter[i]), 1),
            "depth_m": round(float(depth_m[i]), 1),
            "depth_to_diameter": round(float(depth_m[i] / diameter[i]), 3),
        }
        if coords is not None:
            ys, xs, ylab, xlab = coords
            item[ylab] = round(float(np.interp(r * f, np.arange(len(ys)), ys)), 5)
            item[xlab] = round(float(np.interp(c * f, np.arange(len(xs)), xs)), 5)
        largest.append(item)
    return {
        "count": int(keep.sum()),
        "per_1000_km2": round(float(1000 * keep.sum() / area_km2), 2) if area_km2 > 0 else None,
        "median_diameter_m": round(float(np.median(diameter)), 1),
        "median_depth_to_diameter": round(float(np.median(depth_m / diameter)), 3),
        "threshold_depth_m": round(threshold, 2),
        "pixel_m": round(float(dyy), 1),
        "largest": largest,
        "_marks": [
            (float(r) * f, float(c) * f, float(rad))
            for (r, c), rad in zip(centres[keep], radius_px, strict=True)
        ],
    }


def _summary(slope: np.ndarray, rough: np.ndarray, pixel_m: float) -> dict:
    s = slope[np.isfinite(slope)]
    r = rough[np.isfinite(rough)]
    if not s.size:
        return {}
    out = {
        "pixel_m": round(pixel_m, 1),
        "slope_median_deg": round(float(np.median(s)), 2),
        "slope_p90_deg": round(float(np.percentile(s, 90)), 2),
        "slope_max_deg": round(float(np.percentile(s, 99.9)), 1),
        "roughness_tri_m": round(float(np.median(r)), 2) if r.size else None,
    }
    for lim in SLOPE_LIMITS:
        out[f"flat_lt{lim}_pct"] = round(100 * float((s < lim).mean()), 1)
    return out


def profile(z: np.ndarray, dy: float, dx: np.ndarray) -> dict:
    """Slope/roughness at fixed baselines, so DEMs of different resolution compare fairly."""
    native = float(max(dy, float(np.median(dx))))
    out = {}
    for b in BASELINES_M:
        f = max(1, round(b / native))
        if abs(native * f - b) / b > BASELINE_TOLERANCE:
            continue
        n = z.shape[0] // f
        zz = _block_mean(z, f)
        if min(zz.shape) < 32:
            break
        s = slope_deg(zz, dy * f, _rows(dx, f, n))
        finite = s[np.isfinite(s)]
        if finite.size < 500:
            continue
        hist = np.histogram(np.minimum(finite, SLOPE_BINS[-1] - 1e-9), bins=SLOPE_BINS)[0]
        out[str(b)] = {
            **_summary(s, tri(zz), float(native * f)),
            "slope_hist": [round(float(v), 4) for v in hist / finite.size],
        }
    return out


def _coords(ds: xr.Dataset, da: xr.DataArray, body: str) -> tuple | None:
    ydim, xdim = da.dims[-2], da.dims[-1]
    if ydim not in ds.coords or xdim not in ds.coords:
        return None
    crs = _crs(ds)
    geographic = (crs is not None and crs.is_geographic) or str(ydim).lower().startswith("lat")
    ylab, xlab = ("lat", "lon") if geographic else ("y", "x")
    return ds[ydim].values.astype("float64"), ds[xdim].values.astype("float64"), ylab, xlab


def _to_metres(z: np.ndarray, units: str, body: str) -> tuple[np.ndarray, str | None]:
    note = None
    if units.strip().lower() in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
        z = z * 1000.0
    ref = bodies.REFERENCE_RADIUS_M.get(body)
    if ref is not None and np.isfinite(z).any() and abs(float(np.nanmedian(z)) - ref) < 50_000:
        z = z - ref
        note = f"values are planetary radius; shown as height above the {ref / 1000:g} km reference sphere"
    return z, note


def _read(ds: xr.Dataset, da: xr.DataArray) -> tuple[np.ndarray, int]:
    """2-D float array no larger than MAX_SIDE, block-averaged; and the averaging factor."""
    field = da.squeeze(drop=True)
    ny, nx = field.shape
    f = max(1, math.ceil(max(ny, nx) / MAX_SIDE))
    src = da.encoding.get("source")
    if f > 1 and src:
        try:  # average-resampled read straight from the file: no full-size array in memory
            import rasterio
            from rasterio.enums import Resampling

            with rasterio.open(src) as r:
                arr = r.read(
                    1,
                    out_shape=(ny // f, nx // f),
                    resampling=Resampling.average,
                    masked=True,
                ).astype("float64")
                scale, offset = r.scales[0] or 1.0, r.offsets[0] or 0.0
            return np.ma.filled(arr, np.nan) * scale + offset, f
        except Exception:
            pass
    return _block_mean(np.asarray(field.values, dtype="float64"), f), f


def analyze(
    ds: xr.Dataset, name: str, body: str, plot: bool
) -> tuple[dict, list[tuple[str, bytes]]]:
    da = ds[name]
    size = pixel_size(ds, da, body)
    z, f = _read(ds, da)
    z, note = _to_metres(z, str(da.attrs.get("units", "")), body)
    coords = _coords(ds, da, body)
    if coords is not None and f > 1:
        ys, xs, ylab, xlab = coords
        coords = (ys[f // 2 :: f][: z.shape[0]], xs[f // 2 :: f][: z.shape[1]], ylab, xlab)
    flip = coords is not None and len(coords[0]) > 1 and coords[0][0] < coords[0][-1]
    if flip:  # north up, so maps read normally and the light comes from the north-west
        z, coords = z[::-1], (coords[0][::-1], *coords[1:])
    finite = z[np.isfinite(z)]
    if finite.size < 100:
        return {}, []
    out: dict = {
        "variable": name,
        "body": body,
        "elevation_m": {
            "min": round(float(finite.min()), 1),
            "p5": round(float(np.percentile(finite, 5)), 1),
            "median": round(float(np.median(finite)), 1),
            "p95": round(float(np.percentile(finite, 95)), 1),
            "max": round(float(finite.max()), 1),
            "relief_p1_p99": round(float(np.percentile(finite, 99) - np.percentile(finite, 1)), 1),
        },
        "voids_pct": round(100 * float(1 - finite.size / z.size), 2),
    }
    if note:
        out["note"] = note
    figs: list[tuple[str, bytes]] = []
    if size is None:
        out["slope_note"] = "no georeferencing: slopes need the pixel size in metres"
        return out, figs
    dy, dx = size[0] * f, _rows(size[1], f, z.shape[0])
    dx = dx[::-1] if flip else dx
    slope = slope_deg(z, dy, dx)
    rough = tri(z)
    pixel_m = float(max(dy, float(np.median(dx))))
    out["at_analysis_resolution"] = _summary(slope, rough, pixel_m)
    if f > 1:
        out["averaged_from_native_m"] = round(pixel_m / f, 2)
    out["by_baseline_m"] = profile(z, dy, dx)
    rough_m = out["at_analysis_resolution"].get("roughness_tri_m") or 0.0
    out["depressions"] = depressions(z, dy, dx, rough_m, coords)
    marks = out["depressions"].pop("_marks", [])
    if plot:
        figs.append(
            (
                "Shaded relief",
                plots.relief(
                    z,
                    hillshade(z, dy, dx),
                    f"Shaded relief ({body}, {pixel_m:,.0f} m pixels); circles = closed depressions",
                    marks,
                ),
            )
        )
        figs.append(
            (
                "Slope map",
                plots.image2d(
                    np.minimum(slope, 45), f"Slope at {pixel_m:,.0f} m baseline (°)", "slope (°)"
                ),
            )
        )
        figs.append(("Slope distribution", plots.slope_histogram(slope, SLOPE_LIMITS, pixel_m)))
    return out, figs
