"""Analysis of gridded data (xarray.Dataset): NetCDF, HDF, GRIB, Zarr, GeoTIFF, FITS images."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import xarray as xr

from .. import plots
from ..core import ReadOptions, ReadResult
from .stats import MAX_SAMPLE, numeric_stats, time_coverage, trend

LAT_NAMES = {"lat", "latitude", "nav_lat", "lats", "xlat", "lat_0", "gridlat_0"}
LON_NAMES = {"lon", "longitude", "nav_lon", "lons", "long", "xlong", "lon_0", "gridlon_0"}
MAX_MAP_SIDE = 1500
MAX_TREND_VARS = 6


def _match(da: xr.DataArray, names: set[str], std: str, units: tuple[str, ...]) -> bool:
    return (
        str(da.name).lower() in names
        or da.attrs.get("standard_name") == std
        or str(da.attrs.get("units", "")).lower() in units
    )


def find_coord(ds: xr.Dataset, kind: str) -> str | None:
    pool = list(ds.coords) + [v for v in ds.data_vars if ds[v].ndim <= 2]
    for name in pool:
        da = ds[name]
        if kind == "lat" and _match(
            da, LAT_NAMES, "latitude", ("degrees_north", "degree_north", "degrees_n")
        ):
            return str(name)
        if kind == "lon" and _match(
            da, LON_NAMES, "longitude", ("degrees_east", "degree_east", "degrees_e")
        ):
            return str(name)
        if (
            kind == "time"
            and name in ds.coords
            and (
                np.issubdtype(da.dtype, np.datetime64)
                or str(name).lower() in ("time", "t", "valid_time", "date")
                or da.attrs.get("standard_name") == "time"
            )
        ):
            return str(name)
    return None


def _time_index(ds: xr.Dataset, name: str) -> pd.DatetimeIndex | None:
    try:
        idx = ds.indexes[name] if name in ds.indexes else pd.Index(np.atleast_1d(ds[name].values))
        if hasattr(idx, "to_datetimeindex"):  # cftime calendars
            idx = idx.to_datetimeindex()
        idx = pd.DatetimeIndex(idx)
        return idx
    except Exception:
        return None


def strided(
    da: xr.DataArray, max_elems: int = MAX_SAMPLE, keep: tuple[str, ...] = ()
) -> xr.DataArray:
    if da.size <= max_elems:
        return da
    dims = [d for d in da.dims if d not in keep and da.sizes[d] > 1]
    kept = math.prod(da.sizes[d] for d in keep if d in da.sizes) or 1
    factor = (da.size / kept / max(1, max_elems / kept)) ** (1 / max(1, len(dims)))
    step = max(1, math.ceil(factor))
    return da.isel({d: slice(None, None, step) for d in dims})


def subset(ds: xr.Dataset, opts: ReadOptions) -> tuple[xr.Dataset, list[str]]:
    notes: list[str] = []
    if opts.var:
        if opts.var not in ds.data_vars:
            raise KeyError(
                f"--var {opts.var!r} not found. Available: {', '.join(map(str, ds.data_vars))}"
            )
        ds = ds[[opts.var]]
    lat, lon = find_coord(ds, "lat"), find_coord(ds, "lon")
    if opts.bbox:
        w, s, e, n = opts.bbox
        if lat and lon and ds[lat].ndim == 1 and ds[lon].ndim == 1:
            lonv = ds[lon].values
            if lonv.max() > 180 and w < 0:
                w, e = w % 360, e % 360
            lat_sl = slice(s, n) if ds[lat].values[0] < ds[lat].values[-1] else slice(n, s)
            lon_sl = slice(w, e) if lonv[0] < lonv[-1] else slice(e, w)
            ds = ds.sel({lat: lat_sl, lon: lon_sl})
            notes.append(f"subset to bbox W={w} S={s} E={e} N={n}")
        else:
            notes.append("--bbox ignored: no 1-D lat/lon coordinates")
    t = find_coord(ds, "time")
    if (opts.start or opts.end) and t:
        ds = ds.sel({t: slice(opts.start, opts.end)})
        notes.append(f"subset to time {opts.start or '...'} - {opts.end or '...'}")
    return ds, notes


def _var_summary(name: str, da: xr.DataArray, stats: dict) -> dict:
    enc = da.encoding
    out = {
        "name": name,
        "dims": "x".join(f"{d}:{da.sizes[d]}" for d in da.dims),
        "dtype": str(enc.get("dtype", da.dtype)),
        "units": str(da.attrs.get("units", "")),
        "long_name": str(da.attrs.get("long_name", da.attrs.get("standard_name", ""))),
        "missing_pct": stats.get("missing_pct"),
    }
    for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
        val = enc.get(key, da.attrs.get(key))
        if val is not None:
            out[key] = val.item() if hasattr(val, "item") else str(val)
    return out


def _spatial(ds: xr.Dataset, lat: str | None, lon: str | None, meta: dict) -> dict | None:
    if lat and lon:
        la, lo = ds[lat].values, ds[lon].values
        out = {
            "bbox": [
                float(np.nanmin(lo)),
                float(np.nanmin(la)),
                float(np.nanmax(lo)),
                float(np.nanmax(la)),
            ],
            "bbox_order": "W, S, E, N (degrees)",
        }
        if la.ndim == 1 and la.size > 1 and lo.size > 1:
            out["resolution_deg"] = [
                float(np.median(np.abs(np.diff(lo)))),
                float(np.median(np.abs(np.diff(la)))),
            ]
        return out
    if meta.get("crs") and hasattr(ds, "rio"):
        try:
            out = {
                "crs": meta["crs"],
                "bbox_native": [float(v) for v in ds.rio.bounds()],
                "resolution_native": [abs(float(r)) for r in ds.rio.resolution()],
            }
            out["bbox"] = [float(v) for v in ds.rio.transform_bounds("EPSG:4326")]
            out["bbox_order"] = "W, S, E, N (degrees)"
            return out
        except Exception:
            return out if "out" in locals() else None
    sky = [h["wcs"] for h in meta.get("hdus", []) if h.get("wcs")]
    return {"sky_footprint_deg": sky[0]} if sky else None


def _main_var(ds: xr.Dataset) -> str | None:
    numeric = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number) and ds[v].ndim >= 2]
    return max(numeric, key=lambda v: ds[v].size) if numeric else None


def _map_slice(da: xr.DataArray, ds: xr.Dataset, lat: str | None, lon: str | None):
    if (
        lat
        and lon
        and ds[lat].ndim == 1
        and ds[lon].ndim == 1
        and {ds[lat].dims[0], ds[lon].dims[0]} <= set(da.dims)
    ):
        ydim, xdim = ds[lat].dims[0], ds[lon].dims[0]
    elif lat and lon and ds[lat].ndim == 2 and set(ds[lat].dims) <= set(da.dims):
        ydim, xdim = ds[lat].dims
    else:
        ydim, xdim = da.dims[-2], da.dims[-1]
    fixed = {d: 0 for d in da.dims if d not in (ydim, xdim)}
    sl = da.isel(fixed).transpose(ydim, xdim)
    step = max(1, math.ceil(max(sl.shape) / MAX_MAP_SIDE))
    sl = sl.isel({ydim: slice(None, None, step), xdim: slice(None, None, step)})
    la = lo = None
    if lat and lon and lat in sl.coords and lon in sl.coords and sl[lat].ndim == sl[lon].ndim:
        la, lo = sl[lat].values, sl[lon].values
    return sl.values, la, lo, fixed


def _area_mean(da: xr.DataArray, tdim: str, lat: str | None, ds: xr.Dataset) -> pd.Series:
    da = strided(da, 20_000_000, keep=(tdim,))
    others = [d for d in da.dims if d != tdim]
    if lat and ds[lat].ndim == 1 and ds[lat].dims[0] in others and lat in da.coords:
        w = np.cos(np.deg2rad(da[lat]))
        series = da.weighted(w.fillna(0)).mean(others)
    else:
        series = da.mean(others, skipna=True)
    return series.to_series()


def analyze_grid(res: ReadResult, opts: ReadOptions) -> tuple[dict, list[tuple[str, bytes]]]:
    ds, notes = subset(res.data, opts)
    lat, lon, tname = find_coord(ds, "lat"), find_coord(ds, "lon"), find_coord(ds, "time")
    stats, variables = {}, []
    for name, da in ds.data_vars.items():
        if not np.issubdtype(da.dtype, np.number) or da.size == 0:
            variables.append(
                {"name": str(name), "dims": "x".join(map(str, da.dims)), "dtype": str(da.dtype)}
            )
            continue
        sample = strided(da)
        st = numeric_stats(sample.values)
        if sample.size < da.size:
            st["sampled"] = f"{sample.size:,} of {da.size:,} values"
        stats[str(name)] = st
        variables.append(_var_summary(str(name), da, st))
    summary = {
        "dimensions": {str(k): int(v) for k, v in ds.sizes.items()},
        "coordinates": [str(c) for c in ds.coords],
        "variables": variables,
        "attributes": {str(k): str(v)[:300] for k, v in list(ds.attrs.items())[:40]},
    }
    coverage: dict = {}
    tindex = _time_index(ds, tname) if tname else None
    if tindex is not None:
        coverage["time"] = time_coverage(tindex)
    if sp := _spatial(ds, lat, lon, res.metadata):
        coverage["space"] = sp

    figs: list[tuple[str, bytes]] = []
    trends: list[dict] = []
    main = opts.var if opts.var in ds.data_vars else _main_var(ds)
    tdim = ds[tname].dims[0] if tname and ds[tname].ndim == 1 else None
    if tdim and tindex is not None and ds.sizes.get(tdim, 0) > 1:
        candidates = [
            v for v in ds.data_vars if tdim in ds[v].dims and np.issubdtype(ds[v].dtype, np.number)
        ]
        candidates = ([main] if main in candidates else []) + [v for v in candidates if v != main]
        for v in candidates[:MAX_TREND_VARS]:
            s = _area_mean(ds[v], tdim, lat, ds)
            s.index = (
                tindex[: len(s)]
                if len(s) == len(tindex)
                else pd.to_datetime(s.index, errors="coerce")
            )
            t = trend(s, str(v), str(ds[v].attrs.get("units", "")))
            if t:
                trends.append(t)
            if opts.plots and v == main:
                lines = {str(v): (t["slope_per_year"], t["intercept"])} if t else None
                figs.append(
                    (
                        f"Area-mean time series: {v}",
                        plots.time_series(
                            {str(v): s}, f"Area-mean {v}", str(ds[v].attrs.get("units", "")), lines
                        ),
                    )
                )
    if opts.plots and main:
        da = ds[main]
        arr, la, lo, fixed = _map_slice(da, ds, lat, lon)
        when = ", ".join(f"{k}[0]" for k in fixed) or "single field"
        if tdim in fixed and tindex is not None and len(tindex):
            when = str(tindex[0])[:19]
        units = str(da.attrs.get("units", ""))
        figs.insert(
            0, (f"Map: {main} ({when})", plots.grid_map(arr, la, lo, f"{main} - {when}", units))
        )
    return {
        "summary": summary,
        "coverage": coverage,
        "statistics": stats,
        "trends": trends,
        "notes": notes,
        "main_variable": main,
    }, figs
