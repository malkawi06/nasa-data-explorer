"""HDF4 / HDF-EOS2 (e.g. MODIS) via pyhdf. Scientific datasets become xarray variables
with fill values and the HDF4 calibration convention applied."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

import numpy as np
import xarray as xr

from ..core import ReadOptions, ReadResult
from ..registry import reader
from ._cf import clean_attrs, decode

GEO_NAMES = {"latitude": "lat", "longitude": "lon"}
# MODIS land tiles: the sinusoidal grid on the MODIS sphere (MODIS Land / HDF-EOS2 spec).
SINUSOIDAL = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
_GRID = re.compile(r"GROUP=GRID_\d+(.*?)END_GROUP=GRID_\d+", re.S)
_PAIR = r"\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)"


def _grids(struct: str) -> list[dict]:
    """Grid definitions from an HDF-EOS2 StructMetadata.0 attribute."""
    out = []
    for block in _GRID.findall(struct or ""):
        f = {k: v for k, v in re.findall(r"^\s*(\w+)=(.+?)\s*$", block, re.M)}
        ul = re.match(_PAIR, f.get("UpperLeftPointMtrs", ""))
        lr = re.match(_PAIR, f.get("LowerRightMtrs", ""))
        if ul and lr and f.get("XDim", "").isdigit() and f.get("YDim", "").isdigit():
            out.append(
                {
                    "name": f.get("GridName", "").strip('"'),
                    "nx": int(f["XDim"]),
                    "ny": int(f["YDim"]),
                    "ul": (float(ul[1]), float(ul[2])),
                    "lr": (float(lr[1]), float(lr[2])),
                    "projection": f.get("Projection", ""),
                }
            )
    return out


def _dms(v: float) -> float:
    """HDF-EOS packed degrees DDDMMMSSS.SS (GCTP_GEO corners) to decimal degrees."""
    sign, v = (-1 if v < 0 else 1), abs(v)
    d, rest = divmod(v, 1e6)
    m, sec = divmod(rest, 1e3)
    return sign * (d + m / 60 + sec / 3600)


def _georeference(ds: xr.Dataset, struct: str, meta: dict) -> xr.Dataset:
    """Pixel-centre coordinates (and the CRS) of an HDF-EOS2 grid, so tiles get a map and a
    bounding box. Only single-grid files: several grids at different resolutions in one file
    would need one dataset each."""
    grids = _grids(struct)
    if len(grids) != 1:
        if grids:
            meta["georeference"] = f"{len(grids)} HDF-EOS grids in one file: not georeferenced"
        return ds
    g = grids[0]
    ydim = next((d for d in ds.dims if str(d).startswith("YDim") and ds.sizes[d] == g["ny"]), None)
    xdim = next((d for d in ds.dims if str(d).startswith("XDim") and ds.sizes[d] == g["nx"]), None)
    if ydim is None or xdim is None:
        return ds
    (x0, y0), (x1, y1) = g["ul"], g["lr"]
    proj = g["projection"]
    if "GEO" in proj and "SNSOID" not in proj:  # corners in packed degrees
        x0, y0, x1, y1 = (_dms(v) for v in (x0, y0, x1, y1))
        names = ("lat", "lon")
    elif "SNSOID" in proj:
        names = ("y", "x")
    else:
        meta["georeference"] = f"HDF-EOS projection {proj} not supported"
        return ds
    ys = y0 + (np.arange(g["ny"]) + 0.5) * (y1 - y0) / g["ny"]
    xs = x0 + (np.arange(g["nx"]) + 0.5) * (x1 - x0) / g["nx"]
    ds = ds.rename({ydim: names[0], xdim: names[1]}).assign_coords({names[0]: ys, names[1]: xs})
    meta["grid_name"] = g["name"]
    if names == ("y", "x"):
        meta["crs"] = SINUSOIDAL
        with contextlib.suppress(ImportError):  # rioxarray gives the bounding box in degrees
            import rioxarray  # noqa: F401

            ds = ds.rio.write_crs(SINUSOIDAL)
    return ds


@reader(
    "hdf4",
    category="Earth science gridded",
    extensions=(".hdf", ".hdf4", ".h4", ".he4"),
    magic=(b"\x0e\x03\x13\x01",),
    requires=("pyhdf",),
    extra="hdf4",
)
def read_hdf4(path: Path, opts: ReadOptions) -> ReadResult:
    from pyhdf.SD import SD, SDC

    sd = SD(str(path), SDC.READ)
    file_attrs = sd.attributes()
    names = list(sd.datasets())
    if opts.var and opts.var in names:
        names = [opts.var] + [n for n in names if n.lower() in GEO_NAMES]
    variables: dict[str, xr.DataArray] = {}
    skipped: list[str] = []
    for name in names:
        sds = sd.select(name)
        attrs = sds.attributes()
        if "_FillValue" not in attrs:
            with contextlib.suppress(Exception):  # pyhdf raises when no fill value is set
                if (fill := sds.getfillvalue()) is not None:
                    attrs["_FillValue"] = fill
        dims = [d if d else f"dim_{i}" for i, d in enumerate(sds.dimensions())]
        data = decode(np.asarray(sds.get()), attrs, hdf4_convention=True)
        attrs = {
            k: v
            for k, v in clean_attrs(attrs).items()
            if k not in ("scale_factor", "add_offset", "_FillValue", "valid_range")
        }
        variables[name] = xr.DataArray(data, dims=dims, attrs=attrs)
        sds.endaccess()
    sd.end()
    ds = xr.Dataset()
    for name, da in variables.items():
        try:
            ds[name] = da
        except ValueError:  # conflicting dimension sizes
            skipped.append(name)
    # MODIS swath: promote Latitude/Longitude to coordinates when shapes line up
    for name in list(ds.data_vars):
        if name.lower() in GEO_NAMES:
            ds = ds.set_coords(name)
    meta = {
        "global_attrs": {k: str(v)[:500] for k, v in file_attrs.items()},
        "calibration": "HDF4 convention: value = scale_factor * (raw - add_offset)",
    }
    ds = _georeference(ds, str(file_attrs.get("StructMetadata.0", "")), meta)
    if skipped:
        meta["skipped_variables"] = skipped
    return ReadResult("grid", ds, meta)
