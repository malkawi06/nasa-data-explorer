"""SRTM / NASADEM .hgt tiles: raw big-endian int16 squares of 1201 (3") or 3601 (1") samples,
named after their south-west corner (N29E035.hgt). Read directly instead of through GDAL,
which only accepts the exact tile name: renamed copies such as "N29E035 (1).hgt" or
"03_SRTM_N29E035.hgt" still work, and no geospatial library is needed."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

from ..core import NotThisFormat, ReadOptions, ReadResult
from ..registry import reader

SIDES = {1201 * 1201 * 2: 1201, 3601 * 3601 * 2: 3601}
_TILE = re.compile(r"([NS])(\d{2})([EW])(\d{3})", re.I)


@reader("srtm-hgt", category="Geospatial", extensions=(".hgt",), priority=5)
def read_hgt(path: Path, opts: ReadOptions) -> ReadResult:
    n = SIDES.get(path.stat().st_size)
    if n is None:
        raise NotThisFormat("not a 1201x1201 or 3601x3601 SRTM tile")
    m = _TILE.search(path.name)
    if m is None:
        raise NotThisFormat("the file name has no tile corner such as N29E035")
    lat0 = int(m.group(2)) * (1 if m.group(1).upper() == "N" else -1)
    lon0 = int(m.group(4)) * (1 if m.group(3).upper() == "E" else -1)
    z = np.fromfile(path, dtype=">i2").reshape(n, n).astype("float32")
    z[z == -32768] = np.nan  # SRTM voids
    da = xr.DataArray(
        z,
        dims=("lat", "lon"),
        coords={"lat": np.linspace(lat0 + 1, lat0, n), "lon": np.linspace(lon0, lon0 + 1, n)},
        name="elevation",
        attrs={"units": "m", "long_name": "elevation above the EGM96 geoid (SRTM)"},
    )
    meta = {"crs": "EPSG:4326", "body": "Earth", "tile": m.group(0).upper(), "samples": n}
    return ReadResult("grid", da.to_dataset(), meta)
