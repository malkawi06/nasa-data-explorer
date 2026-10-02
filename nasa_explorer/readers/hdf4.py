"""HDF4 / HDF-EOS2 (e.g. MODIS) via pyhdf. Scientific datasets become xarray variables
with fill values and the HDF4 calibration convention applied."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ..core import ReadOptions, ReadResult
from ..registry import reader
from ._cf import clean_attrs, decode

GEO_NAMES = {"latitude": "lat", "longitude": "lon"}


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
        if "_FillValue" not in attrs and (fill := sds.getfillvalue()) is not None:
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
    if skipped:
        meta["skipped_variables"] = skipped
    return ReadResult("grid", ds, meta)
