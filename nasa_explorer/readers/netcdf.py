"""NetCDF 3/4 via xarray (netcdf4 -> h5netcdf -> scipy engines)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from ..core import NotThisFormat, ReadOptions, ReadResult
from ..registry import reader

ENGINES = ("netcdf4", "h5netcdf", "scipy")


def open_any(path: Path, **kw) -> xr.Dataset:
    errors = []
    for engine in ENGINES:
        for decode_times in (True, False):
            try:
                return xr.open_dataset(path, engine=engine, decode_times=decode_times, **kw)
            except Exception as exc:  # engine missing or cannot parse
                errors.append(f"{engine}: {exc}")
    raise NotThisFormat("; ".join(errors[-3:]))


@reader(
    "netcdf",
    category="Earth science gridded",
    extensions=(".nc", ".nc4", ".cdf", ".netcdf"),
    magic=(b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n"),  # NetCDF-4 is HDF5
)
def read_netcdf(path: Path, opts: ReadOptions) -> ReadResult:
    ds = open_any(path)
    if not ds.data_vars:
        # NetCDF-4 with all data in groups: let the HDF5 tree reader walk it
        raise NotThisFormat("root group has no variables (data lives in groups)")
    return ReadResult("grid", ds, {"global_attrs": dict(ds.attrs)})
