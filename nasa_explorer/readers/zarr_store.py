"""Zarr stores (directory or .zarr / .zarr.zip)."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from ..core import ReadOptions, ReadResult
from ..registry import reader


def is_zarr_dir(path: Path) -> bool:
    return path.is_dir() and any((path / m).exists() for m in (".zgroup", ".zarray", "zarr.json"))


@reader("zarr", category="Earth science gridded", extensions=(".zarr",), requires=("zarr",))
def read_zarr(path: Path, opts: ReadOptions) -> ReadResult:
    try:
        ds = xr.open_zarr(path, consolidated=None)
    except Exception:
        ds = xr.open_dataset(path, engine="zarr")
    return ReadResult("grid", ds, {"global_attrs": dict(ds.attrs)})
