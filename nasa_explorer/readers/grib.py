"""GRIB 1/2 via cfgrib. Files with several hypercubes are split by cfgrib; the largest
is analysed and the rest are listed."""

from __future__ import annotations

from pathlib import Path

from .._native import preload_before_eccodes
from ..core import ReadOptions, ReadResult
from ..registry import reader


@reader(
    "grib",
    category="Earth science gridded",
    extensions=(".grib", ".grb", ".grib2", ".grb2", ".gb2"),
    magic=(b"GRIB",),
    requires=("cfgrib",),
    extra="grib",
)
def read_grib(path: Path, opts: ReadOptions) -> ReadResult:
    preload_before_eccodes()
    import cfgrib

    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    if not datasets:
        raise ValueError("no GRIB messages could be decoded")
    datasets.sort(key=lambda d: sum(v.size for v in d.data_vars.values()), reverse=True)
    ds = datasets[0]
    meta = {
        "global_attrs": dict(ds.attrs),
        "hypercubes": [sorted(d.data_vars) for d in datasets],
    }
    return ReadResult("grid", ds, meta)
