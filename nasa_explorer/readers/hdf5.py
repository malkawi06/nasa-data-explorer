"""HDF5 / HDF-EOS5. Tries xarray first; otherwise walks the group tree with h5py
(e.g. NISAR, ICESat-2, OCO-2 style files that have no single-dataset view)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from ..core import IN_BROWSER, ReadOptions, ReadResult
from ..registry import reader
from ._cf import clean_attrs

MAX_ROWS = 2000


def _as_grid(path: Path) -> xr.Dataset | None:
    for kw in (
        {"engine": "h5netcdf"},
        {"engine": "h5netcdf", "phony_dims": "sort"},
        *([] if IN_BROWSER else [{"engine": "netcdf4"}]),
    ):
        try:
            ds = xr.open_dataset(path, **kw)
        except Exception:
            continue
        if any(v.ndim >= 2 for v in ds.data_vars.values()):
            return ds
        ds.close()
    return None


def walk(path: Path) -> pd.DataFrame:
    import h5py

    rows: list[dict] = []

    def visit(name: str, obj) -> None:
        if len(rows) >= MAX_ROWS:
            return
        if isinstance(obj, h5py.Dataset):
            attrs = clean_attrs(obj.attrs)
            rows.append(
                {
                    "path": "/" + name,
                    "shape": "x".join(map(str, obj.shape)) or "scalar",
                    "dtype": str(obj.dtype),
                    "size": int(obj.size),
                    "units": str(attrs.get("units", "")),
                    "long_name": str(attrs.get("long_name", attrs.get("description", ""))),
                    "attrs": attrs,
                }
            )

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return pd.DataFrame(rows)


@reader(
    "hdf5",
    category="Earth science gridded",
    extensions=(".h5", ".he5", ".hdf5", ".h5ad"),
    magic=(b"\x89HDF\r\n\x1a\n",),
    requires=("h5py",),
)
def read_hdf5(path: Path, opts: ReadOptions) -> ReadResult:
    ds = _as_grid(path)
    if ds is not None:
        return ReadResult("grid", ds, {"global_attrs": dict(ds.attrs)})
    import h5py

    with h5py.File(path, "r") as f:
        root_attrs = clean_attrs(f.attrs)
        n_groups = sum(1 for _ in _iter_groups(f))
    tree = walk(path)
    meta = {"global_attrs": root_attrs, "n_groups": n_groups, "h5_path": str(path)}
    return ReadResult("tree", tree, meta)


def _iter_groups(f):
    import h5py

    out = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Group) else None)
    return out
