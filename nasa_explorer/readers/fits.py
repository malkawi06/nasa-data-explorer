"""FITS (astronomy): images, multi-extension files and binary/ASCII tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..core import ReadOptions, ReadResult
from ..registry import reader

KEYS = ("OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "EXPTIME", "FILTER", "BUNIT", "ORIGIN")


def _table(hdu) -> pd.DataFrame:
    from astropy.table import Table

    t = Table(hdu.data)
    simple = [c for c in t.colnames if len(t[c].shape) <= 1]
    return t[simple].to_pandas()


@reader(
    "fits",
    category="Astronomy",
    extensions=(".fits", ".fit", ".fts", ".fits.fz"),
    magic=(b"SIMPLE  =",),
    requires=("astropy",),
    extra="astro",
)
def read_fits(path: Path, opts: ReadOptions) -> ReadResult:
    from astropy.io import fits

    hdus_info, images, tables = [], {}, {}
    with fits.open(path, memmap=False) as hdul:
        for i, hdu in enumerate(hdul):
            hdr = hdu.header
            name = hdu.name or f"HDU{i}"
            info = {
                "index": i,
                "name": name,
                "type": type(hdu).__name__,
                **{k.lower(): str(hdr[k]) for k in KEYS if k in hdr},
            }
            if hdu.data is None:
                hdus_info.append(info)
                continue
            if hdu.is_image:
                arr = np.asarray(hdu.data, dtype="float64")
                info["shape"] = list(arr.shape)
                dims = (
                    [f"{name.lower()}_{a}" for a in ("z", "y", "x")[-arr.ndim :]]
                    if arr.ndim <= 3
                    else [f"{name.lower()}_d{j}" for j in range(arr.ndim)]
                )
                attrs = {
                    "units": str(hdr.get("BUNIT", "")),
                    "long_name": str(hdr.get("OBJECT", name)),
                }
                images[f"{name}_{i}" if name in images else name] = xr.DataArray(
                    arr, dims=dims, attrs=attrs
                )
                info["wcs"] = _sky_box(hdr, arr.shape)
            else:
                df = _table(hdu)
                tables[name] = df
                info["rows"] = len(df)
                info["columns"] = list(df.columns)[:40]
            hdus_info.append(info)
    meta = {"hdus": hdus_info}
    if images:
        meta["tables"] = {k: len(v) for k, v in tables.items()}
        return ReadResult("grid", xr.Dataset(images), meta)
    if tables:
        name, df = max(tables.items(), key=lambda kv: len(kv[1]))
        meta["table_used"] = name
        return ReadResult("table", df, meta)
    return ReadResult("binary", None, meta)


def _sky_box(hdr, shape) -> dict | None:
    """RA/Dec footprint when a celestial WCS is present."""
    try:
        from astropy.wcs import WCS

        w = WCS(hdr).celestial
        if not w.has_celestial or len(shape) < 2:
            return None
        ny, nx = shape[-2], shape[-1]
        ra, dec = w.all_pix2world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1], 0)
        return {
            "ra_min": float(np.min(ra)),
            "ra_max": float(np.max(ra)),
            "dec_min": float(np.min(dec)),
            "dec_max": float(np.max(dec)),
        }
    except Exception:
        return None
