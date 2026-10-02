"""GeoTIFF / Cloud-Optimized GeoTIFF / other georeferenced rasters via rioxarray."""

from __future__ import annotations

from pathlib import Path

from ..core import NotThisFormat, ReadOptions, ReadResult
from ..registry import reader


@reader(
    "geotiff",
    category="Geospatial",
    extensions=(".tif", ".tiff", ".cog", ".jp2", ".img", ".vrt"),
    magic=(b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"),
    requires=("rasterio", "rioxarray"),
    extra="geo",
)
def read_geotiff(path: Path, opts: ReadOptions) -> ReadResult:
    import rioxarray

    da = rioxarray.open_rasterio(path, masked=True)
    crs = da.rio.crs
    if crs is None and da.rio.transform().is_identity:
        raise NotThisFormat("TIFF has no georeferencing - treated as a plain image")
    name = da.attrs.get("long_name") if isinstance(da.attrs.get("long_name"), str) else "band_data"
    ds = da.to_dataset(name=name or "band_data")
    meta = {
        "crs": crs.to_string() if crs else None,
        "transform": list(da.rio.transform())[:6],
        "nodata": None if da.rio.nodata is None else float(da.rio.nodata),
        "global_attrs": {k: str(v)[:300] for k, v in da.attrs.items()},
    }
    return ReadResult("grid", ds, meta)
