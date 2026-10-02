"""GeoTIFF / Cloud-Optimized GeoTIFF / other georeferenced rasters via rioxarray."""

from __future__ import annotations

from pathlib import Path

from .. import bodies
from ..core import NotThisFormat, ReadOptions, ReadResult
from ..registry import reader


def open_raster(path: Path, label_text: str = "", require_georef: bool = True) -> ReadResult:
    """Any GDAL raster as a grid. Scale/offset are applied (PDS DEMs store DN × 0.5 m)."""
    import rioxarray

    da = rioxarray.open_rasterio(path, mask_and_scale=True)
    crs = da.rio.crs
    if require_georef and crs is None and da.rio.transform().is_identity:
        raise NotThisFormat("raster has no georeferencing - treated as a plain image")
    name = da.attrs.get("long_name") if isinstance(da.attrs.get("long_name"), str) else "band_data"
    ds = da.to_dataset(name=name or "band_data")
    wkt = crs.to_wkt() if crs else None
    meta = {
        "crs": crs.to_string() if crs else None,
        "crs_wkt": wkt,
        "body": bodies.detect(wkt, label_text),
        "transform": list(da.rio.transform())[:6],
        "nodata": None if da.rio.nodata is None else float(da.rio.nodata),
        "global_attrs": {k: str(v)[:300] for k, v in da.attrs.items()},
    }
    return ReadResult("grid", ds, meta)


@reader(
    "geotiff",
    category="Geospatial",
    extensions=(
        ".tif",
        ".tiff",
        ".cog",
        ".jp2",
        ".img",
        ".vrt",
        ".hgt",
        ".bil",
        ".dem",
        ".grd",
        ".asc",
    ),
    magic=(b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+", b"ncols", b"NCOLS"),
    requires=("rasterio", "rioxarray"),
    extra="geo",
)
def read_geotiff(path: Path, opts: ReadOptions) -> ReadResult:
    return open_raster(path)
