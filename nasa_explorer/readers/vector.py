"""Vector geospatial data: Shapefile, GeoJSON, KML/KMZ, GeoPackage, GML, FlatGeobuf."""

from __future__ import annotations

import zipfile
from pathlib import Path

from ..core import IN_BROWSER, ReadOptions, ReadResult
from ..registry import looks_like_text, reader, text_head


def _sniff(head: bytes) -> bool:
    if head.startswith(b"SQLite format 3"):
        return True
    if not looks_like_text(head):
        return False
    t = text_head(head)[:2048]
    return ('"FeatureCollection"' in t or '"Feature"' in t and '"geometry"' in t) or "<kml" in t


def _list_layers(src: str) -> list[str]:
    """pyogrio when installed, fiona otherwise (e.g. in the Pyodide browser build)."""
    try:
        import pyogrio

        return [str(row[0]) for row in pyogrio.list_layers(src)]
    except ImportError:
        import fiona

        # fiona ships KML read support but leaves it switched off by default
        for driver in ("KML", "LIBKML"):
            fiona.drvsupport.supported_drivers.setdefault(driver, "r")
        return list(fiona.listlayers(src))


def _pure_frames(path: Path, head: bytes) -> dict | None:
    """Browser build: avoid fiona for GeoPackage (crashes Pyodide) and KML (no driver)."""
    if not IN_BROWSER:
        return None
    from ._purevector import read_gpkg, read_kml

    if head.startswith(b"SQLite format 3"):
        return read_gpkg(path)
    if path.suffix.lower() in (".kml", ".kmz") or b"<kml" in head:
        return read_kml(path)
    return None


@reader(
    "vector",
    category="Geospatial",
    extensions=(".shp", ".geojson", ".kml", ".kmz", ".gpkg", ".gml", ".fgb", ".topojson"),
    sniff=_sniff,
    requires=("geopandas",),
    extra="geo",
    priority=30,
)
def read_vector(path: Path, opts: ReadOptions) -> ReadResult:
    import geopandas as gpd

    with open(path, "rb") as fh:
        head = fh.read(512)
    if (frames := _pure_frames(path, head)) is not None:
        return _result(frames)
    src = str(path)
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as zf:
            kml = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
        if kml:
            src = f"/vsizip/{path}/{kml}"
    layers = _list_layers(src)
    frames = {}
    for layer in layers[:50]:
        try:
            frames[layer] = gpd.read_file(src, layer=layer)
        except Exception:
            continue
    if not frames:
        frames[""] = gpd.read_file(src)
    return _result(frames)


def _result(frames: dict) -> ReadResult:
    if not frames:
        raise ValueError("no layers with features")
    layer, gdf = max(frames.items(), key=lambda kv: len(kv[1]))
    meta = {
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "layers": {k: len(v) for k, v in frames.items()},
        "layer_used": layer,
        "geometry_types": gdf.geom_type.value_counts().to_dict(),
    }
    return ReadResult("table", gdf, meta)
