"""Vector geospatial data: Shapefile, GeoJSON, KML/KMZ, GeoPackage, GML, FlatGeobuf."""

from __future__ import annotations

import zipfile
from pathlib import Path

from ..core import ReadOptions, ReadResult
from ..registry import looks_like_text, reader, text_head


def _sniff(head: bytes) -> bool:
    if head.startswith(b"SQLite format 3"):
        return True
    if not looks_like_text(head):
        return False
    t = text_head(head)[:2048]
    return ('"FeatureCollection"' in t or '"Feature"' in t and '"geometry"' in t) or "<kml" in t


@reader(
    "vector",
    category="Geospatial",
    extensions=(".shp", ".geojson", ".kml", ".kmz", ".gpkg", ".gml", ".fgb", ".topojson"),
    sniff=_sniff,
    requires=("geopandas", "pyogrio"),
    extra="geo",
    priority=30,
)
def read_vector(path: Path, opts: ReadOptions) -> ReadResult:
    import geopandas as gpd
    import pyogrio

    src = str(path)
    if path.suffix.lower() == ".kmz":
        with zipfile.ZipFile(path) as zf:
            kml = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
        if kml:
            src = f"/vsizip/{path}/{kml}"
    layers = [str(row[0]) for row in pyogrio.list_layers(src)]
    frames = {}
    for layer in layers[:50]:
        try:
            frames[layer] = gpd.read_file(src, layer=layer)
        except Exception:
            continue
    if not frames:
        frames[""] = gpd.read_file(src)
    layer, gdf = max(frames.items(), key=lambda kv: len(kv[1]))
    meta = {
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "layers": {k: len(v) for k, v in frames.items()},
        "layer_used": layer,
        "geometry_types": gdf.geom_type.value_counts().to_dict(),
    }
    return ReadResult("table", gdf, meta)
