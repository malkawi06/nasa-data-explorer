"""Pure-Python KML/KMZ and GeoPackage readers (sqlite3 + xml + shapely).

Used in the browser build, where fiona has no KML driver and its GeoPackage driver
crashes the Pyodide runtime. Each returns {layer name: GeoDataFrame}.
"""

from __future__ import annotations

import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# GeoPackage geometry header: 'GP', version, flags, srs_id, then an optional envelope
_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _gpkg_wkb(blob: bytes) -> bytes | None:
    if not blob or blob[:2] != b"GP":
        return blob or None
    flags = blob[3]
    if (flags >> 4) & 1:  # empty geometry
        return None
    return blob[8 + _ENVELOPE_BYTES.get((flags >> 1) & 7, 0) :]


def read_gpkg(path: Path) -> dict:
    import geopandas as gpd
    import pandas as pd
    import shapely.wkb

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        layers = con.execute(
            "SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns"
        ).fetchall()
        srs = dict(
            con.execute(
                "SELECT srs_id, organization || ':' || organization_coordsys_id FROM gpkg_spatial_ref_sys"
            ).fetchall()
        )
        frames = {}
        for table, geom_col, srs_id in layers[:50]:
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            geoms = [
                shapely.wkb.loads(w) if (w := _gpkg_wkb(b)) else None for b in df.pop(geom_col)
            ]
            crs = srs.get(srs_id) if srs_id and srs_id > 0 else None
            frames[table] = gpd.GeoDataFrame(
                df.drop(columns=["fid"], errors="ignore"), geometry=geoms, crs=crs
            )
        return frames
    finally:
        con.close()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in (text or "").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _geometry(el):
    from shapely.geometry import GeometryCollection, LineString, Point, Polygon

    name = _local(el.tag)
    if name == "Point":
        c = _coords(_find_text(el, "coordinates"))
        return Point(c[0]) if c else None
    if name in ("LineString", "LinearRing"):
        c = _coords(_find_text(el, "coordinates"))
        return LineString(c) if len(c) > 1 else None
    if name == "Polygon":
        rings = {"outerBoundaryIs": [], "innerBoundaryIs": []}
        for b in el:
            if _local(b.tag) in rings:
                rings[_local(b.tag)] += [
                    _coords(_find_text(r, "coordinates"))
                    for r in b.iter()
                    if _local(r.tag) == "LinearRing"
                ]
        outer = rings["outerBoundaryIs"]
        return Polygon(outer[0], rings["innerBoundaryIs"]) if outer and len(outer[0]) > 2 else None
    if name == "MultiGeometry":
        parts = [g for g in (_geometry(c) for c in el) if g is not None]
        return GeometryCollection(parts) if parts else None
    return None


def _find_text(el, name: str) -> str:
    for child in el.iter():
        if _local(child.tag) == name:
            return child.text or ""
    return ""


def read_kml(path: Path) -> dict:
    import geopandas as gpd

    if path.suffix.lower() == ".kmz" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist() if n.lower().endswith(".kml")), None)
            if name is None:
                raise ValueError("KMZ contains no .kml file")
            root = ET.fromstring(zf.read(name))
    else:
        root = ET.parse(path).getroot()
    rows, geoms = [], []
    for pm in root.iter():
        if _local(pm.tag) != "Placemark":
            continue
        row: dict = {}
        geom = None
        for child in pm:
            tag = _local(child.tag)
            if tag in ("name", "description") and child.text:
                row[tag] = child.text.strip()
            elif tag == "ExtendedData":
                for d in child.iter():
                    t = _local(d.tag)
                    if t == "Data" and d.get("name"):
                        row[d.get("name")] = _find_text(d, "value").strip()
                    elif t == "SimpleData" and d.get("name"):
                        row[d.get("name")] = (d.text or "").strip()
            elif geom is None:
                geom = _geometry(child)
        rows.append(row)
        geoms.append(geom)
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    for col in gdf.columns.drop("geometry"):  # numeric attributes arrive as text
        converted = gdf[col].apply(lambda v: _num(v))
        if converted.notna().sum() == gdf[col].notna().sum():
            gdf[col] = converted
    return {path.stem: gdf}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
