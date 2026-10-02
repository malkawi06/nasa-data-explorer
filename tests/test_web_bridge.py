"""The browser bridge (web/bridge.py) runs the same code in CPython as in Pyodide."""

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

import bridge  # noqa: E402
import build  # noqa: E402


def test_analyse_dir_with_shapefile_parts_and_options(samples, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "WORK", tmp_path)
    src = Path(bridge.new_run("run 1"))
    shp = samples["shapefile"][0]
    for part in shp.parent.glob("points.*"):
        shutil.copy(part, src)
    shutil.copy(samples["netcdf4"][0], src)
    opts = json.dumps(
        {"var": "t2m", "bbox": [35, 25, 45, 35], "start": "", "end": "", "lang": "ar"}
    )
    results = json.loads(bridge.analyse_dir(str(src), opts))
    by_file = {r["file"]: r for r in results}
    assert set(by_file) == {"grid.nc", "points.shp"}  # sidecars are not separate reports
    grid = by_file["grid.nc"]
    assert grid["reader"] == "netcdf" and grid["error"] is None
    assert 'dir="rtl"' in grid["html"]
    assert json.loads(grid["json"])["analysis"]["summary"]["dimensions"]["lat"] == 5


def test_formats_and_missing_modules():
    rows = json.loads(bridge.formats())
    assert {"netcdf", "vector", "pdf-lite"} <= {r["name"] for r in rows}
    msg = ["vector reader disabled: missing geopandas. pip install 'nasa-data-explorer[geo]'"]
    assert bridge.missing_modules(msg) == ["geopandas"]


def test_build_bundles_package(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "DIST", tmp_path / "dist")
    bundle = build.build()
    names = zipfile.ZipFile(bundle).namelist()
    assert "bridge.py" in names and "nasa_explorer/readers/netcdf.py" in names
    assert {"index.html", "app.js", "ai.js", "style.css"} <= {
        p.name for p in (tmp_path / "dist").iterdir()
    }


def _run(tmp_path, monkeypatch, files):
    monkeypatch.setattr(bridge, "WORK", tmp_path)
    src = Path(bridge.new_run("ai"))
    for f in files:
        shutil.copy(f, src)
    return {
        r["file"]: r for r in json.loads(bridge.analyse_dir(str(src), json.dumps({"lang": "en"})))
    }


def test_ai_job_round_trip_updates_report(samples, tmp_path, monkeypatch):
    res = _run(tmp_path, monkeypatch, [samples["netcdf4"][0]])["grid.nc"]
    first = json.loads(bridge.ai_start(res["key"], "en", "gemini", "gemini-2.5-flash"))
    step = first["step"]
    assert step["json"] and step["tokens"] > 100 and "statistics.t2m.mean" in step["prompt"]
    reply = {
        "overview": "grid",
        "findings": [{"text": "warming", "value": 999, "fact": "trends.t2m.slope_per_year"}],
    }
    second = json.loads(bridge.ai_next(first["job"], json.dumps(reply)))
    assert second["step"]["label"] == "review"  # the wrong number triggers one review round
    reply["findings"] = []
    done = json.loads(bridge.ai_next(second["job"], json.dumps(reply)))
    assert done["done"] and "ai-panel" in done["panel"] and "gemini" in done["html"]
    assert json.loads(done["json"])["ai"]["rounds"] == 2


def test_chat_on_document_uses_relevant_pages(samples, tmp_path, monkeypatch):
    res = _run(tmp_path, monkeypatch, [samples["pdf"][0]])["paper.pdf"]
    first = json.loads(
        bridge.chat_start(res["key"], "Which precipitation data (IMERG) was used?", "ar")
    )
    assert "[page 2]" in first["step"]["prompt"] and "Arabic" in first["step"]["prompt"]
    done = json.loads(bridge.ai_next(first["job"], "GPM IMERG [p. 2]"))
    assert done == {"job": first["job"], "done": True, "answer": "GPM IMERG [p. 2]"}


def test_pure_python_kml_and_gpkg_match_gdal(samples):
    """The browser-only readers must agree with the GDAL-backed ones."""
    import geopandas as gpd

    from nasa_explorer.readers._purevector import read_gpkg, read_kml

    for key, reader_fn in (("gpkg", read_gpkg), ("kml", read_kml), ("kmz", read_kml)):
        if key not in samples:
            continue
        path = samples[key][0]
        ours = next(iter(reader_fn(path).values()))
        ref = gpd.read_file(f"/vsizip/{path}/doc.kml") if key == "kmz" else gpd.read_file(path)
        assert len(ours) == len(ref), key
        ours_xy = [(round(g.x, 6), round(g.y, 6)) for g in ours.geometry]
        ref_xy = [(round(g.x, 6), round(g.y, 6)) for g in ref.geometry]
        assert ours_xy == ref_xy, key
        assert ours.crs.to_epsg() == 4326
    gp = next(iter(read_gpkg(samples["gpkg"][0]).values()))
    assert {"name", "value"} <= set(gp.columns)


def test_vector_reader_uses_pure_path_in_browser(samples, monkeypatch):
    from nasa_explorer.core import ReadOptions
    from nasa_explorer.readers import vector

    monkeypatch.setattr(vector, "IN_BROWSER", True)
    for key in ("gpkg", "kml", "kmz"):
        res = vector.read_vector(samples[key][0], ReadOptions())
        assert res.kind == "table" and len(res.data) == 10 and res.metadata["crs"] == "EPSG:4326"
