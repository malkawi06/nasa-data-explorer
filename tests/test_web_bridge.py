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
    assert {"index.html", "app.js", "style.css"} <= {p.name for p in (tmp_path / "dist").iterdir()}
