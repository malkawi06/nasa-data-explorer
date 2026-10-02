"""Terrain of elevation models, Moon/Mars data and the analog finder (with real POWER data)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nasa_explorer import analog, bodies, power
from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import analog_report, iter_inputs, process, process_file

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

DATA = Path(__file__).parent / "data" / "power"
MOON = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
MARS = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=3396190 +units=m +no_defs"
RNG = np.random.default_rng(11)


def _clim(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def _tif(path, z, crs, px, origin=(0.0, 0.0), **kw):
    with rasterio.open(
        path,
        "w",
        driver=kw.pop("driver", "GTiff"),
        height=z.shape[0],
        width=z.shape[1],
        count=1,
        dtype=str(z.dtype),
        crs=crs,
        transform=from_origin(origin[0], origin[1], px, px),
        **kw,
    ) as d:
        d.write(z, 1)
    return path


def _bowl(n, centre, radius_px, depth):
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - centre[1], yy - centre[0]) / radius_px
    return np.where(r < 1, -depth * (1 - r**2), 0.0)


def test_body_detection():
    from rasterio.crs import CRS

    assert bodies.detect(CRS.from_proj4(MOON).to_wkt()) == "Moon"
    assert bodies.detect(CRS.from_proj4(MARS).to_wkt()) == "Mars"
    assert bodies.detect(MOON) == "Moon"  # PROJ string as rasterio's to_string() gives it
    assert bodies.detect(CRS.from_epsg(32636).to_wkt()) == "Earth"
    assert bodies.detect(None, "TARGET_NAME = MARS") == "Mars"
    assert bodies.detect(None, "{'planet': 'Mars'}") == "Mars"


def test_slope_crater_and_valley_on_known_terrain(tmp_path):
    n, px = 300, 30.0
    yy, xx = np.mgrid[:n, :n]
    plane = 1000 + xx * px * math.tan(math.radians(10))
    _tif(tmp_path / "plane_dem.tif", plane.astype("float32"), "EPSG:32636", px, (500000, 3300000))
    t = process_file(tmp_path / "plane_dem.tif", ReadOptions(plots=False), tmp_path / "r").analysis[
        "terrain"
    ]
    assert t["at_analysis_resolution"]["slope_median_deg"] == pytest.approx(10, abs=0.05)

    flat = 800 + _bowl(n, (150, 150), 10, 60) + RNG.normal(0, 0.3, (n, n))
    flat += np.where(np.abs(yy - 60) < 3, -25.0, 0.0)  # a valley that drains off the edges
    _tif(tmp_path / "crater_dem.tif", flat.astype("float32"), "EPSG:32636", px, (500000, 3300000))
    rep = process_file(tmp_path / "crater_dem.tif", ReadOptions(plots=True), tmp_path / "r")
    dep = rep.analysis["terrain"]["depressions"]
    assert dep["count"] == 1  # the crater, not the valley
    big = dep["largest"][0]
    assert big["diameter_m"] == pytest.approx(600, rel=0.15)
    assert big["depth_m"] == pytest.approx(60, abs=3)
    assert [p[0] for p in rep.plots][:3] == ["Shaded relief", "Slope map", "Slope distribution"]


def test_moon_polar_dem_with_scale_offset(tmp_path):
    n = 256
    z = _bowl(n, (128, 128), 30, 120) + RNG.normal(0, 1, (n, n))
    dn = np.round(z / 0.5).astype("int16")
    path = _tif(tmp_path / "LDEM_80S_20M.tif", dn, MOON, 20.0, (-2560, 2560))
    with rasterio.open(path, "r+") as d:
        d.scales, d.offsets = (0.5,), (1737400.0,)
    a = process_file(path, ReadOptions(plots=False), tmp_path / "r").analysis
    assert a["body"] == "Moon"
    assert abs(a["terrain"]["elevation_m"]["median"]) < 5  # radius turned into height
    assert a["coverage"]["space"]["bbox"][1] == pytest.approx(-90, abs=0.01)  # lunar latitudes
    assert "lola" in [p["key"] for p in a["products"]]
    assert "analog" not in a  # analog scores are for Earth sites


def test_isis3_pds4_pds3_and_hgt(tmp_path):
    z = (-2500 + 300 * np.sin(np.arange(128 * 128).reshape(128, 128) / 400)).astype("float32")
    for drv, name in (("ISIS3", "mola.cub"), ("PDS4", "mola.xml")):
        _tif(tmp_path / name, z, MARS, 463, (4_000_000, 1_000_000), driver=drv)
    # PDS3 with an attached label, like LOLA LDEM products
    n, rec = 64, 128
    label = (
        "PDS_VERSION_ID = PDS3\nRECORD_TYPE = FIXED_LENGTH\nRECORD_BYTES = 128\n"
        "FILE_RECORDS = 72\nLABEL_RECORDS = 8\n^IMAGE = 9\nTARGET_NAME = MOON\n"
        "OBJECT = IMAGE\n LINES = 64\n LINE_SAMPLES = 64\n SAMPLE_TYPE = LSB_INTEGER\n"
        " SAMPLE_BITS = 16\n SCALING_FACTOR = 0.5\n OFFSET = 1737400.\nEND_OBJECT = IMAGE\nEND\n"
    )
    raw = np.round(_bowl(n, (32, 32), 10, 40) / 0.5).astype("<i2").tobytes()
    (tmp_path / "LDEM_SMALL.IMG").write_bytes(label.encode().ljust(8 * rec, b" ") + raw)
    hgt = (500 + np.add.outer(np.arange(1201), np.arange(1201)) * 0.2).astype(">i2")
    hgt.tofile(tmp_path / "N31E035.hgt")

    names = {p.name for p in iter_inputs(tmp_path)}
    assert "mola.img" not in names  # PDS4 data file: its label is processed instead
    reps = {r.file: r for r in process(tmp_path, ReadOptions(plots=False), tmp_path / "out")}
    for name in ("mola.cub", "mola.xml"):
        assert reps[name].reader == "planetary-raster" and reps[name].analysis["body"] == "Mars"
        assert reps[name].analysis["terrain"]["at_analysis_resolution"]["pixel_m"] == 463
    pds3 = reps["LDEM_SMALL.IMG"]
    assert pds3.reader == "planetary-raster" and pds3.analysis["body"] == "Moon"
    assert pds3.analysis["summary"]["attributes"] or pds3.analysis.get("terrain") is not None
    assert reps["N31E035.hgt"].analysis["terrain"]["at_analysis_resolution"][
        "pixel_m"
    ] == pytest.approx(92.7, abs=0.5)
    # the Earth tile is compared with the Moon and Mars DEMs processed with it
    rows = reps["N31E035.hgt"].analysis["terrain_analogs"]
    assert {r["body"] for r in rows} >= {"Mars"}


def test_analog_scores_rank_expert_analogs_first():
    feats = {
        k: analog.climate_features(_clim(k))
        for k in ("wadi_rum", "atacama", "dry_valleys", "haughton", "irbid")
    }
    assert feats["atacama"]["precip_mm_yr"] == pytest.approx(8.5, abs=0.3)
    assert feats["wadi_rum"]["elevation_m"] == pytest.approx(1076.35)

    def order(target):
        return [
            k
            for _, k in sorted(
                ((analog.score(f, analog.TARGETS[target])["score"], k) for k, f in feats.items()),
                reverse=True,
            )
        ]

    assert order("moon_south_pole")[:2] == ["dry_valleys", "haughton"]
    assert order("mars_low_latitude")[:2] == ["atacama", "wadi_rum"]
    assert order("mars_polar")[0] == "dry_valleys"
    assert order("mars_low_latitude")[-1] in ("haughton", "irbid")
    sc = analog.score(feats["wadi_rum"], analog.TARGETS["mars_low_latitude"])
    assert sc["coverage_pct"] < 100 and "Vegetation (NDVI)" in sc["missing"]  # no NDVI, no DEM


def test_analog_report_and_cli(tmp_path, monkeypatch, capsys):
    names = ["wadi_rum", "atacama", "dry_valleys", "haughton", "irbid"]
    sites = [{"name": n, "lat": 0.0, "lon": 0.0, "analog_for": ""} for n in names]
    sites[1]["analog_for"] = "Mars"
    feats = [analog.climate_features(_clim(n)) for n in names] + [None]
    rep = analog_report(
        "mars_low_latitude",
        sites + [{"name": "offline", "lat": 1, "lon": 1, "analog_for": ""}],
        feats,
        tmp_path,
    )
    ranking = rep.analysis["ranking"]
    assert ranking["ranking"][0]["name"] == "atacama"
    assert ranking["ranking"][-1]["score"] is None
    assert ranking["validation"]["in_top_third"] == 1
    html = rep.html_path.read_text(encoding="utf-8")
    assert "Site ranking" in html and "atacama" in html and "<img" in html

    def fake(lat, lon, cache=None):  # every built-in site gets Wadi Rum's climate
        return _clim("wadi_rum")

    monkeypatch.setattr(power, "fetch_climatology", fake)
    from nasa_explorer.cli import main

    assert main(["--analog", "moon_south_pole", "--out", str(tmp_path / "cli")]) == 0
    out = capsys.readouterr().out
    assert "1." in out and "report:" in out


def test_power_urls_and_table_features(tmp_path):
    url = power.daily_url(29.57, 395.42 - 360, "2015-01-01", "2024-12-31")
    assert "start=20150101" in url and "community=RE" in url and "T2M_RANGE" in url
    assert "longitude=35.42" in power.climatology_url(29.57, 35.42)
    with pytest.raises(ValueError):
        power.daily_url(95, 0, "2020-01-01", "2020-12-31")
    days = np.arange(730)
    header = "-BEGIN HEADER-\nLocation: Latitude  29.57   Longitude 35.42\nElevation from MERRA-2: Average for 0.5 x 0.625 degree lat/lon region = 1076.35 meters\n-END HEADER-\n"
    rows = "\n".join(
        f"{2020 + d // 365},{1 + (d % 365) // 31 % 12},{1 + d % 28},{18 + 8 * math.sin(d / 58):.2f},{26 + 8 * math.sin(d / 58):.2f},{10 + 8 * math.sin(d / 58):.2f},0.10,40.0,25.0"
        for d in days
    )
    (tmp_path / "POWER_point.csv").write_text(
        header + "YEAR,MO,DY,T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,CLOUD_AMT\n" + rows,
        encoding="utf-8",
    )
    a = process_file(
        tmp_path / "POWER_point.csv", ReadOptions(plots=False), tmp_path / "r"
    ).analysis
    f = a["analog"]["features"]
    assert f["precip_mm_yr"] == pytest.approx(36.5, abs=0.5) and f["abs_lat"] == 29.57
    assert f["t_range_c"] == pytest.approx(16, abs=0.5) and f["elevation_m"] == 1076.35
    assert (
        a["analog"]["scores"]["mars_low_latitude"]["score"]
        > a["analog"]["scores"]["mars_polar"]["score"]
    )


def test_terrain_comparison_is_symmetric_and_bounded():
    prof = {
        "by_baseline_m": {
            "100": {"slope_hist": [0.5, 0.3, 0.2] + [0] * 27, "roughness_tri_m": 2.0}
        },
        "depressions": {"per_1000_km2": 10},
    }
    other = {
        "by_baseline_m": {
            "100": {"slope_hist": [0.0, 0.0, 0.2] + [0.8] + [0] * 26, "roughness_tri_m": 8.0}
        },
        "depressions": {"per_1000_km2": 0.0},
    }
    same = analog.compare_terrain(prof, prof)
    assert same["similarity"] == 100
    diff = analog.compare_terrain(prof, other)
    assert 0 <= diff["similarity"] < 60
    assert analog.compare_terrain(prof, {"by_baseline_m": {"300": {}}}) is None


def test_bridge_analog_run_uses_local_dem(tmp_path, monkeypatch):
    import sys

    web = Path(__file__).resolve().parents[1] / "web"
    monkeypatch.syspath_prepend(str(web))
    sys.modules.pop("bridge", None)
    import bridge

    monkeypatch.setattr(bridge, "WORK", tmp_path / "work")
    src = Path(bridge.new_run("run1"))
    flat = 1076 + _bowl(200, (100, 100), 12, 40) + RNG.normal(0, 0.2, (200, 200))
    _tif(src / "wadi_rum_dem.tif", flat.astype("float32"), "EPSG:4326", 1 / 3600, (35.4, 29.6))
    json.loads(bridge.analyse_dir(str(src), json.dumps({"lang": "en"})))
    urls = json.loads(bridge.power_urls(29.57, 35.42))
    assert urls["daily"].startswith("https://power.larc.nasa.gov/api/temporal/daily/point")
    sites = [
        {"name": "Wadi Rum DEM site", "lat": 29.59, "lon": 35.42, "analog_for": ""},
        {"name": "offline", "lat": 0, "lon": 0, "analog_for": ""},
    ]
    out = json.loads(
        bridge.analog_run(
            "run2", "moon_south_pole", json.dumps(sites), json.dumps([_clim("wadi_rum"), None])
        )
    )
    payload = json.loads(out[0]["json"])
    top = payload["analysis"]["ranking"]["ranking"][0]
    assert top["name"] == "Wadi Rum DEM site"
    assert top["features"]["depressions_per_1000_km2"] is not None  # terrain from the analysed DEM
    assert payload["analysis"]["ranking"]["ranking"][1]["score"] is None
    assert json.loads(bridge.analog_sites())["sites"][0]["name"].startswith("Atacama")


def test_renamed_hgt_tile_and_binary_is_not_text(tmp_path):
    from nasa_explorer.registry import looks_like_text

    z = (800 + _bowl(1201, (600, 600), 9, 120)).astype(">i2")
    z.tofile(tmp_path / "03_SRTM_N29E035 (1).hgt")  # renamed download: GDAL would refuse it
    rep = process_file(
        tmp_path / "03_SRTM_N29E035 (1).hgt", ReadOptions(plots=False), tmp_path / "r"
    )
    assert rep.reader == "srtm-hgt"
    dep = rep.analysis["terrain"]["depressions"]["largest"][0]
    assert dep["lat"] == pytest.approx(29.5, abs=0.01) and dep["lon"] == pytest.approx(
        35.5, abs=0.01
    )
    assert not looks_like_text(z.tobytes()[:4096])
    assert looks_like_text("Temperature, ° and Arabic درجة\tok\n".encode())


def test_ndvi_raster_gets_vegetation_shares(tmp_path):
    ndvi = np.clip(RNG.normal(0.05, 0.03, (200, 200)), -0.2, 1) * 10000  # MODIS stores NDVI x 10000
    _tif(tmp_path / "MOD13Q1_NDVI.tif", ndvi.astype("int16"), "EPSG:4326", 0.0025, (35.0, 30.0))
    a = process_file(
        tmp_path / "MOD13Q1_NDVI.tif", ReadOptions(plots=False), tmp_path / "r"
    ).analysis
    veg = a["vegetation"]
    assert veg["mean"] == pytest.approx(0.05, abs=0.01)
    assert veg["share_pct"]["bare_0_0.1"] > 80
    assert (
        a["analog"]["partial"] == "vegetation"
        and a["analog"]["scores"]["mars_low_latitude"]["score"] == 100
    )


def test_paper_facts():
    from nasa_explorer.readers import _paper

    pages = [
        "A Study of Analogs\nA. Author, B. Author\ndoi:10.1029/2026JE000123\nAbstract\n"
        "Mean annual precipitation is 39 mm and humidity is 42% over 2001-2020, which makes the site "
        "a strong analog.\n1. Introduction\nText.",
        "2. Data\nThe site at 29.57 N, 35.42 E was studied. Trends were not significant (p = 0.34). "
        "Data are available at https://power.larc.nasa.gov.\nReferences\n[1] McKay (2003). doi:10.1016/x\n[2] Lee (2007).",
    ]
    f = _paper.paper_facts(pages)
    assert f["doi"] == "10.1029/2026JE000123"
    assert f["abstract"].startswith("Mean annual precipitation is 39 mm")
    assert {"39 mm", "42%"} <= set(f["key_numbers"][0]["values"])
    assert any("p = 0.34" in n["values"] for n in f["key_numbers"])
    assert f["coordinates"][0]["lat"] == 29.57 and f["coordinates"][0]["lon"] == 35.42
    assert f["study_period"] == {"start": 2001, "end": 2020, "mentions": 1}
    assert f["references"]["count"] == 2 and f["data_availability"][0]["page"] == 2


def test_planetary_radius_stats_and_grouped_quality(tmp_path):
    n = 128
    dn = np.round((_bowl(n, (64, 64), 20, 80) + RNG.normal(0, 0.5, (n, n))) / 0.5).astype("int16")
    path = _tif(tmp_path / "LDEM_test.tif", dn, MOON, 20.0, (-1280, 1280))
    with rasterio.open(path, "r+") as d:
        d.scales, d.offsets = (0.5,), (1737400.0,)
    a = process_file(path, ReadOptions(plots=False), tmp_path / "r").analysis
    st = a["statistics"][a["terrain"]["variable"]]
    assert abs(st["p50"]) < 5 and st["min"] < -60  # heights, not radii
    assert "outliers" not in {q["code"] for q in a["quality"]}  # craters are not outliers
