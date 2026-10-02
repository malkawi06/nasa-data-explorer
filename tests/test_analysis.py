"""Correctness of decoding, coverage, subsetting and trends."""

import numpy as np
import pandas as pd
import pytest

from nasa_explorer.analysis.stats import describe_trend, human_delta, trend
from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file


def test_scale_offset_fill_applied(samples, tmp_path):
    rep = process_file(samples["netcdf_packed"][0], ReadOptions(), tmp_path)
    st = rep.analysis["statistics"]["t2m"]
    assert st["min"] > 289 and st["max"] < 294  # decoded, not raw int16
    assert st["missing_pct"] > 0  # the _FillValue cell is NaN
    var = rep.analysis["summary"]["variables"][0]
    assert var["scale_factor"] == pytest.approx(0.01) and var["_FillValue"] == -32767


def test_hdf4_calibration_and_fill(samples, tmp_path):
    if "hdf4" not in samples:
        pytest.skip("pyhdf missing")
    st = process_file(samples["hdf4"][0], ReadOptions(), tmp_path).analysis["statistics"]["NDVI"]
    assert st["min"] >= 0 and st["max"] <= 1
    assert st["count"] == 20 * 30 - 5


def test_hdf5_tree_fill_and_scale(samples, tmp_path):
    a = process_file(samples["hdf5_tree"][0], ReadOptions(), tmp_path).analysis
    paths = [d["path"] for d in a["summary"]["datasets"]]
    assert "/science/LSAR/GCOV/grids/frequencyA/HHHH" in paths
    st = a["statistics"]["/science/LSAR/GCOV/grids/frequencyA/HHHH"]
    assert st["max"] <= 1.0 and st["count"] == 64 * 64 - 16


def test_grid_coverage_and_trend(samples, tmp_path):
    a = process_file(samples["netcdf4"][0], ReadOptions(), tmp_path).analysis
    assert a["coverage"]["time"]["resolution"] == "1 month"
    assert a["coverage"]["space"]["bbox"] == [30.0, 20.0, 50.0, 40.0]
    assert a["coverage"]["space"]["resolution_deg"] == [2.5, 2.5]
    t = next(t for t in a["trends"] if t["variable"] == "t2m")
    assert t["slope_per_year"] == pytest.approx(0.6, rel=0.1)
    assert t["text"].startswith("increasing, significant at 99%")


def test_grid_subsetting(samples, tmp_path):
    opts = ReadOptions(var="t2m", bbox=(35, 25, 45, 35), start="2002-01-01", end="2002-12-31")
    a = process_file(samples["netcdf4"][0], opts, tmp_path).analysis
    dims = a["summary"]["dimensions"]
    assert dims == {"time": 12, "lat": 5, "lon": 5}
    assert list(a["statistics"]) == ["t2m"]


def test_unknown_var_reported(samples, tmp_path):
    rep = process_file(samples["netcdf4"][0], ReadOptions(var="nope"), tmp_path)
    assert "not found" in rep.error and "t2m" in rep.error


def test_power_header_and_fill(samples, tmp_path):
    a = process_file(samples["power_header_csv"][0], ReadOptions(), tmp_path).analysis
    assert a["coverage"]["time"]["start"].startswith("2010-01-01")
    assert a["statistics"]["T2M"]["min"] > 0  # -999 removed
    assert any("-9999" in n for n in a["notes"])
    assert "NASA/POWER" in a["summary"]["header_text"]


def test_nested_json_dates(samples, tmp_path):
    a = process_file(samples["json_nested"][0], ReadOptions(), tmp_path).analysis
    assert a["coverage"]["time"]["n_steps"] == 30
    assert {"T2M", "RH2M"} <= set(a["statistics"])


def test_document_extraction(samples, tmp_path):
    s = process_file(samples["pdf"][0], ReadOptions(), tmp_path).analysis["summary"]
    assert s["title"].startswith("Warming Trends") and len(s["authors"]) == 3
    assert {"MODIS", "GPM", "GLDAS"} <= set(s["nasa_mentions"])
    assert s["nasa_mentions"]["GPM"] == [2]
    assert any(c["caption"].startswith("Figure 1") and c["page"] == 2 for c in s["figure_captions"])
    assert any("Results" in x for x in s["sections"])


def test_scanned_pdf_ocr(samples, tmp_path):
    import shutil

    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not installed")
    a = process_file(samples["pdf_scanned"][0], ReadOptions(), tmp_path).analysis
    assert a["summary"]["ocr"] is True
    assert "Landsat" in a["summary"]["nasa_mentions"]


def test_image_stats(samples, tmp_path):
    rep = process_file(samples["png"][0], ReadOptions(), tmp_path)
    assert set(rep.analysis["statistics"]) == {"R", "G", "B"}
    assert [t for t, _, _ in rep.plots] == ["Image", "Color histogram"]


def test_trend_wording():
    idx = pd.date_range("2000", periods=30, freq="YS")
    up = trend(
        pd.Series(np.arange(30.0) + np.random.default_rng(0).normal(0, 0.5, 30), idx), "x", "K"
    )
    assert (
        up["slope_per_year"] == pytest.approx(1, rel=0.05)
        and "increasing, significant at 99%" in up["text"]
    )
    flat = trend(pd.Series(np.random.default_rng(1).normal(0, 1, 30), idx), "y")
    assert flat["text"].startswith("no significant trend") or flat["mk_p"] < 0.05
    assert describe_trend({"slope_per_year": -1, "mk_p": 0.07, "units": ""}).startswith(
        "weakly decreasing"
    )
    assert trend(pd.Series([1.0, 2.0], idx[:2]), "short") is None


def test_mann_kendall_matches_pymannkendall_and_scales():
    mk = pytest.importorskip("pymannkendall")
    from nasa_explorer.analysis.stats import _mann_kendall, _sen_slope

    rng = np.random.default_rng(2)
    for x in (
        rng.normal(size=300),
        np.cumsum(rng.normal(size=500)) * 0.3 + np.arange(500) * 0.01,  # autocorrelated
        np.round(rng.gamma(0.5, 3, 400) * (rng.random(400) < 0.3), 1),  # dry days: ties
    ):
        for hr, test in ((False, mk.original_test), (True, mk.hamed_rao_modification_test)):
            ref = test(x)
            got = _mann_kendall(x, hr)
            assert got[0] == ref.trend and got[2] == ref.slope and abs(got[1] - ref.p) < 1e-9
    # long series: the bracketed Sen slope is the exact median of all pairwise slopes
    for x in (rng.normal(size=2200) + np.arange(2200) * 1e-3, rng.integers(0, 3, 2400) * 1.0):
        d = np.concatenate([(x[k:] - x[:-k]) / k for k in range(1, x.size)])
        assert _sen_slope(x) == np.median(d)


def test_human_delta():
    assert human_delta(86400) == "1 day"
    assert human_delta(3600 * 3) == "3 hours"
    assert human_delta(86400 * 31) == "1 month"
    assert human_delta(86400 * 365) == "1 year"
