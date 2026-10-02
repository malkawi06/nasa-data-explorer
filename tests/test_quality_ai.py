"""Trend methods, quality checks, product cards and the verified AI workflows."""

import json

import numpy as np
import pandas as pd
import pytest
from samples import PAPER
from test_interfaces import FakeProvider

from nasa_explorer import ai, verify
from nasa_explorer.analysis.quality import check
from nasa_explorer.analysis.stats import trend
from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file
from nasa_explorer.products import identify

RNG = np.random.default_rng(3)


def test_seasonal_series_uses_seasonal_mk_and_recovers_slope():
    idx = pd.date_range("2000-01-01", periods=240, freq="MS")
    y = (
        10 * np.sin(2 * np.pi * (idx.month - 1) / 12)
        + 0.05 * np.arange(240) / 12
        + RNG.normal(0, 0.3, 240)
    )
    t = trend(pd.Series(y, idx), "x", "K")
    assert t["seasonal"] and t["method"] == "Seasonal Mann-Kendall"
    assert t["slope_per_year"] == pytest.approx(0.05, abs=0.01)


def test_autocorrelated_noise_is_not_called_significant():
    e = RNG.normal(0, 1, 400)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.85 * ar[i - 1] + e[i]
    t = trend(pd.Series(ar, pd.date_range("2000-01-01", periods=400, freq="D")), "ar1")
    assert "Hamed-Rao" in t["method"] and t["lag1_autocorr"] > 0.5


def _stats(**kw):
    base = {
        "count": 100,
        "missing_pct": 0.0,
        "min": 1.0,
        "max": 2.0,
        "mean": 1.5,
        "std": 0.2,
        "p5": 1.1,
        "p25": 1.3,
        "p50": 1.5,
        "p75": 1.7,
        "p95": 1.9,
    }
    return {**base, **kw}


def test_quality_rules():
    a = {
        "summary": {"variables": [{"name": "lst", "units": "K"}, {"name": "ndvi", "units": ""}]},
        "statistics": {
            "lst": _stats(min=-9999.0, max=320.0),
            "ndvi": _stats(min=-2000.0, max=9000.0, p25=1000, p75=6000),
            "precip": _stats(min=-3.0),
            "flat": _stats(std=0.0),
            "gappy": _stats(missing_pct=75.0),
        },
        "coverage": {
            "time": {
                "duplicates": 4,
                "gaps": 2,
                "largest_gap": "3 months",
                "resolution": "1 month",
            },
            "space": {"bbox": [0, -10, 350, 10]},
        },
    }
    codes = {(q["code"], q["variable"]) for q in check("grid", a)}
    assert ("undeclared_fill", "lst") in codes
    assert ("unscaled", "ndvi") in codes
    assert ("negative_values", "precip") in codes
    assert ("constant", "flat") in codes and ("mostly_missing", "gappy") in codes
    assert (
        ("duplicate_times", None) in codes
        and ("time_gaps", None) in codes
        and ("lon_0_360", None) in codes
    )
    assert check("grid", a)[0]["level"] == "error"  # sorted by severity


def test_point_data_duplicates_are_normal():
    a = {
        "statistics": {},
        "coverage": {"time": {"duplicates": 50}, "space": {"bbox": [30, 30, 31, 31]}},
    }
    assert not [q for q in check("table", a) if q["code"] == "duplicate_times"]


def test_products():
    assert identify("MOD11A2.A2020001.h21v05.061.hdf", {})[0]["key"] == "mod11"
    firms = {"summary": {"columns": [{"name": "latitude"}, {"name": "longitude"}, {"name": "frp"}]}}
    assert identify("anything.csv", firms)[0]["key"] == "firms"
    assert identify("random.nc", {}) == []


def test_report_has_quality_trend_map_and_ai_placeholder(samples, tmp_path):
    rep = process_file(samples["netcdf4"][0], ReadOptions(), tmp_path)
    a = rep.analysis
    assert a["trend_map"]["pct_significant_increase"] > 90
    titles = [t for t, _, _ in rep.plots]
    assert "Trend map: t2m" in titles and "Seasonal cycle: t2m" in titles
    html = rep.html_path.read_text(encoding="utf-8")
    assert "ai-panel" in html and "Not run yet" in html and "Quality checks" in html


def test_verify_data_statuses(samples, tmp_path):
    a = process_file(samples["netcdf4"][0], ReadOptions(), tmp_path).analysis
    f = verify.facts(a)
    mean = f["statistics.t2m.mean"]
    r = verify.verify_data(
        {
            "findings": [
                {
                    "text": f"mean is {mean:.1f} K",
                    "value": round(mean, 1),
                    "fact": "statistics.t2m.mean",
                },
                {"text": "mean is 300 K", "value": 300, "fact": "statistics.t2m.mean"},
                {"text": "x", "value": 1, "fact": "statistics.nope.mean"},
                {"text": "t2m peaks at 999 K"},
                {"text": "significant warming (p<0.001)"},
            ]
        },
        f,
    )
    assert [x["status"] for x in r["findings"]] == [
        "verified",
        "mismatch",
        "unsupported",
        "unsupported",
        "qualitative",
    ]


def test_data_workflow_reviews_and_fixes(samples, tmp_path, monkeypatch):
    a = process_file(samples["netcdf4"][0], ReadOptions(), tmp_path).analysis
    mean = verify.facts(a)["statistics.t2m.mean"]
    bad = {
        "overview": "o",
        "findings": [{"text": "mean is 300 K", "value": 300, "fact": "statistics.t2m.mean"}],
    }
    good = {
        "overview": "o",
        "findings": [
            {
                "text": f"mean is {mean:.2f} K",
                "value": round(mean, 2),
                "fact": "statistics.t2m.mean",
            }
        ],
    }
    fake = FakeProvider(replies=["Sure! ```json\n" + json.dumps(bad) + "\n```", json.dumps(good)])
    result = ai.run_workflow(
        ai.data_workflow(a, "grid.nc", "netcdf", "en"), ai.AIClient(fake, verbose=False)
    )
    assert len(fake.calls) == 2 and "statistics.t2m.mean is" in fake.calls[1]
    assert result["rounds"] == 2 and result["fixed_in_review"] == 1
    assert result["findings"][0]["status"] == "verified"


def test_paper_workflow_verifies_pages():
    reply = {
        "findings": [
            {
                "text": "0.45 K per decade warming",
                "page": 1,
                "quote": "warming of 0.45 K per decade",
            },
            {"text": "1.2 K per decade", "page": 2},
        ],
        "nasa_datasets": [{"name": "GPM", "how_used": "rain", "page": 2}],
    }
    fixed = {"findings": [reply["findings"][0]], "nasa_datasets": reply["nasa_datasets"]}
    fake = FakeProvider(replies=[json.dumps(reply), json.dumps(fixed)])
    result = ai.run_workflow(ai.paper_workflow(PAPER, "T", "en"), ai.AIClient(fake, verbose=False))
    assert result["rounds"] == 2 and result["verification"]["verified"] == 2
    assert "1.2" in fake.calls[1]  # the review prompt names the unsupported claim


def test_ai_panel_in_report(samples, tmp_path, monkeypatch):
    reply = {
        "overview": "Monthly grid",
        "findings": [
            {"text": "warming 0.6 K/year", "value": 0.6, "fact": "trends.t2m.slope_per_year"}
        ],
        "next_analyses": ["compare with ERA5"],
    }
    monkeypatch.setattr(
        ai, "get_provider", lambda name=None: FakeProvider(replies=[json.dumps(reply)])
    )
    rep = process_file(samples["netcdf4"][0], ReadOptions(lang="ar"), tmp_path, ai=True)
    html = rep.html_path.read_text(encoding="utf-8")
    assert "b-verified" in html and "مؤكَّد" in html and "compare with ERA5" in html


def test_parse_json_tolerates_noise():
    assert verify.parse_json('Here you go:\n```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert verify.parse_json("no json here") is None


def test_heatwave_is_reported_as_extreme_and_rain_is_not_an_outlier():
    idx = pd.date_range("2000-01-01", periods=240, freq="MS")
    y = np.asarray(
        10 * np.sin(2 * np.pi * (idx.month - 1) / 12)
        + 0.02 * np.arange(240) / 12
        + RNG.normal(0, 0.2, 240)
    )
    hot = np.asarray((idx.year == 2010) & idx.month.isin([6, 7, 8]))
    y[hot] += 3
    t = trend(pd.Series(y, idx), "t2m", "K")
    assert sorted(e["date"] for e in t["extremes"]) == ["2010-06", "2010-07", "2010-08"]
    issues = check("grid", {"statistics": {}, "coverage": {}, "trends": [t]})
    assert any(q["code"] == "extremes" and "2010-07" in q["message"] for q in issues)

    rain = _stats(min=0.0, max=40.0, p5=0.0, p25=4e-14, p50=0.001, p75=0.3, p95=3.0)
    assert not [
        q
        for q in check("grid", {"statistics": {"precip": rain}, "coverage": {}})
        if q["code"] == "outliers"
    ]


def test_trend_claims_must_match_significance():
    f = {
        "trends.frp.slope_per_year": 0.69,
        "trends.frp.mk_p": 0.068,
        "trends.t2m.slope_per_year": 0.043,
        "trends.t2m.mk_p": 0.0001,
    }
    r = verify.verify_data(
        {
            "findings": [
                {
                    "text": "FRP increases by 0.69 per year",
                    "value": 0.69,
                    "fact": "trends.frp.slope_per_year",
                },
                {
                    "text": "FRP rises 0.69/yr but this is not statistically significant",
                    "value": 0.69,
                    "fact": "trends.frp.slope_per_year",
                },
                {
                    "text": "t2m warms 0.043 K/yr, significant",
                    "value": 0.043,
                    "fact": "trends.t2m.slope_per_year",
                },
                {
                    "text": "t2m shows no significant trend (0.043 K/yr)",
                    "value": 0.043,
                    "fact": "trends.t2m.slope_per_year",
                },
                {
                    "text": "اتجاه FRP غير دال إحصائياً",
                    "value": 0.69,
                    "fact": "trends.frp.slope_per_year",
                },
            ]
        },
        f,
    )
    assert [x["status"] for x in r["findings"]] == [
        "mismatch",
        "verified",
        "verified",
        "mismatch",
        "verified",
    ]
    assert "not statistically significant" in r["findings"][0]["note"]


def test_event_table_reports_counts_peak_month_and_hotspots(tmp_path):
    rng = np.random.default_rng(1)
    days = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    weights = np.asarray(1 + 6 * (days.month.isin([7, 8])), dtype=float)
    picked = rng.choice(np.asarray(days), 3000, p=weights / weights.sum())
    centres = np.array([[32.0, 36.0], [30.0, 47.0]])[rng.integers(0, 2, 3000)]
    df = pd.DataFrame(
        {
            "acq_date": pd.to_datetime(picked).strftime("%Y-%m-%d"),
            "latitude": centres[:, 0] + rng.normal(0, 0.2, 3000),
            "longitude": centres[:, 1] + rng.normal(0, 0.2, 3000),
            "frp": rng.gamma(2, 5, 3000),
            "daynight": rng.choice(["D", "N"], 3000),
        }
    )
    path = tmp_path / "events.csv"
    df.to_csv(path, index=False)
    rep = process_file(path, ReadOptions(), tmp_path / "out")
    e = rep.analysis["events"]
    assert e["total"] == 3000 and e["peak_month"] in ("July", "August")
    assert len(e["hotspots"]) == 2 and abs(e["hotspots"]["1"]["share_pct"] - 50) < 5
    assert rep.analysis["trends"][0]["variable"] == "events per day"
    assert set(rep.analysis["categories"]["daynight"]) == {"D", "N"}
    titles = [t for t, _, _ in rep.plots]
    assert "Seasonal cycle of events" in titles and "Time series (mean per date)" not in titles
    assert "events.peak_month_share_pct" in verify.facts(rep.analysis)
