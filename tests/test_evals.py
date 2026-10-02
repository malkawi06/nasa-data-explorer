"""The eval harness itself: cases build, scoring rewards right answers and catches wrong ones."""

import json

from test_interfaces import FakeProvider

from evals.cases import build_cases
from evals.score import score
from nasa_explorer import ai
from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file
from nasa_explorer.registry import read_file


def _run(case, tmp_path, replies):
    rep = process_file(case.path, ReadOptions(plots=False), tmp_path / "r")
    res, _, _ = read_file(case.path, ReadOptions())
    fake = FakeProvider(replies=replies)
    fake.model = (
        f"fake-{len(replies)}-{hash(replies[0])}"  # separate on-disk cache entry per scripted run
    )
    client = ai.AIClient(fake, verbose=False)
    return rep, ai.run_workflow(ai.workflow_for(rep, res, "en"), client)


def test_cases_build_and_quality_checks_see_the_planted_problems(tmp_path):
    cases = {c.name: c for c in build_cases(tmp_path)}
    assert set(cases) == {
        "warming_grid",
        "seasonal_station",
        "autocorrelated_noise",
        "planted_fill",
        "fire_events",
        "cloudy_scene",
        "paper",
    }
    rep = process_file(cases["planted_fill"].path, ReadOptions(plots=False), tmp_path / "r")
    assert any(q["code"] == "undeclared_fill" for q in rep.analysis["quality"])
    rep = process_file(cases["fire_events"].path, ReadOptions(plots=False), tmp_path / "r")
    assert rep.analysis["events"]["peak_month"] in ("July", "August")
    rep = process_file(cases["cloudy_scene"].path, ReadOptions(plots=False), tmp_path / "r")
    assert 7 < rep.analysis["image"]["white_low_saturation_pct"] < 11
    rep = process_file(cases["seasonal_station"].path, ReadOptions(plots=False), tmp_path / "r")
    assert rep.analysis["trends"][0]["seasonal"]
    rep = process_file(cases["autocorrelated_noise"].path, ReadOptions(plots=False), tmp_path / "r")
    assert "Hamed-Rao" in rep.analysis["trends"][0]["method"]


def test_score_good_and_bad_answers(tmp_path):
    case = next(c for c in build_cases(tmp_path) if c.name == "warming_grid")
    rep = process_file(case.path, ReadOptions(plots=False), tmp_path / "r")
    slope = rep.analysis["trends"][0]["slope_per_year"]
    good = {
        "overview": "Steady warming",
        "findings": [
            {
                "text": f"t2m increases {slope:.2f} K/year",
                "value": round(slope, 2),
                "fact": "trends.t2m.slope_per_year",
            }
        ],
    }
    _, result = _run(case, tmp_path, [json.dumps(good)])
    assert score(result, case)["passed"]

    bad = {
        "overview": "Cooling",
        "findings": [
            {"text": "t2m falls 3 K/year", "value": -3, "fact": "trends.t2m.slope_per_year"}
        ],
    }
    _, result = _run(case, tmp_path, [json.dumps(bad), json.dumps(bad)])
    s = score(result, case)
    assert not s["passed"] and s["mismatch"] == 1 and s["rounds"] == 2
