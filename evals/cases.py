"""Eval cases built from synthetic files whose truth is known.

Each `expect_terms` entry is a group of alternatives; the group passes if any alternative
appears in the model's answer (case-insensitive), so wording can vary.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from samples import make_samples  # noqa: E402


@dataclass
class Case:
    name: str
    path: Path
    why: str
    expect_facts: list[str] = field(default_factory=list)  # fact paths the answer should cite
    expect_terms: list[tuple[str, ...]] = field(default_factory=list)
    forbid_terms: list[str] = field(
        default_factory=list
    )  # wording that would be a wrong conclusion


def build_cases(d: Path) -> list[Case]:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    samples = make_samples(d / "samples")

    idx = pd.date_range("2001-01-01", periods=240, freq="MS")
    temp = (
        15
        + 8 * np.sin(2 * np.pi * (idx.month - 4) / 12)
        + 0.04 * np.arange(240) / 12
        + rng.normal(0, 0.4, 240)
    )
    pd.DataFrame({"date": idx, "T2M": temp}).to_csv(d / "seasonal_station.csv", index=False)

    days = pd.date_range("2015-01-01", periods=600, freq="D")
    ar = np.zeros(600)
    noise = rng.normal(0, 1, 600)
    for i in range(1, 600):
        ar[i] = 0.9 * ar[i - 1] + noise[i]
    pd.DataFrame({"date": days, "soil_moisture_anomaly": ar}).to_csv(
        d / "autocorrelated.csv", index=False
    )

    grid = xr.open_dataset(samples["netcdf4"][0]).load()
    grid["t2m"].values[:, :2, :2] = -9999.0  # fill value nobody declared
    grid.to_netcdf(d / "planted_fill.nc")

    return [
        Case(
            "warming_grid",
            samples["netcdf4"][0],
            "steady +0.6 K/year warming everywhere",
            ["trends.t2m.slope_per_year"],
            [("warm", "increas")],
        ),
        Case(
            "seasonal_station",
            d / "seasonal_station.csv",
            "strong annual cycle on top of +0.04/year",
            ["trends.T2M.slope_per_year"],
            [("season", "annual cycle")],
        ),
        Case(
            "autocorrelated_noise",
            d / "autocorrelated.csv",
            "red noise without a real trend",
            [],
            [("autocorrel", "persistence", "no significant", "not significant")],
            ["significant increase", "significantly increasing", "significant decrease"],
        ),
        Case(
            "planted_fill",
            d / "planted_fill.nc",
            "-9999 left in the data as a real value",
            [],
            [("fill", "-9999", "missing value")],
        ),
        Case(
            "paper",
            samples["pdf"][0],
            "short paper: 0.45 K/decade (p.1), MODIS and GPM IMERG (p.2)",
            [],
            [("0.45",), ("modis",), ("imerg", "gpm")],
        ),
    ]
