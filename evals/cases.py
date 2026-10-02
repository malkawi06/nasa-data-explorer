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

    fire_days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    w = np.asarray(1 + 5 * np.exp(-(((fire_days.dayofyear - 215) / 30) ** 2)))
    picked = rng.choice(np.asarray(fire_days), 4000, p=w / w.sum())
    centres = np.array([[33.0, 36.0], [30.6, 47.2], [35.0, 40.5]])[rng.integers(0, 3, 4000)]
    pd.DataFrame(
        {
            "acq_date": pd.to_datetime(picked).strftime("%Y-%m-%d"),
            "latitude": centres[:, 0] + rng.normal(0, 0.3, 4000),
            "longitude": centres[:, 1] + rng.normal(0, 0.3, 4000),
            "frp": rng.lognormal(2.3, 1.0, 4000).round(1),
            "daynight": rng.choice(["D", "N"], 4000, p=[0.7, 0.3]),
        }
    ).to_csv(d / "fires_events.csv", index=False)

    from PIL import Image

    y, x = np.mgrid[:600, :900]
    scene = np.empty((600, 900, 3))
    scene[:] = (20, 60, 120)  # ocean
    land = (x - 600) ** 2 / 250**2 + (y - 320) ** 2 / 200**2 < 1
    scene[land] = (70, 120, 50)  # vegetation
    scene[land & (x > 650)] = (150, 120, 80)  # bare, dry ground
    scene[
        ((x - 250) ** 2 + (y - 180) ** 2 < 110**2) | ((x - 330) ** 2 + (y - 230) ** 2 < 80**2)
    ] = 242
    scene += rng.normal(0, 6, scene.shape)
    Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8)).save(d / "cloudy_scene.jpg")

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
            "fire_events",
            d / "fires_events.csv",
            "fire detections peaking in July-August in 3 regions",
            [],
            [("august", "july", "summer"), ("hotspot", "cluster", "concentrat", "region")],
            ["significant increase", "significantly increasing"],
        ),
        Case(
            "cloudy_scene",
            d / "cloudy_scene.jpg",
            "clouds over ocean beside green and brown land (~9% cloud); needs a vision provider",
            ["image.white_low_saturation_pct"],
            [("cloud",), ("ocean", "water", "sea"), ("vegetat", "green")],
        ),
        Case(
            "paper",
            samples["pdf"][0],
            "short paper: 0.45 K/decade (p.1), MODIS and GPM IMERG (p.2)",
            [],
            [("0.45",), ("modis",), ("imerg", "gpm")],
        ),
    ]
