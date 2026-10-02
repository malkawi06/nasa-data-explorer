"""NDVI rasters: how much of the ground is bare, sparse or vegetated.

MODIS (MOD13) and other products often store NDVI × 10000 as integers; values far outside
-1..1 are rescaled here, and the report says so.
"""

from __future__ import annotations

import re

import numpy as np
import xarray as xr

NDVI = re.compile(r"(?<![a-z])ndvi(?![a-z])", re.I)
CLASSES = (  # (label, lower bound, upper bound)
    ("water_ice_cloud_lt0", -1.0, 0.0),
    ("bare_0_0.1", 0.0, 0.1),
    ("sparse_0.1_0.3", 0.1, 0.3),
    ("moderate_0.3_0.6", 0.3, 0.6),
    ("dense_ge0.6", 0.6, 1.0001),
)


def is_ndvi(label: str, name: str, da: xr.DataArray) -> bool:
    text = " ".join(
        [label, name, str(da.attrs.get("long_name", "")), str(da.attrs.get("standard_name", ""))]
    )
    return bool(NDVI.search(text))


def analyze(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    out: dict = {}
    if np.percentile(np.abs(v), 99) > 1.5:  # stored as NDVI × 10000
        v = v / 10000.0
        out["note"] = "values look like NDVI × 10000 (MODIS convention); rescaled to -1..1"
    v = v[(v >= -1.0) & (v <= 1.0)]
    if v.size == 0:
        return out
    out["mean"] = round(float(v.mean()), 3)
    out["median"] = round(float(np.median(v)), 3)
    out["share_pct"] = {
        lab: round(100 * float(((v >= lo) & (v < hi)).mean()), 1) for lab, lo, hi in CLASSES
    }
    return out
