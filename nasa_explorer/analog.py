"""How much an Earth location resembles a Moon or Mars base / landing environment.

A target is a list of factors that can be measured on Earth: climate from NASA POWER, terrain
from a DEM, vegetation from NDVI. Each factor maps its value linearly (or log-linearly) from
a "not analog" level (score 0) to a "fully analog" level (score 1). The site score is the
weighted mean over the factors that have data, reported with its coverage, so a missing DEM
or NDVI is visible instead of silently changing the answer.

The levels are transparent, documented assumptions, not physical constants: change them here.
Known analog sites (Atacama, Haughton, Dry Valleys, ...) are scored alongside candidates, so
the ranking can be checked against expert choices.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    unit: str
    zero: float  # value that scores 0 (not analog)
    one: float  # value that scores 1 (fully analog)
    weight: float
    why: str
    log: bool = False


@dataclass(frozen=True)
class Target:
    key: str
    name: str
    body: str
    description: str
    factors: tuple[Factor, ...]
    known_tag: str = ""  # SITES whose analog_for lists this tag are expert-chosen analogs


NDVI_WHY = "The surface there has no vegetation; bare ground is needed for regolith, dust and mobility tests."
TARGETS: dict[str, Target] = {
    t.key: t
    for t in (
        Target(
            "moon_south_pole",
            "Moon - permanent south-polar base",
            "Moon",
            "Cold, dry, airless, cratered ground with the Sun near the horizon (Artemis region). "
            "The Moon has no weather: terrain carries most of the weight, and climate counts only "
            "for cold operations and unweathered ground. Without a DEM the score is a proxy.",
            (
                Factor(
                    "depressions_per_1000_km2",
                    "Crater-like depressions",
                    "per 1000 km²",
                    0.5,
                    20,
                    3,
                    "The lunar highlands are densely cratered; closed depressions in a DEM approximate crater density.",
                    log=True,
                ),
                Factor(
                    "flat_lt10_pct",
                    "Flat ground (<10°)",
                    "%",
                    30,
                    80,
                    2,
                    "Landing and base pads need gentle slopes, typically under about 10°.",
                ),
                Factor("ndvi", "Vegetation (NDVI)", "", 0.4, 0.05, 2, NDVI_WHY),
                Factor(
                    "abs_lat",
                    "Latitude",
                    "°",
                    30,
                    70,
                    2,
                    "At the lunar poles the Sun stays near the horizon; high latitudes give low sun "
                    "angles, long shadows and long polar days and nights for lighting and power tests.",
                ),
                Factor(
                    "t_mean_c",
                    "Mean air temperature",
                    "°C",
                    25,
                    -25,
                    1.5,
                    "Shadowed lunar craters stay below about 110 K; cold Earth sites test thermal "
                    "control, batteries and cryogenic handling (an operations factor, not climate).",
                ),
                Factor(
                    "precip_mm_yr",
                    "Precipitation",
                    "mm/yr",
                    500,
                    20,
                    1,
                    "The Moon has no liquid water; very dry sites keep the ground dusty and unweathered.",
                    log=True,
                ),
            ),
            known_tag="Moon",
        ),
        Target(
            "mars_low_latitude",
            "Mars - low/mid-latitude landing site",
            "Mars",
            "Hyper-arid, clear-sky, high-UV desert with gentle landing terrain (Jezero / Gale type).",
            (
                Factor(
                    "precip_mm_yr",
                    "Precipitation",
                    "mm/yr",
                    400,
                    10,
                    3,
                    "Mars is hyper-arid; deserts with under ~25 mm/yr best match its desiccated, oxidized "
                    "soils and the dry limits of life.",
                    log=True,
                ),
                Factor(
                    "rh_pct",
                    "Relative humidity",
                    "%",
                    70,
                    20,
                    1.5,
                    "Very dry air keeps soils desiccated, as on Mars.",
                ),
                Factor(
                    "cloud_pct",
                    "Cloud cover",
                    "%",
                    70,
                    20,
                    0.5,
                    "Clear skies mean intense sunlight and UV, as under Mars' thin, mostly cloud-free "
                    "atmosphere (low weight: ~1° cells near coasts include ocean cloud decks).",
                ),
                Factor(
                    "elevation_m",
                    "Elevation",
                    "m",
                    0,
                    4000,
                    1,
                    "High altitude means thinner air and stronger UV, the closest Earth gets to Mars' thin atmosphere.",
                ),
                Factor(
                    "t_range_c",
                    "Daily temperature swing",
                    "°C",
                    8,
                    20,
                    1,
                    "Mars has large day-night temperature swings; deserts with big daily swings test thermal cycling.",
                ),
                Factor(
                    "t_mean_c",
                    "Mean air temperature",
                    "°C",
                    30,
                    0,
                    0.5,
                    "Mars is cold (mean about -60 °C); colder deserts add freeze-thaw processes "
                    "(low weight: many accepted Mars analogs are hot deserts).",
                ),
                Factor("ndvi", "Vegetation (NDVI)", "", 0.3, 0.08, 2, NDVI_WHY),
                Factor(
                    "flat_lt15_pct",
                    "Gentle ground (<15°)",
                    "%",
                    50,
                    90,
                    1.5,
                    "Landing ellipses need mostly gentle slopes, and rovers avoid steep ground.",
                ),
            ),
            known_tag="Mars",
        ),
        Target(
            "mars_polar",
            "Mars - polar / ice-rich periglacial site",
            "Mars",
            "Frozen, dry ground with polar light cycles and patterned ground (Phoenix type).",
            (
                Factor(
                    "t_mean_c",
                    "Mean air temperature",
                    "°C",
                    5,
                    -20,
                    3,
                    "Ground ice and permafrost need sub-zero mean temperatures, as at the Phoenix site.",
                ),
                Factor(
                    "precip_mm_yr",
                    "Precipitation",
                    "mm/yr",
                    400,
                    50,
                    2,
                    "Polar deserts are dry: little snowfall keeps the ground ice-cemented rather than snow-covered.",
                    log=True,
                ),
                Factor(
                    "abs_lat",
                    "Latitude",
                    "°",
                    40,
                    72,
                    1.5,
                    "High latitude brings polar light cycles and periglacial landforms such as polygons.",
                ),
                Factor(
                    "t_range_c",
                    "Daily temperature swing",
                    "°C",
                    4,
                    12,
                    0.5,
                    "Freeze-thaw and thermal-contraction cracking need temperature cycling.",
                ),
                Factor("ndvi", "Vegetation (NDVI)", "", 0.3, 0.05, 2, NDVI_WHY),
                Factor(
                    "flat_lt10_pct",
                    "Flat ground (<10°)",
                    "%",
                    40,
                    90,
                    1,
                    "Polar landers need flat plains, like the Phoenix landing site.",
                ),
            ),
            known_tag="Mars-polar",
        ),
    )
}

# Known analogs (used by space agencies / field campaigns) and candidates to evaluate.
# Tags: Moon, Mars (low/mid-latitude), Mars-polar; no tag = candidate.
_SITE_ROWS = (
    ("Atacama Desert (Yungay), Chile", -24.08, -70.02, "Mars"),
    ("Death Valley, USA", 36.46, -116.87, "Mars"),
    ("Wadi Rum, Jordan", 29.57, 35.42, "Mars"),
    ("McMurdo Dry Valleys, Antarctica", -77.52, 161.67, "Mars-polar, Moon"),
    ("Haughton Crater, Devon Island, Canada", 75.37, -89.68, "Mars-polar, Moon"),
    ("Mars Desert Research Station, Utah, USA", 38.41, -110.79, "Mars"),
    ("Mauna Kea, Hawaii, USA", 19.82, -155.47, "Moon, Mars"),
    ("Meteor Crater, Arizona, USA", 35.03, -111.02, "Moon"),
    ("Black Point Lava Flow, Arizona, USA", 35.68, -111.45, "Moon"),
    ("Nordlinger Ries crater, Germany", 48.88, 10.56, "Moon"),
    ("Askja, Iceland", 65.05, -16.75, "Moon, Mars"),
    ("Timanfaya, Lanzarote, Spain", 29.0, -13.75, "Moon, Mars"),
    ("Rio Tinto, Spain", 37.70, -6.59, "Mars"),
    ("Svalbard (Bockfjorden), Norway", 79.43, 13.33, "Mars-polar"),
    ("Dome C (Concordia), Antarctica", -75.10, 123.33, "Moon, Mars-polar"),
    ("Namib Desert (Gobabeb), Namibia", -23.56, 15.04, "Mars"),
    ("Tso Kar, Ladakh, India", 33.30, 78.0, "Mars"),
    ("Dallol, Ethiopia", 14.24, 40.30, "Mars"),
    ("Harrat al-Sham basalt plateau, Jordan", 32.30, 37.30, ""),
    ("Qa' al-Jafr basin, Jordan", 30.30, 36.30, ""),
    ("Wadi Araba, Jordan", 30.40, 35.17, ""),
    ("Al-Mudawwara desert, Jordan", 29.32, 36.0, ""),
    ("Azraq basin, Jordan", 31.85, 36.83, ""),
    ("Dead Sea shore (Lisan), Jordan", 31.28, 35.50, ""),
)
SITES: tuple[dict, ...] = tuple(
    {"name": n, "lat": la, "lon": lo, "analog_for": tags} for n, la, lo, tags in _SITE_ROWS
)


def _tags(text: str) -> set[str]:
    return {t.strip() for t in text.split(",") if t.strip()}


def _ramp(value: float, zero: float, one: float, log: bool) -> float:
    if log:
        value, zero, one = (math.log10(max(v, 1e-3)) for v in (value, zero, one))
    t = (value - zero) / (one - zero)
    return min(1.0, max(0.0, t))


def score(features: dict, target: Target) -> dict:
    """0-100 score with every factor's value, partial score, weight and reason."""
    rows, got, total, acc = [], 0.0, 0.0, 0.0
    for f in target.factors:
        total += f.weight
        v = features.get(f.key)
        if v is None or not math.isfinite(float(v)):
            rows.append(
                {"key": f.key, "label": f.label, "value": None, "score": None, "weight": f.weight}
            )
            continue
        s = _ramp(float(v), f.zero, f.one, f.log)
        got += f.weight
        acc += s * f.weight
        rows.append(
            {
                "key": f.key,
                "label": f.label,
                "value": round(float(v), 2),
                "unit": f.unit,
                "score": round(100 * s),
                "weight": f.weight,
                "why": f.why,
            }
        )
    return {
        "target": target.key,
        "score": round(100 * acc / got) if got else None,
        "coverage_pct": round(100 * got / total) if total else 0,
        "factors": rows,
        "missing": [r["label"] for r in rows if r["value"] is None],
    }


def score_all(features: dict) -> dict:
    return {key: score(features, t) for key, t in TARGETS.items()}


# --- features from data ------------------------------------------------------------------


def _monthly(par: dict, name: str, fill: float, how: str = "mean") -> float | None:
    months = [v for k, v in (par.get(name) or {}).items() if k != "ANN" and v not in (None, fill)]
    if not months:
        return None
    return float({"mean": np.mean, "min": np.min, "max": np.max}[how](months))


def climate_features(clim: dict) -> dict:
    """Features from a POWER climatology response (2001-2020 monthly means)."""
    par = clim.get("properties", {}).get("parameter", {})
    fill = clim.get("header", {}).get("fill_value", -999.0)
    lon, lat, *rest = (clim.get("geometry", {}).get("coordinates") or [None, None]) + [None]
    precip = _monthly(par, "PRECTOTCORR", fill)
    out = {
        "precip_mm_yr": None if precip is None else precip * 365.25,
        "rh_pct": _monthly(par, "RH2M", fill),
        "t_mean_c": _monthly(par, "T2M", fill),
        "t_range_c": _monthly(par, "T2M_RANGE", fill),
        "t_min_c": _monthly(par, "T2M_MIN", fill, "min"),
        "t_max_c": _monthly(par, "T2M_MAX", fill, "max"),
        "cloud_pct": _monthly(par, "CLOUD_AMT", fill),
        "solar_kwh_m2_day": _monthly(par, "ALLSKY_SFC_SW_DWN", fill),
        "wind_m_s": _monthly(par, "WS2M", fill),
        "abs_lat": None if lat is None else abs(float(lat)),
        "elevation_m": rest[0] if rest and rest[0] is not None else None,
    }
    return {k: (round(float(v), 3) if v is not None else None) for k, v in out.items()}


_LATLON = re.compile(r"latitude\s+(-?[\d.]+)\s+longitude\s+(-?[\d.]+)", re.I)
_ELEV = re.compile(r"=\s*(-?[\d.]+)\s*meters", re.I)


def table_features(analysis: dict, header: str = "") -> dict | None:
    """Features from a NASA POWER daily/monthly point table (means over its period)."""
    st = analysis.get("statistics", {})
    if "T2M" not in st or "PRECTOTCORR" not in st:
        return None

    def mean(name):
        return st.get(name, {}).get("mean")

    rng = mean("T2M_RANGE")
    if rng is None and mean("T2M_MAX") is not None and mean("T2M_MIN") is not None:
        rng = mean("T2M_MAX") - mean("T2M_MIN")
    solar = mean("ALLSKY_SFC_SW_DWN")
    if solar is not None and re.search(r"ALLSKY_SFC_SW_DWN.*MJ", header):
        solar /= 3.6
    feats = {
        "precip_mm_yr": mean("PRECTOTCORR") * 365.25 if mean("PRECTOTCORR") is not None else None,
        "rh_pct": mean("RH2M"),
        "t_mean_c": mean("T2M"),
        "t_range_c": rng,
        "t_min_c": st.get("T2M_MIN", {}).get("min"),
        "cloud_pct": mean("CLOUD_AMT"),
        "solar_kwh_m2_day": solar,
    }
    if m := _LATLON.search(header):
        feats["abs_lat"] = abs(float(m.group(1)))
        feats["lat"], feats["lon"] = float(m.group(1)), float(m.group(2))
    if m := _ELEV.search(header):
        feats["elevation_m"] = float(m.group(1))
    return {k: (round(float(v), 3) if v is not None else None) for k, v in feats.items()}


def terrain_features(terrain: dict) -> dict:
    res = terrain.get("at_analysis_resolution", {})
    dep = terrain.get("depressions", {})
    return {
        "flat_lt10_pct": res.get("flat_lt10_pct"),
        "flat_lt15_pct": res.get("flat_lt15_pct"),
        "depressions_per_1000_km2": dep.get("per_1000_km2")
        if dep.get("count") is not None
        else None,
    }


# --- ranking and terrain comparison --------------------------------------------------------


def rank(sites: list[dict], features: list[dict | None], target_key: str) -> dict:
    target = TARGETS[target_key]
    rows = []
    for site, feats in zip(sites, features, strict=True):
        if not feats:
            rows.append(
                {**site, "score": None, "coverage_pct": 0, "factors": [], "error": "no data"}
            )
            continue
        rows.append({**site, **score(feats, target), "features": feats})
    rows.sort(key=lambda r: -1 if r["score"] is None else -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    scored = [r for r in rows if r["score"] is not None]
    known = [r for r in scored if target.known_tag in _tags(r.get("analog_for", ""))]
    check = None
    if scored and known:
        third = max(1, math.ceil(len(scored) / 3))
        top = sum(1 for r in known if r["rank"] <= third)
        check = {
            "known_analogs": len(known),
            "in_top_third": top,
            "text": f"{top} of {len(known)} expert-chosen {target.known_tag} analogs rank in the top third",
        }
    return {
        "target": target.key,
        "target_name": target.name,
        "body": target.body,
        "description": target.description,
        "ranking": rows,
        "validation": check,
        "robustness": robustness(rows, target),
    }


WEIGHT_JITTER = 0.30  # each weight x U(0.7, 1.3)
LEVEL_JITTER = 0.20  # each 0 / 100 level moved by up to 20% of the ramp length
DRAWS = 1000


def _auc(scores: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Share of (known analog, other site) pairs where the known analog scores higher
    (ties count half), for each row of `scores`."""
    k, o = scores[..., known], scores[..., ~known]
    diff = k[..., :, None] - o[..., None, :]
    return ((diff > 0) + 0.5 * (diff == 0)).mean(axis=(-1, -2))


def robustness(rows: list[dict], target: Target, draws: int = DRAWS, seed: int = 0) -> dict | None:
    """How much the ranking depends on the hand-set weights and levels: re-score every site
    under random weights (±30%) and levels (±20%). Adds each site's 90% score and rank range,
    and compares the score with single-factor baselines on the known analogs (in-sample: the
    levels were set knowing these sites, so this checks consistency, not predictive skill)."""
    scored = [r for r in rows if r.get("score") is not None and r.get("features")]
    if len(scored) < 3:
        return None
    fs = target.factors
    logs = np.array([f.log for f in fs])

    def tr(a):
        return np.where(logs, np.log10(np.maximum(a, 1e-3)), a)

    x = np.array(
        [
            [
                np.nan if r["features"].get(f.key) is None else float(r["features"][f.key])
                for f in fs
            ]
            for r in scored
        ]
    )
    have = np.isfinite(x)
    x = tr(np.nan_to_num(x))
    zero, one = tr(np.array([f.zero for f in fs])), tr(np.array([f.one for f in fs]))
    span = one - zero
    rng = np.random.default_rng(seed)
    w = np.array([f.weight for f in fs]) * rng.uniform(
        1 - WEIGHT_JITTER, 1 + WEIGHT_JITTER, (draws, len(fs))
    )
    z = zero + span * rng.uniform(-LEVEL_JITTER, LEVEL_JITTER, (draws, len(fs)))
    o = one + span * rng.uniform(-LEVEL_JITTER, LEVEL_JITTER, (draws, len(fs)))
    part = np.clip((x[None] - z[:, None]) / (o - z)[:, None], 0, 1)  # draws x sites x factors
    ww = w[:, None, :] * have[None]
    sc = 100 * (part * ww).sum(-1) / ww.sum(-1)
    ranks = (-sc).argsort(axis=1).argsort(axis=1) + 1
    lo_s, hi_s = np.percentile(sc, [5, 95], axis=0)
    lo_r, hi_r = np.percentile(ranks, [5, 95], axis=0)
    for i, r in enumerate(scored):
        r["score_range"] = [round(float(lo_s[i])), round(float(hi_s[i]))]
        r["rank_range"] = [int(round(lo_r[i])), int(round(hi_r[i]))]
    out = {
        "draws": draws,
        "weights_pm_pct": round(100 * WEIGHT_JITTER),
        "levels_pm_pct": round(100 * LEVEL_JITTER),
    }
    known = np.array([target.known_tag in _tags(r.get("analog_for", "")) for r in scored])
    if known.all() or not known.any():
        out["text"] = (
            f"Scores re-computed {draws} times with weights ±{out['weights_pm_pct']}% and levels "
            f"±{out['levels_pm_pct']}%: the ranges show how much each score depends on them."
        )
        return out
    nominal = float(_auc(np.array([r["score"] for r in scored], float), known))
    aucs = _auc(sc, known)
    single = {}
    for j, f in enumerate(fs):  # each factor alone, on the sites that have it
        if have[:, j].sum() >= 3 and known[have[:, j]].any() and (~known[have[:, j]]).any():
            alone = np.clip((x[have[:, j], j] - zero[j]) / span[j], 0, 1)
            single[f.label] = float(_auc(alone, known[have[:, j]]))
    best = max(single.items(), key=lambda kv: kv[1]) if single else None
    out.update(
        auc=round(nominal, 2),
        auc_range=[round(float(v), 2) for v in np.percentile(aucs, [5, 95])],
        best_single_factor={"label": best[0], "auc": round(best[1], 2)} if best else None,
    )
    beat = (
        f"; the best single factor ({best[0]}) alone gives {best[1]:.2f}"
        + (" - the combined score adds nothing over it here" if best[1] >= nominal else "")
        if best
        else ""
    )
    out["text"] = (
        f"Known {target.known_tag} analogs outscore the other sites in {nominal:.0%} of pairs "
        f"(90% range {out['auc_range'][0]:.0%}-{out['auc_range'][1]:.0%} when weights vary "
        f"±{out['weights_pm_pct']}% and levels ±{out['levels_pm_pct']}%){beat}. In-sample check: "
        "the levels were set knowing these sites, so it shows consistency, not predictive skill."
    )
    return out


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0 = same distribution, 1 = disjoint)."""
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _ratio_similarity(a, b, floor: float = 0.0) -> float | None:
    if a is None or b is None:
        return None
    a, b = float(a) + floor, float(b) + floor
    if a <= 0 or b <= 0:
        return None
    return math.exp(-abs(math.log(a / b)))


def compare_terrain(earth: dict, other: dict) -> dict | None:
    """How alike two DEMs' terrain is, at the finest baseline both were measured at."""
    pa, pb = earth.get("by_baseline_m", {}), other.get("by_baseline_m", {})
    common = sorted(set(pa) & set(pb), key=float)
    if not common:
        return None
    k = common[0]
    slope_sim = 1 - _jsd(np.asarray(pa[k]["slope_hist"]), np.asarray(pb[k]["slope_hist"]))
    rough = _ratio_similarity(pa[k].get("roughness_tri_m"), pb[k].get("roughness_tri_m"))
    craters = _ratio_similarity(
        earth.get("depressions", {}).get("per_1000_km2"),
        other.get("depressions", {}).get("per_1000_km2"),
        floor=0.5,
    )
    parts = [x for x in (slope_sim, rough, craters) if x is not None]
    return {
        "baseline_m": float(k),
        "similarity": round(100 * float(np.mean(parts))),
        "slope_distribution": round(100 * slope_sim),
        "roughness": None if rough is None else round(100 * rough),
        "crater_density": None if craters is None else round(100 * craters),
    }
