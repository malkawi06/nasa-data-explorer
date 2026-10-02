"""Which planetary body a dataset describes (from its CRS or label text), and its radius.

Distances, slopes and maps all depend on the body: a degree of longitude is 111 km on Earth
but 30 km on the Moon, and a lunar polar DEM must never be reprojected onto Earth coordinates.
"""

from __future__ import annotations

import re

MEAN_RADIUS_M = {
    "Earth": 6_371_008.8,
    "Moon": 1_737_400.0,
    "Mars": 3_389_500.0,
    "Mercury": 2_439_400.0,
    "Venus": 6_051_800.0,
}
# Sphere radii that planetary products use as their zero level when they store radius
# instead of height (LOLA: 1737.4 km sphere; MOLA MEGDR: 3396.19 km sphere).
REFERENCE_RADIUS_M = {"Moon": 1_737_400.0, "Mars": 3_396_190.0}
# Semi-major axes seen in CRS definitions, matched within 0.5 %.
_AXES = {
    "Moon": (1_737_400.0, 1_737_150.0),
    "Mars": (3_396_190.0, 3_396_000.0, 3_389_500.0, 3_393_400.0),
    "Mercury": (2_439_400.0, 2_440_000.0),
    "Venus": (6_051_800.0, 6_051_000.0),
}
_NAME = re.compile(r"(?<![a-z])(moon|lunar|luna|mars|martian|mercury|venus)(?![a-z])", re.I)
_CANON = {"moon": "Moon", "lunar": "Moon", "luna": "Moon", "mars": "Mars", "martian": "Mars"}
_AXIS = re.compile(r"(?:ELLIPSOID|SPHEROID)\[\"[^\"]*\",\s*([0-9.]+)", re.I)
_TARGET = re.compile(
    r"(?:TARGET_NAME|planet|target_body|target|body)['\"]?\s*[=:]\s*['\"]?([A-Za-z]+)", re.I
)
_PROJ_RADIUS = re.compile(r"\+(?:R|a)=([0-9.]+)")


def _named(text: str) -> str | None:
    if m := _NAME.search(text):
        word = m.group(1).lower()
        return _CANON.get(word, word.capitalize())
    return None


def detect(crs_wkt: str | None = None, label: str = "") -> str:
    """'Earth' unless the CRS names or measures another body, or a PDS label targets one."""
    if m := _TARGET.search(label or ""):
        name = m.group(1).capitalize()
        if name in MEAN_RADIUS_M:
            return name
    wkt = crs_wkt or ""
    if name := _named(wkt):
        return name
    if m := _AXIS.search(wkt) or _PROJ_RADIUS.search(wkt):
        a = float(m.group(1))
        for body, axes in _AXES.items():
            if any(abs(a - x) / x < 0.005 for x in axes):
                return body
    return "Earth"


def radius_m(body: str) -> float:
    return MEAN_RADIUS_M.get(body, MEAN_RADIUS_M["Earth"])
