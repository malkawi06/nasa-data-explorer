"""Recognise well-known NASA products and attach a card with the facts people get wrong.

Matching uses the file name, global attributes / header text and (for tables) column names.
Cards are also given to the AI as trusted context, so its advice reflects real caveats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote


@dataclass(frozen=True)
class Product:
    key: str
    name: str
    pattern: str  # regex over file name + attributes + header text
    columns: frozenset[str] = frozenset()  # all must be present (case-insensitive) to match a table
    resolution: str = ""
    caveats: tuple[str, ...] = ()
    search: str = ""  # Earthdata search term
    extra_links: tuple[tuple[str, str], ...] = field(default_factory=tuple)


CATALOG: tuple[Product, ...] = (
    Product(
        "firms",
        "FIRMS active fire detections (MODIS / VIIRS)",
        r"firms|modis_c6|viirs_snpp|j1_viirs|active.?fire",
        frozenset({"latitude", "longitude", "frp"}),
        "375 m (VIIRS) / 1 km (MODIS) pixels, per overpass",
        (
            "Detections are thermal anomalies (hotspots), not burned area; one fire can produce many rows.",
            "Confidence is 0-100 for MODIS but l/n/h (low/nominal/high) for VIIRS.",
            "Check the daynight column: night detections are fewer and have different brightness.",
            "Cloud cover and overpass timing mean 'no detection' is not 'no fire'.",
        ),
        "FIRMS",
        (("FIRMS map", "https://firms.modaps.eosdis.nasa.gov/map/"),),
    ),
    Product(
        "power",
        "NASA POWER meteorology / solar point data",
        r"nasa/power|power\.larc|-begin header-",
        frozenset(),
        "0.5° × 0.625° (MERRA-2) / 1° (CERES) grid cell, not a station",
        (
            "Values are modelled/satellite-derived for the grid cell, not station observations.",
            "-999 marks missing values.",
            "Units differ per parameter (e.g. T2M °C, ALLSKY_SFC_SW_DWN kWh/m²/day).",
        ),
        "NASA POWER",
        (("POWER data viewer", "https://power.larc.nasa.gov/data-access-viewer/"),),
    ),
    Product(
        "mod11",
        "MODIS Land Surface Temperature (MOD11/MYD11)",
        r"m[oy]d11",
        resolution="1 km, daily (A1) or 8-day (A2)",
        caveats=(
            "Raw LST uses scale 0.02 → Kelvin; 0 is fill.",
            "Clear-sky only: cloudy pixels are missing, which biases means warm/cold by season.",
            "Use the QC_Day/QC_Night layers to drop low-quality pixels.",
            "Sinusoidal projection tiles (hXXvYY), not lat/lon.",
        ),
        search="MOD11A2",
    ),
    Product(
        "mod13",
        "MODIS Vegetation Indices NDVI/EVI (MOD13/MYD13)",
        r"m[oy]d13",
        resolution="250 m-1 km, 16-day composites",
        caveats=(
            "Scale factor 0.0001 (valid -2000..10000 raw).",
            "16-day composites: the date is the composite start, not the observation day.",
            "Use pixel reliability / VI quality layers.",
        ),
        search="MOD13Q1",
    ),
    Product(
        "mcd64",
        "MODIS Burned Area (MCD64A1)",
        r"mcd64",
        resolution="500 m, monthly",
        caveats=("Values are burn day-of-year; 0 = unburned, -1/-2 = unmapped/water.",),
        search="MCD64A1",
    ),
    Product(
        "imerg",
        "GPM IMERG precipitation",
        r"imerg|3b-hhr|3b-day|3b-mo|gpm_3",
        resolution="0.1°, 30-min / daily / monthly",
        caveats=(
            "Early/Late/Final runs differ; use Final for research, Early/Late for near-real-time.",
            "Half-hourly files are mm/hr, not mm per half hour.",
            "Less reliable over snow/ice and complex terrain.",
        ),
        search="GPM IMERG",
    ),
    Product(
        "merra2",
        "MERRA-2 reanalysis",
        r"merra-?2|merra2_",
        resolution="0.5° × 0.625°, hourly to monthly",
        caveats=(
            "Reanalysis = model + assimilated observations, not direct measurement.",
            "Temperatures in Kelvin.",
            "Collections (e.g. M2T1NXSLV) differ in variables and time step.",
        ),
        search="MERRA-2",
    ),
    Product(
        "grace",
        "GRACE / GRACE-FO terrestrial water storage",
        r"grace|grctellus|tellus|mascon|lwe_thickness",
        resolution="~300 km effective (0.25-1° grids), monthly",
        caveats=(
            "Values are anomalies relative to a baseline (often 2004-2009 mean), not absolute storage.",
            "Gap between GRACE (to 2017-06) and GRACE-FO (from 2018-06).",
            "Apply scaling factors / land mask for land-only analyses.",
        ),
        search="GRACE-FO mascon",
    ),
    Product(
        "smap",
        "SMAP soil moisture",
        r"smap|spl[234]",
        resolution="9-36 km, daily (AM/PM passes)",
        caveats=(
            "Top ~5 cm soil moisture in m³/m³.",
            "AM (descending) and PM (ascending) retrievals are separate fields.",
            "Use retrieval_qual_flag; dense vegetation and frozen soil degrade retrievals.",
        ),
        search="SMAP L3",
    ),
    Product(
        "icesat2",
        "ICESat-2 photon/height products (ATLxx)",
        r"atl0[0-9]|atl1[0-9]|icesat-?2",
        resolution="along-track points per beam (gt1l…gt3r)",
        caveats=(
            "Data are grouped per beam; strong and weak beams differ.",
            "Heights are ellipsoidal (WGS84) unless a geoid is applied.",
        ),
        search="ICESat-2",
    ),
    Product(
        "gedi",
        "GEDI lidar footprints",
        r"gedi0[12]",
        resolution="25 m footprints along tracks",
        caveats=(
            "Filter with quality_flag == 1 and degrade_flag == 0.",
            "Data are per beam groups (BEAMxxxx).",
        ),
        search="GEDI L2A",
    ),
    Product(
        "landsat",
        "Landsat Collection 2 (Level-2)",
        r"l[ct]0[89]_|landsat|_sr_b\d|_st_b\d",
        resolution="30 m, 16-day revisit",
        caveats=(
            "Surface reflectance: value × 0.0000275 - 0.2.",
            "Surface temperature: value × 0.00341802 + 149.0 → Kelvin.",
            "Mask clouds with the QA_PIXEL band.",
        ),
        search="Landsat Collection 2 Level-2",
    ),
    Product(
        "hls",
        "Harmonized Landsat Sentinel-2 (HLS)",
        r"hls\.[ls]30",
        resolution="30 m, 2-3 day combined revisit",
        caveats=("Reflectance scale factor 0.0001.", "Use the Fmask layer for clouds/shadows."),
        search="HLS",
    ),
    Product(
        "oco2",
        "OCO-2 / OCO-3 XCO2",
        r"oco[23]",
        resolution="~2 km soundings along track",
        caveats=(
            "Use xco2_quality_flag == 0.",
            "XCO2 is a column average in ppm, not surface concentration.",
        ),
        search="OCO-2 L2 Lite",
    ),
    Product(
        "tempo",
        "TEMPO air quality (NO2, O3, HCHO)",
        r"tempo_",
        resolution="~2 × 4.75 km, hourly daytime, North America",
        caveats=(
            "Columns in molecules/cm²; use main_data_quality_flag == 0.",
            "Daytime only, North America only.",
        ),
        search="TEMPO NO2",
    ),
    Product(
        "nisar",
        "NISAR SAR products",
        r"nisar",
        resolution="~10-50 m depending on product",
        caveats=(
            "Backscatter is linear power (γ0); convert with 10·log10 for dB.",
            "Data live in HDF5 groups per frequency/polarisation.",
        ),
        search="NISAR",
    ),
    Product(
        "viirs_bm",
        "VIIRS Black Marble night lights (VNP46)",
        r"vnp46|vj146|black.?marble",
        resolution="500 m, daily/monthly",
        caveats=(
            "Use the gap-filled / quality-flagged radiance; moonlight and clouds affect daily values.",
        ),
        search="VNP46A2",
    ),
    Product(
        "gistemp",
        "GISTEMP global temperature anomalies",
        r"gistemp|glb\.ts|ts\+dsst",
        resolution="global/zonal means, monthly",
        caveats=(
            "Values are anomalies vs 1951-1980, not absolute temperatures.",
            "'***' marks missing values.",
        ),
        search="GISTEMP",
        extra_links=(("GISTEMP", "https://data.giss.nasa.gov/gistemp/"),),
    ),
)


def _haystack(file: str, analysis: dict) -> str:
    s = analysis.get("summary", {})
    parts = [
        file,
        str(s.get("attributes", "")),
        str(s.get("root_attributes", "")),
        s.get("header_text", ""),
        str(s.get("title", "")),
    ]
    return " ".join(parts).lower()


def identify(file: str, analysis: dict) -> list[dict]:
    text = _haystack(file, analysis)
    columns = {
        str(c.get("name", "")).lower() for c in analysis.get("summary", {}).get("columns", [])
    }
    cards = []
    for p in CATALOG:
        by_text = re.search(p.pattern, text) is not None
        by_columns = bool(p.columns) and p.columns <= columns
        if not (by_text or by_columns):
            continue
        links = [
            (
                "Earthdata Search",
                f"https://search.earthdata.nasa.gov/search?q={quote(p.search or p.name)}",
            ),
            ("Worldview", "https://worldview.earthdata.nasa.gov/"),
            *p.extra_links,
        ]
        cards.append(
            {
                "key": p.key,
                "name": p.name,
                "resolution": p.resolution,
                "caveats": list(p.caveats),
                "links": [{"title": t, "url": u} for t, u in links],
                "matched_by": "columns" if by_columns and not by_text else "name/attributes",
            }
        )
    return cards[:3]
