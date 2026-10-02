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
        r"nasa/power|nasa power|power\.larc|-begin header-",
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
        r"m[oy]d13|modis.{0,20}ndvi|ndvi.{0,20}modis",
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
    # --- elevation models and planetary data (Moon / Mars) -----------------------------------
    Product(
        "lola",
        "LRO LOLA lunar topography (LDEM / polar DEMs)",
        r"(?<![a-z])(lola|ldem|ldsm|lunar.?orbiter.?laser)",
        resolution="global grids 4-512 pixels/degree; polar DEMs 5-20 m/pixel",
        caveats=(
            "Heights are relative to a 1737.4 km reference sphere: the Moon has no sea level.",
            "Polar products use polar stereographic projection in metres; scale is exact only at the pole.",
            "High-resolution grids interpolate between laser tracks: check for striping and smooth gaps.",
            "Slopes depend on baseline: lander hazards are judged at metre scale, finer than most grids.",
        ),
        search="LOLA LDEM",
        extra_links=(
            ("LOLA PDS", "https://pds-geosciences.wustl.edu/missions/lro/lola.htm"),
            ("LRO QuickMap", "https://quickmap.lroc.asu.edu/"),
        ),
    ),
    Product(
        "diviner",
        "LRO Diviner lunar surface temperature",
        r"diviner|(?<![a-z])dlre(?![a-z])",
        resolution="~200-250 m/pixel maps; polar summer/winter and day/night products",
        caveats=(
            "Temperatures are in kelvin and depend strongly on local time and season.",
            "Permanently shadowed regions can stay below ~110 K, cold enough to trap water ice.",
            "Polar maps mix seasons: compare like with like (same season, same local time).",
        ),
        search="Diviner LRO",
        extra_links=(("Diviner", "https://www.diviner.ucla.edu/"),),
    ),
    Product(
        "lroc",
        "LRO Camera (NAC / WAC) images and mosaics",
        r"lroc|(?<![a-z])(nac|wac)(?![a-z])|lunar.?reconnaissance.?orbiter.?camera"
        r"|(?<![a-z0-9])m\d{9,10}(le|re|lc|rc|me|ce)(?![a-z0-9])",  # M1105555625LE
        resolution="NAC ~0.5 m/pixel; WAC ~100 m/pixel",
        caveats=(
            "Illumination changes with every image; near the poles long shadows hide much of the ground.",
            "Pixel values are reflectance (I/F), not elevation: use LOLA or NAC DTMs for slopes.",
        ),
        search="LROC",
        extra_links=(("LRO QuickMap", "https://quickmap.lroc.asu.edu/"),),
    ),
    Product(
        "mola",
        "MGS MOLA Mars topography (MEGDR)",
        r"(?<![a-z])(mola|megdr|mars.?orbiter.?laser)|(?<![a-z0-9])meg[tracs]\d{2}[ns]\d{3}[a-z]{2}",
        resolution="global grid up to 128 pixels/degree (~463 m)",
        caveats=(
            "Heights are relative to the Mars areoid (equipotential surface); Mars has no sea level.",
            "At ~463 m pixels, slopes are much gentler than what a lander sees.",
            "Longitudes are often 0-360° East.",
        ),
        search="MOLA MEGDR",
        extra_links=(("MOLA PDS", "https://pds-geosciences.wustl.edu/missions/mgs/mola.html"),),
    ),
    Product(
        "hirise",
        "MRO HiRISE images and DTMs",
        r"hirise|(?<![a-z])dte[ep][ce]?_|(?<![a-z0-9])(psp|esp|aeb|ptf)_\d{6}_\d{4}",
        resolution="images 25-50 cm/pixel; DTMs ~1-2 m posting",
        caveats=(
            "DTMs cover only small stereo footprints (a few km wide).",
            "DTM elevations are tied to MOLA; small offsets between DTMs are normal.",
        ),
        search="HiRISE",
        extra_links=(("HiRISE", "https://www.uahirise.org/"),),
    ),
    Product(
        "ctx",
        "MRO Context Camera (CTX)",
        r"(?<![a-z])ctx(?![a-z])|context.?camera"
        r"|(?<![a-z0-9])[a-z]\d{2}_\d{6}_\d{4}_x[a-z]_\d{2}[ns]\d{3}w",  # B01_009861_1753_XI_04S352W
        resolution="~6 m/pixel",
        caveats=("Single images are not elevation; slopes need a stereo-derived DTM.",),
        search="MRO CTX",
    ),
    Product(
        "themis",
        "Mars Odyssey THEMIS infrared / thermal inertia",
        r"themis",
        resolution="~100 m/pixel infrared",
        caveats=(
            "Night-time temperature and thermal inertia separate dust (low) from rock and bedrock (high).",
            "Values depend on local time and season of each image.",
        ),
        search="THEMIS",
    ),
    Product(
        "crism",
        "MRO CRISM spectral / mineral maps",
        r"crism|(?<![a-z0-9])(frt|hrl|hrs|frs|fft|msp|hsp)[0-9a-f]{8}",
        resolution="~18-36 m/pixel (targeted), ~100-200 m (mapping)",
        caveats=(
            "Summary parameters (e.g. D2300, OLINDEX) indicate minerals; they are not abundances.",
            "Check for atmospheric and detector artefacts before interpreting faint signals.",
        ),
        search="CRISM",
    ),
    Product(
        "hrsc",
        "Mars Express HRSC DTMs and mosaics (ESA)",
        r"(?<![a-z])hrsc(?![a-z])|(?<![a-z0-9])h\d{4}_\d{4}_(nd|da|dt|s1|s2|p1|p2|re|gr|bl|ir)",
        resolution="DTMs ~50-200 m; images ~12.5 m/pixel",
        caveats=("Heights are relative to the Mars areoid, like MOLA.",),
        search="HRSC",
    ),
    Product(
        "srtm",
        "SRTM / NASADEM elevation (Earth)",
        r"srtm|nasadem|(?<![a-z])[ns]\d{2}[ew]\d{3}(?![0-9])",
        resolution="1 arc-second (~30 m) or 3 arc-seconds (~90 m); 60°N-56°S",
        caveats=(
            "Heights are above the EGM96 geoid, in metres.",
            "Radar voids in steep terrain and sand seas; NASADEM fills many of them.",
            "It is a surface model: forest canopy and buildings raise the heights.",
        ),
        search="NASADEM",
        extra_links=(("OpenTopography", "https://opentopography.org/"),),
    ),
    Product(
        "copdem",
        "Copernicus DEM GLO-30 / GLO-90 (Earth)",
        r"copernicus.?dem|cop.?dem|glo.?30|glo.?90|dem_cop",
        resolution="~30 m or ~90 m, global",
        caveats=(
            "Heights are above the EGM2008 geoid.",
            "It is a surface model (includes trees and buildings).",
        ),
        search="Copernicus DEM",
    ),
    Product(
        "aster_gdem",
        "ASTER GDEM (Earth)",
        r"aster.?gdem|astgtm",
        resolution="1 arc-second (~30 m), 83°N-83°S",
        caveats=("Cloud artefacts (pits and bumps) remain in some tiles; check the NUM layer.",),
        search="ASTER GDEM",
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
        " ".join(s.get("nasa_mentions", {})),  # papers: the datasets they use
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
