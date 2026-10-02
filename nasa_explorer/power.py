"""NASA POWER point data: daily series and 2001-2020 climatology for any latitude/longitude.

The browser fetches the same URLs (built here, in Python) itself; the CLI uses requests.
Community "RE" keeps irradiance in kWh/m²/day, the unit the analog thresholds use.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

API = "https://power.larc.nasa.gov/api/temporal"
DAILY_PARAMS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2M_RANGE",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN",
    "CLOUD_AMT",
)
CLIMATE_PARAMS = DAILY_PARAMS
TIMEOUT = 90


def _check(lat: float, lon: float) -> tuple[float, float]:
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 360):
        raise ValueError(f"latitude {lat} / longitude {lon} out of range")
    return round(lat, 4), round(((lon + 180) % 360) - 180, 4)


def daily_url(lat: float, lon: float, start: str, end: str) -> str:
    """start/end as YYYY-MM-DD or YYYYMMDD."""
    lat, lon = _check(lat, lon)
    s, e = start.replace("-", ""), end.replace("-", "")
    return (
        f"{API}/daily/point?parameters={','.join(DAILY_PARAMS)}&community=RE"
        f"&longitude={lon}&latitude={lat}&start={s}&end={e}&format=CSV"
    )


def climatology_url(lat: float, lon: float) -> str:
    lat, lon = _check(lat, lon)
    return (
        f"{API}/climatology/point?parameters={','.join(CLIMATE_PARAMS)}&community=RE"
        f"&longitude={lon}&latitude={lat}&format=JSON"
    )


def default_period(years: int = 10) -> tuple[str, str]:
    """The last `years` complete calendar years (POWER lags a few days behind)."""
    last = date.today().year - 1
    return f"{last - years + 1}-01-01", f"{last}-12-31"


def _get(url: str, cache: Path | None) -> bytes:
    import requests

    key = cache / (hashlib.sha256(url.encode()).hexdigest()[:24]) if cache else None
    if key and key.exists():
        return key.read_bytes()
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    if key:
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_bytes(resp.content)
    return resp.content


def fetch_daily(lat: float, lon: float, start: str, end: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    la, lo = _check(lat, lon)
    path = out_dir / f"POWER_daily_{la}_{lo}_{start.replace('-', '')}_{end.replace('-', '')}.csv"
    path.write_bytes(_get(daily_url(lat, lon, start, end), None))
    return path


def fetch_climatology(lat: float, lon: float, cache: Path | None = None) -> dict:
    return json.loads(_get(climatology_url(lat, lon), cache))
