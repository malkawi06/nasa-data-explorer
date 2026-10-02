"""NASA POWER point data: daily series for any latitude/longitude.

The browser fetches the same URLs (built here, in Python) itself; the CLI uses requests.
Community "RE" keeps irradiance in kWh/m²/day.
"""

from __future__ import annotations

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


def default_period(years: int = 10) -> tuple[str, str]:
    """The last `years` complete calendar years (POWER lags a few days behind)."""
    last = date.today().year - 1
    return f"{last - years + 1}-01-01", f"{last}-12-31"


def _get(url: str) -> bytes:
    import requests

    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content


def fetch_daily(lat: float, lon: float, start: str, end: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    la, lo = _check(lat, lon)
    path = out_dir / f"POWER_daily_{la}_{lo}_{start.replace('-', '')}_{end.replace('-', '')}.csv"
    path.write_bytes(_get(daily_url(lat, lon, start, end)))
    return path
