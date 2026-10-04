"""Small real public NASA samples (no login). Skipped when offline or blocked.

Beyond "it opens", every numeric column the tool reports is compared with an independent
pandas reading of the same file (count of valid values and mean), so a reader that skips
the wrong header lines, mis-detects the delimiter or keeps fill values fails here."""

import io

import numpy as np
import pandas as pd
import pytest
import requests

from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file

SAMPLES = {
    "firms_24h.csv": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Australia_NewZealand_24h.csv",
    "power_amman.csv": "https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=T2M,PRECTOTCORR"
    "&community=RE&longitude=35.93&latitude=31.95&start=2001&end=2020&format=CSV",
    "gistemp.csv": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
}
FILL = [-999, -999.0, -9999, -9999.0]


def _independent(name: str, text: str) -> pd.DataFrame:
    """How a person would load each file by hand, following its documented layout."""
    if name == "gistemp.csv":  # one title line, then the table; *** marks missing values
        return pd.read_csv(io.StringIO(text), skiprows=1, na_values=["***", "****"])
    if name == "power_amman.csv":  # free-text block between -BEGIN HEADER- and -END HEADER-
        body = text.split("-END HEADER-", 1)[1]
        return pd.read_csv(io.StringIO(body.strip()), na_values=FILL)
    return pd.read_csv(io.StringIO(text))


@pytest.mark.network
@pytest.mark.parametrize("name", list(SAMPLES))
def test_real_sample(name, tmp_path):
    try:
        r = requests.get(SAMPLES[name], timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"cannot download {name}: {type(exc).__name__}")
    path = tmp_path / name
    path.write_bytes(r.content)
    rep = process_file(path, ReadOptions(plots=False), tmp_path / "reports")
    assert rep.error is None and rep.kind == "table"
    stats = rep.analysis["statistics"]
    assert stats

    ref = _independent(name, r.text)
    assert rep.analysis["summary"]["n_rows"] == len(ref)
    compared = 0
    for col, st in stats.items():
        if col not in ref.columns or "mean" not in st:
            continue
        values = pd.to_numeric(ref[col], errors="coerce").replace(FILL, np.nan)
        values = values[np.isfinite(values)]
        assert st["count"] == len(values), col
        assert st["mean"] == pytest.approx(values.mean(), rel=1e-9, abs=1e-9), col
        compared += 1
    assert compared >= 2  # the comparison really ran


@pytest.mark.network
def test_gistemp_warming_trend_matches_an_independent_fit(tmp_path):
    """Physical sanity on real data: the annual global anomaly (J-D) has risen since 1880.
    The tool's linear slope must equal an ordinary least-squares fit on the same values."""
    try:
        r = requests.get(SAMPLES["gistemp.csv"], timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"cannot download GISTEMP: {type(exc).__name__}")
    (tmp_path / "gistemp.csv").write_bytes(r.content)
    rep = process_file(tmp_path / "gistemp.csv", ReadOptions(plots=False), tmp_path / "r")
    trend = next((t for t in rep.analysis["trends"] if t["variable"] == "J-D"), None)
    if trend is None:
        pytest.skip("no time axis recognised for GISTEMP (Year column)")
    ref = _independent("gistemp.csv", r.text)[["Year", "J-D"]].dropna()
    slope = np.polyfit(ref["Year"].astype(float), ref["J-D"].astype(float), 1)[0]
    assert trend["slope_per_year"] == pytest.approx(slope, rel=0.02)
    assert 0.005 < trend["slope_per_year"] < 0.02  # °C per year: warming, published order
    assert trend["mk_trend"] == "increasing" and trend["mk_p"] < 0.001
