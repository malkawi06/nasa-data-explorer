"""Small real public NASA samples (no login). Skipped when offline or blocked."""

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
    rep = process_file(path, ReadOptions(), tmp_path / "reports")
    assert rep.error is None and rep.kind == "table"
    assert rep.analysis["statistics"]
