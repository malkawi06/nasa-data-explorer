"""Awkward real-world inputs found while testing the website: each one once broke or misled."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process_file

RNG = np.random.default_rng(5)


def _run(path, tmp_path):
    return process_file(path, ReadOptions(plots=False), tmp_path / "r")


def _codes(rep):
    return {q["code"] for q in rep.analysis.get("quality", [])}


def test_nested_json_lists_are_flattened_into_events(tmp_path):
    events = [
        {
            "id": f"E{i}",
            "categories": [{"id": "wildfires"}],
            "geometry": [
                {
                    "date": f"2024-06-{1 + i % 28:02d}T00:00:00Z",
                    "coordinates": [-120.0 + i % 5, 40.0],
                }
            ],
        }
        for i in range(120)
    ]
    (tmp_path / "eonet.json").write_text(json.dumps({"events": events}), encoding="utf-8")
    rep = _run(tmp_path / "eonet.json", tmp_path)
    assert rep.error is None
    assert rep.analysis["events"]["total"] == 120 and rep.analysis["events"]["hotspots"]


def test_semicolon_decimal_comma_and_day_first_dates(tmp_path):
    days = pd.date_range("2018-01-01", periods=900, freq="D")
    df = pd.DataFrame({"Datum": days.strftime("%d/%m/%Y"), "Temp": RNG.normal(12, 3, 900).round(2)})
    (tmp_path / "eu.csv").write_text(df.to_csv(sep=";", index=False, decimal=","), encoding="utf-8")
    rep = _run(tmp_path / "eu.csv", tmp_path)
    assert rep.analysis["summary"]["n_rows"] == 900
    assert rep.analysis["coverage"]["time"]["start"].startswith("2018-01-01")
    assert abs(rep.analysis["statistics"]["Temp"]["mean"] - 12) < 0.5


def test_date_column_found_by_its_values_whatever_the_name(tmp_path):
    df = pd.DataFrame(
        {"التاريخ": pd.date_range("2020-01-01", periods=800).strftime("%Y-%m-%d"), "x": range(800)}
    )
    (tmp_path / "ar.csv").write_text("﻿" + df.to_csv(index=False), encoding="utf-8")
    rep = _run(tmp_path / "ar.csv", tmp_path)
    assert rep.analysis["coverage"]["time"]["column"] == "التاريخ"


def test_epoch_milliseconds_and_varying_places_are_events(tmp_path):
    t0 = pd.Timestamp("2022-01-01").value // 10**6
    feats = [
        {
            "type": "Feature",
            "properties": {
                "mag": 4.5 + i % 7 / 10,
                "time": t0 + i * 3_600_000 * 20,
                "place": f"zone {i % 9}",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(RNG.uniform(-180, 180)), float(RNG.uniform(-60, 60))],
            },
        }
        for i in range(1200)
    ]
    (tmp_path / "q.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8"
    )
    rep = _run(tmp_path / "q.geojson", tmp_path)
    assert rep.analysis["coverage"]["time"]["start"].startswith("2022-01-01")
    assert rep.analysis["events"]["total"] == 1200


def test_units_row_thousands_and_empty_columns(tmp_path):
    rows = "\n".join(
        f'2021-{1 + i % 12:02d}-{1 + i // 12:02d},{20 + i % 5}.5,"{1000 + i:,}",,'
        for i in range(300)
    )
    (tmp_path / "m.csv").write_text("date,temp,rain,,\n,degC,mm,,\n" + rows, encoding="utf-8")
    rep = _run(tmp_path / "m.csv", tmp_path)
    a = rep.analysis
    assert a["summary"]["n_rows"] == 300 and a["summary"]["n_columns"] == 3
    assert a["statistics"]["rain"]["max"] == 1299
    assert any("degC" in n for n in a["notes"])


def test_single_numeric_column_csv_is_a_table(tmp_path):
    (tmp_path / "flux.csv").write_text(
        "flux\n" + "\n".join(str(v) for v in RNG.lognormal(0, 1, 50)), encoding="utf-8"
    )
    assert _run(tmp_path / "flux.csv", tmp_path).kind == "table"


def test_several_stations_side_by_side_are_not_events(tmp_path):
    days = pd.date_range("2015-01-01", periods=1500, freq="D")
    df = pd.concat(
        pd.DataFrame(
            {"date": days, "station": s, "lat": 30 + k, "lon": 35.0, "t": RNG.normal(20, 2, 1500)}
        )
        for k, s in enumerate(["A", "B", "C"])
    )
    df.to_csv(tmp_path / "st.csv", index=False)
    rep = _run(tmp_path / "st.csv", tmp_path)
    assert "events" not in rep.analysis or not rep.analysis["events"]
    assert rep.analysis["coverage"]["time"]["series_column"] == "station"
    assert "duplicate_times" not in _codes(rep)


def test_short_record_is_not_called_a_trend(tmp_path):
    days = pd.date_range("2022-01-01", periods=365)
    pd.DataFrame({"date": days, "t": 20 + 8 * np.sin(np.arange(365) / 58)}).to_csv(
        tmp_path / "y.csv", index=False
    )
    rep = _run(tmp_path / "y.csv", tmp_path)
    assert "short_record" in _codes(rep)
    assert rep.analysis["trends"][0]["text"].startswith("record too short")


def test_empty_header_only_and_truncated_files_explain_themselves(tmp_path, samples):
    (tmp_path / "e.csv").write_bytes(b"")
    (tmp_path / "h.csv").write_text("date,value\n", encoding="utf-8")
    nc = samples["netcdf4"][0].read_bytes()
    (tmp_path / "t.nc").write_bytes(nc[: len(nc) // 3])
    assert "empty_file" in _codes(_run(tmp_path / "e.csv", tmp_path))
    assert "no_rows" in _codes(_run(tmp_path / "h.csv", tmp_path))
    rep = _run(tmp_path / "t.nc", tmp_path)
    assert "unreadable" in _codes(rep)


def test_saturated_16bit_image_is_not_a_fill_value(tmp_path):
    arr = RNG.poisson(500, (64, 64)).astype(np.uint16)
    arr[10:12, 10:12] = 65535
    Image.fromarray(arr).save(tmp_path / "stars.png")
    assert "undeclared_fill" not in _codes(_run(tmp_path / "stars.png", tmp_path))


def test_excel_with_several_sheets_says_which_was_used(tmp_path):
    with pd.ExcelWriter(tmp_path / "w.xlsx") as xw:
        pd.DataFrame({"a": range(50), "b": range(50)}).to_excel(xw, sheet_name="big", index=False)
        pd.DataFrame({"c": range(5)}).to_excel(xw, sheet_name="small", index=False)
    rep = _run(tmp_path / "w.xlsx", tmp_path)
    assert any("'big'" in n and "'small'" in n for n in rep.analysis["notes"])


def test_zero_inflated_rain_does_not_produce_absurd_sigmas(tmp_path):
    days = pd.date_range("2010-01-01", periods=3650, freq="D")
    rain = np.where(RNG.random(3650) < 0.15, RNG.gamma(0.7, 6, 3650), 0.0)
    pd.DataFrame({"date": days, "rain": rain}).to_csv(tmp_path / "rain.csv", index=False)
    rep = _run(tmp_path / "rain.csv", tmp_path)
    zs = [abs(e["z"]) for t in rep.analysis["trends"] for e in t.get("extremes", [])]
    assert all(z < 30 for z in zs)


def test_csv_with_a_blank_line_after_every_row(tmp_path):
    # CSV text re-written in text mode on Windows ends every row with \r\r\n
    days = pd.date_range("2018-01-01", periods=400).strftime("%d/%m/%Y")
    text = "# station 7\r\nDatum;Temp\r\n" + "".join(
        f"{d};{i % 17},5\r\n" for i, d in enumerate(days)
    )
    (tmp_path / "w.csv").write_bytes(text.replace("\r\n", "\r\r\n").encode())
    rep = _run(tmp_path / "w.csv", tmp_path)
    assert rep.analysis["summary"]["n_rows"] == 400
    assert rep.analysis["coverage"]["time"]["column"] == "Datum"


def test_exactly_linear_series_is_a_significant_trend(tmp_path):
    days = pd.date_range("2020-01-01", periods=800)
    pd.DataFrame({"date": days, "count": range(800)}).to_csv(tmp_path / "c.csv", index=False)
    t = _run(tmp_path / "c.csv", tmp_path).analysis["trends"][0]
    assert t["mk_trend"] == "increasing" and t["mk_p"] < 0.001


def test_jpeg2000_is_named_and_explained_in_the_browser(monkeypatch):
    from nasa_explorer.analysis import quality
    from nasa_explorer.readers.unknown import detect_signature

    assert (
        detect_signature(b"\x00\x00\x00\x0cjP  \r\n\x87\n\x00\x00\x00\x14ftypjp2 ") == "JPEG 2000"
    )
    assert detect_signature(b"\xffO\xffQ\x00\x2f") == "JPEG 2000"
    monkeypatch.setattr(quality, "IN_BROWSER", True)
    (issue,) = quality.unreadable(5000, "JPEG 2000", ["GDAL: not a supported format"])
    assert issue["code"] == "unsupported_in_browser" and "gdal_translate" in issue["message"]


def test_infinite_values_are_missing_not_a_crash(tmp_path):
    rows = "\n".join(
        f"{'inf' if i % 5 == 0 else i},{'-inf' if i == 3 else i * 2}" for i in range(40)
    )
    (tmp_path / "inf.csv").write_text("a,b\n" + rows, encoding="utf-8")
    rep = process_file(tmp_path / "inf.csv", ReadOptions(), tmp_path / "r")
    assert rep.error is None
    assert rep.analysis["statistics"]["a"]["count"] == 32 and np.isfinite(
        rep.analysis["statistics"]["a"]["max"]
    )


def test_sidecar_files_are_not_reported_as_data(tmp_path):
    from nasa_explorer.pipeline import iter_inputs

    for name in ("a.tif", "a.tif.aux.xml", "a.tif.ovr", "e.grib2", "e.grib2.5b7b6.idx", "p.shp"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "p.dbf").write_bytes(b"x")
    (tmp_path / "lonely.dbf").write_bytes(b"x")  # no .shp: it is data on its own
    names = {p.name for p in iter_inputs(tmp_path)}
    assert names == {"a.tif", "e.grib2", "p.shp", "lonely.dbf"}


def test_archive_member_cannot_escape_into_a_sibling_folder(tmp_path):
    from nasa_explorer.archives import _safe_target

    (tmp_path / "run").mkdir()
    with pytest.raises(ValueError):
        _safe_target(tmp_path / "run", "../run_evil/x.csv")
    assert _safe_target(tmp_path / "run", "sub/x.csv").name == "x.csv"
