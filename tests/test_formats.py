"""Every supported format: detected, read, analysed and reported without errors."""

import json

import pytest

from nasa_explorer.core import ReadOptions
from nasa_explorer.pipeline import process, process_file
from nasa_explorer.registry import read_file


def _cases(samples):
    return {k: v for k, v in samples.items() if v[1] != "archive"}


def test_every_format_reads_and_reports(samples, tmp_path):
    failures = []
    for name, (path, reader_name, kind) in _cases(samples).items():
        res, used, problems = read_file(path, ReadOptions())
        rep = process_file(path, ReadOptions(), tmp_path / "reports", label=name)
        ok = (
            used == reader_name
            and res.kind == kind
            and rep.error is None
            and rep.html_path.exists()
        )
        if not ok:
            failures.append(
                f"{name}: reader={used} (want {reader_name}) kind={res.kind} (want {kind}) "
                f"error={rep.error} problems={problems}"
            )
        data = json.loads(rep.json_path.read_text(encoding="utf-8"))
        assert data["analysis"] is not None
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "name", ["netcdf4", "netcdf3", "hdf5_xarray", "zarr", "grib", "hdf4", "geotiff", "fits_image"]
)
def test_gridded_have_map_and_stats(samples, tmp_path, name):
    if name not in samples:
        pytest.skip(f"{name} writer not installed")
    rep = process_file(samples[name][0], ReadOptions(), tmp_path)
    assert rep.analysis["statistics"]
    # elevation models (the GeoTIFF sample is dem.tif) show shaded relief instead of a plain map
    assert any(t.startswith(("Map", "Shaded relief")) for t, _, _ in rep.plots)


@pytest.mark.parametrize("name", ["csv", "tsv", "txt_delimited", "xlsx", "parquet", "json_records"])
def test_tables_detect_time_space(samples, tmp_path, name):
    rep = process_file(samples[name][0], ReadOptions(), tmp_path)
    cov = rep.analysis["coverage"]
    assert cov["time"]["resolution"] == "1 day"
    assert cov["space"]["bbox"][0] >= 35
    titles = [t for t, _, _ in rep.plots]
    assert "Scatter map" in titles and any(t.startswith("Time series") for t in titles)
    assert "Correlation heatmap" in titles


def test_archives_expand_every_member(samples, tmp_path):
    for name, want in (("zip", 2), ("gz", 1), ("targz", 2)):
        reps = process(samples[name][0], out_dir=tmp_path / name)
        assert len(reps) == want, name
        assert all(r.error is None for r in reps)
        assert (tmp_path / name / "index.html").exists()


def test_folder_run_writes_index(samples, tmp_path):
    folder = samples["csv"][0].parent
    reps = process(folder, out_dir=tmp_path / "all")
    index = (tmp_path / "all" / "index.html").read_text(encoding="utf-8")
    assert len(reps) >= len(samples) - 2
    assert "fires.csv" in index
    # shapefile sidecars are not reported separately
    assert not any(r.file.endswith((".dbf", ".shx", ".prj")) for r in reps)


def test_unknown_never_crashes(samples, tmp_path):
    rep = process_file(samples["unknown"][0], ReadOptions(), tmp_path)
    s = rep.analysis["summary"]
    assert s["signature"] == "unrecognised" and s["first_bytes_hex"].startswith("00 01 02")


def test_corrupt_file_with_known_extension(tmp_path):
    bad = tmp_path / "broken.nc"
    bad.write_bytes(b"CDF\x01" + b"\xff" * 50)
    rep = process_file(bad, ReadOptions(), tmp_path / "r")
    assert rep.reader == "unknown" and rep.html_path.exists()


def test_json_summary_is_strict_json(samples, tmp_path):
    """No NaN/Infinity tokens: the website and other tools parse these files strictly."""
    rep = process_file(samples["netcdf_packed"][0], ReadOptions(), tmp_path)
    text = rep.json_path.read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
