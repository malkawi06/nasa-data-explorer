"""Every supported format: detected, read, analysed and reported without errors."""

import json

import numpy as np
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


def _modis_tile(path, struct, dims, shape):
    sd_mod = pytest.importorskip("pyhdf.SD")
    sd = sd_mod.SD(str(path), sd_mod.SDC.WRITE | sd_mod.SDC.CREATE)
    sd.attr("StructMetadata.0").set(sd_mod.SDC.CHAR8, struct)
    sds = sd.create("NDVI", sd_mod.SDC.INT16, shape)  # no fill value set, like some products
    sds.dim(0).setname(dims[0])
    sds.dim(1).setname(dims[1])
    sds[:] = np.arange(shape[0] * shape[1]).reshape(shape).astype("int16")
    sds.endaccess()
    sd.end()


def _grid_struct(name, nx, ny, ul, lr, proj):
    return (
        f'GROUP=GridStructure\n\tGROUP=GRID_1\n\t\tGridName="{name}"\n\t\tXDim={nx}\n\t\tYDim={ny}\n'
        f"\t\tUpperLeftPointMtrs=({ul[0]:.6f},{ul[1]:.6f})\n\t\tLowerRightMtrs=({lr[0]:.6f},{lr[1]:.6f})\n"
        f"\t\tProjection={proj}\n\tEND_GROUP=GRID_1\nEND_GROUP=GridStructure\nEND"
    )


def test_modis_sinusoidal_tile_gets_its_bounding_box(tmp_path):
    pytest.importorskip("rioxarray")
    tile = 1111950.5196666666  # h21v05: the MODIS tile over Jordan and the Levant
    x0, y0 = -20015109.354 + 21 * tile, 10007554.677 - 5 * tile
    struct = _grid_struct("G", 60, 60, (x0, y0), (x0 + tile, y0 - tile), "GCTP_SNSOID")
    path = tmp_path / "MOD13A2.A2020001.h21v05.061.hdf"
    _modis_tile(path, struct, ("YDim:G", "XDim:G"), (60, 60))
    rep = process_file(path, ReadOptions(plots=False), tmp_path / "r")
    assert rep.reader == "hdf4" and rep.error is None
    west, south, east, north = rep.analysis["coverage"]["space"]["bbox"]
    assert (south, north) == (pytest.approx(30, abs=1e-3), pytest.approx(40, abs=1e-3))
    assert west == pytest.approx(34.64, abs=0.01) and east == pytest.approx(52.22, abs=0.01)


def test_modis_climate_grid_in_packed_degrees(tmp_path):
    struct = _grid_struct("CMG", 36, 18, (-180e6, 90e6), (180e6, -90e6), "GCTP_GEO")
    path = tmp_path / "MOD13C2.A2020001.061.hdf"
    _modis_tile(path, struct, ("YDim:CMG", "XDim:CMG"), (18, 36))
    rep = process_file(path, ReadOptions(plots=False), tmp_path / "r")
    assert rep.analysis["coverage"]["space"]["bbox"] == [-175.0, -85.0, 175.0, 85.0]  # cell centres
