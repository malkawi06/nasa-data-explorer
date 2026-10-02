"""End-to-end test of the website on the REAL Pyodide runtime (downloads ~40 MB+).

Opt-in: set E2E_PYODIDE=1 (the CI 'pyodide-e2e' job does). Verifies that every
browser-supported format is read by the expected reader inside Pyodide.
"""

import os
import shutil

import pytest

pw = pytest.importorskip("playwright.sync_api")

from .harness import Server, chromium_path  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("E2E_PYODIDE") != "1", reason="set E2E_PYODIDE=1 to run"
)

EXPECTED = {  # sample key -> reader expected inside the browser
    "csv": "delimited-text",
    "xlsx": "excel",
    "json_nested": "json",
    "parquet": "parquet",
    "netcdf4": "netcdf",
    "netcdf3": "netcdf",
    "hdf5_tree": "hdf5",
    "geotiff": "geotiff",
    "geojson": "vector",
    "kml": "vector",
    "fits_image": "fits",
    "png": "image",
    "pdf": "pdf-lite",
    "docx": "docx",
    "html": "html",
    "markdown": "markdown",
}


def test_formats_in_real_pyodide(samples, tmp_path):
    server = Server(tmp_path / "work")
    files = [str(samples[k][0]) for k in EXPECTED if k in samples]
    shp = samples.get("shapefile")
    if shp:  # a shapefile needs all of its parts in one drop
        files += [str(p) for p in shp[0].parent.glob("points.*")]
    try:
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium_path())
            ctx = browser.new_context()
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{server.url}/index.html")
            page.wait_for_selector(".dot.ready, .dot.error", timeout=420_000)
            status = page.inner_text("#status-text")
            assert page.locator(".dot.ready").count(), f"Pyodide failed to start: {status}"
            page.set_input_files("#file-input", files)
            n_cards = sum(k in samples for k in EXPECTED) + (1 if shp else 0)
            page.wait_for_function(
                f"document.querySelectorAll('.file-item:not(.pending)').length >= {n_cards}",
                timeout=600_000,
            )
            cards = page.eval_on_selector_all(
                ".card",
                "cs => cs.map(c => [c.querySelector('.file').textContent, "
                "c.querySelector('.meta').textContent, c.querySelector('.err').hidden])",
            )
            browser.close()
    finally:
        server.close()
        shutil.rmtree(tmp_path / "work", ignore_errors=True)
    by_file = {name: (meta.split(" · ")[0], ok) for name, meta, ok in cards}
    wrong = []
    for key, reader in EXPECTED.items():
        if key not in samples:
            continue
        name = samples[key][0].name
        got = by_file.get(name)
        if got is None or got[0] != reader or not got[1]:
            wrong.append(f"{name}: got {got}, want {reader}")
    if shp:
        assert by_file.get("points.shp", ("", False))[0] == "vector"
    assert not wrong, "\n".join(wrong)
    assert not errors, errors
