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
            _offline_routes(ctx)
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
                ".file-item",
                "cs => cs.map(c => [c.querySelector('.fi-name').textContent, "
                "c.querySelector('.fi-meta').textContent, !c.querySelector('.fi-state.bad')])",
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


def _offline_routes(ctx) -> None:
    """With PYODIDE_DIR set (an extracted Pyodide release), serve Pyodide from disk and fetch
    PyPI wheels through Python, so the test runs where the CDN is blocked."""
    root = os.environ.get("PYODIDE_DIR")
    if not root:
        return
    import mimetypes
    from pathlib import Path

    import requests

    def local(route):
        name = route.request.url.split("/full/")[1].split("?")[0]
        path = Path(root) / name
        if not path.exists():
            return route.fulfill(status=404)
        kind = (
            "text/javascript" if name.endswith((".mjs", ".js")) else mimetypes.guess_type(name)[0]
        )
        route.fulfill(
            status=200, body=path.read_bytes(), content_type=kind or "application/octet-stream"
        )

    def remote(route):
        resp = requests.get(route.request.url, timeout=60)
        route.fulfill(
            status=resp.status_code,
            body=resp.content,
            headers={
                "content-type": resp.headers.get("content-type", "application/octet-stream"),
                "access-control-allow-origin": "*",
            },
        )

    ctx.route("https://cdn.jsdelivr.net/pyodide/**", local)
    ctx.route(lambda u: u.startswith("https://") and "cdn.jsdelivr.net/pyodide" not in u, remote)
