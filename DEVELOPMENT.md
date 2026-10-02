# nasa-data-explorer: development notes

Generic "drop any NASA/scientific file in, get a report" tool for Space Apps hackathons.

## Commands
- Setup: `uv venv .venv && . .venv/bin/activate && uv pip install -e ".[all,dev]"` (+ `tesseract` binary for OCR)
- Test: `pytest -q` (network tests auto-skip offline) · Lint: `ruff check . && ruff format --check .`
- Run: `nasa-explore PATH [--var --bbox W,S,E,N --start --end --ai --ask Q --lang ar]`

## Structure
- `nasa_explorer/registry.py`: `@reader` decorator and detection (extension that agrees with the
  signature → magic/sniff → remaining extension matches, sorted by `priority`). Missing deps
  disable just that reader.
- `nasa_explorer/readers/*.py`: one file per format family, auto-imported (modules starting
  with `_` are helpers). Each reader returns `core.ReadResult(kind, data, metadata, pages)`.
- `analysis/`: one analyzer per kind (grid / table / document / image / tree / binary).
  Each returns `(json-able dict, [(title, png)])`.
- `pipeline.py`: file, archive and folder handling, report/JSON/chunk writing, index.html.
- `report.py`: HTML rendering. `plots.py`: matplotlib with the fixed CVD-safe palette.
- `ai.py`: providers, cache and backoff, plus the AI *workflows*. These are generators that
  yield `Step` prompts and receive replies, driven by `run_workflow` (CLI) or by `web/bridge.py`
  (browser BYOK). `verify.py` flattens the analysis into fact paths and labels claims.
  `ai_view.py` renders the AI panel, quality list and product cards.
- `analysis/quality.py`: rule-based checks, run for every file. `products.py`: the NASA product
  catalog.
- `bodies.py`: which planetary body a CRS/label describes. `analysis/terrain.py`: DEM slope,
  roughness, depressions and the per-baseline profile. `analog.py`: Moon/Mars analog targets,
  factor scoring, built-in sites, terrain comparison. `power.py`: NASA POWER URLs and downloads
  (the browser fetches the same URLs itself).
- `qa.py`: retrieval (sentence-transformers or TF-IDF).
- `evals/`: AI eval cases with known truths (`python -m evals.run --provider X`).
  `tests/e2e/`: browser tests. The harness stubs Pyodide and proxies bridge calls to CPython.
- `web/`: the static site (GitHub Pages via `.github/workflows/pages.yml`; `netlify.toml` also works).
  - Pyodide runs in a Web Worker (`worker.js`), so the page never freezes. `app.js` is the UI:
    a file list on the side and the selected report in the main area. The report is rendered into
    a shadow root, so its CSS stays scoped and the page scrolls normally.
  - `bridge.py` is the only JS↔Python surface (JSON strings in and out). `build.py` zips the
    package into `web/dist/`.
  - The Pyodide version, the package batches and `MODULE_SOURCES` live at the top of `worker.js`.

## Gotchas
- **eccodes vs GDAL/cartopy**: always call `_native.preload_before_eccodes()` before importing
  eccodes/cfgrib. Otherwise you get a segfault or an abort at exit.
- NetCDF-4 files carry the HDF5 signature; the netcdf reader claims it for `.nc`, the hdf5 reader for magic-only.
- HDF4 uses `scale * (raw - offset)` (MODIS), CF uses `raw * scale + offset`. See `readers/_cf.py`.
- Tabular analysis treats -9999/-999/... as missing (NASA POWER, FIRMS conventions) and logs it in notes.
- Report f-strings must stay Python 3.10 compatible (no nested same-type quotes).
- **Pyodide fatal crashes cannot be caught.** The `IN_BROWSER` flag in `core.py` routes around them:
  - The netCDF4 engine overflows the JS stack, so the browser uses h5netcdf and scipy only.
  - fiona crashes on GeoPackage and has no KML driver, so the browser uses
    `readers/_purevector.py`.
  - `pd.read_parquet` fails when pyarrow loads after pandas, so the reader uses
    `pyarrow.parquet` directly.

  Verify browser changes on real Pyodide with `E2E_PYODIDE=1`, not only the stubbed UI test.
- The browser build has no pyogrio, pymupdf, eccodes or pyhdf. Readers must keep heavy imports
  inside the function and declare them in `requires`, so the registry disables them cleanly.
  The page then loads the missing modules on demand and retries (`MODULE_SOURCES` in `worker.js`).
- JSON output must be strict, with no NaN or Infinity (`report.dumps` turns them into null).
  Browsers' `JSON.parse` rejects them.
- Playwright: use `context.route`, not `page.route`, or requests made from the worker are not
  intercepted.
- Always pass `encoding="utf-8"` to file reads and writes: reports contain → × and Arabic, and
  Windows defaults to cp1252.
- The AI cache key includes the provider *model* name. Scripted FakeProvider runs in one test
  share a cache dir, so give each run its own `model`.
- AI prompts ask for JSON. Keep keys stable (`findings[].fact/value`, `page`/`quote`),
  because `verify.py`, `ai_view.py` and `evals/score.py` depend on them.
- Sending file content to a provider is opt-in only (`AI_SEND_IMAGES=1` / `--ai-images`, the web
  "Send images" box) and only for `ai.VISION` providers. `Step.image` carries it; keep the default
  path summary-only.
- Never reproject Moon/Mars rasters to EPSG:4326: bounds go to the body's own geodetic CRS
  (`grid._geographic_bounds`). PROJ refuses or silently mixes bodies otherwise.
- Terrain numbers depend on pixel size: always report the baseline and compare DEMs only via
  `by_baseline_m` (same baseline).
- `tests/data/power/*.json` are real POWER climatology responses (2001-2020); the analog ranking
  tests rely on them, so do not replace them with synthetic data.
- Tests force `NASA_EXPLORER_EMBEDDINGS=tfidf` and clear AI keys; AI is tested with a fake provider.
