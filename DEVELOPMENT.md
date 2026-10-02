# nasa-data-explorer

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
- `ai.py`: providers, cache, backoff and prompts. `qa.py`: retrieval (sentence-transformers or TF-IDF).

## Gotchas
- **eccodes vs GDAL/cartopy**: always call `_native.preload_before_eccodes()` before importing
  eccodes/cfgrib. Otherwise you get a segfault or an abort at exit.
- NetCDF-4 files carry the HDF5 signature; the netcdf reader claims it for `.nc`, the hdf5 reader for magic-only.
- HDF4 uses `scale * (raw - offset)` (MODIS), CF uses `raw * scale + offset`. See `readers/_cf.py`.
- Tabular analysis treats -9999/-999/... as missing (NASA POWER, FIRMS conventions) and logs it in notes.
- Report f-strings must stay Python 3.10 compatible (no nested same-type quotes).
- Tests force `NASA_EXPLORER_EMBEDDINGS=tfidf` and clear AI keys; AI is tested with a fake provider.
