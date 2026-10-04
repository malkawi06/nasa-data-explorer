# nasa-data-explorer

Drop in almost any scientific data file and understand it in minutes. It covers Earth science,
Moon and Mars data, NASA POWER climate and research papers, with optional AI explanations whose
numbers are checked against the data. It runs in the browser:
https://malkawi06.github.io/nasa-data-explorer/

For every file it detects the format (by extension, then by magic bytes), reads it,
and writes an HTML report plus a JSON summary with:

- **Summary**: variables/columns, dimensions, dtypes, units, `long_name`, missing %, and
  fill values decoded (`_FillValue`, `scale_factor`, `add_offset`, `valid_range`).
- **Coverage**: time range and resolution, plus the spatial bounding box and resolution
  (from lat/lon or a CRS).
- **Statistics**: min/max/mean/std and the 5/25/50/75/95th percentiles.
- **Quick plots**: a map of the first time step (with cartopy coastlines when available), an
  area-mean time series, a scatter map, histograms, a correlation heatmap and colour histograms.
- **Trends**: the linear slope per year plus a Mann-Kendall test, stated in plain words
  ("increasing, significant at 99% (p<0.001); slope +0.61 K/year").
- **AI layer (optional)**: plain-language explanations, structured paper summaries with
  page citations, and Q&A over a folder.

## Install

```bash
git clone https://github.com/malkawi06/nasa-data-explorer && cd nasa-data-explorer
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[all]"            # everything
pip install -e ".[geo,docs]"       # or pick only the extras you need
```

| Extra | Enables |
|---|---|
| core (no extra) | CSV/TSV/TXT, Excel, JSON, Parquet, NetCDF, HDF5, Zarr, PNG/JPG/TIFF, Markdown/TXT |
| `[geo]` | GeoTIFF/COG, Shapefile, GeoJSON, KML/KMZ, GeoPackage, cartopy maps |
| `[hdf4]` | HDF4 / HDF-EOS2 (MODIS) |
| `[grib]` | GRIB 1/2 |
| `[astro]` | FITS |
| `[docs]` | PDF (text, tables, OCR), DOCX, HTML |
| `[ai]` | Anthropic and Gemini SDKs, plus sentence-transformers embeddings |
| `[app]` | Streamlit drag-and-drop app |

If a library is missing, only that reader is disabled, and the report tells you which
`pip install` command fixes it. OCR for scanned PDFs also needs the `tesseract` binary
(`apt install tesseract-ocr`, `brew install tesseract`, or the Windows installer from UB Mannheim).

## CLI

```bash
nasa-explore data/                                   # every file in a folder -> reports/index.html
nasa-explore MOD11A2.hdf --var LST_Day_1km
nasa-explore era5.grib --bbox 34,29,40,34 --start 2020-01-01 --end 2020-12-31
nasa-explore paper.pdf --ai --lang ar                 # Arabic report + AI paper summary
nasa-explore data/ --ask "Which datasets measure soil moisture?"
nasa-explore LDEM_80S_20M.IMG                         # lunar DEM: slopes, flat ground, craters
nasa-explore --power 29.57,35.42                      # NASA POWER daily data for a point (last 10 years)
nasa-explore --formats                                # what's enabled in this environment
nasa-explore --app                                    # Streamlit drag-and-drop UI
```

Options: `--out DIR` (default `reports/`), `--no-plots`, `-v`.

## Jupyter / Colab

```python
from nasa_explorer import explore

rep = explore("file.nc")  # renders the report inline
rep = explore("file.nc", var="t2m", bbox=(35, 29, 40, 34), start="2020-01-01")
rep.analysis["trends"]  # the same data as the JSON summary
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb). It also runs in Google Colab.

## Supported formats

| Category | Reader | Extensions | Detected by signature | Read as |
|---|---|---|---|---|
| Tabular | delimited-text | .csv .tsv .txt .dat .tab .asc | consistent delimiter (skips free-text headers such as NASA POWER's) | DataFrame |
| Tabular | excel | .xlsx .xlsm .xls | OLE header | DataFrame (largest sheet) |
| Tabular | json | .json | `{` / `[` | DataFrame (largest table inside the JSON) |
| Tabular | parquet | .parquet .pq | `PAR1` | DataFrame |
| Earth science | netcdf | .nc .nc4 .cdf | `CDF\x01/\x02/\x05`, HDF5 | xarray.Dataset |
| Earth science | hdf5 | .h5 .he5 .hdf5 | HDF5 | Dataset, or a walk of the h5py group tree (e.g. NISAR) |
| Earth science | hdf4 | .hdf .hdf4 .h4 | `\x0e\x03\x13\x01` | Dataset (MODIS calibration applied) |
| Earth science | grib | .grib .grb .grib2 .grb2 | `GRIB` | Dataset (cfgrib) |
| Earth science | zarr | .zarr, or a folder with `.zgroup` / `zarr.json` | – | Dataset |
| Geospatial | geotiff | .tif .tiff .cog .jp2 .img .vrt .hgt .bil .dem .grd .asc | TIFF, ESRI ASCII grid | Dataset (rioxarray) |
| Planetary | planetary-raster | .lbl .img .cub .xml .vic | PDS3 / PDS4 / ISIS3 / VICAR labels | Dataset on the right body (Moon, Mars, ...) |
| Geospatial | vector | .shp .geojson .kml .kmz .gpkg .gml .fgb | GeoJSON, KML, SQLite | GeoDataFrame |
| Astronomy | fits | .fits .fit .fts | `SIMPLE  =` | Dataset (images, multi-extension) or DataFrame (tables) |
| Images | image | .png .jpg .jpeg .tif .gif .bmp .webp | PNG, JPEG, GIF, BMP, TIFF | Dataset (y, x, band) |
| Documents | pdf | .pdf | `%PDF` | text, title, authors, sections, tables, captions, OCR fallback |
| Documents | docx / html / markdown / text | .docx .html .md .txt | HTML / text | the same document fields |
| Archives | – | .zip .tar .tar.gz .tgz .gz .bz2 .xz | ZIP, gzip, bzip2, xz, tar | each member is processed |
| Anything else | unknown | – | – | size, signature and first bytes (it never crashes) |

Document readers also list the NASA missions and datasets they mention (MODIS, VIIRS, GPM, GRACE,
SMAP, ICESat-2, MERRA-2, FIRMS, and others), with page numbers.

## Elevation models and Moon / Mars data


**Terrain of any elevation model** (SRTM, Copernicus DEM, LOLA, MOLA, HiRISE DTMs, ...), found
by file or variable name:
- Slope, roughness (TRI) and the share of flat ground (<5°, <10°, <15°, <25°), in metres on the
  right body: a degree is 111 km on Earth but 30 km on the Moon.
- Closed, crater-like depressions (sink filling, so valleys that drain do not count): number,
  density per 1000 km², diameter, depth and depth/diameter.
- Shaded relief with the depressions circled, a slope map and the slope distribution.
- A profile at fixed baselines (10 m ... 10 km). Slopes depend on pixel size, so DEMs are only
  compared at the same baseline.

**Planetary data**: PDS3 (.IMG with attached or detached .LBL), PDS4 (.xml + data), ISIS3 cubes
and VICAR are read through GDAL. The body comes from the CRS or the label; maps stay in lunar /
Martian coordinates; LOLA/MOLA radii are turned into heights above the reference sphere.

NASA POWER daily data for any point: `nasa-explore --power LAT,LON` (command line).

## AI layer (optional)

The AI layer is configured with environment variables or a `.env` file (gitignored; see
`.env.example`):

| `AI_PROVIDER` | Needs | Default model |
|---|---|---|
| `ollama` | Ollama running locally (free, offline) | `qwen2.5:7b`, or any model you have pulled |
| `gemini` | `GEMINI_API_KEY` (free tier) | `gemini-3.8-flash` |
| `groq` | `GROQ_API_KEY` (free tier) | `llama-3.3-70b-versatile` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |

- If `AI_PROVIDER` is unset, the first configured provider in the table order is used.
  `AI_MODEL` overrides the model.
- A 429 response is retried with exponential backoff, honouring `Retry-After`. Before each call
  the tool prints the estimated token count, and responses are cached on disk in
  `~/.cache/nasa_explorer/ai`, so the same input is never sent twice.
- For data files, only the computed summary and statistics are sent, never the raw file.
- **Images (opt-in):** `--ai-images` (or `AI_SEND_IMAGES=1`) also sends a downscaled 1024 px
  JPEG of a plain image to Gemini or Anthropic, so the model can see it. Without it the model
  sees only the measurements below.

### Images
- Every plain image gets **measurements computed by code**: brightness, contrast, clipping,
  saturation, colourfulness, sharpness, edge density, dominant colours, and the share of
  white unsaturated pixels (cloud, snow or background), green-dominant and blue-dominant pixels.
  EXIF capture time and GPS position are read when present.
- With vision on, each thing the model says it sees is listed under *What the AI sees*, with a
  numbered box drawn on the image. These are badged 👁 *visual, not machine-checked*. Numbers
  must still cite a measurement (`image.white_low_saturation_pct`) and are verified.

### Verified AI output
- **Its own place in every report:** the AI analysis is a highlighted panel at the top. Without
  AI it shows how to run it.
- **Structured answers:** the model returns JSON. For data files that is overview, key findings,
  issues, next analyses, visualizations, hackathon ideas and caveats. For papers it is problem,
  method, data used, findings, limitations, NASA datasets and ideas.
- **Every number is traced (not proven true):**
  - For data files, each finding must cite a *fact path* such as `statistics.t2m.mean`, and the
    value is compared with what the tool computed.
  - For papers, each claim must cite a page, and its numbers, quote or dataset name must appear
    on that page.
  - Each finding gets a badge: ✓ verified, ✗ wrong, ? not verified, or · no number.
  - "Verified" means consistent with what the tool computed, not that the computation or the
    sentence's wording is right. A number is only matched against the cited fact and its own
    variable, plus dates and counts. A finding that cites nothing and names no variable stays
    unverified. In the tests, findings with planted wrong numbers must come out "verified" in under 5%
    of cases and the correct ones in over 85%
    (`tests/test_quality_ai.py::test_planted_wrong_numbers_are_rarely_verified`).
  - File text sent to the model is marked as data, not instructions. This lowers the risk of
    prompt injection from a paper but does not remove it.
- **Review round:** if any claim fails, the model gets one round to fix it, with the exact
  problems listed. Claims it cannot support stay flagged and are never hidden.
- **Trusted context:** the model also receives the automatic quality checks and the recognised
  NASA product notes, so its advice reflects real caveats.
- **One implementation:** the workflows (`ai.data_workflow`, `ai.paper_workflow`) are generators.
  The CLI and the website run the same code; only the HTTP call differs.

### Quality checks and product cards (no AI needed)
- **Rule-based checks:**
  - Undeclared fill values (`-9999`, `9.96e36`), and NDVI-style values with no scale factor applied.
  - Impossible ranges for K/°C/% and negative rainfall.
  - Constant or mostly-missing variables, and outliers.
  - Duplicate or missing time steps, and 0-360 longitudes.
- **Product cards:** the tool recognises 30 well-known products (among them MODIS LST/NDVI, FIRMS, NASA POWER, IMERG,
  MERRA-2, GRACE, SMAP, ICESat-2, GEDI, Landsat C2, HLS, OCO-2, TEMPO, NISAR, Black Marble,
  GISTEMP, LOLA, MOLA, HiRISE and SRTM). Each card lists the resolution, the caveats people usually miss (for example the
  Landsat C2 scale factors, or FIRMS confidence codes) and Earthdata/Worldview links.

### Trends done right
- **Seasonality:** a strong annual cycle is removed first, and monthly series use the
  **Seasonal Mann-Kendall** test.
- **Autocorrelation:** positively autocorrelated series switch to the **Hamed-Rao** corrected
  test. Plain Mann-Kendall would call red noise "significant".
- **Gridded data:** the report adds a per-pixel **trend map**, with dots where p<0.05, and the
  **mean annual cycle**.

- `--ask` retrieves passages with sentence-transformers when it is available, or with a built-in
  TF-IDF index (works offline, no install). With a provider it writes an answer citing
  `[file, p. N]`. Without one it lists the best passages.
- Without any provider, everything except the AI text works.

## Web app (GitHub Pages, runs in the browser)

`web/` is a static site that runs the same `nasa_explorer` package in the visitor's browser with
[Pyodide](https://pyodide.org). There is no server: files are never uploaded, and hosting is free.

**Deploy (GitHub Pages):**
1. Repo *Settings → Pages → Build and deployment → Source*: choose **GitHub Actions** (once).
2. Every push to `main` runs `.github/workflows/pages.yml`: `python web/build.py`, then publishes
   `web/dist` to `https://<user>.github.io/nasa-data-explorer/`.

Any static host works the same way (`netlify.toml` is kept for Netlify).

**Run it locally:**
```bash
python web/build.py
python -m http.server -d web/dist 8000
```
Then open http://localhost:8000.

**What works in the browser:**
- CSV/TSV/TXT, Excel, JSON, Parquet, NetCDF, HDF5, Zarr, GeoTIFF, Shapefile/GeoJSON/KML/GPKG, FITS,
  images, PDF (text through pypdf), DOCX, HTML, Markdown, and ZIP/TAR/GZ archives.
- Extra libraries load the first time a format needs them.

**AI on the website (bring your own key):**
- Open *AI settings* and pick a provider:
  - Gemini or Groq: both have free tiers.
  - Anthropic.
  - Ollama on your own machine: start it with `OLLAMA_ORIGINS=https://<user>.github.io`.
- The key stays in your browser. It is kept for the session only, unless you tick "remember",
  and it is sent only to that provider. There is no server.
- *Send images* (off by default) lets Gemini or Claude see a downscaled copy of an image file.
- Each report card has three tabs:
  - **Report**
  - **🤖 AI analysis**: verified findings with badges. The downloadable report includes them.
  - **Ask this file**: a chat that cites fact paths or pages.

**Desktop CLI only:**
- GRIB, HDF4, OCR of scanned PDFs, and PDF table extraction. These need native libraries
  Pyodide does not have.
- Files larger than about 600 MB, because of the browser's memory limits.
- The AI layer, so that API keys never sit in a public page.

The first visit downloads about 50 MB (Python, about 12 MB, plus the scientific stack). After that the browser
caches it.

## Adding a new reader

Create one file in `nasa_explorer/readers/`. It is imported automatically:

```python
# nasa_explorer/readers/envi.py
from pathlib import Path
from ..core import ReadOptions, ReadResult
from ..registry import reader


@reader(
    "envi",
    category="Geospatial",
    extensions=(".hdr",),
    magic=(b"ENVI",),
    requires=("rasterio",),
    extra="geo",
)
def read_envi(path: Path, opts: ReadOptions) -> ReadResult:
    import rioxarray

    ds = rioxarray.open_rasterio(path.with_suffix("")).to_dataset(name="band_data")
    return ReadResult("grid", ds, {"crs": str(ds.rio.crs)})
```

Return `ReadResult("grid", xarray.Dataset)`, `("table", DataFrame)`,
`("document", pages=[...])` or `("image", ...)`, and the analysis, plots and report all work
unchanged. Raise `NotThisFormat` to let the next candidate reader try the file.

## Hackathon-day checklist

1. **Before the day**:
   - Run `pip install -e ".[all]"` and `nasa-explore --formats` (everything should say `ok`).
   - Run `pytest -q`.
   - Pull a local model (`ollama pull qwen2.5:7b`) or put a free Gemini or Groq key in `.env`.
2. **Get the data**: download the challenge's files into `data/`. Archives can stay zipped.
3. **First pass**: run `nasa-explore data/` and open `reports/index.html`. Scan formats, time
   ranges, bounding boxes and missing %.
4. **Focus**: run `nasa-explore data/file.nc --var X --bbox W,S,E,N --start ... --end ...` for your
   region and period.
5. **Read the papers**: run `nasa-explore papers/ --ai`, then
   `nasa-explore papers/ --ask "what data did they use for ...?"`.
6. **Sanity checks**: read the *Notes* section (fill values, sampling, skipped variables) and the
   AI's "suspicious" list.
7. **Reuse**: the `reports/*.json` files hold the numbers for your slides or app. The plot PNGs
   are saved under `reports/<file>_plots/`.
8. **Arabic audience?** Add `--lang ar`.

## Development

```bash
pip install -e ".[all,dev]"
python -m playwright install chromium   # for the website tests
pytest -q                     # synthetic samples for every format; real NASA samples skip offline
ruff check . && ruff format --check .
E2E_PYODIDE=1 pytest tests/e2e/test_web_pyodide.py   # website on the real Pyodide runtime
```

**CI** (`.github/workflows/ci.yml`) has two parts:
- Lint and tests on Linux (Python 3.11 and 3.12) and on Windows. These runs include the real
  NASA samples.
- A job that runs the website on real Pyodide in Chromium.

### AI eval set
`evals/` holds files whose truth is known:
- a warming grid
- a seasonal station series
- autocorrelated noise with no real trend
- a grid with a planted `-9999` fill value
- a short paper

Use it to compare models and prompts:

```bash
python -m evals.run --provider gemini              # or groq / anthropic / ollama, --model ..., --lang ar
```

Each case is scored on:
- the share of numbers verified
- whether the expected facts are cited
- whether the expected points are mentioned (season, autocorrelation, fill value…)
- whether any wrong conclusion appears

Results are saved in `evals/results/`.

### Known native-library quirk

The eccodes wheel (GRIB) bundles C libraries that clash with GDAL and cartopy if those load
after it. `nasa_explorer/_native.py` loads them first. If you write your own GRIB scripts,
import `pyogrio` and `cartopy.crs` before `eccodes`.

## License

MIT
