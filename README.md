# nasa-data-explorer

Drop in almost any scientific data file and understand it in minutes. Built as a
generic helper for the **NASA Space Apps Challenge 2026**, not as a solution to one
challenge.

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
| Geospatial | geotiff | .tif .tiff .cog .jp2 .img .vrt | TIFF | Dataset (rioxarray) |
| Geospatial | vector | .shp .geojson .kml .kmz .gpkg .gml .fgb | GeoJSON, KML, SQLite | GeoDataFrame |
| Astronomy | fits | .fits .fit .fts | `SIMPLE  =` | Dataset (images, multi-extension) or DataFrame (tables) |
| Images | image | .png .jpg .jpeg .tif .gif .bmp .webp | PNG, JPEG, GIF, BMP, TIFF | Dataset (y, x, band) |
| Documents | pdf | .pdf | `%PDF` | text, title, authors, sections, tables, captions, OCR fallback |
| Documents | docx / html / markdown / text | .docx .html .md .txt | HTML / text | the same document fields |
| Archives | – | .zip .tar .tar.gz .tgz .gz .bz2 .xz | ZIP, gzip, bzip2, xz, tar | each member is processed |
| Anything else | unknown | – | – | size, signature and first bytes (it never crashes) |

Document readers also list the NASA missions and datasets they mention (MODIS, VIIRS, GPM, GRACE,
SMAP, ICESat-2, MERRA-2, FIRMS, and others), with page numbers.

## AI layer (optional)

The AI layer is configured with environment variables or a `.env` file (gitignored; see
`.env.example`):

| `AI_PROVIDER` | Needs | Default model |
|---|---|---|
| `ollama` | Ollama running locally (free, offline) | `qwen2.5:7b`, or any model you have pulled |
| `gemini` | `GEMINI_API_KEY` (free tier) | `gemini-2.5-flash` |
| `groq` | `GROQ_API_KEY` (free tier) | `llama-3.3-70b-versatile` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |

- If `AI_PROVIDER` is unset, the first configured provider in the table order is used.
  `AI_MODEL` overrides the model.
- A 429 response is retried with exponential backoff, honouring `Retry-After`. Before each call
  the tool prints the estimated token count, and responses are cached on disk in
  `~/.cache/nasa_explorer/ai`, so the same input is never sent twice.
- For data files, only the computed summary and statistics are sent, never the raw file.
- Papers are split into page-tagged chunks and summarised map-reduce style. Every claim
  cites `(p. N)`.
- `--ask` retrieves passages with sentence-transformers when it is available, or with a built-in
  TF-IDF index (works offline, no install). With a provider it writes an answer citing
  `[file, p. N]`. Without one it lists the best passages.
- Without any provider, everything except the AI text works.

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
pytest -q                     # synthetic samples for every format; real NASA samples skip offline
ruff check . && ruff format --check .
```

### Known native-library quirk

The eccodes wheel (GRIB) bundles C libraries that clash with GDAL and cartopy if those load
after it. `nasa_explorer/_native.py` loads them first. If you write your own GRIB scripts,
import `pyogrio` and `cartopy.crs` before `eccodes`.

## License

MIT
