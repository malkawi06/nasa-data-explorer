"""Analysis for documents, plain images, HDF5 trees and unreadable binaries."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .. import plots
from ..core import ReadOptions, ReadResult
from ..readers._cf import clean_attrs, decode
from .stats import MAX_SAMPLE, numeric_stats

STOP = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "from",
        "at",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "their",
        "they",
        "which",
        "using",
        "used",
        "use",
        "can",
        "may",
        "also",
        "than",
        "into",
        "between",
        "over",
        "under",
        "both",
        "such",
        "not",
        "but",
        "all",
        "each",
        "more",
        "most",
        "other",
        "some",
        "there",
        "here",
        "has",
        "have",
        "had",
        "will",
        "would",
        "should",
        "could",
        "et",
        "al",
        "fig",
        "figure",
        "table",
    ]
)


def analyze_document(res: ReadResult, opts: ReadOptions):
    m = res.metadata
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", " ".join(res.pages).lower())
    keywords = [w for w, _ in Counter(w for w in words if w not in STOP).most_common(25)]
    summary = {
        "title": m.get("title", ""),
        "authors": m.get("authors", []),
        "pages": m.get("n_pages", len(res.pages)),
        "words": m.get("n_words", 0),
        "sections": m.get("sections", []),
        "figure_captions": m.get("figure_captions", []),
        "table_captions": m.get("table_captions", []),
        "tables": m.get("tables", [])[:10],
        "nasa_mentions": m.get("nasa_mentions", {}),
        "keywords": keywords,
        "ocr": m.get("ocr", False),
    }
    notes = list(m.get("notes", []))
    if m.get("page_note"):
        notes.append(m["page_note"])
    preview = res.pages[0][:1500] if res.pages else ""
    return {
        "summary": summary,
        "coverage": {},
        "statistics": {},
        "trends": [],
        "notes": notes,
        "preview": preview,
    }, []


def analyze_image(res: ReadResult, opts: ReadOptions):
    da = res.data["image"]
    arr = da.values
    bands = [str(b) for b in da["band"].values]
    stats = {b: numeric_stats(arr[..., i]) for i, b in enumerate(bands)}
    summary = {k: v for k, v in res.metadata.items() if k != "exif"}
    summary["bands"] = bands
    summary["dtype"] = str(arr.dtype)
    if "exif" in res.metadata:
        summary["exif"] = res.metadata["exif"]
    figs = []
    if opts.plots:
        figs.append(("Image", plots.thumbnail(arr, "Image preview")))
        figs.append(("Color histogram", plots.color_histogram(arr, bands, "Color histogram")))
    return {
        "summary": summary,
        "coverage": {},
        "statistics": stats,
        "trends": [],
        "notes": [],
    }, figs


def _sample_h5(ds) -> np.ndarray:
    if ds.shape == ():
        return np.asarray(ds[()])
    step = max(1, math.ceil((ds.size / MAX_SAMPLE) ** (1 / ds.ndim)))
    return ds[tuple(slice(None, None, step) for _ in ds.shape)]


def analyze_tree(res: ReadResult, opts: ReadOptions):
    import h5py

    tree = res.data
    stats, notes, figs = {}, [], []
    if tree.empty:
        return {
            "summary": {"datasets": []},
            "coverage": {},
            "statistics": {},
            "trends": [],
            "notes": ["no datasets"],
        }, []
    numeric = tree[tree["dtype"].str.match(r"^(u?int|float)\d+")]
    if opts.var:
        numeric = numeric[numeric["path"].str.endswith("/" + opts.var.lstrip("/"))]
        if numeric.empty:
            raise KeyError(f"--var {opts.var!r} matches no dataset path")
    best = None
    with h5py.File(res.metadata["h5_path"], "r") as f:
        for _, row in numeric.head(200).iterrows():
            ds = f[row["path"]]
            vals = decode(np.asarray(_sample_h5(ds)), dict(ds.attrs))
            st = numeric_stats(vals)
            if vals.size < ds.size:
                st["sampled"] = f"{vals.size:,} of {ds.size:,} values"
            stats[row["path"]] = st
            if ds.ndim >= 2 and min(ds.shape[-2:]) > 1 and (best is None or ds.size > best[1]):
                best = (row["path"], ds.size)
        if opts.plots and best:
            ds = f[best[0]]
            idx = (0,) * (ds.ndim - 2)
            ny, nx = ds.shape[-2:]
            step = max(1, math.ceil(max(ny, nx) / 1500))
            arr = decode(
                np.asarray(ds[idx + (slice(None, None, step), slice(None, None, step))]),
                dict(ds.attrs),
            )
            figs.append(
                (
                    f"Largest 2-D dataset: {best[0]}",
                    plots.image2d(arr, best[0], str(clean_attrs(ds.attrs).get("units", ""))),
                )
            )
    if len(numeric) > 200:
        notes.append(f"statistics computed for the first 200 of {len(numeric)} numeric datasets")
    datasets = tree.drop(columns=["attrs"]).to_dict("records")
    for d in datasets:
        if d["path"] in stats:
            d["missing_pct"] = stats[d["path"]].get("missing_pct")
    summary = {
        "n_datasets": len(tree),
        "n_groups": res.metadata.get("n_groups"),
        "datasets": datasets,
        "root_attributes": res.metadata.get("global_attrs", {}),
    }
    notes.append(
        "xarray could not open this HDF5 file as one dataset; the group tree was walked with h5py"
    )
    return {
        "summary": summary,
        "coverage": {},
        "statistics": stats,
        "trends": [],
        "notes": notes,
    }, figs


def analyze_binary(res: ReadResult, opts: ReadOptions):
    return {
        "summary": dict(res.metadata),
        "coverage": {},
        "statistics": {},
        "trends": [],
        "notes": [],
    }, []
