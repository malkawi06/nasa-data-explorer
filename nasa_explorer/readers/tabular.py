"""Tabular formats: CSV/TSV/delimited TXT, Excel, JSON, Parquet."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..core import NotThisFormat, ReadOptions, ReadResult
from ..registry import looks_like_text, reader, text_head

_DELIMS = [",", "\t", ";", "|"]
_SCAN_LINES = 400
_CONSISTENT = 8  # rows with the same field count needed to call it a table


def _looks_delimited(head: bytes) -> bool:
    if not looks_like_text(head):
        return False
    lines = text_head(head).splitlines()[:-1] or text_head(head).splitlines()
    return _find_table_start(lines) is not None


def _fields(line: str, delim: str | None) -> int:
    return len(line.split()) if delim is None else line.count(delim) + 1


def _find_table_start(lines: list[str]) -> tuple[int, str | None] | None:
    """Return (first table line, delimiter); None = whitespace. Skips free-text headers
    such as NASA POWER's -BEGIN HEADER- block or '#' comments."""
    need = min(_CONSISTENT, max(2, len(lines) - 1))
    for delim in [*_DELIMS, None]:
        for start in range(0, max(1, min(len(lines) - need + 1, 120))):
            block = [ln for ln in lines[start : start + need]]
            if any(not ln.strip() or ln.lstrip().startswith("#") for ln in block):
                continue
            counts = {_fields(ln, delim) for ln in block}
            if len(counts) == 1 and counts.pop() > 1:
                if delim is None and not _numeric_share(block[1:]):
                    break  # prose also splits on whitespace; demand numbers
                return start, delim
    return None


def _numeric_share(lines: list[str]) -> bool:
    toks = [t for ln in lines for t in ln.split()]
    num = sum(bool(re.fullmatch(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", t)) for t in toks)
    return bool(toks) and num / len(toks) > 0.5


@reader(
    "delimited-text",
    category="Tabular",
    extensions=(".csv", ".tsv", ".txt", ".dat", ".tab", ".asc"),
    sniff=_looks_delimited,
    priority=50,
)
def read_delimited(path: Path, opts: ReadOptions) -> ReadResult:
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [next(fh, "") for _ in range(_SCAN_LINES)]
    lines = [ln.rstrip("\r\n") for ln in lines if ln]
    found = _find_table_start(lines)
    if found is None:
        raise NotThisFormat("no consistent delimiter found")
    start, delim = found
    sep = r"\s+" if delim is None else delim
    kw = {"engine": "python"} if delim is None else {"engine": "c", "low_memory": False}
    df = pd.read_csv(path, sep=sep, skiprows=start, encoding_errors="replace", **kw)
    df.columns = [str(c).strip() for c in df.columns]
    meta = {"delimiter": {None: "whitespace", "\t": "tab"}.get(delim, delim)}
    if start:
        meta["header_text"] = "\n".join(lines[:start])[:4000]
    return ReadResult("table", df, meta)


@reader(
    "excel",
    category="Tabular",
    extensions=(".xlsx", ".xlsm", ".xls"),
    magic=(b"\xd0\xcf\x11\xe0",),
    requires=("openpyxl",),
)
def read_excel(path: Path, opts: ReadOptions) -> ReadResult:
    sheets = pd.read_excel(path, sheet_name=None)
    if not sheets:
        raise NotThisFormat("workbook has no sheets")
    name, df = max(sheets.items(), key=lambda kv: kv[1].size)
    meta = {"sheets": {k: list(v.shape) for k, v in sheets.items()}, "sheet_used": name}
    return ReadResult("table", df, meta)


def _is_json(head: bytes) -> bool:
    return looks_like_text(head) and text_head(head)[:1] in "{["


def _best_frame(obj, depth: int = 0) -> pd.DataFrame | None:
    """Find the largest table-like structure inside parsed JSON."""
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj[:50]):
        return pd.json_normalize(obj)
    if isinstance(obj, list) and obj and all(isinstance(x, list) for x in obj[:50]):
        return (
            pd.DataFrame(obj[1:], columns=obj[0])
            if all(isinstance(c, str) for c in obj[0])
            else pd.DataFrame(obj)
        )
    if not isinstance(obj, dict) or depth > 6:
        return None
    vals = list(obj.values())
    # dict of equal-keyed dicts, e.g. NASA POWER {"T2M": {"20200101": 1.2, ...}, ...}
    if len(vals) > 0 and all(isinstance(v, dict) for v in vals):
        keys = set(vals[0])
        if (
            len(keys) > 1
            and all(set(v) == keys for v in vals)
            and all(not isinstance(x, dict | list) for x in vals[0].values())
        ):
            return pd.DataFrame(obj).rename_axis("key").reset_index()
    found = [f for f in (_best_frame(v, depth + 1) for v in vals) if f is not None]
    return max(found, key=lambda f: f.size) if found else None


@reader("json", category="Tabular", extensions=(".json",), sniff=_is_json, priority=40)
def read_json(path: Path, opts: ReadOptions) -> ReadResult:
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if (
        isinstance(obj, dict)
        and obj.get("type") in ("FeatureCollection", "Feature")
        and "parameter" not in json.dumps(obj.get("properties", ""))[:2000]
    ):
        raise NotThisFormat("GeoJSON - handled by the vector reader")
    df = _best_frame(obj)
    if df is None:
        df = pd.json_normalize(obj if isinstance(obj, list) else [obj])
    meta = {}
    if isinstance(obj, dict):
        meta["top_level_keys"] = list(obj)[:50]
    return ReadResult("table", df, meta)


@reader(
    "parquet",
    category="Tabular",
    extensions=(".parquet", ".pq"),
    magic=(b"PAR1",),
    requires=("pyarrow",),
)
def read_parquet(path: Path, opts: ReadOptions) -> ReadResult:
    return ReadResult("table", pd.read_parquet(path))
