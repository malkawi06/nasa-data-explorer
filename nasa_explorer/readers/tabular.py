"""Tabular formats: CSV/TSV/delimited TXT, Excel, JSON, Parquet."""

from __future__ import annotations

import csv
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
    if delim is None:
        return len(line.split())
    return len(next(csv.reader([line], delimiter=delim), []))  # quoted "1,234" is one field


def _start_for(lines: list[str], delim: str | None, need: int) -> int | None:
    for start in range(0, max(1, min(len(lines) - need + 1, 120))):
        block = lines[start : start + need]
        if any(not ln.strip() or ln.lstrip().startswith("#") for ln in block):
            continue
        counts = {_fields(ln, delim) for ln in block}
        if len(counts) == 1 and counts.pop() > 1:
            if delim is None and not _numeric_share(block[1:]):
                return None  # prose also splits on whitespace; demand numbers
            return start
    return None


def _find_table_start(lines: list[str]) -> tuple[int, str | None] | None:
    """Return (first table line, delimiter); None = whitespace. Skips free-text headers
    such as NASA POWER's -BEGIN HEADER- block or '#' comments. The delimiter whose table
    starts earliest wins, so decimal commas in a ';' file do not pass for a ',' table."""
    need = min(_CONSISTENT, max(2, len(lines) - 1))
    found = [
        (st, i, d) for i, d in enumerate(_DELIMS) if (st := _start_for(lines, d, need)) is not None
    ]
    if found:
        start, _, delim = min(found)
        return start, delim
    if (start := _start_for(lines, None, need)) is not None:
        return start, None
    body = [ln.strip() for ln in lines[1 : need + 1]]
    if len(body) >= 2 and all(_NUMBER.fullmatch(t) for t in body):
        return 0, ","  # a single column of numbers under a header
    return None


_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_THOUSANDS = re.compile(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?")
_DECIMAL_COMMA = re.compile(r"[-+]?\d+,\d+")


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
    body = [t for ln in lines[start + 1 :] for t in (ln.split(delim) if delim else ln.split())]
    if (
        delim != ","
        and body
        and sum(bool(_DECIMAL_COMMA.fullmatch(t.strip())) for t in body) > 0.2 * len(body)
    ):
        kw["decimal"] = ","
    df = pd.read_csv(path, sep=sep, skiprows=start, encoding_errors="replace", **kw)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    meta = {"delimiter": {None: "whitespace", "\t": "tab"}.get(delim, delim)}
    if kw.get("decimal"):
        meta["decimal"] = "comma"
    if start:
        meta["header_text"] = "\n".join(lines[:start])[:4000]
    meta.update(tidy(df))
    return ReadResult("table", df, meta)


def tidy(df: pd.DataFrame) -> dict:
    """In-place clean-up of common spreadsheet habits; returns what was done, for the report:
    empty filler columns, a units row under the header, numbers written as text ("1,234")."""
    done: dict = {}
    empty = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if empty:
        df.drop(columns=empty, inplace=True)
        done["dropped_empty_columns"] = len(empty)
    text_cols = [c for c in df.columns if _is_text(df[c])]
    if len(df) > 2 and text_cols:
        first = df.iloc[0]
        rest = {c: pd.to_numeric(_unthousand(df[c].iloc[1:]), errors="coerce") for c in text_cols}
        numeric_below = [c for c in text_cols if rest[c].notna().mean() > 0.9]
        if (
            numeric_below
            and all(
                pd.isna(first[c]) or not _NUMBER.fullmatch(str(first[c]).strip())
                for c in numeric_below
            )
            and any(isinstance(first[c], str) for c in numeric_below)
        ):
            done["units"] = {str(c): str(first[c]) for c in df.columns if isinstance(first[c], str)}
            df.drop(index=df.index[0], inplace=True)
            df.reset_index(drop=True, inplace=True)
    for c in [c for c in df.columns if _is_text(df[c])]:
        vals = df[c].dropna().astype(str).str.strip()
        if (
            len(vals)
            and vals.map(lambda v: bool(_THOUSANDS.fullmatch(v) or _NUMBER.fullmatch(v))).mean()
            > 0.95
        ):
            df[c] = pd.to_numeric(_unthousand(df[c]), errors="coerce")
            done.setdefault("numbers_from_text", []).append(str(c))
    return done


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _unthousand(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"(?<=\d),(?=\d{3}\b)", "", regex=True)


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
    meta.update(tidy(df))
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


def _flatten_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Cells holding lists (e.g. NASA EONET "geometry": [{...}]) cannot be counted or compared.
    One-element lists of objects are expanded into columns, [lon, lat] pairs become coordinates,
    anything else is kept as JSON text."""
    for col in [c for c in df.columns if df[c].map(lambda v: isinstance(v, list | dict)).any()]:
        cells = df[col]
        first = cells.map(lambda v: v[0] if isinstance(v, list) and v else v)
        if (
            first.map(lambda v: isinstance(v, dict) or v is None).all()
            and cells.map(lambda v: not isinstance(v, list) or len(v) <= 1).all()
        ):
            inner = pd.json_normalize([v or {} for v in first]).add_prefix(f"{col}.")
            df = pd.concat([df.drop(columns=col), _flatten_lists(inner)], axis=1)
        elif (
            str(col).endswith("coordinates")
            and cells.map(
                lambda v: (
                    isinstance(v, list)
                    and len(v) in (2, 3)
                    and all(isinstance(x, int | float) for x in v)
                )
            ).all()
        ):
            df = df.drop(columns=col)
            df["longitude"] = cells.map(lambda v: v[0]).to_numpy()
            df["latitude"] = cells.map(lambda v: v[1]).to_numpy()
        else:
            df[col] = cells.map(lambda v: json.dumps(v) if isinstance(v, list | dict) else v)
    return df


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
    df = _flatten_lists(df)
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
    # pyarrow directly: pd.read_parquet breaks in Pyodide when pyarrow loads after pandas
    import pyarrow.parquet as pq

    return ReadResult("table", pq.read_table(path).to_pandas())
