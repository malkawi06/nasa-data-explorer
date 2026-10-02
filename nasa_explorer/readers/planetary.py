"""Planetary rasters from the PDS and USGS ISIS: PDS3 (.IMG/.LBL), PDS4 (.xml + data),
ISIS3 cubes (.cub) and VICAR. All are opened through GDAL, which also turns their map
projection (lunar polar stereographic, Mars equirectangular, ...) into a CRS."""

from __future__ import annotations

import re
from pathlib import Path

from ..core import NotThisFormat, ReadOptions, ReadResult
from ..registry import looks_like_text, read_head, reader
from .geotiff import open_raster

LABEL_EXTS = (".lbl", ".xml")
DATA_EXTS = (".img", ".dat", ".raw", ".qub")
PDS3_MAGIC = (b"PDS_VERSION_ID", b"CCSD3ZF", b"ODL_VERSION_ID", b"NJPL1I00PDS")
KEYWORDS = (
    "TARGET_NAME",
    "MISSION_NAME",
    "INSTRUMENT_NAME",
    "INSTRUMENT_ID",
    "PRODUCT_ID",
    "DATA_SET_ID",
    "MAP_PROJECTION_TYPE",
    "MAP_RESOLUTION",
    "MAP_SCALE",
    "SCALING_FACTOR",
    "OFFSET",
    "UNIT",
    "START_TIME",
    "STOP_TIME",
)
_KEYWORD = re.compile(r"^\s*(" + "|".join(KEYWORDS) + r")\s*=\s*(.+?)\s*$", re.M)


def _is_pds4(head: bytes) -> bool:
    return looks_like_text(head) and b"pds.nasa.gov/pds4" in head and b"Product_" in head


def label_for(path: Path) -> Path:
    """The file GDAL must open: the label for detached-label products, else the file itself."""
    if path.suffix.lower() in DATA_EXTS:
        for sib in sorted(path.parent.iterdir()):
            if sib == path or sib.stem.lower() != path.stem.lower():
                continue
            ext = sib.suffix.lower()
            if ext == ".lbl" or (ext == ".xml" and _is_pds4(read_head(sib))):
                return sib
    return path


def _label_text(path: Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(64_000)
    text = head.decode("latin-1", errors="replace")
    end = text.find("\nEND\n") if "PDS_VERSION_ID" in text else -1
    return text[: end if end > 0 else len(text)]


def _keywords(text: str) -> dict:
    out: dict = {}
    for key, value in _KEYWORD.findall(text):
        out.setdefault(key, value.strip().strip('"')[:120])
    if m := re.search(
        r"<target_name>([^<]+)</target_name>|<name>(Moon|Mars|[A-Z][a-z]+)</name>", text
    ):
        out.setdefault("TARGET_NAME", (m.group(1) or m.group(2)).strip())
    return out


@reader(
    "planetary-raster",
    category="Planetary",
    extensions=(".lbl", ".img", ".cub", ".xml", ".vic", ".dat", ".qub"),
    magic=(*PDS3_MAGIC, b"Object = IsisCube", b"LBLSIZE="),
    sniff=_is_pds4,
    requires=("rasterio", "rioxarray"),
    extra="geo",
    priority=30,
)
def read_planetary(path: Path, opts: ReadOptions) -> ReadResult:
    target = label_for(path)
    text = _label_text(target)
    pds4 = "pds.nasa.gov/pds4" in text
    if not (
        pds4
        or any(text.startswith(m.decode()) for m in PDS3_MAGIC)
        or "IsisCube" in text[:200]
        or text.startswith("LBLSIZE=")
        or target != path
    ):
        raise NotThisFormat("no PDS / ISIS / VICAR label")
    try:
        res = open_raster(target, label_text=text, require_georef=False)
    except NotThisFormat:
        raise
    except Exception as exc:  # e.g. a PDS table or spectrum, not an image
        raise NotThisFormat(f"GDAL could not open it as a raster: {exc}") from exc
    keys = _keywords(text)
    fmt = (
        "PDS4"
        if pds4
        else "ISIS3"
        if "IsisCube" in text[:200]
        else "VICAR"
        if text.startswith("LBLSIZE=")
        else "PDS3"
    )
    res.metadata.update(format=fmt, label=keys, label_file=target.name)
    res.metadata["header_text"] = "\n".join(f"{k} = {v}" for k, v in keys.items())
    return res
