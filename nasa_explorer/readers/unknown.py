"""Fallback for anything no reader understood: size, signature and first bytes."""

from __future__ import annotations

from pathlib import Path

from ..core import ReadOptions, ReadResult

SIGNATURES = {
    b"PK\x03\x04": "ZIP container",
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ": "xz",
    b"7z\xbc\xaf": "7-Zip",
    b"Rar!": "RAR",
    b"\x89HDF": "HDF5",
    b"\x0e\x03\x13\x01": "HDF4",
    b"CDF": "NetCDF classic",
    b"GRIB": "GRIB",
    b"SIMPLE": "FITS",
    b"%PDF": "PDF",
    b"\x89PNG": "PNG",
    b"\xff\xd8\xff": "JPEG",
    b"\x00\x00\x00\x0cjP  \r\n\x87\n": "JPEG 2000",
    b"\xffO\xffQ": "JPEG 2000",  # bare codestream (.j2k / .j2c)
    b"II*\x00": "TIFF",
    b"MM\x00*": "TIFF",
    b"PAR1": "Parquet",
    b"SQLite format 3": "SQLite / GeoPackage",
    b"\x7fELF": "ELF executable",
    b"ARROW1": "Arrow IPC",
    b"\x93NUMPY": "NumPy .npy",
    b"\x80": "Python pickle (not opened: unsafe)",
}


def detect_signature(head: bytes) -> str:
    for magic, name in SIGNATURES.items():
        if head.startswith(magic):
            return name
    if len(head) > 262 and head[257:262] == b"ustar":
        return "tar archive"
    return "unrecognised"


def read_unknown(path: Path, opts: ReadOptions) -> ReadResult:
    head = b""
    if path.is_file():
        with open(path, "rb") as fh:
            head = fh.read(64)
    meta = {
        "signature": detect_signature(head),
        "first_bytes_hex": head.hex(" "),
        "first_bytes_ascii": "".join(chr(b) if 32 <= b < 127 else "." for b in head),
    }
    return ReadResult("binary", None, meta)
