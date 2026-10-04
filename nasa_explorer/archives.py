"""Archive handling: .zip, .tar(.gz/.bz2/.xz), .gz/.bz2/.xz single files."""

from __future__ import annotations

import bz2
import gzip
import lzma
import tarfile
import zipfile
from pathlib import Path

from .core import IN_BROWSER
from .registry import file_suffixes, read_head, readers

ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".gz",
    ".bz2",
    ".xz",
)
_STREAMS = {b"\x1f\x8b": gzip.open, b"BZh": bz2.open, b"\xfd7zXZ": lzma.open}
MAX_MEMBERS = 500
# Unpacked bytes per archive: a tiny "zip bomb" can expand to terabytes. The browser runs in
# 32-bit WebAssembly with a few GB of memory at most, so its limit is lower.
MAX_UNPACKED = (1 if IN_BROWSER else 8) * 1024**3


def _check_size(total: int) -> None:
    if total > MAX_UNPACKED:
        raise ValueError(
            f"archive unpacks to more than {MAX_UNPACKED / 1024**3:.0f} GB; extract it yourself "
            "and analyse the files you need"
        )


def is_archive(path: Path) -> bool:
    if path.is_dir():
        return False
    suffixes = file_suffixes(path)
    if any(s in r.extensions for s in suffixes for r in readers()):
        return False  # .docx / .xlsx / .kmz are zips but have their own readers
    if any(s in ARCHIVE_SUFFIXES for s in suffixes):
        return True
    head = read_head(path, 512)
    return (
        head.startswith(b"PK\x03\x04")
        or any(head.startswith(m) for m in _STREAMS)
        or head[257:262] == b"ustar"
    )


def _safe_target(dest: Path, name: str) -> Path:
    target = (dest / name).resolve()
    if not target.is_relative_to(dest.resolve()):  # a string prefix lets "run_x/" pass for "run/"
        raise ValueError(f"unsafe path in archive: {name}")
    return target


def extract(path: Path, dest: Path) -> list[Path]:
    """Extract into dest and return the extracted paths (path traversal is rejected)."""
    dest.mkdir(parents=True, exist_ok=True)
    head = read_head(path, 512)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()][:MAX_MEMBERS]
            _check_size(sum(m.file_size for m in members))  # declared sizes, checked again below
            for m in members:
                _safe_target(dest, m.filename)
                zf.extract(m, dest)
        return [dest / m.filename for m in members]
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            members = [m for m in tf.getmembers() if m.isfile()][:MAX_MEMBERS]
            _check_size(sum(m.size for m in members))
            for m in members:
                _safe_target(dest, m.name)
            # names are checked above; the "data" filter (Python 3.10.12+) also refuses links
            safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tf.extractall(dest, members=members, **safe)
        return [dest / m.name for m in members]
    for magic, opener in _STREAMS.items():
        if head.startswith(magic):
            name = path.name
            for sfx in (".gz", ".bz2", ".xz", ".tgz"):
                if name.lower().endswith(sfx):
                    name = name[: -len(sfx)] + (".tar" if sfx == ".tgz" else "")
                    break
            out = dest / (name if name != path.name else name + ".out")
            written = 0
            with opener(path, "rb") as src, open(out, "wb") as dst:
                while block := src.read(1 << 20):  # stream: the size is unknown until the end
                    written += len(block)
                    if written > MAX_UNPACKED:
                        dst.close()
                        out.unlink()
                        _check_size(written)
                    dst.write(block)
            return [out]
    raise ValueError("not a supported archive")
