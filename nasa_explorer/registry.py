"""Plugin-style reader registry and format detection.

A reader is a function decorated with :func:`reader`. Detection order:
1. readers whose extension matches AND whose signature agrees (or declare none),
2. readers whose magic bytes / sniff match (handles missing or wrong extensions),
3. remaining extension matches.
Each candidate is tried in turn; a reader may raise NotThisFormat to pass.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .core import MissingDependency, NotThisFormat, ReadOptions, ReadResult

HEAD_BYTES = 2048

ReaderFunc = Callable[[Path, ReadOptions], ReadResult]


@dataclass(frozen=True)
class ReaderSpec:
    name: str
    func: ReaderFunc
    category: str
    extensions: tuple[str, ...]
    magic: tuple[bytes, ...]
    sniff: Callable[[bytes], bool] | None
    requires: tuple[str, ...]
    extra: str | None
    priority: int

    def signature_matches(self, head: bytes) -> bool:
        if any(head.startswith(m) for m in self.magic):
            return True
        return bool(self.sniff and self.sniff(head))

    def has_signature(self) -> bool:
        return bool(self.magic or self.sniff)

    def missing(self) -> list[str]:
        return [m for m in self.requires if importlib.util.find_spec(m) is None]


_REGISTRY: list[ReaderSpec] = []


def reader(
    name: str,
    *,
    category: str,
    extensions: tuple[str, ...] = (),
    magic: tuple[bytes, ...] = (),
    sniff: Callable[[bytes], bool] | None = None,
    requires: tuple[str, ...] = (),
    extra: str | None = None,
    priority: int = 10,
) -> Callable[[ReaderFunc], ReaderFunc]:
    """Register a reader. Extensions are lower-case and include the dot.

    priority orders readers that match the same file (lower wins); generic sniffers
    such as delimited text or plain prose use high values so specific formats go first.
    """

    def deco(fn: ReaderFunc) -> ReaderFunc:
        _REGISTRY.append(
            ReaderSpec(name, fn, category, extensions, magic, sniff, requires, extra, priority)
        )
        return fn

    return deco


def readers() -> list[ReaderSpec]:
    from . import readers as _  # noqa: F401  (imports every reader module once)

    return list(_REGISTRY)


def read_head(path: Path, n: int = HEAD_BYTES) -> bytes:
    if path.is_dir():
        return b""
    with open(path, "rb") as fh:
        return fh.read(n)


def file_suffixes(path: Path) -> list[str]:
    """Most specific first: '.tar.gz' then '.gz'."""
    sfx = [s.lower() for s in path.suffixes]
    return ["".join(sfx[i:]) for i in range(len(sfx))]


def candidates(path: Path, head: bytes | None = None) -> list[ReaderSpec]:
    head = read_head(path) if head is None else head
    suffixes = file_suffixes(path)
    regs = sorted(readers(), key=lambda r: r.priority)
    by_ext = [r for s in suffixes for r in regs if s in r.extensions]
    agree = [r for r in by_ext if not r.has_signature() or r.signature_matches(head)]
    by_magic = [r for r in regs if r.signature_matches(head)]
    ordered: list[ReaderSpec] = []
    for r in agree + by_magic + by_ext:
        if r not in ordered:
            ordered.append(r)
    return ordered


def read_file(path: Path, opts: ReadOptions | None = None) -> tuple[ReadResult, str, list[str]]:
    """Try candidate readers in order. Returns (result, reader name, problems)."""
    opts = opts or ReadOptions()
    head = read_head(path)
    problems: list[str] = []
    for spec in candidates(path, head):
        missing = spec.missing()
        if missing:
            hint = f"pip install 'nasa-data-explorer[{spec.extra}]'" if spec.extra else ""
            problems.append(
                f"{spec.name} reader disabled: missing {', '.join(missing)}. {hint}".strip()
            )
            continue
        try:
            return spec.func(path, opts), spec.name, problems
        except (NotThisFormat, MissingDependency) as exc:
            problems.append(f"{spec.name}: {exc}")
        except Exception as exc:  # a broken file must never crash a folder run
            problems.append(f"{spec.name} failed: {type(exc).__name__}: {exc}")
    from .readers.unknown import read_unknown

    return read_unknown(path, opts), "unknown", problems


# --- signature helpers shared by readers -------------------------------------------------


def looks_like_text(head: bytes) -> bool:
    if not head:
        return False
    if b"\x00" in head:
        return False
    # binary data can be valid UTF-8 byte by byte (e.g. SRTM heights 0x03 0x20 ...); text has
    # almost no control characters besides tab, newline, form feed and carriage return
    if sum(b < 9 or 13 < b < 32 or b == 127 for b in head) > 0.01 * len(head):
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # tolerate a multi-byte char cut at the end of the buffer
        if exc.start < len(head) - 4:
            try:
                head.decode("latin-1")
            except UnicodeDecodeError:
                return False
    return True


def text_head(head: bytes) -> str:
    return head.decode("utf-8", errors="ignore").lstrip("﻿").lstrip()


__all__ = [
    "ReaderSpec",
    "candidates",
    "read_file",
    "reader",
    "readers",
    "looks_like_text",
    "text_head",
    "NotThisFormat",
]
