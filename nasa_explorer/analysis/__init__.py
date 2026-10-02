"""Dispatch a ReadResult to the analysis for its kind."""

from __future__ import annotations

from ..core import ReadOptions, ReadResult
from .grid import analyze_grid
from .other import analyze_binary, analyze_document, analyze_image, analyze_tree
from .table import analyze_table

_DISPATCH = {
    "grid": analyze_grid,
    "table": analyze_table,
    "document": analyze_document,
    "image": analyze_image,
    "tree": analyze_tree,
    "binary": analyze_binary,
}


def analyze(res: ReadResult, opts: ReadOptions) -> tuple[dict, list[tuple[str, bytes]]]:
    """Return (JSON-able analysis dict, [(plot title, PNG bytes)])."""
    return _DISPATCH[res.kind](res, opts)


__all__ = ["analyze"]
