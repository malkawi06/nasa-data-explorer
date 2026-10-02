"""Shared data types passed between readers, analysis and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["grid", "table", "document", "image", "tree", "binary"]


@dataclass
class ReadOptions:
    """User-selected subsetting, passed to readers and analysis."""

    var: str | None = None
    bbox: tuple[float, float, float, float] | None = None  # W, S, E, N
    start: str | None = None
    end: str | None = None
    lang: str = "en"
    plots: bool = True


@dataclass
class ReadResult:
    """What every reader returns.

    kind     -> grid: xarray.Dataset | table: pandas.DataFrame (GeoDataFrame ok)
                document: pages (text per page) | image: xarray.Dataset (y, x, band)
                tree: DataFrame listing HDF5 datasets | binary: nothing readable
    """

    kind: Kind
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[str] = field(default_factory=list)


class NotThisFormat(Exception):
    """A reader raises this to let the next candidate reader try the file."""


class MissingDependency(ImportError):
    """Raised when a reader's optional library is not installed."""
