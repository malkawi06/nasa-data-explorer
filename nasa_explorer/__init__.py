"""nasa-data-explorer: drop in any scientific data file, get an instant analysis report."""

from __future__ import annotations

from pathlib import Path

from .core import ReadOptions, ReadResult
from .registry import read_file, reader

__version__ = "0.1.0"


def explore(
    path: str | Path,
    *,
    var: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    start: str | None = None,
    end: str | None = None,
    lang: str = "en",
    ai: bool = False,
    out_dir: str | Path = "reports",
    display: bool = True,
):
    """Analyse a file or folder, write reports and (in Jupyter) display them inline.

    Returns the FileReport for a single file, or a list for folders/archives.
    """
    from .pipeline import process

    opts = ReadOptions(var=var, bbox=bbox, start=start, end=end, lang=lang)
    reports = process(path, opts, out_dir, ai=ai)
    if display:
        try:
            from IPython import get_ipython
            from IPython.display import HTML
            from IPython.display import display as show

            if get_ipython() is not None:
                for r in reports:
                    show(HTML(r._repr_html_()))
        except ImportError:
            pass
    return reports[0] if len(reports) == 1 else reports


__all__ = ["explore", "read_file", "reader", "ReadOptions", "ReadResult"]
