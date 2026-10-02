"""Native-library load order.

The eccodes wheel (GRIB) bundles its own C libraries. If GDAL (pyogrio) or GEOS/PROJ
(cartopy) load *after* it, the process can segfault or abort at exit. Loading them
first avoids the clash, so call this before importing eccodes / cfgrib.
"""

import contextlib
import importlib


def preload_before_eccodes() -> None:
    for mod in ("pyogrio", "cartopy.crs"):
        with contextlib.suppress(Exception):
            importlib.import_module(mod)
