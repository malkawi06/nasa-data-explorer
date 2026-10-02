"""Reader plugins. Every module in this package is imported automatically, so adding a
format means dropping one new file here that uses the ``@reader`` decorator."""

import importlib
import pkgutil

for _mod in pkgutil.iter_modules(__path__):
    if not _mod.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_mod.name}")
