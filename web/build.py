"""Netlify build step: bundle the nasa_explorer package for the browser.

Produces web/dist/ with the static page plus nasa_explorer.zip (package + bridge.py),
which Pyodide unpacks at startup. Uses only the standard library.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = WEB / "dist"
STATIC = ("index.html", "app.js", "style.css")


def build() -> Path:
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir(parents=True)
    for name in STATIC:
        shutil.copy(WEB / name, DIST / name)
    bundle = DIST / "nasa_explorer.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((ROOT / "nasa_explorer").rglob("*.py")):
            zf.write(path, path.relative_to(ROOT).as_posix())
        zf.write(WEB / "bridge.py", "bridge.py")
    return bundle


if __name__ == "__main__":
    out = build()
    print(f"built {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
