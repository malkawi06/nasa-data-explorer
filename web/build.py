"""Static-site build step: bundle the nasa_explorer package for the browser.

Produces web/dist/ with the static page plus nasa_explorer.zip (package + bridge.py),
which Pyodide unpacks at startup. Uses only the standard library.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = WEB / "dist"
STATIC = ("index.html", "app.js", "ai.js", "worker.js", "style.css")


# Every reference to a file we publish gets ?v=<content hash>: hosts and CDNs (GitHub Pages
# caches ~10 min) then can never pair a new page with an old script or Python package.
VERSIONED = {
    "index.html": ('href="style.css"', 'src="app.js"', ">v dev<"),
    "app.js": ('"./ai.js"', '"./worker.js"'),
    "worker.js": ('"nasa_explorer.zip"',),
}


def _version() -> str:
    h = hashlib.sha256()
    for path in sorted(
        [*(ROOT / "nasa_explorer").rglob("*.py"), *(WEB / n for n in STATIC), WEB / "bridge.py"]
    ):
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def build() -> Path:
    shutil.rmtree(DIST, ignore_errors=True)
    DIST.mkdir(parents=True)
    version = _version()
    for name in STATIC:
        text = (WEB / name).read_text(encoding="utf-8")
        for ref in VERSIONED.get(name, ()):
            if ref not in text:
                raise RuntimeError(f"{name}: expected reference {ref} not found")
            new = f">v {version[:7]}<" if ref == ">v dev<" else f"{ref[:-1]}?v={version}{ref[-1]}"
            text = text.replace(ref, new)
        (DIST / name).write_text(text, encoding="utf-8")
    bundle = DIST / "nasa_explorer.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((ROOT / "nasa_explorer").rglob("*.py")):
            zf.write(path, path.relative_to(ROOT).as_posix())
        zf.write(WEB / "bridge.py", "bridge.py")
    return bundle


if __name__ == "__main__":
    out = build()
    print(f"built {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
