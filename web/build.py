"""Static-site build step: bundle the nasa_explorer package for the browser.

Produces web/dist/ with the static page plus nasa_explorer.zip (package + bridge.py),
which Pyodide unpacks at startup. Uses only the standard library.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = WEB / "dist"
STATIC = ("index.html", "app.js", "ai.js", "worker.js", "style.css")


# Pure-Python packages the browser installs with micropip: pinned (a new release on PyPI can
# never change the site) and hash-checked. `build.py --wheels` publishes them next to the page,
# so the deployed site does not depend on PyPI at all; without it the browser fetches exactly
# these versions from PyPI. name -> (version, wheel file, sha256)
WHEELS = {
    "pymannkendall": (
        "1.4.3",
        "pymannkendall-1.4.3-py3-none-any.whl",
        "7d988efffae26c12c04b95f77042b946c8581da7526d94b2bbfb7416fb55681f",
    ),
    "h5netcdf": (
        "1.8.1",
        "h5netcdf-1.8.1-py3-none-any.whl",
        "a76ed7cfc9b8a8908ea7057c4e57e27307acff1049b7f5ed52db6c2247636879",
    ),
    "et-xmlfile": (
        "2.0.0",
        "et_xmlfile-2.0.0-py3-none-any.whl",
        "7a91720bc756843502c3b7504c77b8fe44217c85c537d85037f0f536151b2caa",
    ),
    "openpyxl": (
        "3.1.5",
        "openpyxl-3.1.5-py2.py3-none-any.whl",
        "5282c12b107bffeef825f4617dc029afaf41d0ea60823bbb665ef3079dc79de2",
    ),
    "markdown": (
        "3.11",
        "markdown-3.11-py3-none-any.whl",
        "cd6c89e7eb308c8b332ed673215a52d208a43f8bacc030b1419376129408719e",
    ),
    "arabic-reshaper": (
        "3.0.1",
        "arabic_reshaper-3.0.1-py3-none-any.whl",
        "41c5adc2420f85758eada7e880251c4b6a2adbd83377bd27e5d4eba71f648bc7",
    ),
    "rioxarray": (
        "0.23.0",
        "rioxarray-0.23.0-py3-none-any.whl",
        "5f8dad0577792a3cb5b90dda66339f161f149a1ad5c03fba0747cc67e29b54af",
    ),
    "python-docx": (
        "1.2.0",
        "python_docx-1.2.0-py3-none-any.whl",
        "3fd478f3250fbbbfd3b94fe1e985955737c145627498896a8a6bf81f4baf66c7",
    ),
    "pypdf": (
        "6.19.0",
        "pypdf-6.19.0-py3-none-any.whl",
        "7e5d6e730e7dae87d560a2cee218b852f6498c8be61966f3cd02ead971e48d14",
    ),
}


def _download_wheel(name: str, version: str, filename: str, sha256: str, dest: Path) -> None:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as r:
        url = next(u["url"] for u in json.load(r)["urls"] if u["filename"] == filename)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"{filename}: sha256 does not match the pinned hash")
    (dest / filename).write_bytes(data)


# Every reference to a file we publish gets ?v=<content hash>: hosts and CDNs (GitHub Pages
# caches ~10 min) then can never pair a new page with an old script or Python package.
VERSIONED = {
    "index.html": ('href="style.css"', 'src="app.js"', ">v dev<"),
    "app.js": ('"./ai.js"', '"./worker.js"'),
    "worker.js": ('"nasa_explorer.zip"', '"wheels/index.json"'),
}


def _version() -> str:
    h = hashlib.sha256()
    for path in sorted(
        [*(ROOT / "nasa_explorer").rglob("*.py"), *(WEB / n for n in STATIC), WEB / "bridge.py"]
    ):
        h.update(path.read_bytes())
    h.update(repr(sorted(WHEELS.items())).encode())
    return h.hexdigest()[:12]


def build(wheels: bool = False) -> Path:
    """wheels=True downloads the pinned wheels into dist/wheels (network needed)."""
    shutil.rmtree(DIST, ignore_errors=True)
    (DIST / "wheels").mkdir(parents=True)
    index = {}
    for name, (version, filename, sha256) in WHEELS.items():
        if wheels:
            _download_wheel(name, version, filename, sha256, DIST / "wheels")
        index[name] = f"wheels/{filename}" if wheels else f"{name}=={version}"
    (DIST / "wheels" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
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
    out = build(wheels="--wheels" in sys.argv[1:])
    print(f"built {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
