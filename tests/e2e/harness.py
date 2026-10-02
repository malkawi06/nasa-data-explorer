"""Serve web/dist for browser tests.

With `fake_pyodide=True` the Pyodide module is replaced by a stub whose `bridge` calls go
over synchronous XHR to this server, which runs the real web/bridge.py in CPython. That
exercises the page's JavaScript and the bridge together without downloading Pyodide.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))

import bridge  # noqa: E402
import build  # noqa: E402

BRIDGE_API = {"formats", "new_run", "analyse_dir", "ai_start", "ai_next", "chat_start"}

STUB = """
const BASE = location.origin;
function call(fn, args) {
  const x = new XMLHttpRequest();
  x.open("POST", `${BASE}/__bridge/${fn}`, false);
  x.send(JSON.stringify(args));
  if (x.status !== 200) throw new Error(x.responseText);
  return x.responseText;
}
export async function loadPyodide() {
  return {
    loadPackage: async () => {}, runPythonAsync: async () => {}, unpackArchive() {},
    FS: { writeFile(path, data) {
      const x = new XMLHttpRequest();
      x.open("POST", `${BASE}/__fs?path=${encodeURIComponent(path)}`, false);
      x.send(data);
    } },
    pyimport: () => new Proxy({}, { get: (_, fn) => (...args) => call(fn, args) }),
  };
}
"""


def chromium_path() -> str | None:
    env = os.environ.get("CHROMIUM_PATH")
    if env:
        return env
    for p in sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux*/chrome")):
        return str(p)
    return None  # let Playwright use its own downloaded browser


class Server:
    def __init__(self, work: Path):
        bridge.WORK = work
        build.build()
        handler = functools.partial(_Handler, directory=str(build.DIST))
        self.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        shutil.rmtree(build.DIST, ignore_errors=True)


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep test output quiet
        pass

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        try:
            if url.path == "/__fs":
                path = Path(unquote(parse_qs(url.query)["path"][0]))
                if bridge.WORK.resolve() not in path.resolve().parents:
                    raise PermissionError(path)
                path.write_bytes(body)
                out = ""
            else:
                fn = url.path.rsplit("/", 1)[-1]
                if fn not in BRIDGE_API:
                    raise PermissionError(fn)
                out = getattr(bridge, fn)(*json.loads(body))
            self._reply(200, out)
        except Exception as exc:
            self._reply(500, f"{type(exc).__name__}: {exc}")

    def _reply(self, code: int, text: str) -> None:
        data = text.encode()
        self.send_response(code)
        self.send_header("content-type", "text/plain; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
