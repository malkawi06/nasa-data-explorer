"""HTML + JSON report writers."""

from __future__ import annotations

import base64
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Template

from .i18n import labels

CSS = """
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5}
 img.plot{background:#fcfcfb}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Tahoma,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:14px;margin:16px 0 6px;color:var(--ink2)}
.meta{color:var(--ink2);margin-bottom:12px}.meta b{color:var(--ink)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:start;vertical-align:top}
th{color:var(--ink2);font-weight:600;background:transparent}
td.num{text-align:end}
img.plot{max-width:100%;border-radius:6px;border:1px solid var(--line)}
.notes li{color:var(--ink2)}.pill{display:inline-block;padding:1px 8px;border-radius:10px;border:1px solid var(--line);font-size:12px;color:var(--ink2);margin:2px}
.trend{font-weight:600}.ai{white-space:normal}.ai pre{white-space:pre-wrap}
pre.preview{white-space:pre-wrap;font-size:12px;color:var(--ink2);max-height:320px;overflow:auto}
a{color:var(--accent)}
"""

PAGE = Template("""<!doctype html>
<html lang="{{ lang }}" dir="{{ 'rtl' if lang == 'ar' else 'ltr' }}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }}</title>
<style>{{ css }}</style></head><body><main>{{ body }}</main></body></html>""")


def fmt(v: Any) -> str:
    if isinstance(v, float | np.floating):
        if not math.isfinite(v):
            return "–"
        return f"{v:.4g}" if abs(v) < 1e6 or v == 0 else f"{v:.3e}"
    if isinstance(v, int | np.integer) and not isinstance(v, bool):
        return f"{v:,}"
    return html.escape(str(v))


def to_html(obj: Any, depth: int = 0) -> str:
    """Render nested dict/list data as tables."""
    if isinstance(obj, dict):
        if not obj:
            return "–"
        rows = "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{to_html(v, depth + 1)}</td></tr>"
            for k, v in obj.items()
        )
        return f"<table>{rows}</table>"
    if isinstance(obj, list):
        if not obj:
            return "–"
        if all(isinstance(x, dict) for x in obj):
            cols: list[str] = []
            for x in obj:
                cols += [k for k in x if k not in cols]
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
            body = "".join(
                "<tr>"
                + "".join(
                    f"<td class='{'num' if isinstance(x.get(c), int | float) else ''}'>{to_html(x.get(c, ''), depth + 1)}</td>"
                    for c in cols
                )
                + "</tr>"
                for x in obj
            )
            return f"<table><tr>{head}</tr>{body}</table>"
        if all(isinstance(x, list) for x in obj):  # raw table rows
            return (
                "<table>"
                + "".join("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>" for r in obj)
                + "</table>"
            )
        return ", ".join(fmt(x) for x in obj)
    return fmt(obj)


def stats_table(stats: dict) -> str:
    if not stats:
        return ""
    keys = ["count", "missing_pct", "min", "p5", "p25", "p50", "mean", "p75", "p95", "max", "std"]
    head = "<th></th>" + "".join(f"<th>{k}</th>" for k in keys)
    rows = "".join(
        f"<tr><th>{html.escape(name)}</th>"
        + "".join(f"<td class='num'>{fmt(s.get(k, ''))}</td>" for k in keys)
        + "</tr>"
        for name, s in stats.items()
    )
    return f"<table><tr>{head}</tr>{rows}</table>"


def _md(text: str) -> str:
    try:
        import markdown

        return markdown.markdown(text, extensions=["tables"])
    except ImportError:
        return f"<pre>{html.escape(text)}</pre>"


def render_body(rep: dict, plots: list[tuple[str, str, bytes]], lang: str) -> str:
    L = labels(lang)
    a = rep["analysis"]
    out = [
        f"<h1>{html.escape(rep['file'])}</h1>",
        f"<div class='meta'><b>{L['format']}:</b> {html.escape(rep['reader'])} · <b>{L['category']}:</b> "
        f"{html.escape(rep['category'])} · <b>{L['kind']}:</b> {rep['kind']} · <b>{L['size']}:</b> {rep['size_human']}"
        f" · <b>{L['generated']}:</b> {rep['generated']}</div>",
    ]
    if rep.get("error"):
        out.append(f"<div class='card'><b>Error:</b> {html.escape(rep['error'])}</div>")
    if a.get("trends"):
        out.append(
            f"<h2>{L['trends']}</h2><div class='card'><ul>"
            + "".join(
                f"<li><b>{html.escape(t['variable'])}</b>: <span class='trend'>{html.escape(t['text'])}</span> (n={t['n']})</li>"
                for t in a["trends"]
            )
            + "</ul></div>"
        )
    if rep.get("ai"):
        out.append(
            f"<h2>{L['ai']}</h2><div class='card ai'><div class='meta'>{html.escape(rep['ai'].get('provider', ''))}"
            f" / {html.escape(rep['ai'].get('model', ''))}</div>{_md(rep['ai']['text'])}</div>"
        )
    if plots:
        out.append(f"<h2>{L['plots']}</h2>")
        for title, fname, png in plots:
            b64 = base64.b64encode(png).decode()
            out.append(
                f"<div class='card'><h3>{html.escape(title)}</h3><img class='plot' alt='{html.escape(title)}' "
                f"src='data:image/png;base64,{b64}'><div class='meta'>{html.escape(fname)}</div></div>"
            )
    s = dict(a.get("summary", {}))
    lists = [
        (label, s.pop(key))
        for key, label in (
            ("variables", L["variables"]),
            ("columns", L["columns"]),
            ("datasets", L["datasets"]),
            ("tables", L["tables"]),
            ("figure_captions", L["figures"]),
            ("sections", L["sections"]),
        )
        if s.get(key)
    ]
    for key in ("variables", "columns", "datasets", "tables", "figure_captions", "sections"):
        s.pop(key, None)
    mentions = s.pop("nasa_mentions", None)
    out.append(f"<h2>{L['summary']}</h2><div class='card'>{to_html(s)}</div>")
    if mentions:
        out.append(
            f"<h3>{L['nasa']}</h3><div class='card'>"
            + "".join(
                f"<span class='pill'>{html.escape(k)} (p. {', '.join(map(str, v[:8]))})</span>"
                for k, v in mentions.items()
            )
            + "</div>"
        )
    for label, value in lists:
        out.append(f"<h3>{label}</h3><div class='card'>{to_html(value)}</div>")
    if a.get("coverage"):
        out.append(f"<h2>{L['coverage']}</h2><div class='card'>{to_html(a['coverage'])}</div>")
    if a.get("statistics"):
        out.append(
            f"<h2>{L['statistics']}</h2><div class='card'>{stats_table(a['statistics'])}</div>"
        )
    if a.get("trends"):
        rows = [{k: v for k, v in t.items() if k not in ("text", "intercept")} for t in a["trends"]]
        out.append(f"<h3>{L['details']}</h3><div class='card'>{to_html(rows)}</div>")
    if a.get("preview"):
        out.append(
            f"<h2>{L['preview']}</h2><div class='card'><pre class='preview'>{html.escape(a['preview'])}</pre></div>"
        )
    notes = a.get("notes", []) + rep.get("problems", [])
    if notes:
        out.append(
            f"<h2>{L['notes']}</h2><ul class='notes'>"
            + "".join(f"<li>{html.escape(n)}</li>" for n in notes)
            + "</ul>"
        )
    return "\n".join(out)


def page(title: str, body: str, lang: str) -> str:
    return PAGE.render(title=title, body=body, css=CSS, lang=lang)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    return str(o)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, ensure_ascii=False, indent=1)


def write_index(entries: list[dict], out_dir: Path, lang: str) -> Path:
    L = labels(lang)
    rows = "".join(
        f"<tr><td><a href='{html.escape(e['html'])}'>{html.escape(e['file'])}</a></td><td>{html.escape(e['reader'])}</td>"
        f"<td>{html.escape(e['category'])}</td><td class='num'>{e['size_human']}</td><td>{html.escape(e.get('headline', ''))}</td></tr>"
        for e in entries
    )
    body = (
        f"<h1>{L['index']}</h1><div class='meta'>{len(entries)} · {datetime.now():%Y-%m-%d %H:%M}</div>"
        f"<div class='card'><table><tr><th>{L['file']}</th><th>{L['format']}</th><th>{L['category']}</th>"
        f"<th>{L['size']}</th><th>{L['summary']}</th></tr>{rows}</table></div>"
    )
    path = out_dir / "index.html"
    path.write_text(page(L["index"], body, lang), encoding="utf-8")
    return path
