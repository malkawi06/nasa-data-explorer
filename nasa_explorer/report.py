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

from . import ai_view
from .i18n import labels

CSS = """
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5}
 img.plot{background:#fcfcfb}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Tahoma,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
.muted{color:var(--muted)}.small{font-size:12px}
.rep-head h1{font-size:22px;margin:0 0 6px;overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0 8px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-width:0}
.kpi-label{font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;overflow-wrap:anywhere}
.kpi-value{font-size:17px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.kpi-sub{font-size:12px;color:var(--ink2);margin-top:2px;overflow-wrap:anywhere}
.kpi-warn{border-color:#fab219}.kpi-warn .kpi-value{color:#b37400}.kpi-ok .kpi-value{color:#0ca30c}
.toc{position:sticky;top:0;z-index:5;display:flex;gap:4px;overflow-x:auto;padding:8px 0;margin:6px 0 4px;
 background:var(--bg);border-bottom:1px solid var(--line)}
.toc a{white-space:nowrap;text-decoration:none;color:var(--ink2);font-size:13px;padding:4px 10px;border-radius:14px}
.toc a:hover{background:var(--card);color:var(--ink)}
section,.anchor{scroll-margin-top:52px}
.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px}
.fig{margin:0}.fig figcaption{font-weight:600;font-size:13px;margin-bottom:8px;color:var(--ink2)}
.fold{margin:10px 0}.fold>summary{cursor:pointer;color:var(--ink2);font-weight:600;font-size:14px;padding:4px 0}
.trend-list{margin:0;padding-inline-start:18px}.trend-list li{margin:4px 0}
.card.err{border-color:#d03b3b}
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

CSS += ai_view.CSS

PAGE = Template("""<!doctype html>
<html lang="{{ lang }}" dir="{{ 'rtl' if lang == 'ar' else 'ltr' }}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ title }}</title>
<style>{{ css }}</style></head><body><main>{{ body }}</main></body></html>""")


def fmt(v: Any) -> str:
    if isinstance(v, float | np.floating):
        if not math.isfinite(v):
            return "–"
        if v != 0 and abs(v) < 1e-12:  # float noise around zero
            return "≈0"
        return f"{v:.4g}" if abs(v) < 1e6 or v == 0 else f"{v:.3e}"
    if isinstance(v, int | np.integer) and not isinstance(v, bool):
        return f"{v:,}"
    text = str(v)
    if text.endswith(" 00:00:00"):
        text = text[:-9]
    return html.escape(text)


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


def _bbox_text(b) -> str:
    if not b or len(b) != 4:
        return ""
    w, s_, e, n = b
    ew = lambda x: f"{round(abs(x), 2):g}°{'W' if x < 0 else 'E'}"  # noqa: E731
    ns = lambda y: f"{round(abs(y), 2):g}°{'S' if y < 0 else 'N'}"  # noqa: E731
    return f"{ew(w)} – {ew(e)}, {ns(s_)} – {ns(n)}"


def _date(v) -> str:
    return str(v or "")[:10]


def _tile(label: str, value: str, sub: str = "", cls: str = "") -> str:
    # numbers, units and dates stay left-to-right inside Arabic (right-to-left) reports
    sub_html = f"<div class='kpi-sub'><bdi dir='ltr'>{html.escape(sub)}</bdi></div>" if sub else ""
    return (
        f"<div class='kpi {cls}'><div class='kpi-label'>{html.escape(label)}</div>"
        f"<div class='kpi-value'><bdi dir='ltr'>{html.escape(value)}</bdi></div>{sub_html}</div>"
    )


def kpi_tiles(rep: dict, L: dict) -> str:
    """The few facts people look for first, before any table."""
    a, kind = rep["analysis"], rep["kind"]
    s, cov = a.get("summary", {}), a.get("coverage", {})
    tiles = []
    if t := cov.get("time"):
        tiles.append(
            _tile(
                L["kpi_time"],
                f"{_date(t.get('start'))} → {_date(t.get('end'))}",
                f"{t.get('resolution', '')} · {t.get('n_steps', '')} {L['steps']}",
            )
        )
    if (b := (cov.get("space") or {}).get("bbox")) and len(b) == 4:
        res = (cov.get("space") or {}).get("resolution_deg")
        tiles.append(
            _tile(L["kpi_region"], _bbox_text(b), f"{res[0]:g}° × {res[1]:g}°" if res else "")
        )
    if kind == "grid":
        dims = s.get("dimensions", {})
        tiles.append(
            _tile(
                L["kpi_size"],
                f"{len(s.get('variables', []))} {L['variables'].lower()}",
                " × ".join(f"{k} {v:,}" for k, v in list(dims.items())[:4]),
            )
        )
    elif kind == "table":
        tiles.append(
            _tile(
                L["kpi_size"],
                f"{s.get('n_rows', 0):,} {L['rows']}",
                f"{s.get('n_columns', 0)} {L['columns'].lower()}",
            )
        )
    elif kind == "document":
        tiles.append(
            _tile(
                L["kpi_size"],
                f"{s.get('pages', 0)} {L['pages']}",
                f"{s.get('words', 0):,} {L['words']}",
            )
        )
        if s.get("nasa_mentions"):
            tiles.append(
                _tile(
                    L["nasa"], f"{len(s['nasa_mentions'])}", ", ".join(list(s["nasa_mentions"])[:4])
                )
            )
    elif kind == "image":
        tiles.append(
            _tile(L["kpi_size"], f"{s.get('width')} × {s.get('height')}", f"{s.get('mode', '')}")
        )
    elif kind == "tree":
        tiles.append(_tile(L["kpi_size"], f"{s.get('n_datasets', 0)} {L['datasets'].lower()}", ""))
    if trends := a.get("trends"):
        t = trends[0]
        sig = t.get("mk_p", 1) < 0.05
        arrow = ("↑" if t["slope_per_year"] > 0 else "↓") if sig else "→"
        unit = f" {t['units']}" if t.get("units") else ""
        tiles.append(
            _tile(
                f"{L['trends']}: {t['variable']}",
                f"{arrow} {t['slope_per_year']:+.3g}{unit}/yr",
                L["significant"] if sig else L["not_significant"],
            )
        )
    if (q := a.get("quality")) is not None:
        counts = {
            lvl: sum(1 for i in q if i["level"] == lvl) for lvl in ("error", "warning", "info")
        }
        serious = counts["error"] + counts["warning"]
        if serious:
            tiles.append(
                _tile(
                    L["quality"],
                    f"⚠ {serious}",
                    f"{counts['error']} {L['error'].lower()} · "
                    f"{counts['warning']} {L['warning'].lower()}",
                    "kpi-warn",
                )
            )
        else:
            tiles.append(
                _tile(L["quality"], "✓", f"{counts['info']} {L['info'].lower()}", "kpi-ok")
            )
    return f"<div class='kpis'>{''.join(tiles)}</div>" if tiles else ""


def _section(sid: str, title: str, body: str) -> str:
    return f"<section id='{sid}'><h2>{html.escape(title)}</h2>{body}</section>"


def _fold(title: str, body: str, open_: bool = False) -> str:
    return f"<details class='fold'{' open' if open_ else ''}><summary>{html.escape(title)}</summary>{body}</details>"


def render_body(rep: dict, plots: list[tuple[str, str, bytes]], lang: str) -> str:
    L = labels(lang)
    a = rep["analysis"]
    sections: list[tuple[str, str, str]] = []  # (id, nav label, html)

    sections.append(("ai", L["ai"], ai_view.render_ai(rep.get("ai"), lang)))
    quality = ai_view.render_quality(a.get("quality"), lang)
    products = ai_view.render_products(a.get("products"), lang)
    if quality or products:
        sections.append(("quality", L["quality"], quality + products))
    if a.get("trends"):
        items = "".join(
            f"<li><b>{html.escape(t['variable'])}</b>: <span class='trend'>{html.escape(t['text'])}</span> "
            f"<span class='muted'>(n={t['n']})</span></li>"
            for t in a["trends"]
        )
        body = f"<div class='card'><ul class='trend-list'>{items}</ul></div>"
        if a.get("trend_map"):
            body += _fold(L["trend_map"], f"<div class='card'>{to_html(a['trend_map'])}</div>")
        rows = [
            {k: v for k, v in t.items() if k not in ("text", "intercept", "extremes")}
            for t in a["trends"]
        ]
        body += _fold(L["details"], f"<div class='card'>{to_html(rows)}</div>")
        sections.append(("trends", L["trends"], _section("trends", L["trends"], body)))
    if plots:
        figs = "".join(
            f"<figure class='card fig'><figcaption>{html.escape(title)}</figcaption><img class='plot' "
            f"alt='{html.escape(title)}' src='data:image/png;base64,{base64.b64encode(png).decode()}'>"
            f"<div class='muted small'>{html.escape(fname)}</div></figure>"
            for title, fname, png in plots
        )
        sections.append(
            ("plots", L["plots"], _section("plots", L["plots"], f"<div class='plots'>{figs}</div>"))
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
    body = ""
    if mentions:
        pills = "".join(
            f"<span class='pill'>{html.escape(k)} (p. {', '.join(map(str, v[:8]))})</span>"
            for k, v in mentions.items()
        )
        body += f"<h3>{L['nasa']}</h3><div class='card'>{pills}</div>"
    for i, (label, value) in enumerate(lists):
        long_ = isinstance(value, list) and len(value) > 15
        block = f"<div class='card'>{to_html(value)}</div>"
        body += (
            _fold(f"{label} ({len(value)})", block)
            if long_ or i > 0
            else f"<h3>{label}</h3>{block}"
        )
    if a.get("statistics"):
        body += f"<h3>{L['statistics']}</h3><div class='card'>{stats_table(a['statistics'])}</div>"
    if a.get("coverage"):
        body += _fold(L["coverage"], f"<div class='card'>{to_html(a['coverage'])}</div>")
    body += _fold(L["attributes"], f"<div class='card'>{to_html(s)}</div>")
    if a.get("preview"):
        body += _fold(
            L["preview"],
            f"<div class='card'><pre class='preview'>{html.escape(a['preview'])}</pre></div>",
            True,
        )
    sections.append(("data", L["summary"], _section("data", L["summary"], body)))

    notes = a.get("notes", []) + rep.get("problems", [])
    if notes:
        lis = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
        sections.append(
            ("notes", L["notes"], _section("notes", L["notes"], f"<ul class='notes'>{lis}</ul>"))
        )

    chips = [rep["reader"], rep["category"], rep["kind"], rep["size_human"]]
    head = (
        f"<header class='rep-head'><h1>{html.escape(rep['file'])}</h1><div class='chips'>"
        + "".join(
            f"<span class='pill'><bdi dir='ltr'>{html.escape(str(c))}</bdi></span>" for c in chips
        )
        + f"<span class='muted small'>{L['generated']}: <bdi dir='ltr'>{html.escape(rep['generated'])}</bdi></span></div></header>"
    )
    error = (
        f"<div class='card err'><b>{L['error']}:</b> {html.escape(rep['error'])}</div>"
        if rep.get("error")
        else ""
    )
    nav = (
        "<nav class='toc' aria-label='Sections'>"
        + "".join(f"<a href='#{sid}'>{html.escape(label)}</a>" for sid, label, _ in sections)
        + "</nav>"
    )
    # the AI and quality blocks render their own headings; give them anchors
    body_html = "".join(
        f"<div id='{sid}' class='anchor'>{h}</div>" if not h.startswith("<section") else h
        for sid, _, h in sections
    )
    return head + error + kpi_tiles(rep, L) + nav + body_html


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


def _finite(obj: Any) -> Any:
    """NaN/Infinity are not valid JSON (browsers and most parsers reject them): use null."""
    if isinstance(obj, float | np.floating):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _finite(obj.tolist())
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(
        _finite(obj), default=_json_default, ensure_ascii=False, indent=1, allow_nan=False
    )


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
