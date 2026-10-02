"""HTML for the AI panel, quality checks and product cards (shared by reports and the website)."""

from __future__ import annotations

import html

from .i18n import labels

CSS = """
.ai-panel{border:2px solid var(--accent);border-radius:10px;padding:16px;margin:14px 0;background:var(--card)}
.ai-panel h2{margin-top:0;border:0;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.ai-panel h3{margin-top:14px}
.ai-panel ul{margin:4px 0;padding-inline-start:20px}.ai-panel li{margin:4px 0}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
.badge{display:inline-block;font-size:11.5px;font-weight:600;padding:1px 7px;border-radius:9px;border:1px solid;white-space:nowrap;margin-inline-start:6px}
.b-verified{color:#006300;border-color:#0ca30c}.b-mismatch{color:#a32b2b;border-color:#d03b3b}
.b-unsupported{color:#7a5300;border-color:#fab219}.b-qualitative{color:var(--muted);border-color:var(--line)}
@media (prefers-color-scheme:dark){.b-verified{color:#4fd14f}.b-mismatch{color:#ff8a8a}.b-unsupported{color:#ffcf66}}
.why{display:block;font-size:12px;color:var(--muted)}
.lvl{display:inline-block;min-width:64px;font-size:12px;font-weight:600}
.lvl-error{color:#d03b3b}.lvl-warning{color:#b37400}.lvl-info{color:var(--ink2)}
.qlist{list-style:none;padding:0;margin:0}.qlist li{padding:5px 0;border-bottom:1px solid var(--line)}
.qlist li:last-child{border:0}
.sev{font-size:11px;color:var(--muted);margin-inline-start:6px}
"""

SYMBOL = {"verified": "✓", "mismatch": "✗", "unsupported": "?", "qualitative": "·"}
LEVEL = {"error": "⛔", "warning": "⚠", "info": "ℹ"}


def _e(x) -> str:
    return html.escape(str(x))


def _badge(item: dict, L: dict) -> str:
    st = item.get("status")
    if not st:
        return ""
    why = (
        f"<span class='why'>{_e(item['note'])}</span>"
        if item.get("note") and st != "verified"
        else ""
    )
    title = f" title='{_e(item.get('note', ''))}'" if item.get("note") else ""
    return f"<span class='badge b-{st}'{title}>{SYMBOL.get(st, '')} {_e(L.get(st, st))}</span>{why}"


def _page(item: dict) -> str:
    return (
        f" <span class='sev'>(p. {_e(item['page'])})</span>"
        if item.get("page") not in (None, "")
        else ""
    )


def _items(title: str, items, L: dict) -> str:
    if not items:
        return ""
    lis = []
    for it in items:
        if isinstance(it, dict):
            text = it.get("text") or it.get("name", "")
            if it.get("how_used"):
                text = f"{it.get('name', '')}: {it['how_used']}"
            sev = f"<span class='sev'>[{_e(it['severity'])}]</span>" if it.get("severity") else ""
            lis.append(f"<li>{_e(text)}{_page(it)}{sev}{_badge(it, L)}</li>")
        else:
            lis.append(f"<li>{_e(it)}</li>")
    return f"<h3>{_e(title)}</h3><ul>{''.join(lis)}</ul>"


def _chips(ai: dict, L: dict) -> str:
    v = ai.get("verification") or {}
    chips = [
        f"<span class='badge b-{k}'>{SYMBOL[k]} {v[k]} {_e(L[k])}</span>"
        for k in SYMBOL
        if v.get(k)
    ]
    if ai.get("rounds", 1) > 1:
        fixed = ai.get("fixed_in_review")
        extra = f" · {fixed} {_e(L['fixed'])}" if fixed else ""
        chips.append(f"<span class='badge b-qualitative'>{_e(L['reviewed'])}{extra}</span>")
    return f"<div class='chips'>{''.join(chips)}</div>" if chips else ""


def render_ai(ai: dict | None, lang: str) -> str:
    """The prominent AI analysis card. Placeholder when AI has not run."""
    L = labels(lang)
    head = f"<h2>🤖 {_e(L['ai'])}"
    if not ai:
        return f"<section class='ai-panel'>{head}</h2><p class='meta'>{_e(L['ai_not_run'])}</p></section>"
    who = f"<span class='sev'>{_e(ai.get('provider', ''))} / {_e(ai.get('model', ''))}</span>"
    out = [
        f"<section class='ai-panel'>{head} {who}</h2>",
        f"<p class='meta'>{_e(L['ai_checked'])}</p>",
        _chips(ai, L),
    ]
    if ai.get("parse_error") or ai.get("text"):
        from .report import _md

        out.append(f"<p class='meta'>{_e(L['parse_error'])}</p>" if ai.get("parse_error") else "")
        out.append(_md(ai.get("raw") or ai.get("text") or ""))
        return "".join(out) + "</section>"
    if ai.get("overview"):
        out.append(f"<h3>{_e(L['overview'])}</h3><p>{_e(ai['overview'])}</p>")
    keys = (
        (
            "problem",
            "method",
            "data_used",
            "findings",
            "limitations",
            "nasa_datasets",
            "hackathon_ideas",
        )
        if ai.get("kind") == "paper"
        else (
            "findings",
            "quality_issues",
            "next_analyses",
            "visualizations",
            "hackathon_ideas",
            "caveats",
        )
    )
    for key in keys:
        out.append(_items(L[key], ai.get(key), L))
    return "".join(out) + "</section>"


def render_quality(issues: list[dict] | None, lang: str) -> str:
    L = labels(lang)
    if issues is None:
        return ""
    if not issues:
        return f"<h2>{_e(L['quality'])}</h2><div class='card'>✓ {_e(L['no_issues'])}</div>"
    lis = "".join(
        f"<li><span class='lvl lvl-{q['level']}'>{LEVEL[q['level']]} {_e(L[q['level']])}</span> {_e(q['message'])}</li>"
        for q in issues
    )
    return f"<h2>{_e(L['quality'])}</h2><div class='card'><ul class='qlist'>{lis}</ul></div>"


def render_products(cards: list[dict] | None, lang: str) -> str:
    if not cards:
        return ""
    L = labels(lang)
    out = [f"<h2>{_e(L['product'])}</h2>"]
    for c in cards:
        links = " · ".join(
            f"<a href='{_e(x['url'])}' target='_blank' rel='noopener'>{_e(x['title'])}</a>"
            for x in c["links"]
        )
        cav = "".join(f"<li>{_e(x)}</li>" for x in c["caveats"])
        out.append(
            f"<div class='card'><b>{_e(c['name'])}</b><div class='meta'>{_e(L['resolution'])}: {_e(c['resolution'])}"
            f"</div><ul>{cav}</ul><div class='meta'>{links}</div></div>"
        )
    return "".join(out)
