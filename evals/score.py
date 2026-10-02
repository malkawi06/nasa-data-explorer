"""Score one AI result against an eval case."""

from __future__ import annotations

import json

from .cases import Case


def _cited_facts(result: dict) -> set[str]:
    return {
        str(f.get("fact"))
        for f in result.get("findings", []) or []
        if isinstance(f, dict) and f.get("fact")
    }


def score(result: dict, case: Case) -> dict:
    text = json.dumps(result, ensure_ascii=False).lower()
    v = result.get("verification", {})
    checked = v.get("verified", 0) + v.get("mismatch", 0) + v.get("unsupported", 0)
    precision = v.get("verified", 0) / checked if checked else 1.0
    cited = _cited_facts(result)
    facts_hit = (
        sum(f in cited for f in case.expect_facts) / len(case.expect_facts)
        if case.expect_facts
        else 1.0
    )
    terms_hit = (
        sum(any(t.lower() in text for t in group) for group in case.expect_terms)
        / len(case.expect_terms)
        if case.expect_terms
        else 1.0
    )
    forbidden = [t for t in case.forbid_terms if t.lower() in text]
    json_ok = not result.get("parse_error")
    return {
        "case": case.name,
        "json_ok": json_ok,
        "verified": v.get("verified", 0),
        "mismatch": v.get("mismatch", 0),
        "unsupported": v.get("unsupported", 0),
        "precision": round(precision, 3),
        "facts_hit": round(facts_hit, 3),
        "terms_hit": round(terms_hit, 3),
        "forbidden": forbidden,
        "rounds": result.get("rounds", 1),
        "passed": json_ok
        and precision >= 0.8
        and facts_hit == 1
        and terms_hit == 1
        and not forbidden,
    }
