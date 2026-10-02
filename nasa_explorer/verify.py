"""Check AI output against computed facts (data files) or page text (papers).

Nothing here calls a model: it flattens the analysis into citable facts, parses the
model's JSON and labels every claim:
  verified     - every number matches a fact / appears on the cited page
  mismatch     - the cited fact exists but the value differs
  unsupported  - a number (or page) could not be traced
  qualitative  - no number to check
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

CONFIDENCE_RE = re.compile(r"(significan\w*|confidence|level)\s*(at\s*)?(the\s*)?$", re.I)
NUM_RE = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
REL_TOL = 0.005  # plus half a unit of the last digit written, so rounding is accepted
MAX_FACTS = 400


# --- facts ------------------------------------------------------------------------------


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if len(out) >= MAX_FACTS:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif (
        isinstance(obj, list | tuple)
        and obj
        and all(isinstance(x, int | float) for x in obj)
        and len(obj) <= 8
    ):
        for i, v in enumerate(obj):
            out[f"{prefix}[{i}]"] = v
    elif (
        isinstance(obj, bool)
        or isinstance(obj, int | float)
        and math.isfinite(obj)
        or isinstance(obj, str)
        and len(obj) <= 80
    ):
        out[prefix] = obj


def facts(analysis: dict) -> dict[str, Any]:
    """Citable facts: statistics, trends, coverage, trend map and table/grid sizes."""
    out: dict[str, Any] = {}
    s = analysis.get("summary", {})
    for key in ("n_rows", "n_columns", "dimensions", "pages", "words", "n_datasets"):
        if key in s:
            _flatten(f"summary.{key}", s[key], out)
    if isinstance(s.get("variables"), list):
        out["summary.n_variables"] = len(s["variables"])
    _flatten("coverage", analysis.get("coverage", {}), out)
    for t in analysis.get("trends", []):
        brief = {
            k: t[k]
            for k in (
                "slope_per_year",
                "slope_per_decade",
                "mk_p",
                "r2",
                "n",
                "method",
                "units",
                "seasonal_strength",
                "lag1_autocorr",
            )
            if k in t
        }
        _flatten(f"trends.{t['variable']}", brief, out)
    if analysis.get("trend_map"):
        _flatten("trend_map", analysis["trend_map"], out)
    for key in ("events", "correlations", "categories"):
        if analysis.get(key):
            _flatten(key, analysis[key], out)
    for name, st in analysis.get("statistics", {}).items():
        _flatten(f"statistics.{name}", {k: v for k, v in st.items() if k != "sampled"}, out)
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def facts_text(f: dict[str, Any]) -> str:
    return "\n".join(f"{k} = {_fmt(v)}" for k, v in f.items())


# --- parsing ----------------------------------------------------------------------------


def parse_json(text: str) -> dict | None:
    """Extract the JSON object from a model reply (fences, prose and trailing commas tolerated)."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = t[start : end + 1]
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def numbers_in(text: str) -> list[tuple[str, float]]:
    """Numbers to check. Bounds such as 'p<0.05' are thresholds, not claims, so they are skipped."""
    found = []
    text = text or ""
    for m in NUM_RE.finditer(text):
        before = text[: m.start()].rstrip()
        if before.endswith(("<", ">", "≤", "≥", "<=", ">=")):
            continue
        after = text[m.end() : m.end() + 25]
        if m.group(0) in ("90", "95", "99") and (
            CONFIDENCE_RE.search(before[-40:])
            or re.match(r"\s*%\s*(confidence|level|significance)", after, re.I)
        ):
            continue  # "significant at 95%" names a confidence level
        raw = m.group(0).replace("−", "-").replace(",", "")
        try:
            found.append((m.group(0), float(raw)))
        except ValueError:
            continue
    return found


def _tolerance(raw: str, value: float) -> float:
    """Half a unit in the last digit the model wrote, but at least REL_TOL relative."""
    digits = raw.split(".")[1] if "." in raw else ""
    digits = re.sub(r"[eE].*", "", digits)
    return max(0.5 * 10 ** (-len(digits)) if "." in raw else 0.5, REL_TOL * abs(value), 1e-12)


def close(raw: str, value: float, target: Any) -> bool:
    if not isinstance(target, int | float) or isinstance(target, bool):
        return False
    tol = _tolerance(raw, value)
    return any(abs(value - target * scale) <= tol for scale in (1.0, 100.0))


def _numeric_facts(f: dict[str, Any]) -> list[float]:
    vals = [v for v in f.values() if isinstance(v, int | float) and not isinstance(v, bool)]
    for v in f.values():  # years and dates inside coverage strings
        if isinstance(v, str):
            vals += [float(x) for x in re.findall(r"\b(1[89]\d\d|2[01]\d\d)\b", v)]
    return vals


# --- data-file claims -------------------------------------------------------------------


def verify_data(result: dict, f: dict[str, Any]) -> dict:
    """Label each finding; returns the result with `status`/`note` added and a summary."""
    pool = _numeric_facts(f)
    counts = {"verified": 0, "mismatch": 0, "unsupported": 0, "qualitative": 0}
    for item in result.get("findings", []) or []:
        if not isinstance(item, dict):
            continue
        status, note = _check_finding(item, f, pool)
        item["status"], item["note"] = status, note
        counts[status] += 1
    result["verification"] = counts
    return result


TREND_PATH = re.compile(
    r"^trends\.(.+)\.(slope_per_year|slope_per_decade|sens_slope_per_step|mk_p|r2|regression_p)$"
)
NOT_SIGNIFICANT = re.compile(
    r"not\s+(statistically\s+)?significant|no\s+(statistically\s+)?significant|insignificant|non-?significant|"
    r"not\s+robust|no\s+clear\s+trend|غير\s+دال|ليس\s+دال|غير\s+معنوي|لا\s+يوجد\s+اتجاه",
    re.I,
)


def _significance_problem(path: str, text: str, f: dict[str, Any]) -> str | None:
    """A trend claim must agree with its Mann-Kendall test: the number can be right while the
    sentence is misleading (a non-significant slope presented as a trend, or the reverse)."""
    m = TREND_PATH.match(path)
    if not m:
        return None
    p = f.get(f"trends.{m.group(1)}.mk_p")
    if not isinstance(p, int | float):
        return None
    says_not = bool(NOT_SIGNIFICANT.search(text))
    if p >= 0.05 and not says_not:
        return f"the {m.group(1)} trend is not statistically significant (p={p:.2g}); the sentence must say so"
    if p < 0.05 and says_not:
        return f"the {m.group(1)} trend IS significant (p={p:.2g}), contrary to the sentence"
    return None


def _check_finding(item: dict, f: dict[str, Any], pool: list[float]) -> tuple[str, str]:
    path = str(item.get("fact") or "").strip()
    value = item.get("value")
    nums = numbers_in(str(item.get("text", "")))
    if path:
        if path not in f:
            return "unsupported", f"cited fact '{path}' does not exist"
        target = f[path]
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and not close(str(value), float(value), target)
        ):
            return "mismatch", f"{path} is {_fmt(target)}, not {_fmt(value)}"
        bad = [
            r for r, v in nums if not close(r, v, target) and not any(close(r, v, p) for p in pool)
        ]
        if bad:
            return "unsupported", f"number(s) {', '.join(bad)} not found in the computed facts"
        if problem := _significance_problem(path, str(item.get("text", "")), f):
            return "mismatch", problem
        return "verified", f"{path} = {_fmt(target)}"
    if not nums:
        return "qualitative", ""
    # without a cited path, prefer facts about the variable the sentence names
    text = str(item.get("text", "")).lower()
    named = {k: v for k, v in f.items() if "." in k and k.split(".")[1].lower() in text}
    if named:
        pool = _numeric_facts(
            {**named, **{k: v for k, v in f.items() if k.startswith(("coverage", "summary"))}}
        )
    bad = [r for r, v in nums if not any(close(r, v, p) for p in pool)]
    if bad:
        return "unsupported", f"number(s) {', '.join(bad)} not found in the computed facts"
    return "verified", "numbers match computed facts"


# --- paper claims -----------------------------------------------------------------------

PAPER_SECTIONS = ("problem", "method", "data_used", "findings", "limitations", "nasa_datasets")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("−", "-")).lower()


def verify_paper(result: dict, pages: list[str]) -> dict:
    """Every claim must cite a real page whose text contains its numbers (or its quote)."""
    norm_pages = [_norm(p) for p in pages]
    counts = {"verified": 0, "mismatch": 0, "unsupported": 0, "qualitative": 0}
    for section in PAPER_SECTIONS:
        for item in result.get(section, []) or []:
            if not isinstance(item, dict):
                continue
            status, note = _check_paper_claim(item, norm_pages)
            item["status"], item["note"] = status, note
            counts[status] += 1
    result["verification"] = counts
    return result


def _check_paper_claim(item: dict, pages: list[str]) -> tuple[str, str]:
    try:
        page = int(item.get("page"))
    except (TypeError, ValueError):
        return "unsupported", "no page cited"
    if not 1 <= page <= len(pages):
        return "mismatch", f"page {page} does not exist (document has {len(pages)})"
    text = pages[page - 1]
    neighbours = " ".join(pages[max(0, page - 2) : page + 1])  # tolerate off-by-one citations
    quote = _norm(str(item.get("quote", "")))
    if quote and len(quote) > 12 and quote in text:
        return "verified", f"quote found on p. {page}"
    name = _norm(str(item.get("name", "")))
    if name:
        if name in text:
            return "verified", f"'{item['name']}' appears on p. {page}"
        if name in neighbours:
            return "verified", f"'{item['name']}' appears next to p. {page}"
        return "unsupported", f"'{item['name']}' not found on p. {page}"
    nums = numbers_in(str(item.get("text", "")))
    if not nums:
        return (
            ("verified", f"quote on p. {page}")
            if quote and quote in neighbours
            else ("qualitative", f"p. {page}")
        )
    page_vals = [v for _, v in numbers_in(text)]
    near_vals = [v for _, v in numbers_in(neighbours)]
    missing = [r for r, v in nums if not any(close(r, v, pv) for pv in page_vals)]
    if not missing:
        return "verified", f"numbers found on p. {page}"
    if not [r for r, v in nums if not any(close(r, v, pv) for pv in near_vals)]:
        return "verified", f"numbers found next to p. {page} (page cited off by one)"
    return "unsupported", f"{', '.join(missing)} not found on p. {page}"


def review_problems(result: dict) -> list[str]:
    """Human-readable list of claims that need fixing, for the review pass."""
    out = []
    for section in ("findings", *PAPER_SECTIONS):
        for i, item in enumerate(result.get(section, []) or [], 1):
            if isinstance(item, dict) and item.get("status") in ("mismatch", "unsupported"):
                out.append(f'{section}[{i}] "{str(item.get("text", ""))[:160]}": {item["note"]}')
    return list(dict.fromkeys(out))
