"""Optional AI layer with swappable backends.

Provider is chosen by env AI_PROVIDER (ollama | gemini | groq | anthropic); if unset, the
first configured one wins in that order. AI_MODEL overrides the default model.
Only computed summaries are ever sent for data files - never the raw file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import requests

if TYPE_CHECKING:
    from .core import ReadResult
    from .pipeline import FileReport

log = logging.getLogger("nasa_explorer.ai")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

CHUNK_TOKENS = 6000
MAX_RETRIES = 6
TIMEOUT = 180


class AIUnavailable(RuntimeError):
    pass


class RateLimited(RuntimeError):
    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited (429)")
        self.retry_after = retry_after


class Provider(Protocol):
    name: str
    model: str

    def complete(self, prompt: str, system: str, json_mode: bool = False) -> str: ...


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English; Arabic tokenizes denser, so be conservative."""
    return max(1, int(len(text) / 3.5))


def _raise_for(resp: requests.Response) -> None:
    if resp.status_code == 429:
        ra = resp.headers.get("retry-after")
        raise RateLimited(float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
    resp.raise_for_status()


# --- backends ---------------------------------------------------------------------------


class Ollama:
    name = "ollama"

    def __init__(self, model: str | None = None):
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = model or "qwen2.5:7b"

    @classmethod
    def available(cls) -> bool:
        try:
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            return requests.get(
                f"{host if host.startswith('http') else 'http://' + host}/api/tags", timeout=1.5
            ).ok
        except requests.RequestException:
            return False

    def resolve_model(self) -> None:
        """Fall back to any pulled model if the requested one is not installed."""
        tags = requests.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])
        names = [m["name"] for m in tags]
        if not names:
            raise AIUnavailable(
                "Ollama is running but no models are pulled: run `ollama pull qwen2.5:7b`"
            )
        if self.model not in names and f"{self.model}:latest" not in names:
            log.warning("Ollama model %s not pulled; using %s", self.model, names[0])
            self.model = names[0]

    def complete(self, prompt: str, system: str, json_mode: bool = False) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            timeout=TIMEOUT * 3,
            json={
                "model": self.model,
                "stream": False,
                **({"format": "json"} if json_mode else {}),
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        _raise_for(resp)
        return resp.json()["message"]["content"]


class Gemini:
    name = "gemini"

    def __init__(self, model: str | None = None):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model or "gemini-3.8-flash"

    def newest_flash(self) -> str | None:
        """Model names get retired; pick the newest generally available 'flash' model."""
        names = [
            m.name.removeprefix("models/")
            for m in self.client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])
        ]
        flash = [n for n in names if re.fullmatch(r"gemini-[\d.]+-flash(-\d+)?", n)]
        flash.sort(key=lambda n: (-float(re.match(r"gemini-([\d.]+)-", n).group(1)), len(n)))
        return flash[0] if flash else None

    def complete(self, prompt: str, system: str, json_mode: bool = False) -> str:
        from google.genai import errors

        try:
            return self._generate(prompt, system, json_mode)
        except errors.APIError as exc:
            if getattr(exc, "code", None) != 404:
                raise
            replacement = self.newest_flash()
            if not replacement or replacement == self.model:
                raise
            log.warning("Gemini model %s is unavailable; switching to %s", self.model, replacement)
            self.model = replacement
            return self._generate(prompt, system, json_mode)

    def _generate(self, prompt: str, system: str, json_mode: bool) -> str:
        from google.genai import errors, types

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    response_mime_type="application/json" if json_mode else None,
                ),
            )
        except errors.APIError as exc:
            if getattr(exc, "code", None) == 429:
                raise RateLimited() from exc
            raise
        return resp.text or ""


class Groq:
    name = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None):
        self.key = os.environ["GROQ_API_KEY"]
        self.model = model or "llama-3.3-70b-versatile"

    def complete(self, prompt: str, system: str, json_mode: bool = False) -> str:
        resp = requests.post(
            self.url,
            timeout=TIMEOUT,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": 0.2,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        _raise_for(resp)
        return resp.json()["choices"][0]["message"]["content"]


class Anthropic:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=0)  # we do our own backoff
        self.model = model or "claude-sonnet-5-5"

    def complete(self, prompt: str, system: str, json_mode: bool = False) -> str:
        import anthropic

        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.RateLimitError as exc:
            ra = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise RateLimited(float(ra) if ra else None) from exc
        return "".join(b.text for b in msg.content if b.type == "text")


BACKENDS: dict[str, tuple[type, Callable[[], bool]]] = {
    "ollama": (Ollama, Ollama.available),
    "gemini": (Gemini, lambda: bool(os.environ.get("GEMINI_API_KEY"))),
    "groq": (Groq, lambda: bool(os.environ.get("GROQ_API_KEY"))),
    "anthropic": (Anthropic, lambda: bool(os.environ.get("ANTHROPIC_API_KEY"))),
}


def get_provider(name: str | None = None) -> Provider:
    name = (name or os.environ.get("AI_PROVIDER") or "").strip().lower()
    model = os.environ.get("AI_MODEL") or None
    if name:
        if name not in BACKENDS:
            raise AIUnavailable(f"unknown AI_PROVIDER {name!r}; choose from {', '.join(BACKENDS)}")
        cls, ok = BACKENDS[name]
        if not ok():
            raise AIUnavailable(f"{name} is not configured (missing key or Ollama not running)")
    else:
        name = next((n for n, (_, ok) in BACKENDS.items() if ok()), "")
        if not name:
            raise AIUnavailable(
                "no AI provider configured: start Ollama, or set GEMINI_API_KEY / GROQ_API_KEY / "
                "ANTHROPIC_API_KEY in the environment or .env"
            )
        cls = BACKENDS[name][0]
    try:
        provider = cls(model)
    except ImportError as exc:
        raise AIUnavailable(
            f"{name}: SDK missing ({exc.name}); pip install 'nasa-data-explorer[ai]'"
        ) from exc
    if isinstance(provider, Ollama):
        provider.resolve_model()
    return provider


# --- client with cache, token estimate and backoff -----------------------------------


def cache_dir() -> Path:
    d = Path(os.environ.get("NASA_EXPLORER_CACHE", Path.home() / ".cache" / "nasa_explorer" / "ai"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class AIClient:
    def __init__(self, provider: Provider | None = None, verbose: bool = True):
        self.provider = provider or get_provider()
        self.verbose = verbose
        self.tokens_sent = 0

    def ask(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        key = hashlib.sha256(
            f"{self.provider.name}|{self.provider.model}|{json_mode}|{system}|{prompt}".encode()
        ).hexdigest()
        path = cache_dir() / f"{key}.json"
        if path.exists():
            if self.verbose:
                print(f"  [ai] cache hit ({self.provider.name}/{self.provider.model})")
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        est = estimate_tokens(system + prompt)
        if self.verbose:
            print(f"  [ai] {self.provider.name}/{self.provider.model}: ~{est:,} input tokens")
        text = self._with_backoff(lambda: self.provider.complete(prompt, system, json_mode))
        self.tokens_sent += est
        path.write_text(
            json.dumps(
                {"text": text, "provider": self.provider.name, "model": self.provider.model},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return text

    def _with_backoff(self, fn: Callable[[], str]) -> str:
        delay = 2.0
        for attempt in range(MAX_RETRIES):
            try:
                return fn()
            except (RateLimited, requests.ConnectionError, requests.Timeout) as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = getattr(exc, "retry_after", None) or delay + random.uniform(0, 1)
                log.warning("AI call failed (%s); retrying in %.1fs", exc, wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")


# --- workflows -------------------------------------------------------------------------
# Each workflow is a generator: it yields Step prompts and receives the model's reply.
# The CLI drives it with AIClient (run_workflow); the browser drives the same generator
# through web/bridge.py and calls the provider from JavaScript (bring-your-own-key).


@dataclass
class Step:
    label: str
    system: str
    prompt: str
    json_mode: bool = True

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.system + self.prompt)


Workflow = Generator[Step, str, dict]

SYSTEM = (
    "You are a careful NASA Earth & space science data analyst helping a hackathon team. "
    "Be concrete and quantitative, never invent numbers, and say when something is uncertain."
)


def _lang_rule(lang: str) -> str:
    return (
        "Write every text value in Arabic (Modern Standard Arabic); keep JSON keys, variable names, "
        "units, numbers and mission names in English."
        if lang == "ar"
        else "Write in clear, plain English."
    )


DATA_SCHEMA = """{
  "overview": "2-4 sentences: what this data is and what it shows",
  "findings": [{"text": "one finding with its number", "value": <number or null>, "fact": "<exact fact path or null>"}],
  "quality_issues": [{"text": "problem and why it matters", "severity": "high|medium|low"}],
  "next_analyses": ["concrete analysis to run next"],
  "visualizations": ["chart/map that would work in a demo"],
  "hackathon_ideas": ["idea that uses this data"],
  "caveats": ["limitation of the data or of these statistics"]
}"""


def _data_context(analysis: dict, file: str, reader: str) -> tuple[str, dict]:
    from . import verify

    f = verify.facts(analysis)
    s = analysis.get("summary", {})
    variables = s.get("variables") or s.get("columns") or s.get("datasets") or []
    var_lines = [
        f"- {v.get('name', v.get('path'))}: {v.get('long_name', '')} [{v.get('units', '')}] dtype={v.get('dtype', '')}"
        for v in variables[:40]
        if isinstance(v, dict)
    ]
    quality = [f"- [{q['level']}] {q['message']}" for q in analysis.get("quality", [])]
    products = [
        f"- {p['name']} ({p['resolution']}): " + " ".join(p["caveats"])
        for p in analysis.get("products", [])
    ]
    context = f"""FILE: {file} (format: {reader})

VARIABLES / COLUMNS:
{chr(10).join(var_lines) or "- (none)"}

FACTS (computed by the tool; cite these exact paths):
{verify.facts_text(f)}

AUTOMATIC QUALITY CHECKS (trusted):
{chr(10).join(quality) or "- none raised"}

RECOGNISED NASA PRODUCT (trusted reference notes):
{chr(10).join(products) or "- not recognised"}

NOTES: {"; ".join(analysis.get("notes", [])) or "none"}"""
    return context, f


def data_workflow(analysis: dict, file: str, reader: str, lang: str) -> Workflow:
    """Draft -> verify every number against the facts -> one review round if needed."""
    from . import verify

    context, f = _data_context(analysis, file, reader)
    rules = (
        "Rules: only the summary below was computed - you never see the raw data. Every finding that "
        'contains a number MUST set "fact" to one exact path from FACTS and "value" to that number. '
        "Do not compute new numbers except simple unit conversions you state explicitly. "
        "Reflect the automatic quality checks and product notes in quality_issues/caveats."
    )
    raw = yield Step(
        "draft",
        SYSTEM,
        f"{context}\n\n{rules}\nReturn ONLY JSON with this shape:\n{DATA_SCHEMA}\n{_lang_rule(lang)}",
    )
    result = verify.parse_json(raw)
    if result is None:
        return {"kind": "data", "parse_error": True, "raw": raw, "rounds": 1}
    verify.verify_data(result, f)
    problems = verify.review_problems(result)
    rounds = 1
    if problems:
        fix = yield Step(
            "review",
            SYSTEM,
            (
                f"{context}\n\nYour previous answer:\n{json.dumps(result, ensure_ascii=False)}\n\n"
                "An automatic checker found these problems:\n- "
                + "\n- ".join(problems)
                + "\n\nReturn the corrected full JSON (same shape). Fix or drop each flagged claim; "
                f"cite exact FACTS paths. {_lang_rule(lang)}"
            ),
        )
        fixed = verify.parse_json(fix)
        if fixed is not None:
            result, rounds = verify.verify_data(fixed, f), 2
            result["fixed_in_review"] = len(problems) - len(verify.review_problems(result))
    result.update(kind="data", rounds=rounds)
    return result


PAPER_SCHEMA = """{
  "problem": [{"text": "...", "page": <int>, "quote": "short exact quote"}],
  "method": [{"text": "...", "page": <int>, "quote": "..."}],
  "data_used": [{"text": "...", "page": <int>, "quote": "..."}],
  "findings": [{"text": "finding with exact numbers", "page": <int>, "quote": "exact sentence containing the numbers"}],
  "limitations": [{"text": "...", "page": <int>, "quote": "..."}],
  "nasa_datasets": [{"name": "mission/dataset", "how_used": "...", "page": <int>}],
  "hackathon_ideas": ["5 ideas for using this paper in a NASA Space Apps project"]
}"""


def _page_chunks(pages: list[str], max_tokens: int | None = None) -> list[str]:
    max_tokens = max_tokens or CHUNK_TOKENS
    chunks, cur = [], ""
    for i, text in enumerate(pages, 1):
        block = f"\n[page {i}]\n{text.strip()}\n"
        if cur and estimate_tokens(cur + block) > max_tokens:
            chunks.append(cur)
            cur = ""
        cur += block[: max_tokens * 3]
    if cur:
        chunks.append(cur)
    return chunks


def paper_workflow(pages: list[str], title: str, lang: str) -> Workflow:
    """Map-reduce over page-tagged chunks; every claim cites a page and is checked against it."""
    from . import verify

    chunks = _page_chunks(pages)
    cite = (
        "Every item must cite the page from the [page N] markers and include a short exact quote."
    )
    if len(chunks) > 1:
        notes = []
        for i, ch in enumerate(chunks, 1):
            notes.append(
                (
                    yield Step(
                        f"map {i}/{len(chunks)}",
                        SYSTEM,
                        (
                            f"Extract from this part of the paper '{title}': problem, methods, data/missions used, key findings "
                            f"with exact numbers, limitations. Terse English bullet points, each ending with (p. N) and an exact quote.\n\n{ch}"
                        ),
                        json_mode=False,
                    )
                )
            )
        material = "\n\n".join(f"## Notes from part {i}\n{n}" for i, n in enumerate(notes, 1))
    else:
        material = chunks[0] if chunks else ""
    raw = yield Step(
        "summary",
        SYSTEM,
        (
            f"Paper: {title} ({len(pages)} pages)\n\n{material}\n\n{cite} Do not invent facts. "
            f"Return ONLY JSON with this shape:\n{PAPER_SCHEMA}\n{_lang_rule(lang)}"
        ),
    )
    result = verify.parse_json(raw)
    if result is None:
        return {"kind": "paper", "parse_error": True, "raw": raw, "rounds": 1}
    verify.verify_paper(result, pages)
    rounds = 1
    if (problems := verify.review_problems(result)) and len(chunks) == 1:
        fix = yield Step(
            "review",
            SYSTEM,
            (
                f"Paper text:\n{material}\n\nYour previous answer:\n{json.dumps(result, ensure_ascii=False)}\n\n"
                "A checker could not find these claims on the cited pages:\n- "
                + "\n- ".join(problems)
                + f"\n\nReturn the corrected full JSON: fix the page/quote, or drop the claim. {_lang_rule(lang)}"
            ),
        )
        fixed = verify.parse_json(fix)
        if fixed is not None:
            result, rounds = verify.verify_paper(fixed, pages), 2
    result.update(kind="paper", rounds=rounds)
    return result


def chat_workflow(question: str, context: str, lang: str) -> Workflow:
    answer = yield Step(
        "chat",
        SYSTEM,
        (
            f"{context}\n\nQuestion: {question}\n\nAnswer only from the material above. Cite fact paths "
            f"like [statistics.t2m.mean] or pages like [p. 3]. If the material does not contain the answer, say so. "
            f"{_lang_rule(lang)}"
        ),
        json_mode=False,
    )
    return {"kind": "chat", "answer": answer}


def run_workflow(workflow: Workflow, client: AIClient) -> dict:
    try:
        step = next(workflow)
        while True:
            if client.verbose:
                print(f"  [ai] step: {step.label}")
            step = workflow.send(client.ask(step.prompt, step.system, step.json_mode))
    except StopIteration as done:
        return done.value


def workflow_for(rep: FileReport, res: ReadResult, lang: str) -> Workflow:
    if res.kind == "document":
        return paper_workflow(res.pages, res.metadata.get("title") or rep.file, lang)
    return data_workflow(rep.analysis, rep.file, rep.reader, lang)


def interpret(rep: FileReport, res: ReadResult, lang: str) -> dict | None:
    """Called by the pipeline with --ai. Never raises: AI problems become a note."""
    try:
        client = AIClient()
        result = run_workflow(workflow_for(rep, res, lang), client)
        return {"provider": client.provider.name, "model": client.provider.model, **result}
    except Exception as exc:
        rep.analysis.setdefault("notes", []).append(f"AI layer skipped: {exc}")
        return None


def chat_context(
    analysis: dict, file: str, reader: str, pages: list[str] | None, question: str, k: int = 5
) -> str:
    """Material for a question about one file: computed facts, or the most relevant pages."""
    if not pages:
        return _data_context(analysis, file, reader)[0]
    from .qa import TfidfEmbedder

    emb = TfidfEmbedder()
    matrix = emb.fit(pages)
    best = sorted((-float(score), i) for i, score in enumerate(matrix @ emb.query(question)))[:k]
    blocks = [f"[page {i + 1}]\n{pages[i][:3000]}" for _, i in sorted(best, key=lambda x: x[1])]
    return f"DOCUMENT: {file}\n\n" + "\n\n".join(blocks)
