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
import time
from collections.abc import Callable
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

    def complete(self, prompt: str, system: str) -> str: ...


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

    def complete(self, prompt: str, system: str) -> str:
        resp = requests.post(
            f"{self.host}/api/chat",
            timeout=TIMEOUT * 3,
            json={
                "model": self.model,
                "stream": False,
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
        self.model = model or "gemini-2.5-flash"

    def complete(self, prompt: str, system: str) -> str:
        from google.genai import errors, types

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
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

    def complete(self, prompt: str, system: str) -> str:
        resp = requests.post(
            self.url,
            timeout=TIMEOUT,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": 0.2,
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

    def complete(self, prompt: str, system: str) -> str:
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

    def ask(self, prompt: str, system: str = "") -> str:
        key = hashlib.sha256(
            f"{self.provider.name}|{self.provider.model}|{system}|{prompt}".encode()
        ).hexdigest()
        path = cache_dir() / f"{key}.json"
        if path.exists():
            if self.verbose:
                print(f"  [ai] cache hit ({self.provider.name}/{self.provider.model})")
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        est = estimate_tokens(system + prompt)
        if self.verbose:
            print(f"  [ai] {self.provider.name}/{self.provider.model}: ~{est:,} input tokens")
        text = self._with_backoff(lambda: self.provider.complete(prompt, system))
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


# --- prompts ---------------------------------------------------------------------------


def _lang_rule(lang: str) -> str:
    return (
        "Write the entire answer in Arabic (Modern Standard Arabic); keep variable names, units and mission names in English."
        if lang == "ar"
        else "Write in clear English."
    )


SYSTEM = (
    "You are a careful NASA Earth & space science data analyst helping a hackathon team. "
    "Be concrete, quantitative and honest about uncertainty. Use Markdown headings and bullet points."
)


def explain_data(analysis: dict, file: str, reader: str, lang: str, client: AIClient) -> str:
    brief = {k: analysis.get(k) for k in ("summary", "coverage", "statistics", "trends", "notes")}
    payload = json.dumps(brief, default=str)[:24000]
    prompt = f"""File: {file} (format: {reader}). Below is ONLY the computed summary and statistics - not the raw data.

{payload}

Explain:
1. What this data is and what it shows, in plain language.
2. Anything suspicious (fill values left in, impossible ranges, gaps, odd units, outliers, sampling).
3. Next analyses worth running and the best visualizations for a hackathon demo.
{_lang_rule(lang)}"""
    return client.ask(prompt, SYSTEM)


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


def summarize_paper(pages: list[str], title: str, lang: str, client: AIClient) -> str:
    """Map-reduce summary; every claim must cite (p. N)."""
    chunks = _page_chunks(pages)
    cite = "Cite the page for every claim as (p. N) using the [page N] markers."
    if len(chunks) > 1:
        notes = []
        for i, ch in enumerate(chunks, 1):
            print(f"  [ai] map step {i}/{len(chunks)}")
            notes.append(
                client.ask(
                    f"Extract from this part of the paper '{title}': problem, methods, data/missions used, key findings "
                    f"with exact numbers, limitations. {cite} Use terse bullet points in English.\n\n{ch}",
                    SYSTEM,
                )
            )
        material = "\n\n".join(f"## Notes from part {i}\n{n}" for i, n in enumerate(notes, 1))
    else:
        material = chunks[0] if chunks else ""
    prompt = f"""Paper: {title}

{material}

Produce a structured summary with these sections:
## Problem
## Method
## Data used
## Key findings (with numbers)
## Limitations
## NASA datasets / missions mentioned (with how they were used)
## 5 ideas for using this paper in a NASA Space Apps hackathon project
{cite} Do not invent facts that are not in the text. {_lang_rule(lang)}"""
    return client.ask(prompt, SYSTEM)


def interpret(rep: FileReport, res: ReadResult, lang: str) -> dict | None:
    """Called by the pipeline with --ai. Never raises: AI problems become a note."""
    try:
        client = AIClient()
        if res.kind == "document":
            text = summarize_paper(res.pages, res.metadata.get("title") or rep.file, lang, client)
        else:
            text = explain_data(rep.analysis, rep.file, rep.reader, lang, client)
        return {"provider": client.provider.name, "model": client.provider.model, "text": text}
    except Exception as exc:
        rep.analysis.setdefault("notes", []).append(f"AI layer skipped: {exc}")
        return None
