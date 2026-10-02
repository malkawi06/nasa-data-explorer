"""Optional AI layer with swappable backends.

Provider is chosen by env AI_PROVIDER (ollama | gemini | groq | anthropic); if unset, the
first configured one wins in that order. AI_MODEL overrides the default model.
Only computed summaries are ever sent for data files - never the raw file. The one exception is
opt-in: with AI_SEND_IMAGES=1 a downscaled copy of a plain image goes to a vision-capable provider.
"""

from __future__ import annotations

import base64
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
VISION = {"gemini", "anthropic"}  # providers whose default models accept images
IMAGE_TOKENS = 1500  # rough cost of one 1024 px image
PREVIEW_SIDE = 480  # small copy stored with the result so the report can draw the boxes


class AIUnavailable(RuntimeError):
    pass


class RateLimited(RuntimeError):
    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited (429)")
        self.retry_after = retry_after


class Provider(Protocol):
    name: str
    model: str

    def complete(
        self, prompt: str, system: str, json_mode: bool = False, image: str | None = None
    ) -> str: ...


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

    def complete(
        self, prompt: str, system: str, json_mode: bool = False, image: str | None = None
    ) -> str:
        from google.genai import errors

        try:
            return self._generate(prompt, system, json_mode, image)
        except errors.APIError as exc:
            if getattr(exc, "code", None) != 404:
                raise
            replacement = self.newest_flash()
            if not replacement or replacement == self.model:
                raise
            log.warning("Gemini model %s is unavailable; switching to %s", self.model, replacement)
            self.model = replacement
            return self._generate(prompt, system, json_mode, image)

    def _generate(self, prompt: str, system: str, json_mode: bool, image: str | None) -> str:
        from google.genai import errors, types

        contents = (
            [types.Part.from_bytes(data=base64.b64decode(image), mime_type="image/jpeg"), prompt]
            if image
            else prompt
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=contents,
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

    def complete(
        self, prompt: str, system: str, json_mode: bool = False, image: str | None = None
    ) -> str:
        import anthropic

        content: str | list = prompt
        if image:
            content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image},
                },
                {"type": "text", "text": prompt},
            ]
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": content}],
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

    def ask(
        self, prompt: str, system: str = "", json_mode: bool = False, image: str | None = None
    ) -> str:
        key = hashlib.sha256(
            f"{self.provider.name}|{self.provider.model}|{json_mode}|{system}|{prompt}|{image or ''}".encode()
        ).hexdigest()
        path = cache_dir() / f"{key}.json"
        if path.exists():
            if self.verbose:
                print(f"  [ai] cache hit ({self.provider.name}/{self.provider.model})")
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        est = estimate_tokens(system + prompt) + (IMAGE_TOKENS if image else 0)
        if self.verbose:
            print(f"  [ai] {self.provider.name}/{self.provider.model}: ~{est:,} input tokens")
        extra = {"image": image} if image else {}
        text = self._with_backoff(
            lambda: self.provider.complete(prompt, system, json_mode, **extra)
        )
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
    image: str | None = None  # base64 JPEG, only for opt-in visual analysis

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.system + self.prompt) + (IMAGE_TOKENS if self.image else 0)


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
  "findings": [{"text": "one insight with its number", "value": <number or null>, "fact": "<exact fact path or null>"}],
  "quality_issues": [{"text": "problem and why it matters", "severity": "high|medium|low"}],
  "next_analyses": ["concrete analysis to run next"],
  "visualizations": ["chart/map that would work in a demo"],
  "hackathon_ideas": ["idea that uses this data"],
  "caveats": ["limitation of the data or of these statistics"]
}"""


IMAGE_SCHEMA = """{
  "overview": "2-4 sentences: what the image shows and what it could be used for",
  "image_type": "satellite true-colour | satellite false-colour | chart/plot | map | astronomy | photo | diagram | other",
  "findings": [{"text": "insight backed by a measurement", "value": <number or null>, "fact": "<exact fact path or null>"}],
  "visual_observations": [{"text": "one thing visible in the image", "box": [ymin, xmin, ymax, xmax] or null, "confidence": "high|medium|low"}],
  "quality_issues": [{"text": "problem and why it matters", "severity": "high|medium|low"}],
  "next_analyses": ["concrete analysis to run next"],
  "hackathon_ideas": ["idea that uses this image"],
  "caveats": ["limitation of the image or of these measurements"]
}"""

IMAGE_RULES = {
    True: (
        "You can see the image. Put every claim about its content in visual_observations, each "
        "with a box (0-1000 coordinates, [ymin, xmin, ymax, xmax]) around the region it describes, "
        "or null for the whole image. Describe only what is visible: do not name places, dates, "
        "instruments or missions unless they are written in the image or given in the metadata. "
        "For a chart, read its title, axes and units and describe the shape; any value read off a "
        "chart is an estimate and must say so. Findings are for the measured FACTS only "
        "(e.g. image.white_low_saturation_pct is white, unsaturated pixels: cloud, snow or a white "
        "background - say which, from what you see)."
    ),
    False: (
        "You cannot see the image, only the measurements below; do not describe its content "
        "beyond what they imply and leave visual_observations empty. Explain what the "
        "measurements suggest (type of image, exposure, sharpness, colour make-up) with care."
    ),
}


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


def data_workflow(
    analysis: dict, file: str, reader: str, lang: str, image: str | None = None
) -> Workflow:
    """Draft -> verify every number against the facts -> one review round if needed.
    Plain images get their own schema; `image` (base64 JPEG) lets the model see the picture."""
    from . import verify

    context, f = _data_context(analysis, file, reader)
    is_image = "image" in analysis
    rules = (
        "Rules: only the summary below was computed - you never see the raw data. Every finding that "
        'contains a number MUST set "fact" to one exact path from FACTS and "value" to that number. '
        "Do not compute new numbers except simple unit conversions you state explicitly. "
        "Write 4-7 findings that a scientist would find informative: patterns in time (trends, "
        "seasonality, unusual periods), in space (where values or events concentrate), and "
        "relationships between variables. Do not restate trivial facts (row counts, a maximum "
        "confidence score) unless they matter for interpretation. Round numbers to 2-3 significant "
        "figures. Any statement about a trend MUST say whether it is statistically significant "
        "(significant only when its mk_p < 0.05); never present a non-significant slope as a trend. "
        "Reflect the automatic quality checks and product notes in quality_issues/caveats."
    )
    if is_image:
        rules = (
            'Rules: every finding that contains a number MUST set "fact" to one exact path from '
            'FACTS and "value" to that number. ' + IMAGE_RULES[image is not None]
        )
    schema = IMAGE_SCHEMA if is_image else DATA_SCHEMA
    raw = yield Step(
        "draft",
        SYSTEM,
        f"{context}\n\n{rules}\nReturn ONLY JSON with this shape:\n{schema}\n{_lang_rule(lang)}",
        image=image,
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
            image=image,
        )
        fixed = verify.parse_json(fix)
        if fixed is not None:
            result, rounds = verify.verify_data(fixed, f), 2
            result["fixed_in_review"] = len(problems) - len(verify.review_problems(result))
    if is_image:
        verify.mark_visual(result)
        result.update(saw_image=image is not None)
    result.update(kind="image" if is_image else "data", rounds=rounds)
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
            step = workflow.send(client.ask(step.prompt, step.system, step.json_mode, step.image))
    except StopIteration as done:
        return done.value


def workflow_for(rep: FileReport, res: ReadResult, lang: str, send_image: bool = False) -> Workflow:
    if res.kind == "document":
        return paper_workflow(res.pages, res.metadata.get("title") or rep.file, lang)
    image = vision_image(rep) if send_image else None
    return data_workflow(rep.analysis, rep.file, rep.reader, lang, image)


def vision_image(rep: FileReport) -> str | None:
    """Downscaled JPEG of a plain image for a vision model (None for anything else)."""
    from .analysis.image_metrics import jpeg_b64

    return jpeg_b64(rep.source) if "image" in rep.analysis else None


def interpret(rep: FileReport, res: ReadResult, lang: str) -> dict | None:
    """Called by the pipeline with --ai. Never raises: AI problems become a note."""
    try:
        client = AIClient()
        send = os.environ.get("AI_SEND_IMAGES") == "1" and client.provider.name in VISION
        result = run_workflow(workflow_for(rep, res, lang, send), client)
        if result.get("saw_image"):
            from .analysis.image_metrics import jpeg_b64

            result["preview"] = jpeg_b64(rep.source, PREVIEW_SIDE)
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
