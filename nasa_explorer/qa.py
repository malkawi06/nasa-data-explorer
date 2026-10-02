"""'Ask your files': retrieval over everything processed into a reports folder.

Embeddings: sentence-transformers when installed and the model is available (cached on
disk per chunk), otherwise a dependency-free TF-IDF. Answers cite file names and pages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("nasa_explorer.qa")

TOP_K = 6
CHUNK_WORDS = 350
_TOKEN = re.compile(r"[\w\-]{2,}", re.UNICODE)


def load_chunks(out_dir: Path) -> list[dict]:
    chunks = []
    for f in sorted(Path(out_dir).glob("*.chunks.json")):
        for c in json.loads(f.read_text(encoding="utf-8")):
            words = c["text"].split()
            for i in range(0, max(1, len(words)), CHUNK_WORDS):  # split long pages
                chunks.append({**c, "text": " ".join(words[i : i + CHUNK_WORDS + 50])})
    return chunks


class TfidfEmbedder:
    name = "tfidf"

    def fit(self, texts: list[str]) -> np.ndarray:
        docs = [Counter(_TOKEN.findall(t.lower())) for t in texts]
        df = Counter(w for d in docs for w in d)
        self.vocab = {w: i for i, w in enumerate(df)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in self.vocab])
        return np.vstack([self._vec(d) for d in docs]) if docs else np.zeros((0, 1))

    def _vec(self, counts: Counter) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in counts.items():
            if w in self.vocab:
                v[self.vocab[w]] = 1 + math.log(c)
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v

    def query(self, text: str) -> np.ndarray:
        return self._vec(Counter(_TOKEN.findall(text.lower())))


class SentenceEmbedder:
    def __init__(self, cache: Path):
        from sentence_transformers import SentenceTransformer

        self.model_name = os.environ.get(
            "NASA_EXPLORER_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)
        self.name = f"st:{self.model_name}"
        self.cache = cache

    def fit(self, texts: list[str]) -> np.ndarray:
        self.cache.mkdir(parents=True, exist_ok=True)
        keys = [hashlib.sha1(f"{self.model_name}|{t}".encode()).hexdigest() for t in texts]
        store = self.cache / "embeddings.npz"
        known = dict(np.load(store)) if store.exists() else {}
        todo = [i for i, k in enumerate(keys) if k not in known]
        if todo:
            vecs = self.model.encode(
                [texts[i] for i in todo], normalize_embeddings=True, show_progress_bar=False
            )
            known.update({keys[i]: v for i, v in zip(todo, vecs, strict=True)})
            np.savez(store, **known)
        return np.vstack([known[k] for k in keys])

    def query(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]


def _embedder(out_dir: Path):
    if os.environ.get("NASA_EXPLORER_EMBEDDINGS", "auto") != "tfidf":
        try:
            return SentenceEmbedder(out_dir / ".index")
        except Exception as exc:  # not installed, or model not downloadable offline
            log.info("sentence-transformers unavailable (%s); using TF-IDF", type(exc).__name__)
    return TfidfEmbedder()


def search(question: str, out_dir: Path, k: int = TOP_K) -> list[dict]:
    chunks = load_chunks(out_dir)
    if not chunks:
        raise FileNotFoundError(
            f"no processed files in {out_dir}; run nasa-explore on a folder first"
        )
    emb = _embedder(Path(out_dir))
    matrix = emb.fit([c["text"] for c in chunks])
    scores = matrix @ emb.query(question)
    best = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in best if scores[i] > 0]


def cite(c: dict) -> str:
    return f"[{c['file']}{', p. ' + str(c['page']) if c.get('page') else ''}]"


def ask(
    question: str, out_dir: str | Path = "reports", lang: str = "en", use_ai: bool = True
) -> str:
    hits = search(question, Path(out_dir))
    if not hits:
        return "No relevant passages found."
    if use_ai:
        try:
            from .ai import SYSTEM, AIClient, _lang_rule

            context = "\n\n".join(f"SOURCE {cite(h)}\n{h['text'][:2500]}" for h in hits)
            prompt = (
                f"Question: {question}\n\nAnswer ONLY from the sources below. After every claim cite the "
                f"source exactly as written, e.g. [file.pdf, p. 3]. If the sources do not contain the answer, say so."
                f" {_lang_rule(lang)}\n\n{context}"
            )
            return AIClient().ask(prompt, SYSTEM)
        except Exception as exc:
            log.warning("AI answer unavailable (%s); showing the best matching passages", exc)
    lines = ["Most relevant passages (no AI provider configured):"]
    for h in hits:
        snippet = re.sub(r"\s+", " ", h["text"])[:400]
        lines.append(f"- {cite(h)} (score {h['score']:.2f}): {snippet}")
    return "\n".join(lines)
