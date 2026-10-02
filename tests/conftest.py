import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("NASA_EXPLORER_EMBEDDINGS", "tfidf")
for key in ("AI_PROVIDER", "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(key, None)
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"  # never hit a real local model in tests

from samples import make_samples  # noqa: E402


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    return make_samples(tmp_path_factory.mktemp("samples"))


@pytest.fixture(autouse=True)
def _ai_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("NASA_EXPLORER_CACHE", str(tmp_path / "ai_cache"))
