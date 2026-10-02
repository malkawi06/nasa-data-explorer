"""CLI, notebook API, Arabic output, plug-in registry, AI layer and Q&A."""

import json
from pathlib import Path

import pytest

from nasa_explorer import ai, explore, qa
from nasa_explorer.cli import main
from nasa_explorer.core import ReadOptions, ReadResult
from nasa_explorer.registry import _REGISTRY, read_file, reader


def test_cli_folder_and_ask(samples, tmp_path, capsys):
    folder = tmp_path / "in"
    folder.mkdir()
    for key in ("pdf", "csv", "netcdf4"):
        p = samples[key][0]
        (folder / p.name).write_bytes(p.read_bytes())
    out = tmp_path / "reports"
    assert main([str(folder), "--out", str(out)]) == 0
    assert (out / "index.html").exists()
    assert main(["--ask", "Which precipitation dataset was used?", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "[paper.pdf, p. 2]" in text


def test_cli_bad_bbox():
    with pytest.raises(SystemExit):
        main(["x.nc", "--bbox", "1,2,3"])


def test_explore_api(samples, tmp_path):
    rep = explore(samples["csv"][0], out_dir=tmp_path, display=False)
    assert "<html" in rep._repr_html_()


def test_arabic_report(samples, tmp_path):
    rep = process_ar(samples["csv"][0], tmp_path)
    html = rep.html_path.read_text(encoding="utf-8")
    assert 'dir="rtl"' in html and "الإحصاءات" in html


def process_ar(path, tmp_path):
    from nasa_explorer.pipeline import process_file

    return process_file(path, ReadOptions(lang="ar"), tmp_path)


def test_new_reader_is_one_decorated_function(tmp_path):
    @reader("demo", category="Test", extensions=(".demo",), magic=(b"DEMO",))
    def read_demo(path: Path, opts: ReadOptions) -> ReadResult:
        import pandas as pd

        return ReadResult("table", pd.DataFrame({"a": [1, 2, 3]}))

    try:
        f = tmp_path / "x.anything"
        f.write_bytes(b"DEMO\n")
        res, name, _ = read_file(f)
        assert name == "demo" and res.kind == "table"
    finally:
        _REGISTRY[:] = [r for r in _REGISTRY if r.name != "demo"]


def test_missing_dependency_message(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda m, *a: None if m == "astropy" else real(m, *a)
    )
    f = tmp_path / "x.fits"
    f.write_bytes(b"SIMPLE  =                    T" + b" " * 100)
    _, name, problems = read_file(f)
    assert any("nasa-data-explorer[astro]" in p for p in problems)


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, fail_times=0):
        self.calls, self.fail_times = [], fail_times

    def complete(self, prompt, system):
        self.calls.append(prompt)
        if len(self.calls) <= self.fail_times:
            raise ai.RateLimited(retry_after=0.01)
        return f"answer #{len(self.calls)}"


def test_ai_cache_and_backoff(capsys):
    p = FakeProvider(fail_times=2)
    client = ai.AIClient(p)
    assert client.ask("hello") == "answer #3"  # two 429s, then success
    assert client.ask("hello") == "answer #3"  # cached: provider not called again
    assert len(p.calls) == 3
    out = capsys.readouterr().out
    assert "input tokens" in out and "cache hit" in out


def test_ai_data_prompt_has_no_raw_data(samples, tmp_path, monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(ai, "get_provider", lambda name=None: fake)
    from nasa_explorer.pipeline import process_file

    rep = process_file(samples["netcdf4"][0], ReadOptions(), tmp_path, ai=True)
    data = json.loads(rep.json_path.read_text(encoding="utf-8"))
    assert data["ai"]["provider"] == "fake"
    assert len(fake.calls[0]) < 30000 and "ONLY the computed summary" in fake.calls[0]


def test_ai_paper_map_reduce(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(ai, "CHUNK_TOKENS", 50)
    pages = ["word " * 120, "more " * 120, "end " * 120]
    ai.summarize_paper(pages, "T", "ar", ai.AIClient(fake, verbose=False))
    assert len(fake.calls) == 4  # 3 map steps + 1 reduce
    assert "[page 2]" in fake.calls[1] and "Arabic" in fake.calls[-1]


def test_no_provider_is_a_clear_message():
    with pytest.raises(ai.AIUnavailable, match="no AI provider configured"):
        ai.get_provider()


def test_ai_failure_is_a_note(samples, tmp_path):
    from nasa_explorer.pipeline import process_file

    rep = process_file(samples["csv"][0], ReadOptions(), tmp_path, ai=True)
    assert rep.error is None
    assert any("AI layer skipped" in n for n in rep.analysis["notes"])


def test_qa_without_ai(samples, tmp_path):
    from nasa_explorer.pipeline import process_file

    process_file(samples["pdf"][0], ReadOptions(), tmp_path)
    process_file(samples["csv"][0], ReadOptions(), tmp_path)
    hits = qa.search("GPM IMERG precipitation", tmp_path)
    assert hits[0]["file"] == "paper.pdf" and hits[0]["page"] == 2
    answer = qa.ask("brightness frp statistics", tmp_path, use_ai=False)
    assert "[fires.csv]" in answer
