"""Browser test of the website: upload, reports, AI analysis (BYOK) and chat.

Pyodide is stubbed (see harness.py) and the Gemini endpoint is intercepted, so this runs
offline. tests/e2e/test_web_pyodide.py covers the real Pyodide runtime in CI.
"""

import json

import pytest

pw = pytest.importorskip("playwright.sync_api")

from .harness import STUB, Server, chromium_path  # noqa: E402

GEMINI = "https://generativelanguage.googleapis.com/**"


def _gemini_reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture()
def page(tmp_path):
    server = Server(tmp_path / "work")
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=chromium_path())
        except Exception as exc:
            server.close()
            pytest.skip(f"no Chromium available: {exc}")
        ctx = browser.new_context(viewport={"width": 1100, "height": 900})
        pg = ctx.new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        ctx.route(  # context-level so requests from the Web Worker are intercepted too
            "https://cdn.jsdelivr.net/**",
            lambda r: r.fulfill(status=200, content_type="text/javascript", body=STUB),
        )
        pg.goto(f"{server.url}/index.html")
        pg.wait_for_selector(".dot.ready", timeout=20000)
        pg.errors = errors
        yield pg
        browser.close()
    server.close()


def _report_text(page) -> str:
    return page.evaluate("document.querySelector('.preview').shadowRoot?.innerHTML || ''")


def test_upload_ai_and_chat(page, samples):
    page.set_input_files("#file-input", [str(samples["netcdf4"][0]), str(samples["pdf"][0])])
    page.wait_for_selector(".file-item:not(.pending)", timeout=60000)
    assert page.locator(".file-item").count() == 2
    assert page.locator("#detail").is_visible() and "kpis" in _report_text(page)
    page.locator(".file-item", has_text="paper.pdf").click()
    assert page.locator("#detail .file").inner_text() == "paper.pdf"
    page.locator(".file-item", has_text="grid.nc").click()

    # without a key, running AI opens the settings panel instead of failing
    page.get_by_role("tab", name="AI analysis").click()
    page.locator(".run-ai").click()
    assert page.locator("#ai-settings").get_attribute("open") is not None
    assert "API key" in page.locator(".ai-status").inner_text()

    page.select_option("#ai-provider", "gemini")
    page.fill("#ai-key", "test-key-1234567890")
    page.click("#ai-form button[type=submit]")

    calls: list[dict] = []

    def gemini(route):
        body = json.loads(route.request.post_data)
        calls.append({"key": route.request.headers.get("x-goog-api-key"), "body": body})
        prompt = body["contents"][0]["parts"][0]["text"]
        if "Question:" in prompt:
            return route.fulfill(json=_gemini_reply("The mean is in [statistics.t2m.mean]."))
        if len(calls) == 1:
            return route.fulfill(status=429, headers={"retry-after": "1"}, body="slow down")
        wrong = {
            "overview": "Synthetic grid",
            "findings": [
                {"text": "warming 9.9 K/year", "value": 9.9, "fact": "trends.t2m.slope_per_year"}
            ],
        }
        if "automatic checker" not in prompt:
            return route.fulfill(json=_gemini_reply(json.dumps(wrong)))
        fixed = {
            "overview": "Synthetic grid",
            "findings": [
                {"text": "warming 0.6 K/year", "value": 0.6, "fact": "trends.t2m.slope_per_year"}
            ],
        }
        return route.fulfill(json=_gemini_reply(json.dumps(fixed)))

    page.context.route(GEMINI, gemini)
    page.locator(".run-ai").click()
    page.wait_for_selector(".ai-frame:not([hidden])", timeout=30000)
    status = page.locator(".ai-status").inner_text()
    assert status.startswith("Done: 1 verified"), status
    assert calls[0]["key"] == "test-key-1234567890"
    assert calls[1]["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert len(calls) == 3  # 429 retried, then draft, then review
    panel = page.evaluate("document.querySelector('.ai-frame').shadowRoot.innerHTML")
    assert "b-verified" in panel and "Synthetic grid" in panel
    page.get_by_role("tab", name="Report").click()
    assert "ai-panel" in _report_text(page) and "Synthetic grid" in _report_text(
        page
    )  # report updated too

    page.get_by_role("tab", name="Ask this file").click()
    page.locator(".ask-input").fill("What is the mean temperature?")
    page.locator(".ask-form button").click()
    page.wait_for_function(
        "document.querySelector('.qa .a')?.textContent.includes('statistics')", timeout=20000
    )
    assert not page.errors


def test_arabic_mobile_and_queue_before_ready(page, samples):
    page.set_viewport_size({"width": 390, "height": 900})
    page.click("button[data-lang=ar]")
    page.set_input_files("#file-input", [str(samples["csv"][0])])
    page.wait_for_selector(".file-item:not(.pending)", timeout=60000)
    assert page.evaluate("document.documentElement.dir") == "rtl"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert "تحليل الذكاء الاصطناعي" in page.locator("[data-tab=ai]").inner_text()
    assert 'dir="rtl"' in _report_text(page)
