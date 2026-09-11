"""End-to-end checks in a real browser.

These cover two failures that only appear once the page is actually laid out,
so no amount of unit testing would have caught them:

* a control panel taller than the viewport pushed the Run button below the
  fold. Scrolling down to reach it scrolled the result out of view, so pressing
  Run looked like it produced nothing at all;
* the stylesheet was appended on every render instead of replaced, so switching
  to Arabic and back left the whole layout mirrored under an English label.

Skipped unless Playwright and a browser are installed; they are not part of the
default development set.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = pytest.mark.gui

playwright_api = pytest.importorskip(
    "playwright.sync_api", reason="playwright is not installed"
)

CONCEPT = "probability.distributions"   # one of the tallest control panels
LOAD_MS = 6000


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def server() -> str:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # NiceGUI switches into its own screen-test mode when it sees pytest in the
    # environment, and then demands a port variable we do not set. The server
    # under test is an ordinary run, so the inherited pytest variables go.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    process = subprocess.Popen(
        [sys.executable, "-m", "visualmetrics", "gui", "--port", str(port), "--no-browser"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", env=environment,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                pytest.skip(f"the GUI exited ({process.returncode}): {output[-300:]}")
            try:
                urllib.request.urlopen(base, timeout=2)
                break
            except Exception:
                time.sleep(1)
        else:
            pytest.skip("the GUI did not answer in time")
        yield base
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as pw:
        try:
            instance = pw.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no chromium available: {exc}")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, server):
    page = browser.new_page(viewport={"width": 1400, "height": 760})
    page.goto(f"{server}/lab/{CONCEPT}", wait_until="networkidle")
    page.wait_for_timeout(LOAD_MS)
    yield page
    page.close()


def geometry(page) -> dict:
    return page.evaluate(
        """() => {
            // Found by position, not by label: the button is translated, so
            // matching on the word "Run" breaks the moment the language changes.
            const run = document.querySelector('.vm-controls-actions button');
            const output = document.getElementById('vm-output');
            const panel = document.querySelector('.vm-controls-scroll');
            const r = run.getBoundingClientRect();
            const o = output.getBoundingClientRect();
            return {
                run_visible: r.top >= 0 && r.bottom <= window.innerHeight,
                output_visible: o.bottom > 0 && o.top < window.innerHeight,
                output_text: (output.innerText || '').trim().length,
                page_height: document.documentElement.scrollHeight,
                viewport: window.innerHeight,
                panel_scrolls: panel.scrollHeight > panel.clientHeight + 1,
                direction: getComputedStyle(document.body).direction,
                style_tags: document.querySelectorAll('style').length,
            };
        }"""
    )


class TestLabLayout:
    def test_run_is_reachable_without_scrolling_the_result_away(self, page):
        state = geometry(page)
        assert state["run_visible"], "the Run button is below the fold"
        assert state["output_visible"], "the result is not on screen beside it"

    def test_a_tall_control_panel_does_not_stretch_the_page(self, page):
        state = geometry(page)
        assert state["panel_scrolls"], "this lab should have more controls than fit"
        assert state["page_height"] < state["viewport"] * 2.5, (
            "the controls are driving the page height again"
        )

    def test_pressing_run_leaves_the_result_on_screen(self, page):
        page.evaluate("window.scrollTo(0, 400)")
        page.wait_for_timeout(400)
        page.locator(".vm-controls-actions button").first.click()
        page.wait_for_timeout(LOAD_MS)

        state = geometry(page)
        assert state["output_visible"], "pressing Run scrolled the result out of view"
        assert state["output_text"] > 100, "Run produced an empty result area"


class TestLanguageSwitching:
    def _switch(self, page, index: int) -> None:
        page.locator(".q-select").first.click()
        page.wait_for_timeout(700)
        page.locator(".q-menu .q-item").nth(index).click()
        page.wait_for_timeout(LOAD_MS)

    def test_switching_back_from_arabic_restores_the_direction(self, page):
        assert geometry(page)["direction"] == "ltr"
        self._switch(page, 1)
        assert geometry(page)["direction"] == "rtl"
        self._switch(page, 0)
        assert geometry(page)["direction"] == "ltr", (
            "the layout stayed mirrored after returning to English"
        )

    def test_stylesheets_are_replaced_rather_than_accumulated(self, page):
        before = geometry(page)["style_tags"]
        for index in (1, 2, 0):
            self._switch(page, index)
        assert geometry(page)["style_tags"] <= before, (
            "each render added another stylesheet"
        )

    def test_the_result_survives_a_language_switch(self, page):
        self._switch(page, 1)
        state = geometry(page)
        assert state["output_text"] > 100
        assert state["output_visible"]
