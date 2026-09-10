"""The notebook renderer.

The rule under test: a notebook cell shows the scientific layer, not just a
figure. A result rendered without its badge, its violated assumptions or its
warnings would be exactly the artefact that gets pasted into a paper draft
stripped of context.
"""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics import notebook as nb


@pytest.fixture(scope="module")
def result():
    return vm.lab("inference.hypothesis_testing", scenario="null_true", seed=3)


@pytest.fixture(scope="module")
def flagged():
    """A run that reports trouble, so the warning path is exercised."""
    return vm.lab("timeseries.garch", scenario="integrated_garch", seed=19)


class TestResultHtml:
    def test_the_badge_and_its_caveat_are_rendered(self, result):
        html = nb.result_html(result)
        assert "vm-nb-badge" in html
        assert result.evidence is not None

    def test_the_numbers_are_included(self, result):
        html = nb.result_html(result)
        assert result.metrics[0].label.split()[0] in html

    def test_warnings_are_surfaced(self, flagged):
        assert flagged.warnings
        html = nb.result_html(flagged)
        assert "vm-nb-warn" in html

    def test_violated_assumptions_are_surfaced(self, flagged):
        broken = [a for a in flagged.assumptions if not a.holds]
        assert broken, "this fixture is meant to have a violated assumption"
        html = nb.result_html(flagged)
        assert broken[0].label[:20] in html

    def test_html_is_escaped(self):
        from visualmetrics.core.state import LabResult

        result = LabResult(concept_id="x")
        result.metric("k", "<script>alert(1)</script>", 1.0)
        html = nb.result_html(result)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_the_style_is_scoped_to_the_package(self, result):
        html = nb.result_html(result)
        for rule in html.split("<style>")[1].split("</style>")[0].split("}"):
            selector = rule.split("{")[0].strip()
            if selector and not selector.startswith("@"):
                assert selector.startswith(".vm-nb"), (
                    f"style rule {selector!r} would leak into the notebook's own theme"
                )

    @pytest.mark.parametrize("language", ["en", "ar", "fr"])
    def test_renders_in_every_language(self, result, language):
        assert nb.result_html(result, language=language)


class TestAnimationHtml:
    def test_every_frame_explanation_is_present(self, result):
        animation = result.animations[0]
        html = nb.animation_html(animation)
        for step in animation.steps:
            assert step.what_you_see[:25].replace("'", "&#x27;") in html

    def test_the_purpose_and_summary_travel_with_it(self, result):
        animation = result.animations[0]
        html = nb.animation_html(animation)
        assert animation.purpose[:25].replace("'", "&#x27;") in html
        assert animation.summary[:25].replace("'", "&#x27;") in html

    def test_an_unexplained_animation_says_so(self):
        from visualmetrics.core.state import AnimationSpec

        html = nb.animation_html(AnimationSpec(id="bare", figure=None, steps=[]))
        assert "vm-nb-warn" in html

    @pytest.mark.parametrize("language", ["en", "ar", "fr"])
    def test_the_commentary_headers_are_translated(self, result, language):
        import re

        html = nb.animation_html(result.animations[0], language=language)
        headers = re.findall(r"<th>([^<]+)</th>", html)
        assert len(headers) == 5
        if language != "en":
            assert headers[1] != "What you see"


class TestIntegration:
    def test_setup_is_safe_outside_ipython(self):
        assert nb.setup() is False
        assert nb.in_notebook() is False

    def test_widgets_are_optional_and_named_when_missing(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            from visualmetrics.core.exceptions import MissingDependencyError

            with pytest.raises(MissingDependencyError) as excinfo:
                nb.interact("inference.power")
            assert "ipywidgets" in str(excinfo.value)
        else:
            assert callable(nb.interact)

    def test_widgets_are_built_from_the_same_control_specs(self):
        pytest.importorskip("ipywidgets")
        spec = vm.registry.lab("inference.power").spec
        for control in spec.controls:
            widget = nb.control_widget(control)
            assert widget is not None
