"""Exporters.

The rule these tests protect: an export carries the science with the picture.
A report that lost the evidence badge, the assumptions or the frame commentary
would be exactly the artefact this package refuses to produce.
"""

from __future__ import annotations

import json

import pytest

import visualmetrics as vm
from visualmetrics.core.exceptions import ExportError
from visualmetrics.core.state import LabState
from visualmetrics.export import (
    export_animation,
    export_animation_frames,
    export_html_report,
    export_result,
    load_config,
    load_configs,
    render_animation_html,
    render_html_report,
    result_to_dict,
    save_config,
    save_configs,
)

CONCEPT = "regression.simple_linear"


@pytest.fixture(scope="module")
def result():
    return vm.lab(CONCEPT, seed=5)


class TestFigureExport:
    def test_html_export_writes_one_file_per_panel(self, result, tmp_path):
        written = export_result(result, tmp_path / "figure.html")
        assert len(written) == len(result.panels)
        for path in written:
            assert path.exists() and path.stat().st_size > 1000

    def test_json_export_is_one_document_for_the_whole_result(self, result, tmp_path):
        written = export_result(result, tmp_path / "result.json")
        assert len(written) == 1
        data = json.loads(written[0].read_text(encoding="utf-8"))
        assert data["concept_id"] == CONCEPT
        assert data["evidence"], "the export dropped the evidence label"
        assert data["metrics"] and data["panels"]
        assert data["code"], "the export dropped the reproducible code"

    def test_an_unknown_suffix_is_refused_with_a_useful_message(self, result, tmp_path):
        with pytest.raises(ExportError, match="unsupported export format"):
            export_result(result, tmp_path / "figure.tiff")

    def test_static_formats_report_the_missing_dependency_by_name(self, result, tmp_path):
        pytest.importorskip
        try:
            import kaleido  # noqa: F401
        except Exception:  # noqa: BLE001
            from visualmetrics.core.exceptions import MissingDependencyError

            with pytest.raises(MissingDependencyError) as excinfo:
                export_result(result, tmp_path / "figure.png")
            assert "kaleido" in str(excinfo.value)
            assert "pip install" in excinfo.value.install_command
        else:
            written = export_result(result, tmp_path / "figure.png")
            assert all(p.exists() for p in written)

    def test_result_to_dict_is_json_serialisable(self, result):
        from visualmetrics.export.images import json_dumps

        text = json_dumps(result_to_dict(result))
        assert json.loads(text)["concept_id"] == CONCEPT


class TestHtmlReport:
    def test_the_report_carries_the_scientific_layer(self, result):
        from html import escape

        html = render_html_report(result)
        assert "<!doctype html>" in html.lower()
        for metric in result.metrics[:3]:
            assert escape(metric.label, quote=True) in html
        for assumption in result.assumptions[:2]:
            assert escape(assumption.label, quote=True) in html
        assert escape(result.code.splitlines()[0], quote=True) in html

    def test_the_evidence_badge_and_its_caveat_are_present(self, result):
        from visualmetrics.core.evidence import badge_for
        from visualmetrics.i18n.translator import get_translator

        html = render_html_report(result)
        from html import escape

        tr = get_translator("en")
        badge = badge_for(result.evidence)
        assert escape(tr.t(badge.label_key, ""), quote=True) in html
        caveat = tr.t(badge.caveat_key, "")
        if caveat:
            assert escape(caveat[:40], quote=True) in html

    def test_warnings_are_not_quietly_dropped(self):
        spec = vm.registry.lab("timeseries.garch").spec
        del spec
        garch = vm.lab("timeseries.garch", scenario="integrated_garch", seed=19)
        html = render_html_report(garch)
        from html import escape

        assert garch.warnings
        assert escape(garch.warnings[0][:40], quote=True) in html

    def test_the_animation_commentary_travels_with_the_report(self, result):
        from html import escape

        html = render_html_report(result)
        step = result.animations[0].steps[0]
        assert escape(step.what_you_see[:30], quote=True) in html
        assert escape(step.why[:30], quote=True) in html

    @pytest.mark.parametrize("language", ["en", "ar", "fr"])
    def test_reports_render_in_every_language_with_the_right_direction(self, language):
        html = render_html_report(vm.lab(CONCEPT, seed=5, language=language), language=language)
        assert f'lang="{language}"' in html
        assert ('dir="rtl"' in html) is (language == "ar")

    def test_a_result_without_a_figure_is_refused(self, tmp_path):
        from visualmetrics.core.state import LabResult

        with pytest.raises(ExportError):
            export_html_report(LabResult(concept_id="x"), tmp_path / "r.html")


class TestAnimationExport:
    def test_the_page_includes_every_frame_explanation(self, result):
        animation = result.animations[0]
        html = render_animation_html(animation)
        payload = json.loads(html.split("const VM = ", 1)[1].split(chr(59) + chr(10), 1)[0])
        exported = {s["id"]: s for s in payload["steps"]}
        for step in animation.steps:
            assert exported[step.id]["what_you_see"] == step.what_you_see
            assert exported[step.id]["what_changed"] == step.what_changed
            assert exported[step.id]["interpretation"] == step.interpretation

    def test_the_page_includes_the_purpose_and_the_summary(self, result):
        from html import escape

        animation = result.animations[0]
        html = render_animation_html(animation)
        # Text is HTML-escaped on the way in, so compare against the escaped form.
        assert escape(animation.purpose[:30], quote=True) in html
        assert escape(animation.summary[:30], quote=True) in html

    def test_export_writes_a_standalone_file(self, result, tmp_path):
        path = export_animation(result.animations[0], tmp_path / "anim.html")
        assert path.exists() and path.stat().st_size > 5000

    def test_a_static_suffix_is_refused_because_it_would_lose_the_commentary(
        self, result, tmp_path
    ):
        with pytest.raises(ExportError, match="explanation"):
            export_animation(result.animations[0], tmp_path / "anim.gif")

    def test_an_animation_without_steps_cannot_be_exported(self):
        from visualmetrics.core.state import AnimationSpec

        with pytest.raises(ExportError, match="explain"):
            render_animation_html(AnimationSpec(id="bare", figure=object(), steps=[]))

    def test_frame_export_always_writes_the_commentary_file(self, result, tmp_path):
        try:
            import kaleido  # noqa: F401
        except Exception:  # noqa: BLE001
            pytest.skip("kaleido is not installed; static frames cannot be rendered")
        written = export_animation_frames(result.animations[0], tmp_path / "frames")
        assert any(p.name.endswith("-explanation.json") for p in written)


class TestConfigs:
    def test_a_config_round_trips_through_json(self, tmp_path):
        state = LabState(concept_id=CONCEPT, parameters={"n": 40}, scenario=None, seed=5)
        restored = load_config(save_config(state, tmp_path / "cfg.json"))
        assert restored.concept_id == state.concept_id
        assert restored.seed == state.seed
        assert restored.parameters["n"] == 40

    def test_a_reloaded_config_reproduces_the_same_numbers(self, tmp_path):
        state = LabState(concept_id=CONCEPT, parameters={}, seed=77)
        first = vm.run_state(state).metric_dict()
        restored = load_config(save_config(state, tmp_path / "cfg.json"))
        assert vm.run_state(restored).metric_dict() == first

    def test_the_version_is_stamped(self, tmp_path):
        path = save_config(LabState(concept_id=CONCEPT), tmp_path / "cfg.json")
        assert json.loads(path.read_text(encoding="utf-8"))["visualmetrics_version"]

    def test_a_collection_round_trips(self, tmp_path):
        states = [
            LabState(concept_id=CONCEPT, seed=1),
            LabState(concept_id="inference.clt", seed=2),
        ]
        restored = load_configs(save_configs(states, tmp_path / "lesson.json", name="lesson"))
        assert [s.concept_id for s in restored] == [s.concept_id for s in states]

    def test_a_foreign_file_is_rejected_clearly(self, tmp_path):
        path = tmp_path / "not-a-config.json"
        path.write_text('{"hello": "world"}', encoding="utf-8")
        with pytest.raises(ExportError, match="not a VisualMetrics configuration"):
            load_config(path)

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(ExportError, match="no configuration file"):
            load_config(tmp_path / "absent.json")

    def test_yaml_is_used_when_available(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        del yaml
        state = LabState(concept_id=CONCEPT, seed=9)
        path = save_config(state, tmp_path / "cfg.yaml")
        assert "concept_id" in path.read_text(encoding="utf-8")
        assert load_config(path).seed == 9
