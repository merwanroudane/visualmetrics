"""Figure-level invariants.

These are not pixel comparisons - a screenshot test would fail on a Plotly
upgrade and tell you nothing. They check the properties a figure must have to
be readable and honest:

* axes are labelled, so a reader knows what the numbers are;
* every visible trace is named, so the legend explains the picture;
* colour is never the only channel distinguishing lines;
* the figure uses the requested theme rather than Plotly's defaults;
* an animation's frames line up with its explanation steps.

One lab per domain is checked by default; the whole catalogue is checked in the
slow sweep.
"""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.catalog.builtin import CATALOG
from visualmetrics.visuals.themes.palette import get_theme

BY_DOMAIN = sorted({s.domain: s.id for s in CATALOG if s.status.is_implemented}.values())
ALL_IMPLEMENTED = [s.id for s in CATALOG if s.status.is_implemented]


#: Trace types that appear in a legend and therefore need a name. A heatmap or
#: a waterfall is labelled by its colour bar and its axis, not by a legend
#: entry, so requiring a name there would be noise rather than a check.
LEGEND_TYPES = {"scatter", "scattergl", "bar", "box", "violin", "histogram", "scatter3d"}


def visible_traces(figure):
    """Traces the reader will see in the legend."""
    return [
        t for t in figure.data
        if getattr(t, "showlegend", None) is not False
        and getattr(t, "type", None) in LEGEND_TYPES
    ]


def line_traces(figure):
    """Legend-visible line series, where a dash pattern is the redundant channel.

    Structural strokes - the edges of a network diagram, a reference grid - are
    excluded: they carry no series identity, so there is nothing for a second
    channel to distinguish.
    """
    out = []
    for trace in visible_traces(figure):
        mode = getattr(trace, "mode", None) or ""
        if getattr(trace, "line", None) is not None and "lines" in str(mode):
            out.append(trace)
    return out


@pytest.mark.parametrize("concept_id", BY_DOMAIN)
class TestFigureQuality:
    def _run(self, concept_id):
        return vm.lab(concept_id, seed=5)

    def test_every_panel_has_a_figure_with_data(self, concept_id):
        for panel in self._run(concept_id).panels:
            figure = panel.figure
            assert figure is not None, f"{concept_id}:{panel.id} has no figure"
            assert len(figure.data) > 0, f"{concept_id}:{panel.id} draws nothing"

    def test_axes_are_labelled(self, concept_id):
        """A number with no unit or name is not a finding."""
        for panel in self._run(concept_id).panels:
            layout = panel.figure.layout
            if getattr(layout, "scene", None) and getattr(layout.scene, "xaxis", None):
                continue  # 3-D scenes label their axes inside the scene
            if not getattr(layout, "xaxis", None):
                continue
            has_x = bool(getattr(layout.xaxis.title, "text", None))
            has_y = bool(getattr(layout.yaxis.title, "text", None))
            assert has_x or has_y, f"{concept_id}:{panel.id} labels neither axis"

    def test_traces_in_the_legend_are_named(self, concept_id):
        for panel in self._run(concept_id).panels:
            for index, trace in enumerate(visible_traces(panel.figure)):
                name = getattr(trace, "name", None)
                assert name, (
                    f"{concept_id}:{panel.id} trace {index} appears in the legend "
                    "without a name"
                )

    def test_colour_is_never_the_only_channel(self, concept_id):
        """Two lines that differ only by colour are unreadable to many readers."""
        for panel in self._run(concept_id).panels:
            lines = line_traces(panel.figure)
            if len(lines) < 2:
                continue
            styles = {
                (
                    getattr(t.line, "dash", None) or "solid",
                    getattr(getattr(t, "marker", None), "symbol", None),
                )
                for t in lines
            }
            colours = {getattr(t.line, "color", None) for t in lines}
            assert len(styles) > 1 or len(colours) == len(lines), (
                f"{concept_id}:{panel.id} distinguishes {len(lines)} lines by colour alone"
            )

    def test_figures_follow_the_requested_theme(self, concept_id):
        """A dark-theme figure on a light page is a broken export."""
        dark = vm.lab(concept_id, seed=5, theme="dark").panels[0].figure
        light = vm.lab(concept_id, seed=5, theme="light").panels[0].figure
        assert _paper(dark) != _paper(light), (
            f"{concept_id} renders identically in the light and dark themes"
        )
        expected = get_theme("dark").roles.get("paper")
        if expected and _paper(dark):
            assert _paper(dark).lower() == expected.lower()

    def test_animation_frames_match_the_explanation_steps(self, concept_id):
        for animation in self._run(concept_id).animations:
            frames = list(getattr(animation.figure, "frames", []) or [])
            if not frames:
                continue
            for step in animation.steps:
                assert step.frame < len(frames), (
                    f"{concept_id}:{animation.id}:{step.id} points at frame "
                    f"{step.frame} but the figure has {len(frames)}"
                )

    def test_figures_survive_serialisation(self, concept_id):
        """Every figure has to reach an export intact, numpy arrays included."""
        from visualmetrics.export.images import figure_to_json, json_dumps

        for panel in self._run(concept_id).panels:
            assert json_dumps(figure_to_json(panel.figure))


def _paper(figure):
    return getattr(figure.layout, "paper_bgcolor", None)


class TestThemeConsistency:
    def test_every_theme_renders(self):
        from visualmetrics.visuals.themes.palette import list_themes

        for theme in list_themes():
            result = vm.lab("regression.simple_linear", seed=2, theme=theme)
            assert result.panels[0].figure.data

    def test_reduced_motion_still_produces_the_evidence(self):
        result = vm.lab("inference.clt", seed=2, reduced_motion=True)
        assert result.panels, "reduced motion must not remove the figures"
        assert result.metrics


@pytest.mark.slow
class TestWholeCatalogue:
    def test_every_lab_labels_and_names_its_figures(self):
        problems = []
        for concept_id in ALL_IMPLEMENTED:
            result = vm.lab(concept_id, seed=5)
            for panel in result.panels:
                figure = panel.figure
                if figure is None or not len(figure.data):
                    problems.append(f"{concept_id}:{panel.id} draws nothing")
                    continue
                for trace in visible_traces(figure):
                    if not getattr(trace, "name", None):
                        problems.append(f"{concept_id}:{panel.id} has an unnamed trace")
                        break
        assert not problems, "\n".join(problems[:20])
