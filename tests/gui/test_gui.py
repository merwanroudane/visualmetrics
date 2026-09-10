"""The GUI.

Most of what could break here is decided before any widget exists: which tabs a
result gets, what the theme's CSS says, whether the motion preference is
honoured, whether a planned concept is offered as clickable. Those layers hold
no NiceGUI, so they are tested directly and quickly.

The thin rendering layer is covered by the route-building test, which is marked
``gui`` because it needs the optional extra.
"""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.core.evidence import EvidenceType
from visualmetrics.gui.accessibility.a11y import (
    aria_for_assumption,
    contrast_ratio,
    describe_series,
    figure_description,
    keyboard_shortcuts,
    meets_contrast,
)
from visualmetrics.gui.components.catalog import filter_catalog
from visualmetrics.gui.state.session import (
    LEVELS,
    THEMES,
    Session,
    session_from_query,
    session_to_query,
)
from visualmetrics.gui.state.viewmodel import (
    TAB_ORDER,
    build_lab_view,
    format_value,
)
from visualmetrics.gui.themes.css import is_dark, theme_css, theme_variables
from visualmetrics.visuals.themes.palette import list_themes

CONCEPT = "regression.simple_linear"


@pytest.fixture(scope="module")
def view():
    session = Session().open(CONCEPT)
    return build_lab_view(
        vm.run_state(session.lab_state()), translator=session.translator()
    )


class TestSession:
    def test_defaults_are_valid(self):
        session = Session()
        assert session.language in ("en", "ar", "fr")
        assert session.theme in THEMES
        assert session.level in LEVELS

    def test_nonsense_settings_fall_back_instead_of_raising(self):
        session = Session(language="klingon", theme="neon", level="wizard")
        assert (session.language, session.theme, session.level) == (
            "en", "light", "intermediate",
        )

    def test_only_arabic_is_right_to_left(self):
        assert Session(language="ar").is_rtl
        assert not Session(language="fr").is_rtl

    def test_opening_a_new_concept_drops_the_old_parameters(self):
        session = Session().open("inference.power")
        session.set_parameter("alpha", 0.01)
        session.open("inference.clt")
        assert session.parameters == {}, "parameters from another lab must not leak"

    def test_reopening_the_same_concept_keeps_parameters(self):
        session = Session().open("inference.power")
        session.set_parameter("alpha", 0.01)
        session.open("inference.power")
        assert session.parameters == {"alpha": 0.01}

    def test_choosing_a_scenario_clears_manual_edits(self):
        """Otherwise the preset would be shown while quietly overridden."""
        session = Session().open("inference.power")
        session.set_parameter("alpha", 0.2)
        session.choose_scenario("underpowered")
        assert session.parameters == {}
        assert session.scenario == "underpowered"

    def test_lab_state_carries_every_session_setting(self):
        session = Session(language="fr", theme="dark", level="advanced", seed=7)
        session.open(CONCEPT).set_parameter("n", 50)
        state = session.lab_state()
        assert state.concept_id == CONCEPT
        assert state.language == "fr" and state.theme == "dark"
        assert state.level == "advanced" and state.seed == 7
        assert state.parameters["n"] == 50

    def test_lab_state_without_a_concept_is_refused(self):
        with pytest.raises(ValueError, match="no concept is open"):
            Session().lab_state()

    def test_a_session_round_trips_through_a_query_string(self):
        session = Session(language="ar", theme="dark", seed=99).open(CONCEPT)
        session.choose_scenario("canonical")
        session.set_parameter("n", 250)
        restored = session_from_query(session_to_query(session))
        assert restored.language == "ar"
        assert restored.theme == "dark"
        assert restored.seed == 99
        assert restored.scenario == "canonical"
        assert restored.parameters["n"] == "250"

    def test_a_malformed_link_still_opens(self):
        restored = session_from_query({"lang": "xx", "seed": "not-a-number", "theme": ""})
        assert restored.language == "en"
        assert restored.seed == 42

    def test_to_dict_from_dict_round_trip(self):
        session = Session(language="fr", presentation=True).open(CONCEPT)
        assert Session.from_dict(session.to_dict()).to_dict() == session.to_dict()


class TestViewModel:
    def test_tabs_follow_the_canonical_order(self, view):
        positions = [TAB_ORDER.index(t) for t in view.tab_ids if t in TAB_ORDER]
        assert positions == sorted(positions), "tabs must not depend on lab append order"

    def test_the_badge_is_translated_and_carries_its_caveat(self, view):
        assert view.badge.label
        assert view.badge.caveat, "a badge without its caveat is half the message"
        assert isinstance(view.badge.evidence, EvidenceType)

    def test_metrics_are_formatted_for_reading(self, view):
        assert view.metrics
        for metric in view.metrics:
            assert metric.value and "e-" not in metric.value[:1]

    def test_every_lab_gets_a_figure_tab_and_an_animation_tab(self, view):
        assert "visualize" in view.tab_ids or any(t.panels for t in view.tabs)
        assert "animate" in view.tab_ids

    def test_assumptions_are_booleans(self, view):
        for assumption in view.assumptions:
            assert isinstance(assumption.holds, bool)

    def test_trouble_is_flagged_when_something_failed(self):
        session = Session().open("timeseries.garch")
        session.choose_scenario("integrated_garch")
        result = build_lab_view(
            vm.run_state(session.lab_state()), translator=session.translator()
        )
        assert result.has_trouble, "a violated scenario must be visible in the view model"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, "0.5"), (1234.0, "1234"), (0.0, "0"), (float("nan"), "n/a"), (True, "yes")],
    )
    def test_number_formatting(self, value, expected):
        assert format_value(value) == expected

    def test_very_small_numbers_use_scientific_notation(self):
        assert "e" in format_value(1.1e-15)

    @pytest.mark.parametrize("language", ["en", "ar", "fr"])
    def test_the_view_is_built_in_every_language(self, language):
        session = Session(language=language).open(CONCEPT)
        built = build_lab_view(
            vm.run_state(session.lab_state()), translator=session.translator()
        )
        assert built.title
        assert built.is_rtl is (language == "ar")


class TestThemes:
    @pytest.mark.parametrize("theme", list_themes())
    def test_every_theme_produces_usable_css(self, theme):
        css = theme_css(theme)
        assert "--vm-bg" in css and "--vm-ink" in css
        assert ".vm-badge" in css

    @pytest.mark.parametrize("theme", list_themes())
    def test_body_text_meets_wcag_aa_in_every_theme(self, theme):
        variables = theme_variables(theme)
        assert meets_contrast(variables["--vm-ink"], variables["--vm-bg"]), (
            f"{theme}: contrast is {contrast_ratio(variables['--vm-ink'], variables['--vm-bg']):.2f}"
        )

    def test_high_contrast_is_the_strongest(self):
        def ratio(theme: str) -> float:
            variables = theme_variables(theme)
            return contrast_ratio(variables["--vm-ink"], variables["--vm-bg"])

        assert ratio("high_contrast") >= ratio("light")

    def test_only_the_dark_theme_is_dark(self):
        assert is_dark("dark")
        assert not is_dark("light")

    def test_reduced_motion_disables_animation_outright(self):
        css = theme_css("light", reduced_motion=True)
        assert "animation-duration: 0s !important" in css
        assert "transition-duration: 0s !important" in css

    def test_arabic_gets_a_right_to_left_body_and_an_arabic_font(self):
        css = theme_css("light", language="ar")
        assert "direction: rtl" in css
        assert "Naskh" in css

    def test_presentation_mode_hides_the_controls(self):
        assert ".vm-hide-in-presentation" in theme_css("light", presentation=True)


class TestAccessibility:
    def test_a_figure_always_has_a_text_alternative(self, view):
        for tab in view.tabs:
            for panel in tab.panels:
                assert figure_description(panel, view).strip()

    def test_series_are_described_by_more_than_colour(self):
        described = describe_series("treated", dash="dash", symbol="square")
        assert "dashed" in described and "square" in described

    def test_assumption_status_is_exposed_to_assistive_technology(self, view):
        for assumption in view.assumptions:
            attributes = aria_for_assumption(assumption)
            assert attributes["role"] == "status"
            assert attributes["data-holds"] in ("true", "false")
            assert ("holds" in attributes["aria-label"]
                    or "violated" in attributes["aria-label"])

    def test_the_keyboard_map_has_no_duplicate_actions(self):
        shortcuts = keyboard_shortcuts()
        assert len(set(shortcuts.values())) == len(shortcuts)


class TestCatalogBrowsing:
    def test_planned_concepts_are_listed_by_default(self):
        assert len(filter_catalog()) > len(filter_catalog(include_planned=False))

    def test_they_can_be_hidden_on_request(self):
        assert all(s.status.is_implemented for s in filter_catalog(include_planned=False))

    def test_search_works_from_english(self):
        assert "inference.power" in [s.id for s in filter_catalog("power")]

    def test_search_works_from_arabic(self):
        assert filter_catalog("الانحدار"), "Arabic search returned nothing"

    def test_domain_filter(self):
        assert {s.domain.value for s in filter_catalog(domain="econometrics")} == {
            "econometrics"
        }


@pytest.mark.gui
class TestApplication:
    def test_every_route_is_declared(self):
        from nicegui import app as nicegui_app

        from visualmetrics.gui.app import build_app

        build_app()
        paths = {getattr(route, "path", "") for route in nicegui_app.routes}
        for expected in (
            "/", "/catalog", "/proofs", "/glossary", "/paths", "/doctor",
            "/lab/{concept_id}", "/proof/{proof_id}",
        ):
            assert expected in paths

    def test_building_twice_is_harmless(self):
        from visualmetrics.gui.app import build_app

        build_app()
        build_app()

    def test_the_public_launcher_exists(self):
        assert callable(vm.launch)
