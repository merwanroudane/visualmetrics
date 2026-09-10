"""Core dataclasses: evidence, controls, scenarios, state, registry."""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.core.controls import ControlKind, int_slider, select, slider, toggle
from visualmetrics.core.evidence import EVIDENCE_BADGES, EvidenceType, badge_for
from visualmetrics.core.exceptions import ConceptNotFoundError, InvalidParameterError
from visualmetrics.core.scenarios import ScenarioCategory, scenario
from visualmetrics.core.state import AnimationSpec, AnimationStep, LabResult, LabState, Panel


class TestEvidence:
    def test_only_three_kinds_count_as_proof(self):
        proofs = {e for e in EvidenceType if e.is_proof}
        assert proofs == {
            EvidenceType.FORMAL_PROOF,
            EvidenceType.SYMBOLIC_DERIVATION,
            EvidenceType.GEOMETRIC_PROOF,
        }

    def test_simulation_is_never_a_proof(self):
        assert not EvidenceType.SIMULATION.is_proof
        assert not EvidenceType.NUMERICAL_DEMONSTRATION.is_proof
        assert not EvidenceType.VISUAL_INTUITION.is_proof

    def test_every_kind_has_a_badge_with_a_caveat(self):
        for kind in EvidenceType:
            badge = badge_for(kind)
            assert badge.label_key and badge.caveat_key
            assert badge.icon
        assert set(EVIDENCE_BADGES) == set(EvidenceType)

    def test_badge_accepts_the_string_form(self):
        assert badge_for("simulation").evidence is EvidenceType.SIMULATION


class TestControls:
    def test_slider_coerces_and_clamps(self):
        c = slider("alpha", 0.05, 0.001, 0.2, step=0.001)
        assert c.kind is ControlKind.SLIDER
        assert c.coerce("0.1") == pytest.approx(0.1)
        assert c.coerce(5.0) == pytest.approx(0.2)      # clipped to max
        assert c.coerce(-1.0) == pytest.approx(0.001)   # clipped to min

    def test_int_slider_returns_int(self):
        c = int_slider("n", 100, 10, 1000)
        assert isinstance(c.coerce("250"), int)

    def test_select_rejects_unknown_option(self):
        c = select("kind", "increasing", ["increasing", "decreasing"])
        assert c.coerce("decreasing") == "decreasing"
        with pytest.raises(InvalidParameterError):
            c.coerce("sideways")

    def test_toggle_reads_common_spellings(self):
        c = toggle("robust", True)
        assert c.coerce("false") is False
        assert c.coerce(1) is True


class TestScenarios:
    def test_scenario_carries_its_category(self):
        s = scenario("violation", ScenarioCategory.VIOLATION, hetero=0.9)
        assert s.id == "violation"
        assert s.category is ScenarioCategory.VIOLATION
        assert s.overrides["hetero"] == pytest.approx(0.9)

    def test_categories_cover_the_blueprint_taxonomy(self):
        names = {c.value for c in ScenarioCategory}
        for required in ("canonical", "violation", "counterexample", "boundary", "null"):
            assert required in names


class TestState:
    def test_lab_state_round_trips_through_a_dict(self):
        st = LabState(concept_id="inference.power", parameters={"alpha": 0.01}, seed=7)
        restored = LabState.from_dict(st.to_dict())
        assert restored.concept_id == st.concept_id
        assert restored.parameters == st.parameters
        assert restored.seed == 7

    def test_lab_result_helpers(self):
        r = LabResult()
        r.add_panel(Panel(id="main", figure=None))
        r.metric("size", "Rejection rate", 0.048)
        r.explain("interpret", "What this means", "text")
        r.assume("normality", "Errors are normal", True)
        assert r.metric_dict()["size"] == pytest.approx(0.048)
        assert r.figures() == {"main": None}
        assert r.assumptions[0].holds is True

    def test_animation_step_requires_pedagogical_fields(self):
        step = AnimationStep(
            id="s1",
            frame=0,
            title="Start",
            what_you_see="The sampling distribution",
            what_changed="Nothing yet",
            why="Baseline frame",
            interpretation="Compare with the next frame",
        )
        spec = AnimationSpec(id="a", figure=None, steps=[step], purpose="show drift")
        assert spec.purpose
        assert spec.steps[0].what_you_see


class TestRegistry:
    def test_unknown_concept_raises(self):
        with pytest.raises(ConceptNotFoundError):
            vm.registry.lab("does.not.exist")

    def test_planned_concept_refuses_to_open_honestly(self, planned_ids):
        if not planned_ids:
            pytest.skip("every catalogued concept is implemented")
        with pytest.raises(ConceptNotFoundError) as excinfo:
            vm.registry.lab(planned_ids[0])
        assert excinfo.value.key == "errors.concept_planned"

    def test_short_names_resolve_when_unambiguous(self):
        spec = vm.concept("power")
        assert spec.id == "inference.power"

    def test_specs_are_cheap_stubs_until_the_lab_is_opened(self):
        """A fresh registry holds catalog stubs; opening a lab swaps in the full spec.

        This uses its own registry rather than the shared one, because any
        earlier test that opened this lab would already have replaced the stub.
        """
        from visualmetrics.core.registry import ConceptRegistry

        registry = ConceptRegistry()
        stub = next(s for s in registry.specs() if s.id == "regression.fwl")
        assert len(stub.scenarios) == 0, "the catalog stub should carry no scenarios"
        assert stub.module, "the stub must still say where the lab lives"

        lab = registry.lab("regression.fwl")
        assert len(lab.spec.scenarios) > 0
        assert lab.spec.id == stub.id
        assert lab.spec.domain == stub.domain

        reloaded = next(s for s in registry.specs() if s.id == "regression.fwl")
        assert len(reloaded.scenarios) > 0, "the loaded spec should replace the stub"
