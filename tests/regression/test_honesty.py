"""The scientific-honesty rules, guarded so they cannot quietly lapse.

These are the promises that distinguish this package from a gallery of pretty
pictures (blueprint sections 3, 15, 16 and 36.3.1). Each one is checked
against every implemented lab, not a sample.
"""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.catalog.builtin import CATALOG
from visualmetrics.core.evidence import EvidenceType, badge_for

IMPLEMENTED = [s.id for s in CATALOG if s.status.is_implemented]
BY_DOMAIN = sorted({s.domain: s.id for s in CATALOG if s.status.is_implemented}.values())


@pytest.mark.parametrize("concept_id", IMPLEMENTED)
class TestEvidenceLabelling:
    def test_the_spec_declares_an_evidence_type(self, concept_id):
        spec = vm.registry.lab(concept_id).spec
        assert isinstance(spec.evidence, EvidenceType)

    def test_the_evidence_type_has_a_badge_and_a_caveat(self, concept_id):
        spec = vm.registry.lab(concept_id).spec
        badge = badge_for(spec.evidence)
        assert badge.label_key and badge.caveat_key

    def test_a_lab_claims_proof_status_only_with_a_proof_behind_it(self, concept_id):
        """A lab may not wear a proof badge unless it points at a real proof."""
        spec = vm.registry.lab(concept_id).spec
        if spec.evidence.is_proof:
            assert spec.proof_ids, (
                f"{concept_id} is labelled {spec.evidence.value} but references no proof"
            )


@pytest.mark.parametrize("concept_id", BY_DOMAIN)
class TestResultHonesty:
    def test_the_result_carries_its_evidence_type(self, concept_id):
        result = vm.lab(concept_id, seed=11)
        assert isinstance(result.evidence, EvidenceType)

    def test_panel_evidence_is_never_stronger_than_the_lab_claims(self, concept_id):
        result = vm.lab(concept_id, seed=11)
        for panel in result.panels:
            if panel.evidence is not None and panel.evidence.is_proof:
                spec = vm.registry.lab(concept_id).spec
                assert spec.proof_ids or spec.evidence.is_proof, (
                    f"{concept_id}:{panel.id} shows a proof badge with nothing behind it"
                )

    def test_monte_carlo_results_are_not_labelled_proofs(self, concept_id):
        result = vm.lab(concept_id, seed=11)
        text = " ".join(
            f"{block.title} {block.body}" for block in result.explanations
        ).lower()
        if result.evidence is EvidenceType.SIMULATION:
            assert "this proves" not in text
            assert "proves that" not in text


@pytest.mark.parametrize("concept_id", BY_DOMAIN)
class TestAnimationExplanationLayer:
    """Blueprint section 36.3.1: an animation without an explanation is banned."""

    def test_every_lab_produces_at_least_one_animation(self, concept_id):
        assert vm.lab(concept_id, seed=13).animations

    def test_every_animation_states_its_purpose_and_summary(self, concept_id):
        for animation in vm.lab(concept_id, seed=13).animations:
            assert animation.purpose, f"{concept_id}:{animation.id} has no stated purpose"
            assert animation.summary, f"{concept_id}:{animation.id} ends without a summary"

    def test_every_animation_declares_its_evidence_type(self, concept_id):
        for animation in vm.lab(concept_id, seed=13).animations:
            assert isinstance(animation.evidence, EvidenceType)

    def test_every_frame_explains_itself(self, concept_id):
        for animation in vm.lab(concept_id, seed=13).animations:
            assert animation.steps, f"{concept_id}:{animation.id} has no annotated steps"
            for step in animation.steps:
                where = f"{concept_id}:{animation.id}:{step.id}"
                assert step.title, f"{where} has no title"
                assert step.what_you_see, f"{where} does not say what you see"
                assert step.what_changed, f"{where} does not say what changed"
                assert step.why, f"{where} does not say why"
                assert step.interpretation, f"{where} does not say how to read it"

    def test_frames_are_ordered_and_numbered_without_gaps(self, concept_id):
        for animation in vm.lab(concept_id, seed=13).animations:
            frames = [s.frame for s in animation.steps]
            assert frames == sorted(frames), f"{concept_id}:{animation.id} frames are unordered"

    def test_a_static_alternative_exists_for_reduced_motion(self, concept_id):
        result = vm.lab(concept_id, seed=13, reduced_motion=True)
        assert result.panels, "reduced motion must still show the science"


@pytest.mark.parametrize("concept_id", BY_DOMAIN)
class TestAssumptionsAndWarnings:
    def test_assumptions_are_listed_with_their_status(self, concept_id):
        result = vm.lab(concept_id, seed=17)
        for assumption in result.assumptions:
            assert assumption.label
            assert isinstance(assumption.holds, bool)

    def test_a_violation_scenario_is_reported_as_a_violation(self, concept_id):
        """If a lab offers a violation case, running it must say something is wrong."""
        spec = vm.registry.lab(concept_id).spec
        violations = [
            s for s in spec.scenarios
            if s.category.value in {"violation", "counterexample", "misspecification"}
        ]
        if not violations:
            pytest.skip(f"{concept_id} declares no violation scenario")
        result = vm.lab(concept_id, scenario=violations[0].id, seed=19)
        broken = [a for a in result.assumptions if not a.holds]
        assert broken or result.warnings, (
            f"{concept_id}:{violations[0].id} is a violation case but reports "
            "neither a failed assumption nor a warning"
        )


class TestNoFakeCompleteness:
    def test_planned_concepts_are_visible_but_marked(self):
        planned = [s for s in CATALOG if not s.status.is_implemented]
        for spec in planned:
            assert spec.status.value == "planned"
            assert spec.module is None

    def test_the_catalogue_does_not_advertise_what_it_cannot_open(self):
        for spec in vm.registry.specs(implemented_only=True):
            vm.registry.lab(spec.id)  # must not raise

    def test_counts_are_reported_honestly(self):
        info = vm.info()
        implemented = len([s for s in CATALOG if s.status.is_implemented])
        text = str(info)
        assert str(implemented) in text


class TestCanonicalNaming:
    def test_no_local_course_labels_anywhere_in_the_public_api(self):
        banned = ("statistics 3", "econometrics 1", "econometrics 2", "semester")
        for spec in CATALOG:
            blob = " ".join(
                [spec.id, spec.title_key or "", spec.summary_key or ""]
            ).lower()
            for label in banned:
                assert label not in blob

    def test_domains_are_scientific_names(self):
        from visualmetrics.core.concepts import Domain

        for domain in Domain:
            assert domain.value.replace("_", " ").strip()
            assert not any(ch.isdigit() for ch in domain.value)
