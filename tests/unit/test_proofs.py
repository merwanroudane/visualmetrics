"""The declarative proof engine."""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.catalog.builtin import CATALOG
from visualmetrics.core.evidence import EvidenceType
from visualmetrics.proofs import (
    ProofNotFoundError,
    ProofSpec,
    ProofStep,
    StepKind,
    VisualAction,
    list_proofs,
    proof,
    proof_registry,
)

WRITTEN = [e.id for e in list_proofs(written_only=True)]


def make(**kw) -> ProofSpec:
    base = {
        "id": "t.proof",
        "kind": EvidenceType.FORMAL_PROOF,
        "title": "t",
        "claim": "c",
        "conclusion": "c",
        "steps": (ProofStep("a", "statement"),),
        "limitations": "does not establish anything else",
    }
    base.update(kw)
    return ProofSpec(**base)


class TestHonestyGuards:
    @pytest.mark.parametrize(
        "kind",
        [
            EvidenceType.SIMULATION,
            EvidenceType.NUMERICAL_DEMONSTRATION,
            EvidenceType.VISUAL_INTUITION,
            EvidenceType.EMPIRICAL_EXAMPLE,
        ],
    )
    def test_a_non_proof_kind_cannot_be_labelled_a_proof(self, kind):
        with pytest.raises(ValueError, match="not a proof kind"):
            make(kind=kind)

    def test_limitations_are_mandatory(self):
        with pytest.raises(ValueError, match="must state its limitations"):
            make(limitations="")

    def test_steps_are_mandatory(self):
        with pytest.raises(ValueError, match="no steps"):
            make(steps=())

    def test_duplicate_step_ids_are_rejected(self):
        with pytest.raises(ValueError, match="repeats step id"):
            make(steps=(ProofStep("a", "x"), ProofStep("a", "y")))

    def test_a_step_cannot_depend_on_something_that_does_not_exist(self):
        with pytest.raises(ValueError, match="neither an assumption"):
            make(steps=(ProofStep("a", "x", uses=("ghost",)),))

    def test_a_numerical_check_never_claims_proof_status(self):
        spec = proof("regression.ols.residual_orthogonality")
        outcome = spec.run_checks()[0]
        assert outcome.evidence is EvidenceType.NUMERICAL_DEMONSTRATION
        assert not outcome.evidence.is_proof


class TestRegistry:
    def test_an_unknown_proof_is_reported_not_invented(self):
        with pytest.raises(ProofNotFoundError) as excinfo:
            proof("no.such.proof")
        assert excinfo.value.key == "errors.proof_unknown"

    def test_every_catalogued_proof_loads(self):
        for entry in list_proofs(written_only=True):
            assert proof(entry.id).id == entry.id

    def test_loading_is_cached(self):
        assert proof(WRITTEN[0]) is proof(WRITTEN[0])

    def test_lookup_by_concept(self):
        ids = {e.id for e in proof_registry.for_concept("regression.ols_geometry")}
        assert "regression.ols.residual_orthogonality" in ids


@pytest.mark.parametrize("proof_id", WRITTEN)
class TestEveryProof:
    def test_structure(self, proof_id):
        p = proof(proof_id)
        assert p.kind.is_proof
        assert len(p.steps) >= 4, "a proof with fewer than four steps is a sketch"
        assert p.claim and p.conclusion and p.limitations
        assert p.intuition, "every proof states its idea in plain language first"
        assert p.references, "every proof cites a source"

    def test_the_chain_reaches_a_conclusion(self, proof_id):
        """Some step must actually conclude, and the chain must not end on scaffolding.

        A closing geometric restatement is allowed as a coda, but a proof that
        stops at a definition or a setup line has not finished.
        """
        p = proof(proof_id)
        assert any(s.kind is StepKind.CONCLUSION for s in p.steps)
        assert p.steps[-1].kind not in {
            StepKind.SETUP,
            StepKind.ASSUMPTION,
            StepKind.DEFINITION,
        }

    def test_every_step_says_why_it_is_allowed(self, proof_id):
        p = proof(proof_id)
        for step in p.steps:
            assert step.statement, f"{proof_id}:{step.id} has no statement"
            assert step.justification, f"{proof_id}:{step.id} does not justify itself"

    def test_essential_assumptions_say_what_breaks(self, proof_id):
        p = proof(proof_id)
        for a in p.assumptions:
            assert a.statement
            assert a.if_violated, f"{proof_id}:{a.id} never says what fails without it"

    def test_assumptions_are_actually_used(self, proof_id):
        p = proof(proof_id)
        used = {dep for step in p.steps for dep in step.uses}
        for a in p.assumptions:
            assert a.id in used, (
                f"{proof_id} lists assumption {a.id!r} but no step invokes it - "
                "an unused hypothesis is either decoration or a gap"
            )

    def test_concept_links_point_at_real_labs(self, proof_id):
        known = {s.id for s in CATALOG}
        for cid in proof(proof_id).concept_ids:
            assert cid in known

    def test_prerequisites_resolve(self, proof_id):
        known = set(proof_registry.ids())
        for pre in proof(proof_id).prerequisites:
            assert pre in known

    def test_numerical_checks_pass(self, proof_id):
        for outcome in proof(proof_id).run_checks():
            assert outcome.passed, f"{proof_id}:{outcome.id} -> {outcome.detail}"

    def test_navigation(self, proof_id):
        p = proof(proof_id)
        first = p.steps[0]
        assert p.step(first.id) is first
        assert p.index_of(first.id) == 0
        with pytest.raises(KeyError):
            p.step("no_such_step")


@pytest.mark.parametrize("language", ["en", "ar", "fr"])
class TestRendering:
    def test_renders_in_every_language(self, language):
        r = vm.proof("regression.fwl.theorem", language=language)
        assert r.title and r.claim and r.conclusion and r.limitations
        assert len(r.steps) == len(proof("regression.fwl.theorem"))
        assert r.steps[0].number == 1

    def test_structural_labels_are_translated(self, language):
        r = vm.proof("multivariate.pca.variance_maximization", language=language)
        labels = {s.kind_label for s in r.steps}
        if language == "en":
            assert "Key insight" in labels
        else:
            assert "Key insight" not in labels, "step-kind labels must not stay English"

    def test_rtl_flag_is_set_only_for_arabic(self, language):
        r = vm.proof("inference.clt.moment_generating", language=language)
        assert r.is_rtl is (language == "ar")

    def test_serialises_to_json_friendly_data(self, language):
        d = vm.proof("math.projection.shortest_distance", language=language).to_dict()
        import json

        text = json.dumps(d, ensure_ascii=False)
        assert d["kind"] in {e.value for e in EvidenceType}
        assert d["limitations"] in text

    def test_text_transcript_carries_the_limitations(self, language):
        text = vm.proof("ml.lasso.sparsity_geometry", language=language).to_text()
        assert "NOT" in text or "PAS" in text or "لا يثبته" in text


class TestVisualActions:
    def test_geometric_proofs_carry_drawing_instructions(self):
        for entry in list_proofs(written_only=True):
            p = proof(entry.id)
            if p.kind is EvidenceType.GEOMETRIC_PROOF:
                assert p.has_geometry, f"{p.id} claims to be geometric but draws nothing"

    def test_unknown_actions_are_impossible(self):
        for entry in list_proofs(written_only=True):
            for step in proof(entry.id).steps:
                assert isinstance(step.visual_action, VisualAction)


class TestPublicApi:
    def test_proofs_for_a_concept(self):
        assert "regression.fwl.theorem" in vm.proofs_for("regression.fwl")

    def test_labs_do_not_reference_proofs_that_do_not_exist(self):
        known = set(proof_registry.ids())
        for spec in vm.registry.specs(implemented_only=True):
            for pid in vm.registry.lab(spec.id).spec.proof_ids:
                assert pid in known, f"{spec.id} points at missing proof {pid}"
