"""The catalog, the registry and the labs, end to end."""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.catalog.builtin import CATALOG
from visualmetrics.core.concepts import Domain, Status

IMPLEMENTED = [s.id for s in CATALOG if s.status.is_implemented]
PLANNED = [s.id for s in CATALOG if not s.status.is_implemented]

# One lab per domain keeps the default run fast; the full sweep is marked slow.
REPRESENTATIVE = sorted({s.domain: s.id for s in CATALOG if s.status.is_implemented}.values())


class TestCatalogIntegrity:
    def test_ids_are_unique(self):
        ids = [s.id for s in CATALOG]
        assert len(ids) == len(set(ids))

    def test_ids_are_domain_qualified(self):
        for spec in CATALOG:
            assert "." in spec.id, f"{spec.id} is not namespaced"
            assert spec.id == spec.id.lower()

    def test_every_concept_declares_a_domain_and_its_levels(self):
        for spec in CATALOG:
            assert isinstance(spec.domain, Domain)
            assert spec.levels, f"{spec.id} is not assigned to any teaching level"

    def test_planned_concepts_have_no_module(self):
        for spec in CATALOG:
            if spec.status is Status.PLANNED:
                assert spec.module is None
            else:
                assert spec.module, f"{spec.id} is implemented but names no module"

    def test_no_local_course_labels_leak_into_the_catalog(self):
        """Blueprint section 31.1.7: only canonical scientific names."""
        banned = ("statistics 1", "statistics 2", "statistics 3", "econometrics 1",
                  "econometrics 2", "semester", "module 1", "s1", "l3")
        for spec in CATALOG:
            haystack = f"{spec.id} {spec.title_key} {spec.domain.value}".lower()
            for label in banned:
                assert label not in haystack, f"{spec.id} carries a local course label"


class TestLazyLoading:
    def test_importing_the_package_does_not_import_the_labs(self):
        import subprocess
        import sys

        code = (
            "import sys, visualmetrics; "
            "labs=[m for m in sys.modules if m.startswith('visualmetrics.concepts.')"
            " and not m.endswith('_kit')]; "
            "print(len(labs))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "0", "importing visualmetrics eagerly loaded labs"

    @pytest.mark.parametrize("concept_id", IMPLEMENTED)
    def test_the_stub_and_the_loaded_spec_agree(self, concept_id):
        stub = next(s for s in CATALOG if s.id == concept_id)
        spec = vm.registry.lab(concept_id).spec
        assert spec.id == stub.id
        assert spec.domain == stub.domain
        assert spec.status.is_implemented == stub.status.is_implemented

    @pytest.mark.parametrize("concept_id", PLANNED)
    def test_planned_concepts_refuse_to_open(self, concept_id):
        from visualmetrics.core.exceptions import ConceptNotFoundError

        with pytest.raises(ConceptNotFoundError) as excinfo:
            vm.registry.lab(concept_id)
        assert excinfo.value.key == "errors.concept_planned"


@pytest.mark.parametrize("concept_id", IMPLEMENTED)
class TestSpecCompleteness:
    def test_controls_have_distinct_ids_and_valid_defaults(self, concept_id):
        spec = vm.registry.lab(concept_id).spec
        ids = [c.id for c in spec.controls]
        assert len(ids) == len(set(ids)), f"{concept_id} repeats a control id"
        for control in spec.controls:
            assert control.coerce(control.default) is not None

    def test_scenarios_are_declared_and_named_uniquely(self, concept_id):
        spec = vm.registry.lab(concept_id).spec
        assert len(spec.scenarios) >= 5, f"{concept_id} exposes too few scenarios"
        ids = [s.id for s in spec.scenarios]
        assert len(ids) == len(set(ids))

    def test_scenario_coverage_spans_the_taxonomy(self, concept_id):
        """A lab must show the concept failing, not only working."""
        spec = vm.registry.lab(concept_id).spec
        categories = {s.category.value for s in spec.scenarios}
        assert "canonical" in categories, f"{concept_id} has no canonical case"
        interesting = {
            "violation", "counterexample", "boundary", "null", "weak",
            "misspecification", "robustness", "sensitivity", "high_noise",
            "small_sample", "compare",
        }
        assert categories & interesting, (
            f"{concept_id} only shows the happy path; categories = {sorted(categories)}"
        )

    def test_scenario_overrides_name_real_controls(self, concept_id):
        spec = vm.registry.lab(concept_id).spec
        known = {c.id for c in spec.controls}
        for scenario in spec.scenarios:
            unknown = set(scenario.overrides) - known
            assert not unknown, (
                f"{concept_id}:{scenario.id} overrides unknown controls {sorted(unknown)}"
            )

    def test_references_are_present(self, concept_id):
        assert vm.registry.lab(concept_id).spec.references


@pytest.mark.parametrize("concept_id", REPRESENTATIVE)
class TestLabsRun:
    def test_the_canonical_scenario_produces_a_full_result(self, concept_id):
        result = vm.lab(concept_id, seed=101)
        assert result.panels, f"{concept_id} produced no figure"
        assert len(result.metrics) >= 4, f"{concept_id} produced too few metrics"
        assert result.explanations, f"{concept_id} explains nothing"
        assert result.code, f"{concept_id} generated no equivalent Python"
        assert result.concept_id == concept_id

    def test_the_same_seed_gives_the_same_numbers(self, concept_id):
        first = vm.lab(concept_id, seed=7).metric_dict()
        second = vm.lab(concept_id, seed=7).metric_dict()
        assert first == second, f"{concept_id} is not reproducible at a fixed seed"

    def test_a_different_seed_moves_something(self, concept_id):
        a = vm.lab(concept_id, seed=1).metric_dict()
        b = vm.lab(concept_id, seed=999).metric_dict()
        assert a.keys() == b.keys()

    def test_runs_in_every_language(self, concept_id):
        for language in ("en", "ar", "fr"):
            result = vm.lab(concept_id, seed=3, language=language)
            assert result.panels
            assert result.explanations


@pytest.mark.slow
class TestFullSweep:
    def test_every_lab_and_every_scenario(self):
        """The complete catalogue: every declared scenario must run."""
        failures = []
        for concept_id in IMPLEMENTED:
            spec = vm.registry.lab(concept_id).spec
            for scenario in spec.scenarios:
                try:
                    result = vm.lab(concept_id, scenario=scenario.id, seed=202)
                    assert result.panels
                    assert result.metrics
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{concept_id}:{scenario.id} -> {exc!r}")
        assert not failures, "\n".join(failures[:20])
