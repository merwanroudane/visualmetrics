"""Concept specifications and the Lab protocol.

A *concept* is metadata (cheap, always loaded for the catalog).
A *lab* is the implementation (expensive, loaded on demand).

This split is what keeps ``import visualmetrics`` fast even though the catalog
advertises the full curriculum (blueprint section 67.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .controls import ControlSpec, validate_parameters
from .evidence import EvidenceType
from .scenarios import ScenarioSpec

if TYPE_CHECKING:  # pragma: no cover
    from .state import LabResult, LabState

__all__ = [
    "Domain",
    "Level",
    "LearningMode",
    "Status",
    "Reference",
    "ConceptSpec",
    "Lab",
    "LabBase",
]


class Domain(str, Enum):
    """Canonical scientific domains.

    Blueprint section 31.1.7: local course labels such as "Statistics 3" must
    never appear here. Only universal scientific names.
    """

    MATH = "mathematical_foundations"
    PROBABILITY = "probability_random_variables"
    DESCRIPTIVE = "descriptive_statistics"
    INFERENCE = "inferential_statistics"
    REGRESSION = "regression_linear_models"
    ECONOMETRICS = "econometrics"
    TIMESERIES = "time_series_econometrics"
    PANEL = "panel_data_econometrics"
    CAUSAL = "causal_inference"
    MULTIVARIATE = "multivariate_statistics"
    SPATIAL = "spatial_econometrics"
    ML = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    AI = "modern_ai"
    XAI = "explainable_ai"

    @property
    def label_key(self) -> str:
        return f"domains.{self.value}"


class Level(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PHD = "phd"

    @property
    def rank(self) -> int:
        return {"beginner": 0, "intermediate": 1, "advanced": 2, "phd": 3}[self.value]


class LearningMode(str, Enum):
    LEARN = "learn"
    VISUALIZE = "visualize"
    ANIMATE = "animate"
    EXPERIMENT = "experiment"
    COMPARE = "compare"
    PROVE = "prove"
    DERIVE = "derive"
    SIMULATE = "simulate"
    DIAGNOSE = "diagnose"
    COUNTEREXAMPLE = "counterexample"
    QUIZ = "quiz"
    CODE = "code"
    DATA = "data"
    REFERENCES = "references"


class Status(str, Enum):
    """Lifecycle of a concept (blueprint section 72)."""

    PLANNED = "planned"
    DRAFT = "draft"
    EXPERIMENTAL = "experimental"
    REVIEWED = "reviewed"
    STABLE = "stable"
    DEPRECATED = "deprecated"

    @property
    def is_implemented(self) -> bool:
        return self not in (Status.PLANNED,)


@dataclass(frozen=True, slots=True)
class Reference:
    """A citation attached to a concept."""

    citation: str
    kind: str = "book"  # book | paper | course | docs
    url: str | None = None
    doi: str | None = None


@dataclass(frozen=True)
class ConceptSpec:
    """Everything the catalog knows about a concept without loading its lab."""

    id: str
    domain: Domain
    subdomain: str
    title_key: str
    summary_key: str
    levels: tuple[Level, ...] = (Level.INTERMEDIATE,)
    modes: tuple[LearningMode, ...] = (LearningMode.VISUALIZE,)
    evidence: EvidenceType = EvidenceType.VISUAL_INTUITION
    status: Status = Status.EXPERIMENTAL
    version: int = 1
    controls: tuple[ControlSpec, ...] = ()
    scenarios: tuple[ScenarioSpec, ...] = ()
    prerequisites: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    next_concepts: tuple[str, ...] = ()
    confused_with: tuple[str, ...] = ()
    learning_objectives: tuple[str, ...] = ()
    misconceptions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    required_extras: tuple[str, ...] = ()
    renderers: tuple[str, ...] = ("plotly",)
    references: tuple[Reference, ...] = ()
    proof_ids: tuple[str, ...] = ()
    #: dotted import path of the module providing ``LAB`` (lazy loading)
    module: str | None = None
    curriculum_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or " " in self.id:
            raise ValueError(f"invalid concept id: {self.id!r}")

    @property
    def control_map(self) -> dict[str, ControlSpec]:
        return {c.id: c for c in self.controls}

    @property
    def scenario_map(self) -> dict[str, ScenarioSpec]:
        return {s.id: s for s in self.scenarios}

    @property
    def default_parameters(self) -> dict[str, Any]:
        return {c.id: c.default for c in self.controls}

    def parameters_for(self, scenario_id: str | None = None, **overrides: Any) -> dict[str, Any]:
        """Resolve defaults + scenario overrides + explicit overrides."""
        params = self.default_parameters
        if scenario_id:
            sc = self.scenario_map.get(scenario_id)
            if sc is None:
                from .exceptions import UnsupportedScenarioError

                raise UnsupportedScenarioError(
                    f"concept {self.id!r} has no scenario {scenario_id!r}; "
                    f"available: {sorted(self.scenario_map)}",
                    concept=self.id,
                    scenario=scenario_id,
                )
            params.update(sc.overrides)
        params.update(overrides)
        return validate_parameters(self.controls, params)

    def supports(self, mode: LearningMode | str) -> bool:
        return LearningMode(mode) in self.modes

    def search_text(self) -> str:
        parts = [self.id, self.subdomain, *self.tags, *self.aliases]
        return " ".join(p.replace("_", " ").replace(".", " ") for p in parts if p)


@runtime_checkable
class Lab(Protocol):
    """The implementation behind a concept."""

    spec: ConceptSpec

    def run(self, state: LabState) -> LabResult:  # pragma: no cover - protocol
        ...


class LabBase:
    """Convenience base class implementing state plumbing for a lab."""

    spec: ConceptSpec

    def __init__(self, spec: ConceptSpec) -> None:
        self.spec = spec

    # -- to be implemented by concrete labs -------------------------------
    def compute(self, params: dict[str, Any], state: LabState) -> LabResult:
        raise NotImplementedError

    # -- public entry point ------------------------------------------------
    def run(self, state: LabState) -> LabResult:
        params = self.spec.parameters_for(state.scenario, **state.parameters)
        result = self.compute(params, state)
        result.concept_id = self.spec.id
        result.evidence = result.evidence or self.spec.evidence
        result.resolved_parameters = params
        if not result.code:
            result.code = self.generate_code(params, state)
        return result

    def generate_code(self, params: dict[str, Any], state: LabState) -> str:
        """Default no-code -> code translation (blueprint section 53)."""
        from ..export.code import render_lab_code

        return render_lab_code(self.spec, params, state)
