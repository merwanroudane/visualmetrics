"""Serializable lab, figure and animation state.

Blueprint sections 49.2, 49.3 and 65: no hidden mutable globals. Every figure is
a pure function of a serializable state, and every animation frame can recover
its *pedagogical meaning*, not only its frame index.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence import EvidenceType

__all__ = [
    "LabState",
    "FigureState",
    "Panel",
    "AnimationStep",
    "AnimationSpec",
    "AnimationState",
    "ExplanationBlock",
    "MetricRow",
    "Assumption",
    "LabResult",
]


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, EvidenceType):
        return value.value
    return value


@dataclass
class LabState:
    """Complete reproducible state of one lab view."""

    concept_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    scenario: str | None = None
    seed: int = 42
    language: str = "en"
    theme: str = "light"
    level: str = "intermediate"
    terminology: str = "translated"
    reduced_motion: bool = False
    precision: int = 4
    visualmetrics_version: str | None = None
    backend_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.visualmetrics_version is None:
            from ..version import __version__

            self.visualmetrics_version = __version__
        if "seed" in self.parameters:
            try:
                self.seed = int(self.parameters["seed"])
            except (TypeError, ValueError):
                pass
        else:
            self.parameters.setdefault("seed", self.seed)

    def with_(self, **changes: Any) -> LabState:
        data = asdict(self)
        params = dict(data.pop("parameters"))
        for key in list(changes):
            if key in data:
                data[key] = changes.pop(key)
        params.update(changes)
        return LabState(parameters=params, **data)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabState:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> LabState:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class FigureState:
    """Identity of a rendered figure - used for caching and visual tests."""

    concept_id: str
    figure_id: str
    scenario: str | None
    parameters: dict[str, Any]
    theme: str
    locale: str

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "c": self.concept_id,
                "f": self.figure_id,
                "s": self.scenario,
                "p": _jsonable(self.parameters),
                "t": self.theme,
                "l": self.locale,
            },
            sort_keys=True,
        )
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Panel:
    """One figure plus the metadata the GUI needs to frame it honestly."""

    id: str
    figure: Any
    title_key: str | None = None
    title: str | None = None
    caption: str | None = None
    evidence: EvidenceType | None = None
    tab: str = "visualize"
    description: str | None = None


@dataclass
class AnimationStep:
    """One pedagogically meaningful step of an animation.

    Blueprint section 36.3.1 - motion without synchronized interpretation is
    incomplete, so every step carries the full explanation payload.
    """

    id: str
    frame: int
    title: str
    what_you_see: str = ""
    what_changed: str = ""
    why: str = ""
    interpretation: str = ""
    conclusion: str = ""
    warning: str = ""
    math: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    active_assumptions: tuple[str, ...] = ()
    violated_assumptions: tuple[str, ...] = ()
    highlighted: tuple[str, ...] = ()


@dataclass
class AnimationSpec:
    """An animation together with its mandatory explanation layer."""

    id: str
    figure: Any
    steps: list[AnimationStep] = field(default_factory=list)
    purpose: str = ""
    summary: str = ""
    evidence: EvidenceType = EvidenceType.SIMULATION
    frame_duration_ms: int = 500
    loop: bool = False
    static_alternative: Any = None

    def step_at(self, frame: int) -> AnimationStep | None:
        best: AnimationStep | None = None
        for step in self.steps:
            if step.frame <= frame:
                best = step
            else:
                break
        return best or (self.steps[0] if self.steps else None)


@dataclass
class AnimationState:
    """Playback position *and* meaning (blueprint section 49.3)."""

    concept_id: str
    animation_id: str
    frame: int = 0
    step_id: str | None = None
    playing: bool = False
    speed: float = 1.0
    parameters: dict[str, Any] = field(default_factory=dict)
    explanation_key: str | None = None
    active_assumptions: tuple[str, ...] = ()
    violated_assumptions: tuple[str, ...] = ()
    highlighted_objects: tuple[str, ...] = ()
    outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class ExplanationBlock:
    """A localized chunk of prose shown in one of the knowledge tabs."""

    tab: str
    title: str
    body: str
    kind: str = "text"  # text | warning | math | list | misconception
    items: tuple[str, ...] = ()


@dataclass
class MetricRow:
    """One row of the numeric read-out that accompanies every lab."""

    key: str
    label: str
    value: Any
    reference: Any = None
    note: str = ""
    fmt: str = "{:.4f}"

    def formatted(self, precision: int = 4) -> str:
        if isinstance(self.value, str) or self.value is None:
            return "-" if self.value is None else str(self.value)
        if isinstance(self.value, (int,)) and not isinstance(self.value, bool):
            return str(self.value)
        try:
            return f"{float(self.value):.{precision}f}"
        except (TypeError, ValueError):
            return str(self.value)


@dataclass
class Assumption:
    """Status of one modelling assumption inside the current scenario."""

    id: str
    label: str
    holds: bool
    detail: str = ""
    consequence: str = ""

    def __post_init__(self) -> None:
        # Comparisons on numpy scalars return numpy.bool_, which is not a bool
        # and does not survive json.dumps. Normalize once, here, so every lab
        # and every exporter sees a plain Python boolean.
        if not isinstance(self.holds, bool):
            self.holds = bool(self.holds)


@dataclass
class LabResult:
    """Everything a lab produces for one state."""

    panels: list[Panel] = field(default_factory=list)
    metrics: list[MetricRow] = field(default_factory=list)
    explanations: list[ExplanationBlock] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    animations: list[AnimationSpec] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)
    data: Any = None
    dgp: str = ""
    code: str = ""
    evidence: EvidenceType | None = None
    concept_id: str = ""
    resolved_parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- ergonomic helpers -------------------------------------------------
    def add_panel(self, panel: Panel) -> Panel:
        self.panels.append(panel)
        return panel

    def metric(self, key: str, label: str, value: Any, **kw: Any) -> None:
        self.metrics.append(MetricRow(key=key, label=label, value=value, **kw))

    def explain(self, tab: str, title: str, body: str, **kw: Any) -> None:
        self.explanations.append(ExplanationBlock(tab=tab, title=title, body=body, **kw))

    def assume(self, id: str, label: str, holds: bool, **kw: Any) -> None:
        self.assumptions.append(Assumption(id=id, label=label, holds=holds, **kw))

    @property
    def figure(self) -> Any:
        """The primary figure (first panel)."""
        return self.panels[0].figure if self.panels else None

    def figures(self) -> dict[str, Any]:
        return {p.id: p.figure for p in self.panels}

    def metric_dict(self) -> dict[str, Any]:
        return {m.key: m.value for m in self.metrics}

    def show(self) -> None:
        """Display every panel (notebook / interactive use)."""
        for panel in self.panels:
            fig = panel.figure
            if hasattr(fig, "show"):
                fig.show()

    def summary(self, precision: int = 4) -> str:
        lines = [f"VisualMetrics lab: {self.concept_id}"]
        if self.evidence is not None:
            lines.append(f"evidence: {EvidenceType(self.evidence).value}")
        if self.dgp:
            lines.append(f"DGP: {self.dgp}")
        if self.metrics:
            lines.append("")
            width = max(len(m.label) for m in self.metrics)
            for m in self.metrics:
                row = f"  {m.label:<{width}}  {m.formatted(precision)}"
                if m.reference is not None:
                    try:
                        row += f"   (reference: {float(m.reference):.{precision}f})"
                    except (TypeError, ValueError):
                        row += f"   (reference: {m.reference})"
                if m.note:
                    row += f"   {m.note}"
                lines.append(row)
        if self.assumptions:
            lines.append("")
            lines.append("  assumptions:")
            for a in self.assumptions:
                mark = "ok" if a.holds else "VIOLATED"
                lines.append(f"    [{mark}] {a.label}" + (f" - {a.detail}" if a.detail else ""))
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return (
            f"<LabResult {self.concept_id} panels={len(self.panels)} "
            f"metrics={len(self.metrics)} animations={len(self.animations)}>"
        )
