"""Declarative proof specifications (blueprint section 48).

A :class:`ProofSpec` is a *statement plus a chain of licensed steps*.  It is
deliberately data, not code: the GUI step renderer, the Plotly geometric
renderer and any future SVG/Manim exporter all consume the same declaration,
so a proof never has to be written twice (blueprint section 48.2).

Scientific-honesty rules enforced here (blueprint sections 3 and 15):

* a ``ProofSpec`` may only carry an evidence kind for which
  :attr:`~visualmetrics.core.evidence.EvidenceType.is_proof` is true.  A
  simulation is never a proof, so it can never be dressed as one;
* every proof states what it does **not** establish in ``limitations``;
* a numerical ``ProofCheck`` is reported as a *check of a proved identity*,
  never as evidence for the theorem.  Passing checks confirm the
  implementation, not the mathematics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.concepts import Level, Reference
from ..core.evidence import EvidenceType, badge_for

__all__ = [
    "StepKind",
    "VisualAction",
    "ProofStep",
    "ProofAssumption",
    "ProofCheck",
    "CheckOutcome",
    "ProofSpec",
    "RenderedStep",
    "RenderedProof",
]


class StepKind(str, Enum):
    """What kind of move a step makes.

    The GUI colours steps by kind so a reader can see at a glance where the
    real work happens and where the argument is only book-keeping.
    """

    SETUP = "setup"
    ASSUMPTION = "assumption"
    DEFINITION = "definition"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    SUBSTITUTION = "substitution"
    KEY_INSIGHT = "key_insight"
    CONCLUSION = "conclusion"

    @property
    def label_key(self) -> str:
        return f"proofs.step_kind.{self.value}"


class VisualAction(str, Enum):
    """Renderer-agnostic instruction attached to a step.

    A renderer that does not understand an action must ignore it rather than
    fail: the lightweight in-GUI renderer shows the algebra alone, while the
    Plotly geometric renderer draws the picture (blueprint section 48.2).
    """

    NONE = "none"
    DRAW_VECTOR = "draw_vector"
    DRAW_SUBSPACE = "draw_subspace"
    PROJECT_ONTO_SUBSPACE = "project_onto_subspace"
    MARK_RIGHT_ANGLE = "mark_right_angle"
    DECOMPOSE_VECTOR = "decompose_vector"
    HIGHLIGHT_TRIANGLE = "highlight_triangle"
    DRAW_CONSTRAINT_SET = "draw_constraint_set"
    DRAW_LEVEL_CURVES = "draw_level_curves"
    MARK_TANGENCY = "mark_tangency"
    MARK_CORNER = "mark_corner"
    SHADE_REGION = "shade_region"
    SHADE_TAIL = "shade_tail"
    COMPARE_REGIONS = "compare_regions"
    ROTATE_AXES = "rotate_axes"
    TRACE_PATH = "trace_path"
    HIGHLIGHT_TERM = "highlight_term"
    UNFOLD_CHAIN = "unfold_chain"

    @property
    def label_key(self) -> str:
        return f"proofs.visual_action.{self.value}"


@dataclass(frozen=True)
class ProofAssumption:
    """A named hypothesis the proof leans on.

    Steps reference assumptions by ``id`` through :attr:`ProofStep.uses`, so a
    reader can ask "where exactly did normality get used?" and get an exact
    answer instead of a hand wave.
    """

    id: str
    statement: str
    essential: bool = True
    """``False`` marks a convenience assumption that could be relaxed."""

    if_violated: str = ""
    """What breaks when this hypothesis fails."""


@dataclass(frozen=True)
class ProofStep:
    """One licensed move in the chain."""

    id: str
    statement: str = ""
    equation: str | None = None
    """Display math for the step, written in LaTeX without delimiters."""

    justification: str = ""
    """*Why* this move is legal - the rule, lemma or definition invoked."""

    kind: StepKind = StepKind.ALGEBRA
    visual_action: VisualAction = VisualAction.NONE
    visual_hint: dict[str, Any] = field(default_factory=dict)
    uses: tuple[str, ...] = ()
    """Ids of assumptions and earlier steps this step depends on."""

    def key(self, proof_id: str, part: str) -> str:
        return f"proofs.{proof_id}.step.{self.id}.{part}"


@dataclass(frozen=True)
class CheckOutcome:
    """Result of running a :class:`ProofCheck`."""

    id: str
    passed: bool
    detail: str
    value: float | None = None
    tolerance: float | None = None

    @property
    def evidence(self) -> EvidenceType:
        """Always numerical - a check never upgrades to proof status."""
        return EvidenceType.NUMERICAL_DEMONSTRATION


@dataclass(frozen=True)
class ProofCheck:
    """A numerical sanity check of an identity the proof established.

    This exists to catch *implementation* drift between the proof text and the
    lab that illustrates it.  It is explicitly not evidence for the theorem:
    the theorem is already proved, and a finite numerical experiment could
    never establish it (blueprint section 15).
    """

    id: str
    description: str
    fn: Callable[[], CheckOutcome]

    def run(self) -> CheckOutcome:
        return self.fn()


@dataclass(frozen=True)
class ProofSpec:
    """A complete declarative proof."""

    id: str
    kind: EvidenceType
    title: str
    claim: str
    conclusion: str
    steps: tuple[ProofStep, ...]
    assumptions: tuple[ProofAssumption, ...] = ()
    prerequisites: tuple[str, ...] = ()
    concept_ids: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()
    checks: tuple[ProofCheck, ...] = ()
    """Numerical checks of the identities proved - never evidence for them."""

    level: Level = Level.INTERMEDIATE
    limitations: str = ""
    """What the proof does *not* establish.  Required: see :meth:`__post_init__`."""

    intuition: str = ""
    """One sentence of plain language before any symbol appears."""

    def __post_init__(self) -> None:
        if not self.kind.is_proof:
            raise ValueError(
                f"proof {self.id!r} declares evidence kind {self.kind.value!r}, "
                "which is not a proof kind. Only formal_proof, symbolic_derivation "
                "and geometric_proof may be used for a ProofSpec."
            )
        if not self.steps:
            raise ValueError(f"proof {self.id!r} has no steps")
        if not self.limitations:
            raise ValueError(
                f"proof {self.id!r} must state its limitations: every proof has a "
                "scope, and hiding it is the failure mode this package exists to avoid."
            )
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"proof {self.id!r} repeats step id {step.id!r}")
            seen.add(step.id)
        known = seen | {a.id for a in self.assumptions}
        for step in self.steps:
            for dep in step.uses:
                if dep not in known:
                    raise ValueError(
                        f"proof {self.id!r} step {step.id!r} depends on {dep!r}, "
                        "which is neither an assumption nor another step"
                    )

    # -- navigation (blueprint section 48.3) ------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def step(self, step_id: str) -> ProofStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"proof {self.id!r} has no step {step_id!r}")

    def index_of(self, step_id: str) -> int:
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        raise KeyError(f"proof {self.id!r} has no step {step_id!r}")

    def assumption(self, assumption_id: str) -> ProofAssumption:
        for a in self.assumptions:
            if a.id == assumption_id:
                return a
        raise KeyError(f"proof {self.id!r} has no assumption {assumption_id!r}")

    @property
    def essential_assumptions(self) -> tuple[ProofAssumption, ...]:
        return tuple(a for a in self.assumptions if a.essential)

    @property
    def has_geometry(self) -> bool:
        return any(s.visual_action is not VisualAction.NONE for s in self.steps)

    def run_checks(self) -> tuple[CheckOutcome, ...]:
        """Run every numerical check attached to this proof."""
        return tuple(c.run() for c in self.checks)

    def key(self, part: str) -> str:
        return f"proofs.{self.id}.{part}"

    # -- rendering ---------------------------------------------------------

    def render(self, translator: Any = None, locale: str | None = None) -> RenderedProof:
        """Translate the whole proof into ``locale``.

        Falls back to the English defaults carried on the spec itself, so a
        proof is always readable even before its translation bundle exists -
        the missing keys are recorded by the translator instead of being
        silently papered over (blueprint section 25.6).
        """
        from ..i18n.rtl import protect_latin
        from ..i18n.translator import get_translator

        tr = translator or get_translator(locale)
        loc = getattr(tr, "language", None) or locale or "en"

        def tx(key: str, default: str) -> str:
            # A real translation wins; otherwise fall back to the English text
            # carried on the spec, which is always present.
            text = tr.t(key, None)
            if text is None:
                text = default
            return protect_latin(text, loc) if text else ""

        steps = tuple(
            RenderedStep(
                id=s.id,
                number=i + 1,
                kind=s.kind,
                kind_label=tx(s.kind.label_key, s.kind.value.replace("_", " ").title()),
                statement=tx(s.key(self.id, "statement"), s.statement),
                equation=s.equation,
                justification=tx(s.key(self.id, "justification"), s.justification),
                visual_action=s.visual_action,
                visual_hint=dict(s.visual_hint),
                uses=s.uses,
            )
            for i, s in enumerate(self.steps)
        )
        badge = badge_for(self.kind)
        return RenderedProof(
            id=self.id,
            kind=self.kind,
            evidence_label=tx(badge.label_key, self.kind.value.replace("_", " ").title()),
            evidence_caveat=tx(badge.caveat_key, ""),
            evidence_icon=badge.icon,
            title=tx(self.key("title"), self.title),
            claim=tx(self.key("claim"), self.claim),
            intuition=tx(self.key("intuition"), self.intuition),
            conclusion=tx(self.key("conclusion"), self.conclusion),
            limitations=tx(self.key("limitations"), self.limitations),
            steps=steps,
            assumptions=tuple(
                RenderedAssumption(
                    id=a.id,
                    statement=tx(f"proofs.{self.id}.assumption.{a.id}", a.statement),
                    essential=a.essential,
                    if_violated=tx(f"proofs.{self.id}.assumption.{a.id}.violated", a.if_violated),
                )
                for a in self.assumptions
            ),
            is_rtl=bool(getattr(tr, "is_rtl", False)),
        )


@dataclass(frozen=True)
class RenderedAssumption:
    id: str
    statement: str
    essential: bool
    if_violated: str


@dataclass(frozen=True)
class RenderedStep:
    id: str
    number: int
    kind: StepKind
    kind_label: str
    statement: str
    equation: str | None
    justification: str
    visual_action: VisualAction
    visual_hint: dict[str, Any]
    uses: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "number": self.number,
            "kind": self.kind.value,
            "kind_label": self.kind_label,
            "statement": self.statement,
            "equation": self.equation,
            "justification": self.justification,
            "visual_action": self.visual_action.value,
            "visual_hint": self.visual_hint,
            "uses": list(self.uses),
        }


@dataclass(frozen=True)
class RenderedProof:
    """A proof translated into one locale, ready for any renderer."""

    id: str
    kind: EvidenceType
    evidence_label: str
    evidence_caveat: str
    evidence_icon: str
    title: str
    claim: str
    intuition: str
    conclusion: str
    limitations: str
    steps: tuple[RenderedStep, ...]
    assumptions: tuple[RenderedAssumption, ...]
    is_rtl: bool = False

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "evidence_label": self.evidence_label,
            "evidence_caveat": self.evidence_caveat,
            "evidence_icon": self.evidence_icon,
            "title": self.title,
            "claim": self.claim,
            "intuition": self.intuition,
            "conclusion": self.conclusion,
            "limitations": self.limitations,
            "is_rtl": self.is_rtl,
            "assumptions": [
                {
                    "id": a.id,
                    "statement": a.statement,
                    "essential": a.essential,
                    "if_violated": a.if_violated,
                }
                for a in self.assumptions
            ],
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_text(self) -> str:
        """Plain-text transcript - used by the CLI and by exports."""
        out = [self.title, "=" * len(self.title), "", f"[{self.evidence_icon} {self.evidence_label}]"]
        if self.evidence_caveat:
            out.append(self.evidence_caveat)
        out.append("")
        if self.intuition:
            out += [self.intuition, ""]
        out += ["Claim:", f"  {self.claim}", ""]
        if self.assumptions:
            out.append("Assumptions:")
            for a in self.assumptions:
                tag = "" if a.essential else " (relaxable)"
                out.append(f"  ({a.id}){tag} {a.statement}")
            out.append("")
        for s in self.steps:
            out.append(f"{s.number}. [{s.kind_label}] {s.statement}")
            if s.equation:
                out.append(f"     {s.equation}")
            if s.justification:
                out.append(f"     why: {s.justification}")
        out += ["", "Conclusion:", f"  {self.conclusion}", ""]
        out += ["What this does NOT establish:", f"  {self.limitations}"]
        return "\n".join(out)
