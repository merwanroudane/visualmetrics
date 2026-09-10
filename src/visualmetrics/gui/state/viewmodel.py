"""Turning a :class:`LabResult` into something a renderer can lay out.

The view model decides *what appears where*: which knowledge tabs exist for
this result, which panels and explanation blocks belong to each, how the
evidence badge reads, which assumptions failed. It contains no NiceGUI, so the
layout decisions are testable without a browser and the same view model can
drive the notebook renderer or an export.

Tab order follows the blueprint's taxonomy (section 36.3) rather than the order
a lab happened to append things in, so every lab reads the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...core.evidence import EvidenceType, badge_for
from ...core.state import LabResult

__all__ = ["TAB_ORDER", "Badge", "TabView", "LabView", "build_lab_view", "format_value"]

#: Canonical tab order. A tab appears only if this result has content for it.
TAB_ORDER: tuple[str, ...] = (
    "overview",
    "intuition",
    "visualize",
    "animate",
    "experiment",
    "simulation",
    "diagnostics",
    "compare",
    "counterexample",
    "assumptions",
    "math",
    "proof",
    "interpretation",
    "misconceptions",
    "warning",
    "data",
    "code",
    "quiz",
    "references",
)

#: Tabs the GUI always offers when the underlying content exists, even though
#: no lab writes an explanation block for them.
_STRUCTURAL_TABS = {"visualize", "animate", "assumptions", "code", "references", "quiz"}


def format_value(value: Any, precision: int = 4) -> str:
    """Format a metric for display without turning it into scientific soup."""
    import numpy as np

    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    if number != 0 and (abs(number) < 1e-4 or abs(number) >= 1e7):
        return f"{number:.{max(precision - 1, 1)}e}"
    text = f"{number:.{precision}f}".rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True)
class Badge:
    """The evidence badge, already translated."""

    evidence: EvidenceType
    label: str
    caveat: str
    icon: str
    color_role: str

    @property
    def is_proof(self) -> bool:
        return self.evidence.is_proof


@dataclass
class TabView:
    """One knowledge tab and everything that belongs in it."""

    id: str
    label: str
    panels: list[Any] = field(default_factory=list)
    blocks: list[Any] = field(default_factory=list)
    animations: list[Any] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.panels or self.blocks or self.animations)


@dataclass
class MetricRow:
    key: str
    label: str
    value: str
    raw: Any
    note: str = ""
    reference: str = ""


@dataclass
class AssumptionRow:
    id: str
    label: str
    holds: bool
    detail: str = ""
    consequence: str = ""


@dataclass
class LabView:
    """A whole result, arranged for display."""

    concept_id: str
    title: str
    summary: str
    badge: Badge
    tabs: list[TabView]
    metrics: list[MetricRow]
    assumptions: list[AssumptionRow]
    warnings: list[str]
    code: str
    dgp: str
    is_rtl: bool = False
    scenario: str | None = None
    resolved_parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_assumptions(self) -> list[AssumptionRow]:
        return [a for a in self.assumptions if not a.holds]

    @property
    def has_trouble(self) -> bool:
        """Whether the GUI should draw attention to something being wrong."""
        return bool(self.warnings or self.failed_assumptions)

    def tab(self, tab_id: str) -> TabView | None:
        for tab in self.tabs:
            if tab.id == tab_id:
                return tab
        return None

    @property
    def tab_ids(self) -> list[str]:
        return [t.id for t in self.tabs]


def build_lab_view(
    result: LabResult,
    *,
    translator: Any = None,
    language: str | None = None,
    precision: int = 4,
) -> LabView:
    """Arrange a lab result into tabs, metrics, assumptions and warnings."""
    from ...i18n.rtl import protect_latin
    from ...i18n.translator import get_translator

    tr = translator or get_translator(language)
    locale = getattr(tr, "language", language or "en")

    def tx(key: str, default: str = "") -> str:
        text = tr.t(key, None)
        if text is None:
            text = default
        return protect_latin(text, locale) if text else ""

    evidence = result.evidence or EvidenceType.SIMULATION
    meta = badge_for(evidence)
    badge = Badge(
        evidence=evidence,
        label=tx(meta.label_key, evidence.value.replace("_", " ").title()),
        caveat=tx(meta.caveat_key),
        icon=meta.icon,
        color_role=meta.color_role,
    )

    buckets: dict[str, TabView] = {}

    def bucket(tab_id: str) -> TabView:
        if tab_id not in buckets:
            buckets[tab_id] = TabView(
                id=tab_id, label=tx(f"tabs.{tab_id}", tab_id.replace("_", " ").title())
            )
        return buckets[tab_id]

    for panel in result.panels:
        bucket(getattr(panel, "tab", "visualize") or "visualize").panels.append(panel)
    for block in result.explanations:
        bucket(getattr(block, "tab", "overview") or "overview").blocks.append(block)
    for animation in result.animations:
        bucket("animate").animations.append(animation)
    if result.code:
        bucket("code")
    if result.assumptions:
        bucket("assumptions")

    ordered = [buckets[t] for t in TAB_ORDER if t in buckets]
    # Anything a lab invented that is not in the taxonomy still gets shown,
    # after the known tabs, rather than being silently dropped.
    ordered += [v for k, v in buckets.items() if k not in TAB_ORDER]
    ordered = [t for t in ordered if not t.is_empty or t.id in _STRUCTURAL_TABS]

    metrics = [
        MetricRow(
            key=m.key,
            label=protect_latin(m.label, locale),
            value=format_value(m.value, precision),
            raw=m.value,
            note=protect_latin(getattr(m, "note", "") or "", locale),
            reference=(
                "" if getattr(m, "reference", None) is None
                else format_value(m.reference, precision)
            ),
        )
        for m in result.metrics
    ]

    assumptions = [
        AssumptionRow(
            id=a.id,
            label=protect_latin(a.label, locale),
            holds=bool(a.holds),
            detail=protect_latin(getattr(a, "detail", "") or "", locale),
            consequence=protect_latin(getattr(a, "consequence", "") or "", locale),
        )
        for a in result.assumptions
    ]

    return LabView(
        concept_id=result.concept_id,
        title=tx(f"concepts.{result.concept_id}.title", result.concept_id),
        summary=tx(f"concepts.{result.concept_id}.summary"),
        badge=badge,
        tabs=ordered,
        metrics=metrics,
        assumptions=assumptions,
        warnings=[protect_latin(w, locale) for w in result.warnings],
        code=result.code,
        dgp=result.dgp,
        is_rtl=bool(getattr(tr, "is_rtl", False)),
        resolved_parameters=dict(result.resolved_parameters),
    )
