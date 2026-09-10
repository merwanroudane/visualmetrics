"""Scenario specifications - the antidote to one-canned-example labs.

Every lab exposes a *scenario matrix*: canonical, positive, negative, null,
weak/strong, boundary, violation, counterexample, small/large sample and so on
(blueprint section 33). A scenario is a named set of parameter overrides plus
the pedagogical reason it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["ScenarioCategory", "ScenarioSpec", "scenario", "scenario_coverage"]


class ScenarioCategory(str, Enum):
    CANONICAL = "canonical"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NULL = "null"
    WEAK = "weak"
    STRONG = "strong"
    BOUNDARY = "boundary"
    VIOLATION = "violation"
    COUNTEREXAMPLE = "counterexample"
    SENSITIVITY = "sensitivity"
    SMALL_SAMPLE = "small_sample"
    LARGE_SAMPLE = "large_sample"
    HIGH_NOISE = "high_noise"
    LOW_NOISE = "low_noise"
    ROBUSTNESS = "robustness"
    MISSPECIFICATION = "misspecification"
    REAL_DATA = "real_data"
    COMPARE_METHODS = "compare_methods"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """A named preset of parameter overrides."""

    id: str
    category: ScenarioCategory
    overrides: dict[str, Any] = field(default_factory=dict)
    label_key: str | None = None
    description_key: str | None = None
    teaching_point_key: str | None = None

    @property
    def translation_key(self) -> str:
        return self.label_key or f"scenarios.{self.id}.label"

    @property
    def description_translation_key(self) -> str:
        return self.description_key or f"scenarios.{self.id}.description"


def scenario(id: str, category: Any, /, **overrides: Any) -> ScenarioSpec:
    """Terse constructor, e.g. scenario("weak_iv", "weak", instrument_strength=0.05)."""
    return ScenarioSpec(id=id, category=ScenarioCategory(category), overrides=overrides)


def scenario_coverage(scenarios: Any) -> dict[str, list[str]]:
    """Group scenario ids by category - used by the design-review report."""
    out: dict[str, list[str]] = {}
    for sc in scenarios:
        out.setdefault(sc.category.value, []).append(sc.id)
    return out
