"""Curated learning paths and automatic path building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.registry import registry
from .graph import prerequisite_chain, topological_order

__all__ = ["LearningPath", "LEARNING_PATHS", "build_path", "list_paths", "get_path"]


@dataclass(frozen=True)
class LearningPath:
    """An ordered lesson sequence an instructor can teach from."""

    id: str
    concepts: tuple[str, ...]
    level: str = "intermediate"

    @property
    def title_key(self) -> str:
        return f"paths.{self.id}.title"

    @property
    def summary_key(self) -> str:
        return f"paths.{self.id}.summary"


LEARNING_PATHS: tuple[LearningPath, ...] = (
    LearningPath(
        "foundations_of_inference",
        (
            "probability.distributions",
            "inference.sampling_distributions",
            "inference.lln",
            "inference.clt",
            "inference.confidence_intervals",
            "inference.hypothesis_testing",
            "inference.power",
        ),
        "beginner",
    ),
    LearningPath(
        "estimation_theory",
        (
            "probability.distributions",
            "inference.mle",
            "inference.cramer_rao",
            "inference.bootstrap",
            "inference.bayesian_updating",
        ),
        "advanced",
    ),
    LearningPath(
        "regression_from_scratch",
        (
            "math.projection",
            "regression.simple_linear",
            "regression.ols_geometry",
            "regression.fwl",
            "regression.multicollinearity",
            "regression.restricted",
        ),
        "intermediate",
    ),
    LearningPath(
        "diagnostics_and_remedies",
        (
            "regression.simple_linear",
            "econometrics.heteroskedasticity",
            "econometrics.autocorrelation",
            "econometrics.omitted_variable_bias",
            "regression.multicollinearity",
        ),
        "intermediate",
    ),
    LearningPath(
        "identification_and_causality",
        (
            "econometrics.omitted_variable_bias",
            "causal.dag",
            "causal.potential_outcomes",
            "econometrics.endogeneity_iv",
            "causal.did",
            "causal.rdd",
        ),
        "advanced",
    ),
    LearningPath(
        "time_series_econometrics",
        (
            "timeseries.stationarity",
            "timeseries.arma",
            "timeseries.cointegration",
            "timeseries.var",
            "timeseries.garch",
        ),
        "advanced",
    ),
    LearningPath(
        "machine_learning_essentials",
        (
            "ml.bias_variance",
            "ml.regularization",
            "ml.classification_threshold",
            "ml.gradient_descent",
            "multivariate.pca",
        ),
        "intermediate",
    ),
    LearningPath(
        "from_gradients_to_transformers",
        (
            "ml.gradient_descent",
            "deep_learning.backpropagation",
            "ai.attention",
            "xai.shap",
        ),
        "intermediate",
    ),
)

_BY_ID = {p.id: p for p in LEARNING_PATHS}


def list_paths() -> list[LearningPath]:
    return list(LEARNING_PATHS)


def get_path(path_id: str) -> LearningPath:
    if path_id not in _BY_ID:
        raise KeyError(f"unknown learning path {path_id!r}; available: {sorted(_BY_ID)}")
    return _BY_ID[path_id]


def build_path(target: str, *, level: str | None = None) -> list[str]:
    """Build an ordered path that ends at ``target``.

    Prefers a curated path containing the target; otherwise derives one from the
    prerequisite graph.
    """
    for path in LEARNING_PATHS:
        if target in path.concepts:
            idx = path.concepts.index(target)
            return list(path.concepts[: idx + 1])
    chain = prerequisite_chain(target, depth=5)
    ordered = topological_order([*chain, target])
    if level:
        keep = []
        for cid in ordered:
            try:
                spec = registry.get(cid)
            except Exception:
                continue
            if any(lv.value == level for lv in spec.levels) or cid == target:
                keep.append(cid)
        if keep:
            ordered = keep
    return ordered


def path_payload(path_id: str) -> dict[str, Any]:
    from ..i18n.translator import get_translator

    tr = get_translator()
    path = get_path(path_id)
    steps = []
    for cid in path.concepts:
        try:
            spec = registry.get(cid)
        except Exception:
            continue
        steps.append(
            {
                "id": cid,
                "title": tr.t(spec.title_key),
                "summary": tr.t(spec.summary_key),
                "status": spec.status.value,
            }
        )
    return {
        "id": path.id,
        "title": tr.t(path.title_key),
        "summary": tr.t(path.summary_key),
        "level": path.level,
        "steps": steps,
    }
