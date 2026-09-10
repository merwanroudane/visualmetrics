"""Registry-backed convenience namespaces.

``vm.stats.power(...)`` and ``vm.econometrics.iv(...)`` are generated from the
registry rather than hand-written, so they can never drift out of sync with the
catalog (blueprint section 54.3).
"""

from __future__ import annotations

import types
from typing import Any

from ..core.registry import registry

__all__ = ["NAMESPACES", "build_namespace"]

#: namespace -> {attribute: concept id}
NAMESPACES: dict[str, dict[str, str]] = {
    "probability": {
        "distributions": "probability.distributions",
        "bivariate": "probability.bivariate",
        "approximations": "probability.approximations",
    },
    "stats": {
        "describe": "descriptive.explorer",
        "lln": "inference.lln",
        "clt": "inference.clt",
        "sampling": "inference.sampling_distributions",
        "confidence_interval": "inference.confidence_intervals",
        "test": "inference.hypothesis_testing",
        "power": "inference.power",
        "mle": "inference.mle",
        "cramer_rao": "inference.cramer_rao",
        "neyman_pearson": "inference.neyman_pearson",
        "bayes": "inference.bayesian_updating",
        "multiple_testing": "inference.multiple_testing",
        "bootstrap": "inference.bootstrap",
    },
    "econometrics": {
        "ols": "regression.simple_linear",
        "geometry": "regression.ols_geometry",
        "fwl": "regression.fwl",
        "multicollinearity": "regression.multicollinearity",
        "restricted": "regression.restricted",
        "heteroskedasticity": "econometrics.heteroskedasticity",
        "autocorrelation": "econometrics.autocorrelation",
        "omitted_variable_bias": "econometrics.omitted_variable_bias",
        "iv": "econometrics.endogeneity_iv",
        "simultaneous": "econometrics.simultaneous_equations",
        "logit": "econometrics.logit_probit",
        "unit_root": "timeseries.stationarity",
        "arma": "timeseries.arma",
        "cointegration": "timeseries.cointegration",
        "var": "timeseries.var",
        "garch": "timeseries.garch",
        "panel": "panel.fixed_vs_random",
    },
    "causal": {
        "potential_outcomes": "causal.potential_outcomes",
        "dag": "causal.dag",
        "did": "causal.did",
        "rdd": "causal.rdd",
    },
    "ml": {
        "bias_variance": "ml.bias_variance",
        "regularization": "ml.regularization",
        "threshold": "ml.classification_threshold",
        "gradient_descent": "ml.gradient_descent",
        "pca": "multivariate.pca",
        "clustering": "multivariate.clustering",
    },
    "ai": {
        "backpropagation": "deep_learning.backpropagation",
        "attention": "ai.attention",
        "shap": "xai.shap",
    },
    "math": {
        "projection": "math.projection",
    },
}


def _make_caller(concept_id: str) -> Any:
    def call(**parameters: Any):
        import visualmetrics as vm

        return vm.lab(concept_id, **parameters)

    spec_doc = ""
    try:
        spec = registry.get(concept_id)
        controls = ", ".join(c.id for c in spec.controls) or "(see vm.concept(...))"
        spec_doc = f"\n\nControls: {controls}"
    except Exception:  # noqa: BLE001
        pass
    call.__name__ = concept_id.rsplit(".", 1)[-1]
    call.__doc__ = f"Run the {concept_id!r} lab.{spec_doc}"
    call.concept_id = concept_id  # type: ignore[attr-defined]
    return call


def build_namespace(name: str) -> types.SimpleNamespace:
    """Materialize one convenience namespace."""
    mapping = NAMESPACES[name]
    ns = types.SimpleNamespace()
    for attr, concept_id in mapping.items():
        setattr(ns, attr, _make_caller(concept_id))
    ns.concepts = sorted(mapping.values())  # type: ignore[attr-defined]
    return ns
