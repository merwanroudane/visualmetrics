"""Lazy registry for the proof library.

Same discipline as the concept registry: the index is a cheap dict of ids, and
the module holding the actual :class:`~visualmetrics.proofs.spec.ProofSpec` is
imported only when someone opens that proof.  A proof that is catalogued but
not written is reported as missing, honestly, rather than silently absent
(blueprint sections 16 and 67.2).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from ..core.exceptions import ProofUnavailableError
from .spec import ProofSpec

__all__ = ["ProofNotFoundError", "ProofRegistry", "proof_registry", "proof", "list_proofs"]


class ProofNotFoundError(ProofUnavailableError):
    """Raised when a proof id is unknown or its module is not written yet."""


@dataclass(frozen=True)
class ProofEntry:
    """Catalog stub for one proof: enough to list it, not enough to render it."""

    id: str
    module: str | None
    title: str
    concept_ids: tuple[str, ...] = ()

    @property
    def is_written(self) -> bool:
        return self.module is not None


#: Every proof the package advertises.  ``module=None`` means *planned*.
PROOF_INDEX: tuple[ProofEntry, ...] = (
    ProofEntry(
        "regression.ols.residual_orthogonality",
        "visualmetrics.proofs.library.ols_orthogonality",
        "OLS residuals are orthogonal to the regressors",
        ("regression.ols_geometry",),
    ),
    ProofEntry(
        "regression.fwl.theorem",
        "visualmetrics.proofs.library.fwl",
        "Frisch-Waugh-Lovell theorem",
        ("regression.fwl",),
    ),
    ProofEntry(
        "inference.neyman_pearson.lemma",
        "visualmetrics.proofs.library.neyman_pearson",
        "Neyman-Pearson lemma",
        ("inference.neyman_pearson",),
    ),
    ProofEntry(
        "ml.lasso.sparsity_geometry",
        "visualmetrics.proofs.library.lasso_sparsity",
        "Why the lasso sets coefficients exactly to zero",
        ("ml.regularization",),
    ),
    ProofEntry(
        "multivariate.pca.variance_maximization",
        "visualmetrics.proofs.library.pca_variance",
        "The first principal component is the leading eigenvector",
        ("multivariate.pca",),
    ),
    ProofEntry(
        "deep_learning.backprop.chain_rule",
        "visualmetrics.proofs.library.backprop_chain_rule",
        "Backpropagation is the chain rule applied in reverse",
        ("deep_learning.backpropagation",),
    ),
    ProofEntry(
        "math.projection.shortest_distance",
        "visualmetrics.proofs.library.projection_distance",
        "The orthogonal projection is the closest point in a subspace",
        ("math.projection",),
    ),
    ProofEntry(
        "inference.gauss_markov.blue",
        "visualmetrics.proofs.library.gauss_markov",
        "Gauss-Markov theorem: OLS is the best linear unbiased estimator",
        ("regression.gauss_markov", "regression.ols_geometry"),
    ),
    ProofEntry(
        "inference.clt.moment_generating",
        "visualmetrics.proofs.library.clt",
        "Central limit theorem via the moment generating function",
        ("inference.clt",),
    ),
    ProofEntry(
        "econometrics.iv.consistency",
        "visualmetrics.proofs.library.iv_consistency",
        "Instrumental variables are consistent when OLS is not",
        ("econometrics.endogeneity_iv",),
    ),
    ProofEntry(
        "econometrics.ovb.formula",
        "visualmetrics.proofs.library.omitted_variable_bias",
        "Omitted variable bias: short = long + delta times gamma",
        ("econometrics.omitted_variable_bias",),
    ),
    ProofEntry(
        "inference.cramer_rao.bound",
        "visualmetrics.proofs.library.cramer_rao",
        "Cramer-Rao lower bound",
        ("inference.cramer_rao",),
    ),
    ProofEntry(
        "regression.restricted.f_statistic",
        "visualmetrics.proofs.library.restricted_ols",
        "Restricted least squares and the F statistic",
        ("regression.restricted",),
    ),
)

_BY_ID = {e.id: e for e in PROOF_INDEX}


class ProofRegistry:
    """Loads proofs on demand and caches them."""

    def __init__(self, entries: tuple[ProofEntry, ...] = PROOF_INDEX) -> None:
        self._entries = {e.id: e for e in entries}
        self._cache: dict[str, ProofSpec] = {}

    def entries(self, *, written_only: bool = False) -> tuple[ProofEntry, ...]:
        items = tuple(self._entries.values())
        if written_only:
            items = tuple(e for e in items if e.is_written)
        return items

    def ids(self, *, written_only: bool = False) -> tuple[str, ...]:
        return tuple(e.id for e in self.entries(written_only=written_only))

    def for_concept(self, concept_id: str) -> tuple[ProofEntry, ...]:
        return tuple(e for e in self._entries.values() if concept_id in e.concept_ids)

    def get(self, proof_id: str) -> ProofSpec:
        if proof_id in self._cache:
            return self._cache[proof_id]
        entry = self._entries.get(proof_id)
        if entry is None:
            raise ProofNotFoundError(
                f"unknown proof {proof_id!r}", key="errors.proof_unknown", proof_id=proof_id
            )
        if entry.module is None:
            raise ProofNotFoundError(
                f"proof {proof_id!r} is catalogued but not written yet",
                key="errors.proof_planned",
                proof_id=proof_id,
            )
        module = importlib.import_module(entry.module)
        spec = getattr(module, "PROOF", None)
        if not isinstance(spec, ProofSpec):
            raise ProofNotFoundError(
                f"module {entry.module!r} does not export a ProofSpec named PROOF",
                key="errors.proof_unknown",
                proof_id=proof_id,
            )
        if spec.id != proof_id:
            raise ProofNotFoundError(
                f"module {entry.module!r} exports proof {spec.id!r}, expected {proof_id!r}",
                key="errors.proof_unknown",
                proof_id=proof_id,
            )
        self._cache[proof_id] = spec
        return spec


proof_registry = ProofRegistry()


def proof(proof_id: str) -> ProofSpec:
    """Load one proof by id."""
    return proof_registry.get(proof_id)


def list_proofs(*, written_only: bool = False) -> tuple[ProofEntry, ...]:
    """List catalogued proofs."""
    return proof_registry.entries(written_only=written_only)
