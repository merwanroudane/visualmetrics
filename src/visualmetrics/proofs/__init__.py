"""Declarative proof engine (blueprint section 48).

Proofs are data: one :class:`ProofSpec` per theorem, rendered by whichever
renderer the caller has - the lightweight step list in the GUI, the Plotly
geometric renderer, or a text transcript for the CLI and exports.

Only evidence kinds that really are proofs may be used here; a simulation can
never be dressed up as one.
"""

from .registry import (
    ProofEntry,
    ProofNotFoundError,
    ProofRegistry,
    list_proofs,
    proof,
    proof_registry,
)
from .spec import (
    CheckOutcome,
    ProofAssumption,
    ProofCheck,
    ProofSpec,
    ProofStep,
    RenderedProof,
    RenderedStep,
    StepKind,
    VisualAction,
)

__all__ = [
    "ProofSpec",
    "ProofStep",
    "ProofAssumption",
    "ProofCheck",
    "CheckOutcome",
    "StepKind",
    "VisualAction",
    "RenderedProof",
    "RenderedStep",
    "ProofRegistry",
    "ProofEntry",
    "ProofNotFoundError",
    "proof_registry",
    "proof",
    "list_proofs",
]
