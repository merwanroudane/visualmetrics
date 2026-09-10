"""Shared helpers for proof-library modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.concepts import Level, Reference
from ..core.evidence import EvidenceType
from .spec import (
    CheckOutcome,
    ProofAssumption,
    ProofCheck,
    ProofSpec,
    ProofStep,
    StepKind,
    VisualAction,
)

__all__ = [
    "EvidenceType",
    "Level",
    "Reference",
    "ProofSpec",
    "ProofStep",
    "ProofAssumption",
    "ProofCheck",
    "CheckOutcome",
    "StepKind",
    "VisualAction",
    "step",
    "assume",
    "numeric_check",
    "np",
]


def step(
    step_id: str,
    statement: str,
    *,
    equation: str | None = None,
    why: str = "",
    kind: StepKind = StepKind.ALGEBRA,
    visual: VisualAction = VisualAction.NONE,
    uses: tuple[str, ...] = (),
    **visual_hint: object,
) -> ProofStep:
    """Terse constructor so a proof module reads like the proof itself."""
    return ProofStep(
        id=step_id,
        statement=statement,
        equation=equation,
        justification=why,
        kind=kind,
        visual_action=visual,
        visual_hint=dict(visual_hint),
        uses=uses,
    )


def assume(
    assumption_id: str, statement: str, *, essential: bool = True, if_violated: str = ""
) -> ProofAssumption:
    return ProofAssumption(
        id=assumption_id, statement=statement, essential=essential, if_violated=if_violated
    )


def numeric_check(
    check_id: str,
    description: str,
    compute: Callable[[], tuple[float, str]],
    *,
    tol: float = 1e-8,
) -> ProofCheck:
    """Wrap a callable returning ``(residual, detail)`` into a :class:`ProofCheck`.

    ``residual`` is the deviation from an identity the proof established; the
    check passes when it is within ``tol``.  Verifying an identity numerically
    tests this package's code, not the theorem.
    """

    def run() -> CheckOutcome:
        try:
            value, detail = compute()
        except Exception as exc:  # pragma: no cover - defensive
            return CheckOutcome(check_id, False, f"check raised {exc!r}", None, tol)
        ok = bool(np.isfinite(value)) and abs(value) <= tol
        return CheckOutcome(check_id, ok, detail, float(value), tol)

    return ProofCheck(id=check_id, description=description, fn=run)
