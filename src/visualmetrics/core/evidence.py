"""Scientific classification of visual explanations.

VisualMetrics never lets a simulation masquerade as a proof.  Every figure,
animation and derivation step declares an :class:`EvidenceType`, which the GUI
renders as a visible badge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["EvidenceType", "EvidenceBadge", "EVIDENCE_BADGES", "badge_for"]


class EvidenceType(str, Enum):
    """What kind of scientific claim a visual actually supports."""

    FORMAL_PROOF = "formal_proof"
    SYMBOLIC_DERIVATION = "symbolic_derivation"
    GEOMETRIC_PROOF = "geometric_proof"
    VISUAL_DERIVATION = "visual_derivation"
    VISUAL_INTUITION = "visual_intuition"
    SIMULATION = "simulation"
    NUMERICAL_DEMONSTRATION = "numerical_demonstration"
    COUNTEREXAMPLE = "counterexample"
    EMPIRICAL_EXAMPLE = "empirical_example"

    @property
    def is_proof(self) -> bool:
        """True only for evidence types that actually establish a theorem."""
        return self in {
            EvidenceType.FORMAL_PROOF,
            EvidenceType.GEOMETRIC_PROOF,
            EvidenceType.SYMBOLIC_DERIVATION,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBadge:
    """Presentation metadata for an evidence type."""

    evidence: EvidenceType
    label_key: str
    caveat_key: str
    color_role: str
    icon: str


EVIDENCE_BADGES: dict[EvidenceType, EvidenceBadge] = {
    EvidenceType.FORMAL_PROOF: EvidenceBadge(
        EvidenceType.FORMAL_PROOF, "evidence.formal_proof.label",
        "evidence.formal_proof.caveat", "primary", "📐"),
    EvidenceType.SYMBOLIC_DERIVATION: EvidenceBadge(
        EvidenceType.SYMBOLIC_DERIVATION, "evidence.symbolic_derivation.label",
        "evidence.symbolic_derivation.caveat", "primary", "∑"),
    EvidenceType.GEOMETRIC_PROOF: EvidenceBadge(
        EvidenceType.GEOMETRIC_PROOF, "evidence.geometric_proof.label",
        "evidence.geometric_proof.caveat", "primary", "⊥"),
    EvidenceType.VISUAL_DERIVATION: EvidenceBadge(
        EvidenceType.VISUAL_DERIVATION, "evidence.visual_derivation.label",
        "evidence.visual_derivation.caveat", "secondary", "✎"),
    EvidenceType.VISUAL_INTUITION: EvidenceBadge(
        EvidenceType.VISUAL_INTUITION, "evidence.visual_intuition.label",
        "evidence.visual_intuition.caveat", "info", "👁"),
    EvidenceType.SIMULATION: EvidenceBadge(
        EvidenceType.SIMULATION, "evidence.simulation.label",
        "evidence.simulation.caveat", "info", "🎲"),
    EvidenceType.NUMERICAL_DEMONSTRATION: EvidenceBadge(
        EvidenceType.NUMERICAL_DEMONSTRATION, "evidence.numerical_demonstration.label",
        "evidence.numerical_demonstration.caveat", "info", "#"),
    EvidenceType.COUNTEREXAMPLE: EvidenceBadge(
        EvidenceType.COUNTEREXAMPLE, "evidence.counterexample.label",
        "evidence.counterexample.caveat", "warning", "⚠"),
    EvidenceType.EMPIRICAL_EXAMPLE: EvidenceBadge(
        EvidenceType.EMPIRICAL_EXAMPLE, "evidence.empirical_example.label",
        "evidence.empirical_example.caveat", "baseline", "📊"),
}


def badge_for(evidence: EvidenceType | str) -> EvidenceBadge:
    """Return the badge metadata for an evidence type."""
    return EVIDENCE_BADGES[EvidenceType(evidence)]
