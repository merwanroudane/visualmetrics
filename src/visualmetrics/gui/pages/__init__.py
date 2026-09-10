"""Page renderers. Each takes ``(…route args…, session, rerender)``."""

from .lab import lab_page
from .simple import (
    catalog_page,
    doctor_page,
    glossary_page,
    home_page,
    paths_page,
    proof_page,
    proofs_page,
)

__all__ = [
    "home_page",
    "catalog_page",
    "lab_page",
    "proofs_page",
    "proof_page",
    "glossary_page",
    "paths_page",
    "doctor_page",
]
