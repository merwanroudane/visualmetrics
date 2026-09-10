"""VisualMetrics - a multilingual interactive visual laboratory.

Statistics, statistical inference, econometrics, causal inference, machine
learning and AI - made visible, movable, breakable, comparable and reproducible.

Quick start
-----------
>>> import visualmetrics as vm
>>> vm.configure(language="en", level="intermediate")     # doctest: +SKIP
>>> result = vm.lab("regression.simple_linear", beta1=-1.8, n=120, seed=42)
>>> print(result.summary())                               # doctest: +SKIP
>>> vm.launch()                                           # opens the GUI  # doctest: +SKIP

Everything heavy (plotly, statsmodels, sklearn, torch, manim, nicegui) is
imported lazily, so ``import visualmetrics`` stays fast.
"""

from __future__ import annotations

from typing import Any

from .config import Config, configure, get_config, reset_config
from .core.concepts import ConceptSpec, Domain, LearningMode, Level, Status
from .core.evidence import EvidenceType
from .core.exceptions import (
    ConceptNotFoundError,
    MissingDependencyError,
    VisualMetricsError,
)
from .core.registry import registry
from .core.state import LabResult, LabState
from .version import __version__

__all__ = [
    "__version__",
    "Config",
    "ConceptSpec",
    "ConceptNotFoundError",
    "Domain",
    "EvidenceType",
    "LabResult",
    "LabState",
    "LearningMode",
    "Level",
    "MissingDependencyError",
    "Status",
    "VisualMetricsError",
    "compare",
    "concept",
    "configure",
    "doctor",
    "explain",
    "get_config",
    "info",
    "lab",
    "launch",
    "learning_path",
    "list_concepts",
    "prerequisites",
    "registry",
    "reset_config",
    "run_state",
    "search",
    "glossary",
    "proof",
    "proofs_for",
    "list_proofs",
]


def lab(concept_id: str, *, scenario: str | None = None, **parameters: Any) -> LabResult:
    """Run a concept lab and return its figures, metrics and explanations.

    Parameters
    ----------
    concept_id:
        Full id (``"inference.power"``) or an unambiguous short name (``"power"``).
    scenario:
        Name of a scenario preset; explicit keyword parameters win over it.
    **parameters:
        Any control declared by the concept, plus the session overrides
        ``seed``, ``language``, ``theme``, ``level`` and ``reduced_motion``.
    """
    cfg = get_config()
    session_keys = {
        "language", "theme", "level", "terminology", "reduced_motion", "precision",
    }
    session = {k: parameters.pop(k) for k in list(parameters) if k in session_keys}
    seed = parameters.pop("seed", cfg.seed)
    resolved = registry.resolve(concept_id)
    state = LabState(
        concept_id=resolved,
        parameters=parameters,
        scenario=scenario,
        seed=int(seed),
        language=session.get("language", cfg.language),
        theme=session.get("theme", cfg.theme),
        level=session.get("level", cfg.level),
        terminology=session.get("terminology", cfg.terminology),
        reduced_motion=bool(session.get("reduced_motion", cfg.reduced_motion)),
        precision=int(session.get("precision", cfg.precision)),
    )
    return registry.lab(resolved).run(state)


def run_state(state: LabState) -> LabResult:
    """Re-run an exported :class:`LabState` - the reproducibility entry point."""
    return registry.lab(state.concept_id).run(state)


def concept(concept_id: str) -> ConceptSpec:
    """Return the specification of a concept (controls, scenarios, references)."""
    return registry.get(registry.resolve(concept_id))


def list_concepts(
    domain: str | None = None,
    *,
    level: str | None = None,
    implemented_only: bool = True,
    as_table: bool = False,
) -> Any:
    """List catalogued concepts."""
    specs = registry.specs(domain=domain, level=level, implemented_only=implemented_only)
    if not as_table:
        return specs
    from .i18n.translator import get_translator

    tr = get_translator()
    return [
        {
            "id": s.id,
            "domain": s.domain.value,
            "title": tr.t(s.title_key),
            "status": s.status.value,
            "levels": [lv.value for lv in s.levels],
        }
        for s in specs
    ]


def search(query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search concepts in English, Arabic or French."""
    from .catalog.search import search_concepts

    return search_concepts(query, limit=limit)


def explain(concept_id: str, *, level: str | None = None) -> str:
    """Return the localized explanation of a concept as plain text."""
    from .catalog.explain import explain_concept

    return explain_concept(concept_id, level=level)


def compare(*concept_ids: str, **parameters: Any) -> list[LabResult]:
    """Run several labs with the same parameters for side-by-side comparison."""
    if len(concept_ids) < 2:
        raise ValueError("compare() needs at least two concept ids")
    return [lab(cid, **parameters) for cid in concept_ids]


def prerequisites(concept_id: str, *, depth: int = 3) -> list[str]:
    """What should I learn first?"""
    from .catalog.graph import prerequisite_chain

    return prerequisite_chain(registry.resolve(concept_id), depth=depth)


def learning_path(target: str, *, level: str | None = None) -> list[str]:
    """An ordered path of concepts leading to ``target``."""
    from .catalog.paths import build_path

    return build_path(registry.resolve(target), level=level)


def glossary(term: str | None = None, language: str | None = None) -> Any:
    """Look up a scientific term in English, Arabic or French."""
    from .i18n.glossary import glossary_payload, lookup

    lang = language or get_config().language
    if term is None:
        return glossary_payload(lang)
    return lookup(term)


def doctor(*, as_text: bool = True) -> Any:
    """Report which capabilities are available in this environment."""
    from .cli import build_doctor_report, format_doctor_report

    report = build_doctor_report()
    return format_doctor_report(report) if as_text else report


def info() -> dict[str, Any]:
    """Version, catalog size and configuration in one dictionary."""
    from .backends.capabilities import extras_status

    cfg = get_config()
    return {
        "version": __version__,
        "catalog": registry.stats(),
        "config": cfg.to_dict(),
        "extras": extras_status(),
    }


def launch(
    *,
    host: str | None = None,
    port: int | None = None,
    native: bool | None = None,
    open_browser: bool = True,
    reload: bool = False,
) -> None:
    """Launch the local GUI (requires the ``gui`` extra)."""
    from .gui.app import launch_gui

    launch_gui(host=host, port=port, native=native, open_browser=open_browser, reload=reload)


def proof(proof_id: str, *, language: str | None = None) -> Any:
    """Load a proof and render it in ``language``.

    Returns a :class:`~visualmetrics.proofs.spec.RenderedProof`: claim,
    assumptions, numbered steps with their justifications, conclusion, and an
    explicit statement of what the proof does not establish.
    """
    from .i18n.translator import get_translator
    from .proofs import proof as _proof

    spec = _proof(proof_id)
    return spec.render(get_translator(language or get_config().language))


def proofs_for(concept_id: str) -> tuple[str, ...]:
    """Ids of the proofs attached to a concept."""
    from .proofs import proof_registry

    return tuple(e.id for e in proof_registry.for_concept(concept_id))


def list_proofs(*, written_only: bool = False) -> tuple[Any, ...]:
    """List catalogued proofs (``written_only`` hides the planned ones)."""
    from .proofs import list_proofs as _list

    return _list(written_only=written_only)


def __getattr__(name: str) -> Any:
    """Lazy convenience namespaces: ``vm.stats``, ``vm.econometrics``, ``vm.causal``, ..."""
    from .catalog.namespaces import NAMESPACES

    if name in NAMESPACES:
        from .catalog.namespaces import build_namespace

        ns = build_namespace(name)
        globals()[name] = ns
        return ns
    raise AttributeError(f"module 'visualmetrics' has no attribute {name!r}")


def __dir__() -> list[str]:
    from .catalog.namespaces import NAMESPACES

    return sorted(set(__all__) | set(NAMESPACES))
