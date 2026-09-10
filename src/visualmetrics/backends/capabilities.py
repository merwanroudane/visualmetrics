"""Optional-dependency detection.

Every optional scientific package is probed here once, defensively: a package
that is installed but broken (for example compiled against a different NumPy
ABI) must be reported as unavailable rather than crashing a lab.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "Capability",
    "CAPABILITIES",
    "probe",
    "have",
    "require",
    "capability_report",
    "backend_versions",
]


@dataclass(frozen=True)
class Capability:
    """One optional package and the extra that provides it."""

    package: str
    extra: str
    purpose: str
    import_name: str | None = None

    @property
    def module(self) -> str:
        return self.import_name or self.package.replace("-", "_")


CAPABILITIES: dict[str, Capability] = {
    "plotly": Capability("plotly", "viz", "interactive figures and animations"),
    "matplotlib": Capability("matplotlib", "viz", "static publication figures"),
    "nicegui": Capability("nicegui", "gui", "the desktop/browser GUI"),
    "sympy": Capability("sympy", "symbolic", "symbolic derivations and proof algebra"),
    "statsmodels": Capability("statsmodels", "econometrics", "regression, diagnostics, power"),
    "linearmodels": Capability("linearmodels", "econometrics", "IV/2SLS, panel and system models"),
    "arch": Capability("arch", "econometrics", "ARCH/GARCH, unit roots, bootstrap"),
    "pyfixest": Capability("pyfixest", "econometrics", "fixed effects and modern DiD"),
    "networkx": Capability("networkx", "causal", "causal graph structures"),
    "dowhy": Capability("dowhy", "causal", "identification and refutation workflows"),
    "econml": Capability("econml", "causal", "heterogeneous treatment effects and DML"),
    "sklearn": Capability("scikit-learn", "ai", "machine-learning models and metrics", "sklearn"),
    "shap": Capability("shap", "ai", "Shapley-value explanations"),
    "interpret": Capability("interpret", "ai", "glassbox/blackbox explanations"),
    "torch": Capability("torch", "ai", "trainable neural-network labs"),
    "manim": Capability("manim", "proofs", "high-quality exported proof animations"),
    "ipywidgets": Capability("ipywidgets", "notebook", "notebook widgets"),
    "kaleido": Capability("kaleido", "export", "static PNG/SVG export of Plotly figures"),
    "openpyxl": Capability("openpyxl", "data", "Excel import/export"),
    "pyarrow": Capability("pyarrow", "data", "Parquet import/export"),
    "pyreadstat": Capability("pyreadstat", "data", "Stata/SPSS/SAS import"),
}


@lru_cache(maxsize=None)
def probe(name: str) -> tuple[bool, str]:
    """Return ``(available, version_or_reason)`` for a capability.

    Import errors *and* runtime errors (broken binary builds) are caught.
    """
    cap = CAPABILITIES.get(name)
    module_name = cap.module if cap else name
    if importlib.util.find_spec(module_name) is None:
        return False, "not installed"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - a broken build must not crash us
        return False, f"installed but unusable: {type(exc).__name__}"
    return True, str(getattr(module, "__version__", "unknown"))


def have(name: str) -> bool:
    return probe(name)[0]


def require(name: str, *, feature: str | None = None):
    """Import an optional module or raise a friendly capability error."""
    from ..core.exceptions import MissingDependencyError

    cap = CAPABILITIES.get(name)
    ok, info = probe(name)
    if not ok:
        raise MissingDependencyError(
            cap.package if cap else name,
            cap.extra if cap else "all",
            feature=feature or (cap.purpose if cap else name),
        )
    return importlib.import_module(cap.module if cap else name)


def capability_report() -> dict[str, dict[str, object]]:
    """Structured report used by ``visualmetrics doctor`` and the GUI status bar."""
    report: dict[str, dict[str, object]] = {}
    for name, cap in CAPABILITIES.items():
        ok, info = probe(name)
        report[name] = {
            "package": cap.package,
            "extra": cap.extra,
            "purpose": cap.purpose,
            "available": ok,
            "version": info if ok else None,
            "reason": None if ok else info,
        }
    return report


def backend_versions(*names: str) -> dict[str, str]:
    """Versions of the backends a result depends on - stored in exported state."""
    out: dict[str, str] = {}
    for name in names:
        ok, info = probe(name)
        if ok:
            out[name] = info
    return out


def extras_status() -> dict[str, bool]:
    """True when every package of an extra is importable."""
    by_extra: dict[str, list[str]] = {}
    for name, cap in CAPABILITIES.items():
        by_extra.setdefault(cap.extra, []).append(name)
    return {extra: all(have(n) for n in names) for extra, names in sorted(by_extra.items())}
