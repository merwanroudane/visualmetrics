"""Exception hierarchy for VisualMetrics.

Errors are designed to be *educational*: every error carries a stable
translation key so the GUI can present a localized, plain-language message
while still exposing the technical detail in an expandable panel.
"""

from __future__ import annotations

__all__ = [
    "VisualMetricsError",
    "ConfigurationError",
    "ConceptNotFoundError",
    "DuplicateConceptError",
    "InvalidParameterError",
    "UnsupportedScenarioError",
    "MissingDependencyError",
    "DataError",
    "IdentificationError",
    "NumericalError",
    "ProofUnavailableError",
    "ExportError",
    "TranslationError",
    "PluginError",
]


class VisualMetricsError(Exception):
    """Base class for every VisualMetrics error.

    Parameters
    ----------
    message:
        Technical English message (developer-facing).
    key:
        Translation key inside ``errors.json`` (user-facing).
    context:
        Values interpolated into the localized message.
    """

    default_key = "errors.generic"

    def __init__(self, message: str, *, key: str | None = None, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.key = key or self.default_key
        self.context = context

    def localized(self, translator=None) -> str:
        """Return the localized message, falling back to the technical one."""
        if translator is None:
            return self.message
        text = translator.t(self.key, default=None)
        if text is None:
            return self.message
        try:
            return text.format(**self.context)
        except (KeyError, IndexError):
            return text


class ConfigurationError(VisualMetricsError):
    default_key = "errors.configuration"


class ConceptNotFoundError(VisualMetricsError, KeyError):
    default_key = "errors.concept_not_found"


class DuplicateConceptError(VisualMetricsError):
    default_key = "errors.duplicate_concept"


class InvalidParameterError(VisualMetricsError, ValueError):
    default_key = "errors.invalid_parameter"


class UnsupportedScenarioError(VisualMetricsError):
    default_key = "errors.unsupported_scenario"


class MissingDependencyError(VisualMetricsError, ImportError):
    """Raised when an optional scientific extra is required but absent."""

    default_key = "errors.missing_dependency"

    def __init__(self, package: str, extra: str, *, feature: str | None = None) -> None:
        install = f'pip install "visualmetrics[{extra}]"'
        message = (
            f"'{package}' is required for {feature or 'this capability'} "
            f"but is not installed. Install the '{extra}' extra:  {install}"
        )
        super().__init__(message, package=package, extra=extra, install=install,
                         feature=feature or extra)
        self.package = package
        self.extra = extra
        self.install_command = install


class DataError(VisualMetricsError):
    default_key = "errors.data"


class IdentificationError(VisualMetricsError):
    default_key = "errors.identification"


class NumericalError(VisualMetricsError):
    default_key = "errors.numerical"


class ProofUnavailableError(VisualMetricsError):
    default_key = "errors.proof_unavailable"


class ExportError(VisualMetricsError):
    default_key = "errors.export"


class TranslationError(VisualMetricsError):
    default_key = "errors.translation"


class PluginError(VisualMetricsError):
    default_key = "errors.plugin"
