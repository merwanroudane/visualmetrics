"""Translation service.

No scientific string is hard-coded in a GUI component. Everything goes through
:class:`Translator`, which supports the three first-class languages (en, ar, fr),
three terminology modes, and transparent fallback to English.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = [
    "LANGUAGES",
    "LANGUAGE_NAMES",
    "TERMINOLOGY_MODES",
    "Translator",
    "get_translator",
    "set_language",
    "available_languages",
    "t",
]

LANGUAGES: tuple[str, ...] = ("en", "ar", "fr")
FALLBACK = "en"
RESOURCE_FILES: tuple[str, ...] = (
    "common",
    "concepts",
    "labs",
    "proofs",
    "glossary",
    "errors",
)

LANGUAGE_NAMES: dict[str, dict[str, str]] = {
    "en": {"native": "English", "en": "English", "ar": "الإنجليزية", "fr": "Anglais"},
    "ar": {"native": "العربية", "en": "Arabic", "ar": "العربية", "fr": "Arabe"},
    "fr": {"native": "Français", "en": "French", "ar": "الفرنسية", "fr": "Français"},
}

TERMINOLOGY_MODES: tuple[str, ...] = ("translated", "bilingual", "english_technical")

_lock = threading.RLock()
_current: Translator | None = None


@lru_cache(maxsize=None)
def _load_bundle(language: str) -> dict[str, Any]:
    """Load and flatten every resource file for one language."""
    bundle: dict[str, Any] = {}
    for name in RESOURCE_FILES:
        try:
            path = resources.files("visualmetrics.i18n") / language / f"{name}.json"
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, ModuleNotFoundError, OSError):
            continue
        _flatten_into(data, bundle)
    return bundle


def _flatten_into(data: dict[str, Any], out: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            _flatten_into(value, out, full + ".")
        else:
            out[full] = value


def available_languages() -> list[str]:
    return [lang for lang in LANGUAGES if _load_bundle(lang)]


class Translator:
    """Resolve translation keys for one language and terminology mode."""

    def __init__(self, language: str = "en", terminology: str = "translated") -> None:
        self.language = language if language in LANGUAGES else FALLBACK
        self.terminology = terminology if terminology in TERMINOLOGY_MODES else "translated"
        self._bundle = _load_bundle(self.language)
        self._fallback = _load_bundle(FALLBACK) if self.language != FALLBACK else self._bundle
        self.missing: set[str] = set()

    # -- core lookup -------------------------------------------------------
    def raw(self, key: str, *, language: str | None = None) -> str | None:
        bundle = _load_bundle(language) if language else self._bundle
        value = bundle.get(key)
        if value is None and language is None:
            value = self._fallback.get(key)
        return value if isinstance(value, str) else None

    def has(self, key: str) -> bool:
        return key in self._bundle or key in self._fallback

    def t(self, key: str, default: str | None = "", /, **params: Any) -> str:
        """Translate ``key``.

        ``default`` of ``""`` means "fall back to a humanized key"; pass
        ``None`` explicitly to get ``None`` for a missing key.
        """
        text = self.raw(key)
        if text is None:
            self.missing.add(key)
            if default is None:
                return None  # type: ignore[return-value]
            text = default or _humanize(key)
        if params:
            try:
                text = text.format(**params)
            except (KeyError, IndexError, ValueError):
                pass
        return text

    __call__ = t

    # -- terminology modes -------------------------------------------------
    def term(self, key: str, **params: Any) -> str:
        """Translate a *scientific term* honouring the terminology mode.

        ``bilingual`` renders e.g. ``القوة الإحصائية (Statistical Power)``.
        """
        english = self.raw(key, language="en") or _humanize(key)
        if self.language == "en" or self.terminology == "english_technical":
            return english.format(**params) if params else english
        local = self.raw(key)
        if local is None:
            self.missing.add(key)
            local = english
        if params:
            try:
                local = local.format(**params)
                english = english.format(**params)
            except (KeyError, IndexError, ValueError):
                pass
        if self.terminology == "bilingual" and local != english:
            return f"{local} ({english})"
        return local

    # -- locale helpers ----------------------------------------------------
    @property
    def is_rtl(self) -> bool:
        from .rtl import is_rtl

        return is_rtl(self.language)

    @property
    def direction(self) -> str:
        return "rtl" if self.is_rtl else "ltr"

    def language_name(self, language: str) -> str:
        return LANGUAGE_NAMES.get(language, {}).get("native", language)

    def number(self, value: float, precision: int = 4) -> str:
        """Locale-aware number formatting (Babel when available)."""
        try:
            from babel.numbers import format_decimal

            pattern = "#,##0." + "0" * precision if precision else "#,##0"
            return format_decimal(value, format=pattern, locale=self.language)
        except Exception:
            return f"{value:,.{precision}f}"

    def with_language(self, language: str) -> Translator:
        return Translator(language, self.terminology)

    def with_terminology(self, terminology: str) -> Translator:
        return Translator(self.language, terminology)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Translator {self.language} terminology={self.terminology}>"


def _humanize(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("_", " ").strip().capitalize()


def get_translator(language: str | None = None, terminology: str | None = None) -> Translator:
    """Return the process-wide translator, or a fresh one for an override."""
    global _current
    if language is None and terminology is None:
        with _lock:
            if _current is None:
                from ..config import get_config

                cfg = get_config()
                _current = Translator(cfg.language, cfg.terminology)
            return _current
    with _lock:
        base = _current
    lang = language or (base.language if base else "en")
    term = terminology or (base.terminology if base else "translated")
    return Translator(lang, term)


def set_language(language: str, terminology: str | None = None) -> Translator:
    global _current
    with _lock:
        term = terminology or (_current.terminology if _current else "translated")
        _current = Translator(language, term)
        return _current


def t(key: str, default: str | None = "", /, **params: Any) -> str:
    return get_translator().t(key, default, **params)
