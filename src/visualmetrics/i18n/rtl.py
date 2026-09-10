"""Right-to-left support for Arabic.

Blueprint section 38.2: the page flow mirrors, but mathematics, Latin variable
names and numeric axes must NOT mirror. These helpers make that distinction
explicit instead of leaving it to CSS accidents.
"""

from __future__ import annotations

import re

__all__ = [
    "RTL_LANGUAGES",
    "is_rtl",
    "direction",
    "text_align",
    "flip_side",
    "isolate_ltr",
    "protect_latin",
    "contains_arabic",
    "axis_layout",
    "bidi_css",
]

RTL_LANGUAGES: frozenset[str] = frozenset({"ar", "he", "fa", "ur"})

#: U+2066 LRI ... U+2069 PDI - isolates Latin runs inside Arabic prose so that
#: identifiers such as ``beta_1`` or ``R^2`` are not visually reordered.
_LRI = "⁦"
_PDI = "⁩"

_LATIN_RUN = re.compile(r"([A-Za-z][A-Za-z0-9_.^()\\-]*)")
_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def is_rtl(language: str | None) -> bool:
    if not language:
        return False
    return language.split("-")[0].split("_")[0].lower() in RTL_LANGUAGES


def direction(language: str | None) -> str:
    return "rtl" if is_rtl(language) else "ltr"


def text_align(language: str | None) -> str:
    return "right" if is_rtl(language) else "left"


def flip_side(side: str, language: str | None) -> str:
    """Mirror a layout side for RTL ('left' <-> 'right'); other values pass through."""
    if not is_rtl(language):
        return side
    return {"left": "right", "right": "left", "start": "end", "end": "start"}.get(side, side)


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text or ""))


def isolate_ltr(text: str) -> str:
    """Wrap a whole string in an LTR isolate."""
    return f"{_LRI}{text}{_PDI}"


def protect_latin(text: str, language: str | None = "ar") -> str:
    """Isolate Latin runs inside RTL prose so identifiers keep their order.

    ``"القوة تعتمد على beta_1"`` keeps ``beta_1`` readable left-to-right.
    Strings without Arabic are returned untouched.
    """
    if not is_rtl(language) or not text or not contains_arabic(text):
        return text
    return _LATIN_RUN.sub(lambda m: isolate_ltr(m.group(1)), text)


def axis_layout(language: str | None) -> dict[str, object]:
    """Plotly layout fragment for a locale.

    Numeric axes are never mirrored - only the textual side of the layout is.
    """
    if not is_rtl(language):
        return {}
    return {
        "legend": {"x": 0.01, "xanchor": "left"},
        "yaxis": {"side": "right"},
        "title": {"x": 0.98, "xanchor": "right"},
    }


def bidi_css(language: str | None) -> str:
    """Minimal CSS injected by the GUI when a language is activated."""
    if not is_rtl(language):
        return "body{direction:ltr;}"
    return (
        "body{direction:rtl;text-align:right;}"
        ".vm-ltr,.vm-math,.vm-code,code,pre,.katex{direction:ltr;text-align:left;"
        "unicode-bidi:isolate;}"
        ".vm-plot{direction:ltr;}"
    )
