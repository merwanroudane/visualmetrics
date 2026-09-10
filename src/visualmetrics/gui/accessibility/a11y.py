"""Accessibility helpers (blueprint section 36.9).

Three commitments, expressed as code rather than as intentions:

* nothing is encoded by colour alone - every colour-carrying series also gets a
  dash pattern or a marker shape, which the figure themes already provide and
  which :func:`describe_series` states in words;
* motion is a preference the application obeys, and an animation always has a
  static equivalent that shows the same evidence;
* every figure carries a text description, so a reader using a screen reader
  gets the finding rather than "chart".
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "figure_description",
    "describe_series",
    "keyboard_shortcuts",
    "aria_for_assumption",
    "announce",
    "contrast_ratio",
    "meets_contrast",
]


def _relative_luminance(hex_colour: str) -> float:
    text = hex_colour.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    try:
        channels = [int(text[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return 0.0
    adjusted = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * adjusted[0] + 0.7152 * adjusted[1] + 0.0722 * adjusted[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two hex colours (1.0 to 21.0)."""
    light = _relative_luminance(foreground)
    dark = _relative_luminance(background)
    high, low = max(light, dark), min(light, dark)
    return (high + 0.05) / (low + 0.05)


def meets_contrast(foreground: str, background: str, *, large_text: bool = False) -> bool:
    """WCAG AA: 4.5:1 for body text, 3:1 for large text."""
    return contrast_ratio(foreground, background) >= (3.0 if large_text else 4.5)


def describe_series(label: str, *, dash: str = "solid", symbol: str = "circle") -> str:
    """Name the redundant encoding, so colour is never the only channel."""
    dash_word = {"solid": "solid line", "dash": "dashed line", "dot": "dotted line",
                 "dashdot": "dash-dotted line"}.get(dash, f"{dash} line")
    return f"{label}: {dash_word}, {symbol} markers"


def figure_description(panel: Any, view: Any = None) -> str:
    """A text alternative for one figure.

    Prefers what the lab actually said about the panel; falls back to naming
    the panel and its evidence type, which is still more useful than nothing.
    """
    parts: list[str] = []
    title = getattr(panel, "title", None) or getattr(panel, "id", "figure")
    parts.append(str(title))
    caption = getattr(panel, "caption", None) or getattr(panel, "description", None)
    if caption:
        parts.append(str(caption))
    evidence = getattr(panel, "evidence", None)
    if evidence is not None:
        parts.append(f"Evidence type: {getattr(evidence, 'value', evidence)}.")
    if view is not None and getattr(view, "metrics", None):
        headline = view.metrics[0]
        parts.append(f"{headline.label}: {headline.value}.")
    return " ".join(parts)


def aria_for_assumption(assumption: Any) -> dict[str, str]:
    """Attributes that state an assumption's status without relying on colour."""
    holds = bool(getattr(assumption, "holds", True))
    return {
        "role": "status",
        "aria-label": (
            f"{getattr(assumption, 'label', 'assumption')}: "
            f"{'holds' if holds else 'violated'}"
        ),
        "data-holds": "true" if holds else "false",
    }


def announce(message: str) -> dict[str, str]:
    """Attributes for a live region, so changes are spoken as they happen."""
    return {"role": "status", "aria-live": "polite", "aria-atomic": "true",
            "aria-label": message}


def keyboard_shortcuts() -> dict[str, str]:
    """The keyboard map, kept in one place so help and handlers cannot disagree."""
    return {
        "?": "shortcuts.help",
        "/": "shortcuts.search",
        "g c": "shortcuts.catalog",
        "g h": "shortcuts.home",
        "g p": "shortcuts.proofs",
        "ArrowRight": "shortcuts.next_step",
        "ArrowLeft": "shortcuts.previous_step",
        "Space": "shortcuts.play_pause",
        "r": "shortcuts.rerun",
        "n": "shortcuts.new_seed",
        "s": "shortcuts.next_scenario",
        "c": "shortcuts.toggle_code",
        "t": "shortcuts.cycle_theme",
        "l": "shortcuts.cycle_language",
        "m": "shortcuts.toggle_motion",
        "f": "shortcuts.presentation",
        "e": "shortcuts.export",
        "Escape": "shortcuts.close",
    }
