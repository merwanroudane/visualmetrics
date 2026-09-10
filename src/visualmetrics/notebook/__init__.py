"""Jupyter integration (blueprint section 65).

``setup()`` teaches IPython to render a lab result as its full report - badge,
warnings, violated assumptions and numbers - so a notebook cell shows the
science rather than a bare figure.
"""

from __future__ import annotations

from typing import Any

from .display import (
    animation_html,
    display_animation,
    display_result,
    in_notebook,
    install_formatters,
    result_html,
)

__all__ = [
    "setup",
    "display_result",
    "display_animation",
    "result_html",
    "animation_html",
    "install_formatters",
    "in_notebook",
    "interact",
    "LabWidget",
    "control_widget",
]


def setup() -> bool:
    """Register the rich display for lab results. Safe to call more than once."""
    return install_formatters()


def __getattr__(name: str) -> Any:
    """Widgets are imported on demand so ipywidgets stays optional."""
    if name in {"interact", "LabWidget", "control_widget"}:
        from . import widgets

        return getattr(widgets, name)
    raise AttributeError(f"module 'visualmetrics.notebook' has no attribute {name!r}")
