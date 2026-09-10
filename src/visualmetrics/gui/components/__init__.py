"""Reusable GUI components.

Each one renders a piece of the scientific layer - the badge, the assumptions,
the frame commentary, the proof steps - so that layer cannot be dropped by
accident from any page that shows a result.
"""

from .animation import AnimationPlayer, render_animation
from .catalog import filter_catalog, render_catalog, render_concept_card
from .controls import render_control, render_control_panel, render_scenario_picker
from .layout import apply_session_style, page_shell, render_header, shortcuts_dialog
from .proofview import ProofViewer, render_proof
from .result import (
    render_assumptions,
    render_badge,
    render_metrics,
    render_panel,
    render_result,
    render_tabs,
    render_warnings,
)

__all__ = [
    "page_shell",
    "render_header",
    "apply_session_style",
    "shortcuts_dialog",
    "render_catalog",
    "render_concept_card",
    "filter_catalog",
    "render_control",
    "render_control_panel",
    "render_scenario_picker",
    "render_result",
    "render_tabs",
    "render_panel",
    "render_badge",
    "render_metrics",
    "render_assumptions",
    "render_warnings",
    "render_animation",
    "AnimationPlayer",
    "render_proof",
    "ProofViewer",
]
