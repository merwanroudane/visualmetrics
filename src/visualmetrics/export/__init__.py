"""Export figures, reports, animations and configurations (blueprint section 66).

Every exporter carries the scientific layer with the picture: the evidence
badge and its caveat, the assumptions and their status, the warnings, and for
an animation the per-frame explanation. Exporting a figure stripped of that
layer is exactly the failure mode this package exists to avoid, so no exporter
here offers it.
"""

from .animation import export_animation, export_animation_frames, render_animation_html
from .code import render_lab_code, render_state_script, render_value
from .configs import (
    config_from_dict,
    config_to_dict,
    load_config,
    load_configs,
    save_config,
    save_configs,
)
from .html import export_html_report, render_html_report
from .images import (
    STATIC_FORMATS,
    SUPPORTED_FORMATS,
    export_figure,
    export_result,
    figure_to_html,
    figure_to_json,
    result_to_dict,
)

__all__ = [
    "export_figure",
    "export_result",
    "figure_to_html",
    "figure_to_json",
    "result_to_dict",
    "SUPPORTED_FORMATS",
    "STATIC_FORMATS",
    "export_html_report",
    "render_html_report",
    "export_animation",
    "export_animation_frames",
    "render_animation_html",
    "save_config",
    "load_config",
    "save_configs",
    "load_configs",
    "config_to_dict",
    "config_from_dict",
    "render_lab_code",
    "render_state_script",
    "render_value",
]
