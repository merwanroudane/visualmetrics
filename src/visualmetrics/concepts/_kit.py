"""Shared helpers for concept labs.

Keeps every lab focused on its science: state -> translator/theme plumbing,
narrative assembly, animation-step construction and the standard metric rows
all live here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.concepts import (
    ConceptSpec,
    Domain,
    LabBase,
    LearningMode,
    Level,
    Reference,
    Status,
)
from ..core.controls import int_slider, number, seed_control, select, slider, toggle
from ..core.evidence import EvidenceType
from ..core.scenarios import ScenarioCategory, scenario
from ..core.state import AnimationSpec, AnimationStep, LabResult, LabState, Panel
from ..i18n.rtl import protect_latin
from ..i18n.translator import Translator, get_translator
from ..visuals.plotly import primitives as P
from ..visuals.themes.palette import Theme, get_theme

__all__ = [
    "LabContext",
    "context",
    "make_spec",
    "ref",
    "animation",
    "build_frames",
    "Domain",
    "Level",
    "LearningMode",
    "Status",
    "EvidenceType",
    "ScenarioCategory",
    "scenario",
    "AnimationSpec",
    "AnimationStep",
    "LabBase",
    "LabResult",
    "LabState",
    "Panel",
    "ConceptSpec",
    "P",
    "np",
    "fmt",
    "pct",
    "slider",
    "int_slider",
    "select",
    "toggle",
    "number",
    "seed_control",
]


def fmt(value: Any, digits: int = 3) -> str:
    """Format a number for inline prose (never scientific-notation soup)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "n/a"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".") or "0"


def pct(value: float, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


class LabContext:
    """Per-run bundle of translator, theme and locale helpers."""

    __slots__ = ("level", "locale", "precision", "reduced_motion", "state", "theme", "tr")

    def __init__(self, state: LabState) -> None:
        self.state = state
        self.tr: Translator = get_translator(state.language, state.terminology)
        self.theme: Theme = get_theme(state.theme)
        self.locale = state.language
        self.level = state.level
        self.reduced_motion = state.reduced_motion
        self.precision = state.precision

    # -- text --------------------------------------------------------------
    def t(self, key: str, default: str = "", /, **params: Any) -> str:
        return protect_latin(self.tr.t(key, default, **params), self.locale)

    def term(self, key: str, **params: Any) -> str:
        return protect_latin(self.tr.term(key, **params), self.locale)

    def raw(self, key: str, default: str = "", /, **params: Any) -> str:
        return self.tr.t(key, default, **params)

    @property
    def is_rtl(self) -> bool:
        return self.tr.is_rtl

    def show_math(self) -> bool:
        """Beginners get plain language; advanced levels also get the algebra."""
        return self.level in ("advanced", "phd", "intermediate")

    def show_formal(self) -> bool:
        return self.level in ("advanced", "phd")

    # -- figures -----------------------------------------------------------
    def figure(self, title_key: str = "", /, **kw: Any) -> Any:
        title = self.t(title_key) if title_key else ""
        return P.new_figure(title, theme=self.theme, locale=self.locale, **kw)

    def panel(
        self,
        id: str,
        figure: Any,
        title_key: str = "",
        *,
        tab: str = "visualize",
        evidence: EvidenceType | None = None,
        caption: str = "",
    ) -> Panel:
        return Panel(
            id=id,
            figure=P.finalize(figure, reduced_motion=self.reduced_motion),
            title_key=title_key or None,
            title=self.t(title_key) if title_key else None,
            tab=tab,
            evidence=evidence,
            caption=caption,
        )

    def color(self, role: str) -> str:
        return self.theme.color(role)


def context(state: LabState) -> LabContext:
    return LabContext(state)


def ref(citation: str, *, kind: str = "book", url: str | None = None,
        doi: str | None = None) -> Reference:
    return Reference(citation=citation, kind=kind, url=url, doi=doi)


def make_spec(
    id: str,
    domain: Domain,
    subdomain: str,
    *,
    module: str,
    levels: tuple[str, ...] = ("intermediate",),
    modes: tuple[str, ...] = ("learn", "visualize", "experiment"),
    evidence: EvidenceType = EvidenceType.VISUAL_INTUITION,
    status: Status = Status.STABLE,
    controls: tuple[Any, ...] = (),
    scenarios: tuple[Any, ...] = (),
    **kw: Any,
) -> ConceptSpec:
    """Build a ConceptSpec with the conventional translation keys."""
    return ConceptSpec(
        id=id,
        domain=domain,
        subdomain=subdomain,
        title_key=kw.pop("title_key", f"concepts.{id}.title"),
        summary_key=kw.pop("summary_key", f"concepts.{id}.summary"),
        levels=tuple(Level(lv) for lv in levels),
        modes=tuple(LearningMode(m) for m in modes),
        evidence=evidence,
        status=status,
        controls=tuple(controls),
        scenarios=tuple(scenarios),
        module=module,
        **kw,
    )


def animation(
    id: str,
    figure: Any,
    steps: list[AnimationStep],
    *,
    purpose: str,
    summary: str,
    evidence: EvidenceType = EvidenceType.SIMULATION,
    frame_duration_ms: int = 550,
    static_alternative: Any = None,
) -> AnimationSpec:
    return AnimationSpec(
        id=id,
        figure=figure,
        steps=steps,
        purpose=purpose,
        summary=summary,
        evidence=evidence,
        frame_duration_ms=frame_duration_ms,
        static_alternative=static_alternative,
    )


def build_frames(figure: Any, frames: list[Any], *, duration: int = 550,
                 reduced_motion: bool = False, slider_label: str = "") -> Any:
    """Attach plotly frames plus play/pause/step controls to a figure.

    Reduced-motion mode keeps the scrubber (so the learner can still step
    through) but removes autoplay.
    """
    go = P.require_plotly()
    figure.frames = frames
    steps = [
        {
            "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f.name,
            "method": "animate",
        }
        for f in frames
    ]
    buttons = [
        {
            "label": "Step",
            "method": "animate",
            "args": [None, {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        }
    ]
    if not reduced_motion:
        buttons = [
            {
                "label": "Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                ],
            },
        ]
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.16,
                "xanchor": "left",
                "showactive": False,
                "buttons": buttons,
            }
        ],
        sliders=[
            {
                "active": 0,
                "y": -0.06,
                "x": 0.06,
                "len": 0.9,
                "currentvalue": {"prefix": (slider_label + ": ") if slider_label else ""},
                "steps": steps,
            }
        ],
    )
    del go
    return figure
