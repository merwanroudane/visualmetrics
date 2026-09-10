"""Semantic color system.

Concept modules never write hex codes. They request *roles* - ``color("null")``,
``color("power")``, ``color("type_i")`` - so themes, high-contrast mode and
colour-vision-deficiency palettes work everywhere at once (blueprint 37, 75).

Colour is never the only carrier of meaning: every role also exposes a dash
pattern and a marker symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "Theme",
    "THEMES",
    "get_theme",
    "color",
    "encoding",
    "list_themes",
    "register_theme",
]

#: Blueprint section 37.1 defaults.
BASE_ROLES: dict[str, str] = {
    "primary": "#0F766E",
    "secondary": "#7C3AED",
    "positive": "#16A34A",
    "negative": "#DC2626",
    "warning": "#F59E0B",
    "info": "#0284C7",
    "baseline": "#64748B",
    "null": "#64748B",
    "alternative": "#0284C7",
    "treatment": "#7C3AED",
    "control": "#0F766E",
    "type_i": "#E11D48",
    "type_ii": "#D97706",
    "power": "#16A34A",
    "training": "#2563EB",
    "validation": "#9333EA",
    "test": "#EA580C",
    "truth": "#111827",
    "estimate": "#0F766E",
    "residual": "#DC2626",
    "fitted": "#0284C7",
    "projection": "#7C3AED",
    "highlight": "#F59E0B",
    "muted": "#94A3B8",
    "grid": "#E2E8F0",
    "background": "#FFFFFF",
    "paper": "#FFFFFF",
    "surface": "#F8FAFC",
    "foreground": "#0F172A",
    "axis": "#475569",
}

#: Redundant encodings so the figure survives greyscale printing.
ROLE_ENCODING: dict[str, dict[str, str]] = {
    "null": {"dash": "dash", "symbol": "circle-open"},
    "baseline": {"dash": "dash", "symbol": "circle-open"},
    "alternative": {"dash": "solid", "symbol": "diamond"},
    "truth": {"dash": "solid", "symbol": "star"},
    "estimate": {"dash": "solid", "symbol": "circle"},
    "fitted": {"dash": "solid", "symbol": "circle"},
    "residual": {"dash": "dot", "symbol": "x"},
    "treatment": {"dash": "solid", "symbol": "square"},
    "control": {"dash": "dashdot", "symbol": "triangle-up"},
    "training": {"dash": "solid", "symbol": "circle"},
    "validation": {"dash": "dash", "symbol": "square"},
    "test": {"dash": "dot", "symbol": "diamond"},
    "type_i": {"dash": "solid", "symbol": "x"},
    "type_ii": {"dash": "dot", "symbol": "cross"},
    "power": {"dash": "solid", "symbol": "star"},
    "projection": {"dash": "dot", "symbol": "diamond-open"},
}

DEFAULT_ENCODING = {"dash": "solid", "symbol": "circle"}


@dataclass(frozen=True)
class Theme:
    """A named palette plus typography and plot styling."""

    name: str
    roles: dict[str, str] = field(default_factory=lambda: dict(BASE_ROLES))
    font_family: str = "Inter, Segoe UI, Tahoma, Arial, sans-serif"
    font_size: int = 13
    font_scale: float = 1.0
    line_width: float = 2.4
    marker_size: int = 7
    grid: bool = True
    dark: bool = False
    monochrome: bool = False
    animation_emphasis: float = 1.0
    #: continuous scale for heatmaps / surfaces
    colorscale: str = "Viridis"
    diverging_colorscale: str = "RdBu"

    def color(self, role: str, default: str | None = None) -> str:
        if self.monochrome:
            return self.roles.get("foreground", "#111827")
        return self.roles.get(role, default or self.roles.get("primary", "#0F766E"))

    def encoding(self, role: str) -> dict[str, str]:
        return dict(ROLE_ENCODING.get(role, DEFAULT_ENCODING))

    def scaled_font(self) -> int:
        return int(round(self.font_size * self.font_scale))

    def plotly_layout(self, *, locale: str = "en") -> dict[str, Any]:
        """Layout fragment applied to every VisualMetrics figure."""
        from ...i18n.rtl import axis_layout

        axis = {
            "gridcolor": self.color("grid"),
            "zerolinecolor": self.color("axis"),
            "linecolor": self.color("axis"),
            "showgrid": self.grid,
            "title": {"font": {"size": self.scaled_font()}},
        }
        layout: dict[str, Any] = {
            "template": "plotly_dark" if self.dark else "plotly_white",
            "paper_bgcolor": self.color("paper"),
            "plot_bgcolor": self.color("background"),
            "font": {
                "family": self.font_family,
                "size": self.scaled_font(),
                "color": self.color("foreground"),
            },
            "xaxis": dict(axis),
            "yaxis": dict(axis),
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "font": {"size": self.scaled_font()},
            },
            "margin": {"l": 60, "r": 30, "t": 56, "b": 55},
            "hoverlabel": {"font": {"family": self.font_family, "size": self.scaled_font()}},
            "colorway": [
                self.color("primary"),
                self.color("secondary"),
                self.color("warning"),
                self.color("info"),
                self.color("negative"),
                self.color("positive"),
                self.color("test"),
                self.color("baseline"),
            ],
        }
        layout.update(axis_layout(locale))
        return layout

    def with_(self, **changes: Any) -> Theme:
        return replace(self, **changes)


def _dark_roles() -> dict[str, str]:
    roles = dict(BASE_ROLES)
    roles.update(
        {
            "background": "#0B1220",
            "paper": "#0B1220",
            "surface": "#111827",
            "foreground": "#E2E8F0",
            "grid": "#1E293B",
            "axis": "#94A3B8",
            "muted": "#64748B",
            "truth": "#F8FAFC",
            "primary": "#2DD4BF",
            "secondary": "#A78BFA",
            "positive": "#4ADE80",
            "negative": "#F87171",
            "warning": "#FBBF24",
            "info": "#38BDF8",
            "type_i": "#FB7185",
            "type_ii": "#FBBF24",
            "power": "#4ADE80",
        }
    )
    return roles


def _high_contrast_roles() -> dict[str, str]:
    roles = dict(BASE_ROLES)
    roles.update(
        {
            "background": "#FFFFFF",
            "paper": "#FFFFFF",
            "foreground": "#000000",
            "grid": "#9CA3AF",
            "axis": "#000000",
            "primary": "#004D40",
            "secondary": "#4A148C",
            "positive": "#004D00",
            "negative": "#8B0000",
            "warning": "#7A4F01",
            "info": "#01579B",
            "baseline": "#000000",
            "null": "#000000",
            "muted": "#4B5563",
        }
    )
    return roles


def _colorblind_roles() -> dict[str, str]:
    """Okabe-Ito palette: distinguishable under the common CVD types."""
    roles = dict(BASE_ROLES)
    roles.update(
        {
            "primary": "#0072B2",
            "secondary": "#CC79A7",
            "positive": "#009E73",
            "negative": "#D55E00",
            "warning": "#E69F00",
            "info": "#56B4E9",
            "baseline": "#555555",
            "null": "#555555",
            "alternative": "#0072B2",
            "treatment": "#CC79A7",
            "control": "#009E73",
            "type_i": "#D55E00",
            "type_ii": "#E69F00",
            "power": "#009E73",
            "training": "#0072B2",
            "validation": "#CC79A7",
            "test": "#E69F00",
            "truth": "#000000",
        }
    )
    return roles


def _publication_roles() -> dict[str, str]:
    roles = dict(BASE_ROLES)
    roles.update(
        {
            "primary": "#1F2937",
            "secondary": "#4B5563",
            "positive": "#374151",
            "negative": "#111827",
            "baseline": "#9CA3AF",
            "null": "#9CA3AF",
            "grid": "#E5E7EB",
            "axis": "#111827",
            "foreground": "#111827",
        }
    )
    return roles


THEMES: dict[str, Theme] = {
    "light": Theme(name="light"),
    "dark": Theme(name="dark", roles=_dark_roles(), dark=True, colorscale="Cividis"),
    "high_contrast": Theme(
        name="high_contrast",
        roles=_high_contrast_roles(),
        line_width=3.4,
        marker_size=10,
        font_size=15,
    ),
    "classroom": Theme(
        name="classroom",
        font_size=18,
        font_scale=1.25,
        line_width=3.6,
        marker_size=11,
        animation_emphasis=1.4,
    ),
    "publication": Theme(
        name="publication",
        roles=_publication_roles(),
        font_family="Source Serif Pro, Georgia, Times New Roman, serif",
        font_size=12,
        line_width=1.8,
        marker_size=6,
        grid=False,
        colorscale="Greys",
    ),
    "colorblind": Theme(name="colorblind", roles=_colorblind_roles()),
}
THEMES["custom"] = Theme(name="custom")


def list_themes() -> list[str]:
    return list(THEMES)


def register_theme(theme: Theme) -> Theme:
    """Register a user-defined theme (serializable, blueprint 37.3)."""
    THEMES[theme.name] = theme
    return theme


def get_theme(name: str | Theme | None = None) -> Theme:
    if isinstance(name, Theme):
        return name
    if name is None:
        from ...config import get_config

        name = get_config().theme
    return THEMES.get(name, THEMES["light"])


def color(role: str, theme: str | Theme | None = None) -> str:
    return get_theme(theme).color(role)


def encoding(role: str, theme: str | Theme | None = None) -> dict[str, str]:
    return get_theme(theme).encoding(role)
