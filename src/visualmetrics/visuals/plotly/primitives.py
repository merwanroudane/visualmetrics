"""Reusable Plotly primitives shared by every lab.

Plotly is an *optional* dependency: importing this module is cheap, but calling
into it without plotly installed raises a friendly, localizable
:class:`MissingDependencyError` instead of an ImportError traceback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.exceptions import MissingDependencyError
from ..themes.palette import Theme, get_theme

__all__ = [
    "strip_plotly_transport",
    "require_plotly",
    "go",
    "new_figure",
    "add_curve",
    "add_points",
    "shade_between",
    "shade_tail",
    "add_vline",
    "add_hline",
    "add_annotation",
    "add_bar",
    "add_histogram",
    "add_heatmap",
    "add_arrow",
    "add_arrow3d",
    "add_legend_note",
    "finalize",
    "empty_figure",
    "SUBPLOT",
]


def require_plotly() -> Any:
    """Import plotly.graph_objects or raise a capability error."""
    try:
        import plotly.graph_objects as _go
    except ImportError as exc:  # pragma: no cover - exercised by doctor
        raise MissingDependencyError("plotly", "viz", feature="interactive figures") from exc
    return _go


class _LazyGO:
    """``go.Scatter`` style access that imports plotly on first attribute use."""

    def __getattr__(self, item: str) -> Any:
        return getattr(require_plotly(), item)


go = _LazyGO()


def SUBPLOT() -> Any:
    """Return ``plotly.subplots.make_subplots`` (lazily)."""
    require_plotly()
    from plotly.subplots import make_subplots

    return make_subplots


def new_figure(
    title: str = "",
    *,
    theme: str | Theme | None = None,
    locale: str = "en",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int | None = None,
    showlegend: bool = True,
    **layout: Any,
) -> Any:
    """Create a themed figure."""
    _go = require_plotly()
    th = get_theme(theme)
    fig = _go.Figure()
    base = th.plotly_layout(locale=locale)
    if title:
        base.setdefault("title", {})
        if isinstance(base["title"], dict):
            base["title"]["text"] = title
        else:
            base["title"] = {"text": title}
    if xaxis_title:
        base["xaxis"] = {**base.get("xaxis", {}), "title": {"text": xaxis_title}}
    if yaxis_title:
        base["yaxis"] = {**base.get("yaxis", {}), "title": {"text": yaxis_title}}
    if height:
        base["height"] = height
    base["showlegend"] = showlegend
    base.update(layout)
    fig.update_layout(**base)
    return fig


def empty_figure(message: str, *, theme: str | Theme | None = None) -> Any:
    """A placeholder figure carrying an explicit message (never a blank panel)."""
    fig = new_figure(theme=theme, showlegend=False)
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def add_curve(
    fig: Any,
    x: Any,
    y: Any,
    name: str,
    role: str = "primary",
    *,
    theme: str | Theme | None = None,
    dash: str | None = None,
    width: float | None = None,
    fill: str | None = None,
    fillcolor: str | None = None,
    opacity: float = 1.0,
    mode: str = "lines",
    hovertemplate: str | None = None,
    legendgroup: str | None = None,
    showlegend: bool = True,
    **kw: Any,
) -> Any:
    """Add a themed line, using the role's dash pattern as a redundant cue."""
    _go = require_plotly()
    th = get_theme(theme)
    enc = th.encoding(role)
    line = {
        "color": th.color(role),
        "width": width if width is not None else th.line_width,
        "dash": dash or enc["dash"],
    }
    marker = {"size": th.marker_size, "symbol": enc["symbol"], "color": th.color(role)}
    fig.add_trace(
        _go.Scatter(
            x=np.asarray(x),
            y=np.asarray(y),
            name=name,
            mode=mode,
            line=line,
            marker=marker,
            fill=fill,
            fillcolor=fillcolor,
            opacity=opacity,
            hovertemplate=hovertemplate,
            legendgroup=legendgroup,
            showlegend=showlegend,
            **kw,
        )
    )
    return fig


def add_points(
    fig: Any,
    x: Any,
    y: Any,
    name: str,
    role: str = "primary",
    *,
    theme: str | Theme | None = None,
    size: Any = None,
    symbol: str | None = None,
    opacity: float = 0.85,
    text: Any = None,
    line_width: float = 0.6,
    showlegend: bool = True,
    **kw: Any,
) -> Any:
    _go = require_plotly()
    th = get_theme(theme)
    enc = th.encoding(role)
    marker = {
        "size": size if size is not None else th.marker_size,
        "symbol": symbol or enc["symbol"],
        "color": th.color(role),
        "opacity": opacity,
        "line": {"width": line_width, "color": th.color("paper")},
    }
    fig.add_trace(
        _go.Scatter(
            x=np.asarray(x), y=np.asarray(y), name=name, mode="markers",
            marker=marker, text=text, showlegend=showlegend, **kw,
        )
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


def rgba(role: str, alpha: float, theme: str | Theme | None = None) -> str:
    return _rgba(get_theme(theme).color(role), alpha)


def shade_between(
    fig: Any,
    x: Any,
    y_low: Any,
    y_high: Any,
    name: str,
    role: str = "info",
    *,
    theme: str | Theme | None = None,
    alpha: float = 0.18,
    showlegend: bool = True,
) -> Any:
    """Shade a band (confidence band, prediction band, error region)."""
    _go = require_plotly()
    th = get_theme(theme)
    x = np.asarray(x)
    fig.add_trace(
        _go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([np.asarray(y_high), np.asarray(y_low)[::-1]]),
            fill="toself",
            fillcolor=_rgba(th.color(role), alpha),
            line={"width": 0},
            hoverinfo="skip",
            name=name,
            showlegend=showlegend,
        )
    )
    return fig


def shade_tail(
    fig: Any,
    x: Any,
    y: Any,
    mask: Any,
    name: str,
    role: str = "type_i",
    *,
    theme: str | Theme | None = None,
    alpha: float = 0.35,
    showlegend: bool = True,
) -> Any:
    """Shade the region of a density where ``mask`` is True (rejection area)."""
    _go = require_plotly()
    th = get_theme(theme)
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return fig
    xs = x[mask]
    ys = y[mask]
    fig.add_trace(
        _go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([ys, np.zeros_like(ys)]),
            fill="toself",
            fillcolor=_rgba(th.color(role), alpha),
            line={"width": 0},
            name=name,
            hoverinfo="skip",
            showlegend=showlegend,
        )
    )
    return fig


def add_vline(
    fig: Any,
    x: float,
    label: str = "",
    role: str = "baseline",
    *,
    theme: str | Theme | None = None,
    dash: str = "dash",
    width: float = 2.0,
    annotation_position: str = "top",
) -> Any:
    th = get_theme(theme)
    fig.add_vline(
        x=float(x),
        line={"color": th.color(role), "dash": dash, "width": width},
        annotation_text=label or None,
        annotation_position=annotation_position,
    )
    return fig


def add_hline(
    fig: Any,
    y: float,
    label: str = "",
    role: str = "baseline",
    *,
    theme: str | Theme | None = None,
    dash: str = "dash",
    width: float = 2.0,
    annotation_position: str = "right",
) -> Any:
    th = get_theme(theme)
    fig.add_hline(
        y=float(y),
        line={"color": th.color(role), "dash": dash, "width": width},
        annotation_text=label or None,
        annotation_position=annotation_position,
    )
    return fig


def add_annotation(
    fig: Any,
    x: float,
    y: float,
    text: str,
    *,
    role: str = "foreground",
    theme: str | Theme | None = None,
    arrow: bool = False,
    ax: int = 0,
    ay: int = -30,
    bgcolor: str | None = None,
    **kw: Any,
) -> Any:
    th = get_theme(theme)
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=arrow, ax=ax, ay=ay,
        arrowhead=2, arrowcolor=th.color(role),
        font={"color": th.color(role), "size": th.scaled_font()},
        bgcolor=bgcolor, **kw,
    )
    return fig


def add_bar(
    fig: Any,
    x: Any,
    y: Any,
    name: str,
    role: str = "primary",
    *,
    theme: str | Theme | None = None,
    opacity: float = 0.85,
    text: Any = None,
    **kw: Any,
) -> Any:
    _go = require_plotly()
    th = get_theme(theme)
    fig.add_trace(
        _go.Bar(
            x=np.asarray(x), y=np.asarray(y), name=name,
            marker={"color": th.color(role), "line": {"width": 0.5, "color": th.color("paper")}},
            opacity=opacity, text=text, **kw,
        )
    )
    return fig


def add_histogram(
    fig: Any,
    values: Any,
    name: str,
    role: str = "primary",
    *,
    theme: str | Theme | None = None,
    nbins: int = 40,
    density: bool = True,
    opacity: float = 0.6,
    **kw: Any,
) -> Any:
    _go = require_plotly()
    th = get_theme(theme)
    fig.add_trace(
        _go.Histogram(
            x=np.asarray(values),
            name=name,
            nbinsx=nbins,
            histnorm="probability density" if density else None,
            marker={"color": th.color(role), "line": {"width": 0.4, "color": th.color("paper")}},
            opacity=opacity,
            **kw,
        )
    )
    return fig


def add_heatmap(
    fig: Any,
    z: Any,
    x: Any = None,
    y: Any = None,
    *,
    theme: str | Theme | None = None,
    colorscale: str | None = None,
    name: str = "",
    diverging: bool = False,
    **kw: Any,
) -> Any:
    _go = require_plotly()
    th = get_theme(theme)
    scale = colorscale or (th.diverging_colorscale if diverging else th.colorscale)
    fig.add_trace(
        _go.Heatmap(z=np.asarray(z), x=x, y=y, colorscale=scale, name=name, **kw)
    )
    return fig


def add_arrow(
    fig: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    name: str,
    role: str = "primary",
    *,
    theme: str | Theme | None = None,
    width: float | None = None,
    dash: str | None = None,
    showlegend: bool = True,
    head_scale: float = 0.12,
) -> Any:
    """Draw a 2-D vector as a line plus an explicit arrow head."""
    _go = require_plotly()
    th = get_theme(theme)
    col = th.color(role)
    fig.add_trace(
        _go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", name=name,
            line={"color": col, "width": width or th.line_width, "dash": dash or "solid"},
            showlegend=showlegend,
        )
    )
    dx, dy = x1 - x0, y1 - y0
    norm = float(np.hypot(dx, dy))
    if norm > 1e-12:
        ux, uy = dx / norm, dy / norm
        head = head_scale * norm
        px, py = -uy, ux
        fig.add_trace(
            _go.Scatter(
                x=[x1, x1 - head * ux + 0.45 * head * px, x1 - head * ux - 0.45 * head * px, x1],
                y=[y1, y1 - head * uy + 0.45 * head * py, y1 - head * uy - 0.45 * head * py, y1],
                mode="lines",
                fill="toself",
                fillcolor=col,
                line={"color": col, "width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return fig


def add_arrow3d(
    fig: Any,
    origin: Any,
    tip: Any,
    name: str,
    role: str = "primary",
    *,
    theme: str | Theme | None = None,
    width: float | None = None,
    dash: str | None = None,
    showlegend: bool = True,
) -> Any:
    _go = require_plotly()
    th = get_theme(theme)
    o = np.asarray(origin, dtype=float)
    t = np.asarray(tip, dtype=float)
    fig.add_trace(
        _go.Scatter3d(
            x=[o[0], t[0]], y=[o[1], t[1]], z=[o[2], t[2]],
            mode="lines+markers",
            line={"color": th.color(role), "width": width or 6, "dash": dash or "solid"},
            marker={"size": [1, 5], "color": th.color(role)},
            name=name,
            showlegend=showlegend,
        )
    )
    return fig


def add_legend_note(fig: Any, text: str, *, theme: str | Theme | None = None) -> Any:
    """A caption under the plot - accessibility requirement (never colour alone)."""
    th = get_theme(theme)
    fig.add_annotation(
        text=text, xref="paper", yref="paper", x=0, y=-0.18, showarrow=False,
        xanchor="left", align="left",
        font={"size": max(10, th.scaled_font() - 2), "color": th.color("muted")},
    )
    fig.update_layout(margin={"b": 90})
    return fig


def finalize(fig: Any, *, reduced_motion: bool = False) -> Any:
    """Apply final accessibility touches."""
    if reduced_motion:
        fig.update_layout(transition={"duration": 0})
    fig.update_layout(hovermode="closest")
    return fig


def strip_plotly_transport(figure: Any) -> Any:
    """A copy of ``figure`` without Plotly's own play/pause buttons.

    Any surface that shows an animation beside its explanation layer drives the
    frames itself. Leaving Plotly's transport in place would let a reader
    advance the picture while the commentary stayed behind, which is exactly
    the desynchronisation the explanation layer exists to prevent.

    The frame slider is kept: it moves through the same frames the player
    knows about, and a scrubber with no autoplay cannot get out of step in the
    same way.
    """
    import copy

    if figure is None:
        return figure
    clone = copy.deepcopy(figure)
    try:
        # Direct assignment, not update_layout: Plotly merges list-valued
        # layout properties element by element, so passing an empty list
        # through update_layout leaves the existing buttons in place.
        clone.layout.updatemenus = []
    except Exception:
        pass
    return clone
