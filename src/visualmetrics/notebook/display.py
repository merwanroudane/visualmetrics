"""Showing a lab result inside Jupyter (blueprint section 65).

The notebook gets the same discipline as the GUI: a figure is displayed with
its evidence badge, its assumptions and its warnings, and an animation is
displayed with its frame commentary. ``vm.lab(...)`` therefore renders as a
small report rather than as a bare Plotly figure - which is what makes a
notebook cell honest when it is pasted into a paper draft.

Nothing here imports ipywidgets: plain HTML rendering works in every notebook
front end, including the ones that do not run widgets (nbviewer, GitHub, a PDF
export). The interactive controls live in :mod:`visualmetrics.notebook.widgets`
and are optional.
"""

from __future__ import annotations

import html as _html
from typing import Any

from ..core.evidence import badge_for
from ..core.state import LabResult

__all__ = [
    "display_result",
    "result_html",
    "display_animation",
    "animation_html",
    "install_formatters",
    "in_notebook",
]


def in_notebook() -> bool:
    """Whether we are running inside an IPython kernel with a display hook."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, "kernel")


def _esc(value: Any) -> str:
    return _html.escape(str(value if value is not None else ""), quote=True)


def _style() -> str:
    # Scoped to .vm-nb so a notebook's own theme is not disturbed, and written
    # with colours that stay readable on both light and dark notebook themes.
    return """
<style>
.vm-nb { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         line-height: 1.5; margin: .5rem 0 1rem; }
.vm-nb .vm-nb-badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px;
         border: 1px solid currentColor; font-size: .78rem; font-weight: 600; }
.vm-nb .vm-nb-caveat { opacity: .75; font-size: .82rem; margin: .3rem 0 .6rem; }
.vm-nb .vm-nb-warn { color: #a34a00; }
.vm-nb table { border-collapse: collapse; margin: .4rem 0; }
.vm-nb th, .vm-nb td { text-align: left; padding: .2rem .6rem; border-bottom: 1px solid
         rgba(128,128,128,.35); font-size: .85rem; }
.vm-nb td.num { text-align: right; font-variant-numeric: tabular-nums; }
.vm-nb dt { opacity: .7; font-size: .75rem; text-transform: uppercase;
         letter-spacing: .03em; margin-top: .45rem; }
.vm-nb dd { margin: .1rem 0 0; font-size: .87rem; }
.vm-nb details { margin: .3rem 0; }
</style>
"""


def result_html(result: LabResult, *, language: str | None = None,
                include_style: bool = True) -> str:
    """The scientific layer of a result as HTML - badge, assumptions, warnings, numbers."""
    from ..i18n.translator import get_translator

    tr = get_translator(language)
    parts: list[str] = [_style() if include_style else "", '<div class="vm-nb">']

    evidence = result.evidence
    if evidence is not None:
        badge = badge_for(evidence)
        label = tr.t(badge.label_key, evidence.value.replace("_", " ").title())
        caveat = tr.t(badge.caveat_key, "")
        parts.append(
            f'<div><span class="vm-nb-badge">{badge.icon} {_esc(label)}</span></div>'
        )
        if caveat:
            parts.append(f'<div class="vm-nb-caveat">{_esc(caveat)}</div>')

    if result.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in result.warnings)
        parts.append(
            f'<div class="vm-nb-warn"><strong>'
            f'{_esc(tr.t("ui.warnings", "Warnings"))}</strong><ul>{items}</ul></div>'
        )

    broken = [a for a in result.assumptions if not a.holds]
    if broken:
        rows = "".join(
            f"<tr><td>✗ {_esc(a.label)}</td><td>{_esc(getattr(a, 'consequence', '') or '')}"
            f"</td></tr>"
            for a in broken
        )
        parts.append(
            f'<div class="vm-nb-warn"><strong>'
            f'{_esc(tr.t("assumptions.violated", "Violated"))}</strong>'
            f"<table>{rows}</table></div>"
        )

    if result.metrics:
        rows = "".join(
            f"<tr><th>{_esc(m.label)}</th>"
            f'<td class="num">{_esc(_fmt(m.value))}</td></tr>'
            for m in result.metrics
        )
        parts.append(
            f"<details open><summary>{_esc(tr.t('ui.metrics', 'Numbers'))}</summary>"
            f"<table>{rows}</table></details>"
        )

    if result.assumptions and not broken:
        ok = ", ".join(_esc(a.label) for a in result.assumptions)
        parts.append(
            f'<div class="vm-nb-caveat">'
            f'{_esc(tr.t("ui.assumptions_hold", "Assumptions in force"))}: {ok}</div>'
        )

    parts.append("</div>")
    return "".join(parts)


def animation_html(animation: Any, *, language: str | None = None) -> str:
    """The frame commentary as a table.

    A notebook cannot always play an animation - and an export to PDF never
    can - so the commentary is rendered as text. Dropping it would leave a
    silent picture, which is the one thing this package refuses to produce.
    """
    from ..i18n.translator import get_translator

    tr = get_translator(language)
    if not animation.steps:
        return (
            '<div class="vm-nb vm-nb-warn">'
            + _esc(tr.t(
                "ui.animation_unexplained",
                "This animation carries no per-frame explanation.",
            ))
            + "</div>"
        )

    header = "".join(
        f"<th>{_esc(tr.t(key, default))}</th>"
        for key, default in (
            ("ui.frame", "#"),
            ("ui.what_you_see", "What you see"),
            ("ui.what_changed", "What changed"),
            ("ui.why", "Why"),
            ("ui.how_to_read", "How to read it"),
        )
    )
    rows = "".join(
        f'<tr><td class="num">{step.frame}</td><td>{_esc(step.what_you_see)}</td>'
        f"<td>{_esc(step.what_changed)}</td><td>{_esc(step.why)}</td>"
        f"<td>{_esc(step.interpretation)}</td></tr>"
        for step in animation.steps
    )
    summary = (
        f"<p><strong>{_esc(animation.summary)}</strong></p>" if animation.summary else ""
    )
    purpose = f"<p>{_esc(animation.purpose)}</p>" if animation.purpose else ""
    return (
        f'{_style()}<div class="vm-nb">{purpose}'
        f"<details><summary>{_esc(tr.t('ui.frame_commentary', 'Frame commentary'))}"
        f"</summary><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>"
        f"</details>{summary}</div>"
    )


def display_result(result: LabResult, *, language: str | None = None,
                   figures: bool = True) -> None:
    """Display a whole result in a notebook cell."""
    from IPython.display import HTML, display

    display(HTML(result_html(result, language=language)))
    if figures:
        for panel in result.panels:
            if panel.title:
                display(HTML(f'<div class="vm-nb"><strong>{_esc(panel.title)}</strong></div>'))
            if panel.figure is not None:
                display(panel.figure)
            caption = panel.caption or panel.description
            if caption:
                display(HTML(f'<div class="vm-nb vm-nb-caveat">{_esc(caption)}</div>'))
    for animation in result.animations:
        display(HTML(animation_html(animation, language=language)))
        if figures and animation.figure is not None:
            display(animation.figure)


def display_animation(animation: Any, *, language: str | None = None) -> None:
    """Display one animation with its commentary."""
    from IPython.display import HTML, display

    display(HTML(animation_html(animation, language=language)))
    if animation.figure is not None:
        display(animation.figure)


def install_formatters() -> bool:
    """Teach IPython to render a :class:`LabResult` as its report.

    Called automatically by :func:`visualmetrics.notebook.setup`. Returns
    ``False`` when there is no IPython to teach, so importing the package in a
    plain script stays silent.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    formatter = shell.display_formatter.formatters["text/html"]
    formatter.for_type(LabResult, lambda result: result_html(result))
    return True


def _fmt(value: Any) -> str:
    from ..gui.state.viewmodel import format_value

    return format_value(value)
