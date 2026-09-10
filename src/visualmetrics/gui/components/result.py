"""Rendering a :class:`LabView` (blueprint sections 36.3 and 36.8).

The rule this module exists to enforce: a figure never appears on its own. It
appears with its evidence badge, with the assumptions that were in force, with
any warning the lab raised, and with the numbers behind it. Stripping that
context is precisely the failure this package was built against, so the badge
and the warnings are drawn before the figure, not after it.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ..accessibility.a11y import aria_for_assumption, figure_description

__all__ = [
    "render_badge",
    "render_warnings",
    "render_metrics",
    "render_assumptions",
    "render_panel",
    "render_tabs",
    "render_result",
]


def render_badge(badge: Any, *, translator: Any = None) -> None:
    """The evidence badge and its caveat, always together."""
    with ui.row().classes("items-center gap-2 flex-wrap"):
        ui.html(
            f'<span class="vm-badge" data-proof="{str(badge.is_proof).lower()}" '
            f'title="{_attr(badge.caveat)}">{badge.icon} {_attr(badge.label)}</span>'
        )
        if badge.caveat:
            ui.label(badge.caveat).classes("text-xs vm-muted")


def render_warnings(warnings: list[str], *, translator: Any = None) -> None:
    if not warnings:
        return
    title = translator.t("ui.warnings", "Warnings") if translator else "Warnings"
    with ui.card().classes("w-full vm-surface").props("flat bordered"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("warning").classes("vm-warning")
            ui.label(title).classes("font-semibold vm-warning")
        for message in warnings:
            ui.label(message).classes("text-sm").props('role=alert')


def render_metrics(metrics: list[Any], *, translator: Any = None) -> None:
    if not metrics:
        return
    rows = [
        {
            "label": m.label,
            "value": m.value,
            "reference": m.reference,
            "note": m.note,
        }
        for m in metrics
    ]
    columns = [
        {"name": "label", "label": _t(translator, "ui.quantity", "Quantity"),
         "field": "label", "align": "left"},
        {"name": "value", "label": _t(translator, "ui.value", "Value"),
         "field": "value", "align": "right", "classes": "vm-numeric"},
    ]
    if any(r["reference"] for r in rows):
        columns.append({
            "name": "reference", "label": _t(translator, "ui.expected", "Expected"),
            "field": "reference", "align": "right", "classes": "vm-numeric",
        })
    if any(r["note"] for r in rows):
        columns.append({"name": "note", "label": _t(translator, "ui.note", "Note"),
                        "field": "note", "align": "left"})
    ui.table(rows=rows, columns=columns, row_key="label").classes("w-full").props(
        "flat dense bordered"
    )


def render_assumptions(assumptions: list[Any], *, translator: Any = None) -> None:
    """Assumptions with a status that is readable without colour."""
    if not assumptions:
        return
    for assumption in assumptions:
        attributes = aria_for_assumption(assumption)
        icon = "check_circle" if assumption.holds else "cancel"
        colour = "vm-positive" if assumption.holds else "vm-negative"
        word = _t(translator, "assumptions.holds" if assumption.holds else "assumptions.violated",
                  "holds" if assumption.holds else "violated")
        with ui.card().classes("w-full vm-surface my-1").props(
            f'flat bordered role=status aria-label="{_attr(attributes["aria-label"])}"'
        ):
            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon(icon).classes(colour)
                ui.label(assumption.label).classes("font-medium grow")
                # The word repeats the icon on purpose: status must not be
                # carried by colour or shape alone.
                ui.label(word).classes(f"text-xs {colour}")
            if assumption.detail:
                ui.label(assumption.detail).classes("text-sm vm-muted")
            if not assumption.holds and assumption.consequence:
                ui.label(assumption.consequence).classes("text-sm vm-warning")


def render_panel(panel: Any, view: Any = None) -> None:
    """One figure, with its caption and a text alternative."""
    description = figure_description(panel, view)
    with ui.column().classes("w-full gap-1"):
        title = getattr(panel, "title", None)
        if title:
            ui.label(title).classes("font-medium")
        figure = getattr(panel, "figure", None)
        if figure is None:
            ui.label("—").classes("vm-muted")
        else:
            container = ui.plotly(figure).classes("w-full")
            container.props(f'aria-label="{_attr(description)}" role=img')
        caption = getattr(panel, "caption", None) or getattr(panel, "description", None)
        if caption:
            ui.label(caption).classes("text-sm vm-muted")


def render_tabs(view: Any, *, translator: Any = None, active: str | None = None,
                reduced_motion: bool = False) -> Any:
    """The knowledge tabs for one result."""
    from .animation import render_animation

    if not view.tabs:
        ui.label(_t(translator, "ui.no_content", "This run produced no content.")).classes(
            "vm-muted"
        )
        return None

    with ui.tabs().classes("w-full") as tabs:
        for tab in view.tabs:
            ui.tab(tab.id, label=tab.label)

    first = active if active in view.tab_ids else view.tabs[0].id
    with ui.tab_panels(tabs, value=first).classes("w-full"):
        for tab in view.tabs:
            with ui.tab_panel(tab.id):
                for block in tab.blocks:
                    _render_block(block)
                for panel in tab.panels:
                    render_panel(panel, view)
                for animation in tab.animations:
                    render_animation(
                        animation, translator=translator, reduced_motion=reduced_motion
                    )
                if tab.id == "assumptions":
                    render_assumptions(view.assumptions, translator=translator)
                if tab.id == "code" and view.code:
                    _render_code(view, translator)
                if tab.id == "references":
                    _render_references(view, translator)
    return tabs


def _render_block(block: Any) -> None:
    kind = getattr(block, "kind", "") or ""
    title = getattr(block, "title", "")
    body = getattr(block, "body", "")
    classes = "w-full vm-surface my-1"
    if kind == "warning":
        classes += " vm-warning"
    with ui.card().classes(classes).props("flat bordered"):
        if title:
            ui.label(title).classes("font-semibold")
        # Mathematics is written as LaTeX inside the explanation blocks, so it
        # needs the latex extra to render rather than showing raw backslashes.
        ui.markdown(body, extras=["latex"] if kind == "math" else None)


def _render_code(view: Any, translator: Any) -> None:
    ui.label(_t(translator, "ui.code_parity",
                "This is the exact Python that reproduces the figures above.")).classes(
        "text-sm vm-muted"
    )
    ui.code(view.code, language="python").classes("w-full vm-code")
    if view.dgp:
        ui.label(_t(translator, "ui.dgp", "Data generating process")).classes(
            "font-semibold mt-2"
        )
        ui.code(view.dgp, language="text").classes("w-full vm-code")


def _render_references(view: Any, translator: Any) -> None:
    from ...core.registry import registry

    try:
        spec = registry.lab(view.concept_id).spec
    except Exception:  # noqa: BLE001
        return
    for reference in spec.references:
        with ui.row().classes("items-baseline gap-2"):
            ui.label(reference.citation).classes("text-sm")
            if reference.doi:
                ui.link("doi", f"https://doi.org/{reference.doi}").classes("text-xs")
            elif reference.url:
                ui.link("link", reference.url).classes("text-xs")
    if spec.proof_ids:
        ui.separator().classes("my-2")
        ui.label(_t(translator, "ui.proofs", "Proofs")).classes("font-semibold")
        for proof_id in spec.proof_ids:
            ui.link(proof_id, f"/proof/{proof_id}").classes("text-sm")


def render_result(view: Any, *, translator: Any = None, active_tab: str | None = None,
                  reduced_motion: bool = False) -> None:
    """A complete result: badge, warnings, tabs, then the numbers."""
    with ui.column().classes("w-full gap-3"):
        render_badge(view.badge, translator=translator)
        render_warnings(view.warnings, translator=translator)
        render_tabs(
            view, translator=translator, active=active_tab, reduced_motion=reduced_motion
        )
        with ui.expansion(
            _t(translator, "ui.metrics", "Numbers"), value=True
        ).classes("w-full vm-surface"):
            render_metrics(view.metrics, translator=translator)


def _t(translator: Any, key: str, default: str) -> str:
    return translator.t(key, default) if translator is not None else default


def _attr(text: Any) -> str:
    return str(text or "").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
