"""Browsing the catalog (blueprint sections 36.2 and 16).

Two rules shape this component:

* the catalog shows what exists **and** what is only planned, with the planned
  entries visibly marked and not clickable. Hiding them would flatter the
  package; pretending they open would mislead the reader;
* search works across all three languages at once, including Arabic written
  without diacritics, because a reader should not have to guess which language
  a concept was indexed in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from ...catalog.builtin import CATALOG
from ...core.concepts import Domain

__all__ = ["render_catalog", "render_concept_card", "filter_catalog"]


def filter_catalog(
    query: str = "",
    *,
    domain: str | None = None,
    level: str | None = None,
    include_planned: bool = True,
) -> list[Any]:
    """The catalog narrowed by search text, domain and level."""
    from ...catalog.search import search_concepts

    if query.strip():
        # search_concepts returns ranked summaries; map them back to the specs so
        # the cards below always work from one representation.
        by_id = {spec.id: spec for spec in CATALOG}
        specs = [
            by_id[hit["id"]]
            for hit in search_concepts(query, limit=len(CATALOG))
            if hit["id"] in by_id
        ]
    else:
        specs = list(CATALOG)
    if domain and domain != "all":
        specs = [s for s in specs if s.domain.value == domain]
    if level and level != "all":
        specs = [s for s in specs if not s.levels or level in {
            getattr(item, "value", item) for item in s.levels
        }]
    if not include_planned:
        specs = [s for s in specs if s.status.is_implemented]
    return specs


def render_concept_card(spec: Any, session: Any, on_open: Callable[[str], None]) -> None:
    """One catalog entry. A planned entry says so and refuses to open."""
    tr = session.translator()
    implemented = spec.status.is_implemented
    title = tr.t(spec.title_key, spec.id.split(".")[-1].replace("_", " ").title())
    summary = tr.t(spec.summary_key, "")

    with ui.card().classes("w-full vm-surface").props("flat bordered"):
        with ui.row().classes("w-full items-start justify-between gap-2 no-wrap"):
            with ui.column().classes("gap-1 grow min-w-0"):
                ui.label(title).classes("font-semibold")
                if summary:
                    ui.label(summary).classes("text-sm vm-muted")
                with ui.row().classes("gap-2 items-center flex-wrap"):
                    ui.label(
                        tr.t(spec.domain.label_key, spec.domain.value.replace("_", " "))
                    ).classes("text-xs vm-muted")
                    ui.label(spec.id).classes("text-xs vm-muted vm-ltr")
            if implemented:
                ui.button(
                    tr.t("actions.open", "Open"),
                    on_click=lambda _=None, cid=spec.id: on_open(cid),
                ).props("flat dense no-caps").classes("vm-focusable")
            else:
                # Marked, not hidden, and deliberately not clickable.
                ui.html(
                    f'<span class="vm-badge" data-proof="false">'
                    f'{tr.t("ui.planned", "Planned")}</span>'
                )
        if not implemented:
            ui.label(tr.t(
                "errors.concept_planned_short",
                "Catalogued but not built yet - it is listed so the catalogue does not "
                "overstate what exists.",
            )).classes("text-xs vm-muted")


def render_catalog(session: Any, on_open: Callable[[str], None]) -> None:
    """The searchable, filterable concept list."""
    tr = session.translator()
    state = {"query": "", "domain": "all", "level": "all", "planned": True}

    results = ui.column().classes("w-full gap-2")

    def refresh() -> None:
        specs = filter_catalog(
            state["query"],
            domain=state["domain"],
            level=state["level"],
            include_planned=state["planned"],
        )
        results.clear()
        with results:
            implemented = sum(1 for s in specs if s.status.is_implemented)
            ui.label(tr.t(
                "ui.catalog_count",
                "{implemented} built, {planned} planned",
                implemented=implemented, planned=len(specs) - implemented,
            )).classes("text-xs vm-muted")
            if not specs:
                ui.label(tr.t("ui.no_results", "Nothing matched.")).classes("vm-muted")
            for spec in specs:
                render_concept_card(spec, session, on_open)

    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
        ui.input(
            placeholder=tr.t("ui.search_placeholder", "Search in any language"),
            on_change=lambda e: (state.update(query=e.value or ""), refresh()),
        ).props("dense outlined clearable").classes("grow vm-focusable")
        domains = {"all": tr.t("ui.all_domains", "All domains")}
        domains.update({
            d.value: tr.t(d.label_key, d.value.replace("_", " ").title()) for d in Domain
        })
        ui.select(
            domains, value="all",
            on_change=lambda e: (state.update(domain=e.value), refresh()),
        ).props("dense outlined").classes("w-56 vm-focusable")
        levels = {"all": tr.t("ui.all_levels", "All levels")}
        levels.update({
            name: tr.t(f"levels.{name}", name.title())
            for name in ("beginner", "intermediate", "advanced", "phd")
        })
        ui.select(
            levels, value="all",
            on_change=lambda e: (state.update(level=e.value), refresh()),
        ).props("dense outlined").classes("w-44 vm-focusable")
        ui.switch(
            tr.t("ui.show_planned", "Show planned"), value=True,
            on_change=lambda e: (state.update(planned=e.value), refresh()),
        ).classes("vm-focusable")

    refresh()
