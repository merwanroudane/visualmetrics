"""The lab page: controls on one side, evidence on the other.

Re-running is explicit rather than automatic on every slider tick, because
several labs run Monte Carlo studies that take seconds - silently recomputing
on each pixel of drag would make the GUI feel broken. Instead the page marks
itself stale and the run button says so.

Compare mode (blueprint section 36.7) puts two results side by side under the
same controls, which is how you show that a method breaks: the same data, one
assumption changed.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...core.exceptions import VisualMetricsError
from ..components.controls import render_control_panel
from ..components.layout import page_shell
from ..components.result import render_result
from ..state.session import Session
from ..state.viewmodel import build_lab_view

__all__ = ["lab_page"]


def lab_page(concept_id: str, session: Session, rerender: Any) -> None:
    """Render the lab for ``concept_id``."""
    tr = session.translator()
    if session.concept_id != concept_id:
        session.open(concept_id)

    try:
        lab = _load(concept_id)
    except VisualMetricsError as error:
        with page_shell(session, rerender):
            _render_unavailable(error, tr)
        return

    spec = lab.spec
    with page_shell(session, rerender):
        with ui.row().classes("w-full items-baseline gap-3 flex-wrap"):
            ui.label(tr.t(spec.title_key, concept_id)).classes("text-2xl font-semibold")
            ui.label(concept_id).classes("text-xs vm-muted vm-ltr")
        summary = tr.t(spec.summary_key, "")
        if summary:
            ui.label(summary).classes("vm-muted")

        state: dict[str, Any] = {"stale": False, "view": None, "compare": None}

        # The two columns are created before the callbacks so that everything
        # below draws into the right one. NiceGUI places an element where it is
        # created, so an output container built outside this row would stay
        # outside it.
        with ui.row().classes("vm-lab-row w-full items-start gap-4 no-wrap"):
            # The panel scrolls inside itself. A lab with many scenarios and
            # controls is far taller than the result, and letting it stretch the
            # page put the Run button below the fold - so pressing Run appeared
            # to do nothing, with the freshly drawn result sitting above the
            # viewport.
            controls = ui.column().classes(
                "vm-controls vm-surface p-3 gap-2 vm-hide-in-presentation"
            ).style("flex: 0 0 20rem; max-width: 20rem;")
            output = ui.column().classes("grow min-w-0 gap-3").props('id="vm-output"')

        def run(*, reveal: bool = False) -> None:
            state["stale"] = False
            output.clear()
            with output:
                ui.spinner(size="lg").classes("self-center")
            try:
                result = lab.run(session.lab_state())
                view = build_lab_view(
                    result, translator=session.translator(), precision=session.precision
                )
            except Exception as exc:
                output.clear()
                with output:
                    _render_failure(exc, tr)
                return
            state["view"] = view
            output.clear()
            with output:
                if session.compare_with:
                    _render_comparison(session, lab, view, tr)
                else:
                    render_result(
                        view,
                        translator=session.translator(),
                        reduced_motion=session.reduced_motion,
                    )
                _render_footer(session, spec, view, tr, run)
            if reveal:
                _scroll_to_result()

        def mark_stale() -> None:
            state["stale"] = True
            run_button.props("color=primary")
            run_button.set_text(tr.t("actions.rerun", "Run with these settings"))

        def on_parameter(name: str, value: Any) -> None:
            session.set_parameter(name, value)
            mark_stale()

        def on_scenario(scenario: str | None) -> None:
            session.choose_scenario(scenario)
            rerender()

        def on_reset() -> None:
            session.reset_parameters()
            rerender()

        with controls:
            with ui.column().classes("vm-controls-scroll w-full gap-2"):
                render_control_panel(spec, session, on_parameter, on_scenario, on_reset)
            with ui.column().classes("vm-controls-actions w-full gap-2"):
                run_button = ui.button(
                    tr.t("actions.run", "Run"), on_click=lambda: run(reveal=True)
                ).props("no-caps").classes("w-full vm-focusable")
                _render_compare_picker(session, spec, tr, rerender)

        run()


def _scroll_to_result() -> None:
    """Bring the result into view after an explicit run.

    Without this, a reader who scrolled down to reach the button is left
    looking at empty space beside the controls while the new result sits above
    them.
    """
    try:
        ui.run_javascript(
            "const el = document.getElementById('vm-output');"
            "if (el) el.scrollIntoView({behavior: 'auto', block: 'start'});"
        )
    except Exception:  # pragma: no cover - no client yet during the first build
        pass


def _load(concept_id: str) -> Any:
    from ...core.registry import registry

    return registry.lab(concept_id)


def _render_unavailable(error: VisualMetricsError, tr: Any) -> None:
    """A planned concept says so plainly instead of showing an empty page."""
    with ui.card().classes("w-full vm-surface").props("flat bordered"):
        ui.label(tr.t("ui.not_available", "This lab is not available")).classes(
            "text-lg font-semibold"
        )
        ui.label(error.localized(tr)).classes("vm-muted")
        ui.link(tr.t("nav.catalog", "Back to the catalogue"), "/catalog").classes(
            "vm-focusable"
        )


def _render_failure(exc: Exception, tr: Any) -> None:
    """A run that failed is reported, with the technical detail available."""
    with ui.card().classes("w-full vm-surface").props("flat bordered"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("error").classes("vm-negative")
            ui.label(tr.t("ui.run_failed", "This run did not complete")).classes(
                "font-semibold vm-negative"
            )
        message = exc.localized(tr) if isinstance(exc, VisualMetricsError) else str(exc)
        ui.label(message).classes("text-sm")
        with ui.expansion(tr.t("ui.technical_detail", "Technical detail")).classes("w-full"):
            import traceback

            ui.code("".join(traceback.format_exception(exc))).classes("w-full vm-code")


def _render_compare_picker(session: Session, spec: Any, tr: Any, rerender: Any) -> None:
    """Choose a second scenario to place beside the first."""
    if not spec.scenarios:
        return
    ui.separator().classes("my-2")
    options = {"": tr.t("ui.no_comparison", "No comparison")}
    options.update({s.id: tr.t(s.translation_key, s.id.replace("_", " "))
                    for s in spec.scenarios})

    def pick(event: Any) -> None:
        session.compare_with = getattr(event, "value", "") or None
        rerender()

    ui.select(
        options, value=session.compare_with or "", label=tr.t("tabs.compare", "Compare"),
        on_change=pick,
    ).props("dense outlined").classes("w-full vm-focusable")


def _render_comparison(session: Session, lab: Any, view: Any, tr: Any) -> None:
    """Two results, same controls, side by side."""
    other_state = session.lab_state()
    other_state.scenario = session.compare_with
    other_state.parameters = {}
    try:
        other_view = build_lab_view(
            lab.run(other_state), translator=session.translator(),
            precision=session.precision,
        )
    except Exception as exc:
        _render_failure(exc, tr)
        return

    with ui.row().classes("w-full items-start gap-4 no-wrap"):
        for label, current in (
            (session.scenario or tr.t("ui.current", "Current"), view),
            (session.compare_with, other_view),
        ):
            with ui.column().classes("grow min-w-0 gap-2"):
                ui.label(str(label)).classes("font-semibold")
                render_result(
                    current, translator=session.translator(),
                    reduced_motion=session.reduced_motion,
                )


def _render_footer(session: Session, spec: Any, view: Any, tr: Any, run: Any) -> None:
    """Reproducibility, sharing and the teaching material attached to a concept."""
    from ...education.misconceptions import for_concept

    ui.separator().classes("my-2")
    with ui.row().classes("w-full items-center gap-2 flex-wrap vm-hide-in-presentation"):
        ui.label(f"seed = {session.seed}").classes("text-xs vm-muted vm-numeric")

        def new_seed() -> None:
            import random

            session.seed = random.randrange(1, 2**31)
            run(reveal=True)

        ui.button(
            tr.t("actions.new_seed", "New seed"), icon="casino", on_click=new_seed
        ).props("flat dense no-caps").classes("vm-focusable")
        ui.button(
            tr.t("actions.share", "Copy link"), icon="link",
            on_click=lambda: _copy_link(session, tr),
        ).props("flat dense no-caps").classes("vm-focusable")
        ui.button(
            tr.t("actions.export", "Export"), icon="download",
            on_click=lambda: _export_dialog(session, view, tr),
        ).props("flat dense no-caps").classes("vm-focusable")

    from ...education.objectives import for_concept as objectives_for

    goals = objectives_for(spec.id)
    if goals:
        with ui.expansion(
            tr.t("ui.objectives", "What to take away"), value=False
        ).classes("w-full vm-surface"):
            for goal in goals:
                with ui.row().classes("w-full items-baseline gap-2"):
                    ui.icon("radio_button_unchecked").classes("text-xs vm-muted")
                    with ui.column().classes("gap-0 grow min-w-0"):
                        ui.label(goal.statement).classes("text-sm")
                        if goal.check:
                            ui.label(goal.check).classes("text-xs vm-muted")

    misconceptions = for_concept(spec.id)
    if misconceptions:
        with ui.expansion(
            tr.t("tabs.misconceptions", "Common mistakes"), value=False
        ).classes("w-full vm-surface"):
            for entry in misconceptions:
                with ui.card().classes("w-full vm-surface my-1").props("flat bordered"):
                    ui.label(f'"{entry.claim}"').classes("text-sm vm-warning")
                    ui.label(entry.why_wrong).classes("text-sm")
                    ui.label(entry.correct).classes("text-sm vm-positive")
                    if entry.demo:
                        ui.link(
                            tr.t("ui.see_it", "See it: {concept} / {scenario}",
                                 concept=entry.demo[0], scenario=entry.demo[1]),
                            f"/lab/{entry.demo[0]}?scenario={entry.demo[1]}",
                        ).classes("text-xs vm-ltr vm-focusable")

    from ...education.quiz import quiz_for

    if quiz_for(spec.id):
        with ui.expansion(tr.t("tabs.quiz", "Self-check"), value=False).classes(
            "w-full vm-surface"
        ):
            from ..components.quiz import render_quiz

            render_quiz(spec.id, session)


def _copy_link(session: Session, tr: Any) -> None:
    from urllib.parse import urlencode

    from ..state.session import session_to_query

    query = urlencode(session_to_query(session))
    url = f"/lab/{session.concept_id}?{query}"
    ui.clipboard.write(url) if hasattr(ui, "clipboard") else None
    ui.notify(tr.t("ui.link_copied", "Link copied: {url}", url=url))


def _export_dialog(session: Session, view: Any, tr: Any) -> None:
    """Export the current result, explanation layer included."""
    with ui.dialog() as dialog, ui.card().classes("vm-surface"):
        ui.label(tr.t("actions.export", "Export")).classes("text-lg font-semibold")
        ui.label(tr.t(
            "export.note",
            "Every export carries the evidence badge, the assumptions and the frame "
            "commentary with the figures.",
        )).classes("text-sm vm-muted")
        target = ui.input(
            label=tr.t("export.path", "File path"),
            value=f"{session.concept_id.replace('.', '-')}-report.html",
        ).props("dense outlined").classes("w-full")

        def write() -> None:
            from pathlib import Path

            from ...core.registry import registry
            from ...export.html import export_html_report

            try:
                result = registry.lab(session.concept_id).run(session.lab_state())
                path = export_html_report(
                    result, Path(target.value), language=session.language
                )
                ui.notify(tr.t("export.written", "Written to {path}", path=str(path)))
                dialog.close()
            except Exception as exc:
                ui.notify(str(exc), type="negative")

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button(tr.t("actions.close", "Close"), on_click=dialog.close).props("flat")
            ui.button(tr.t("actions.export", "Export"), on_click=write).props("no-caps")
    dialog.open()
