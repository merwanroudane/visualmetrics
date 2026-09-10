"""The pages that are mostly reading: home, catalogue, proofs, glossary, paths, doctor."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...catalog.builtin import CATALOG
from ..components.catalog import render_catalog
from ..components.layout import page_shell
from ..components.proofview import render_proof
from ..state.session import Session

__all__ = [
    "home_page",
    "catalog_page",
    "proofs_page",
    "proof_page",
    "glossary_page",
    "paths_page",
    "doctor_page",
]


def _counts() -> tuple[int, int]:
    implemented = sum(1 for s in CATALOG if s.status.is_implemented)
    return implemented, len(CATALOG)


def home_page(session: Session, rerender: Any) -> None:
    tr = session.translator()
    implemented, catalogued = _counts()
    with page_shell(session, rerender):
        ui.label(tr.t("app.name", "VisualMetrics")).classes("text-3xl font-semibold")
        ui.label(tr.t(
            "app.description",
            "An interactive visual laboratory for statistics, econometrics, causal "
            "inference and machine learning.",
        )).classes("vm-muted")

        with ui.card().classes("w-full vm-surface").props("flat bordered"):
            ui.label(tr.t("ui.honesty_title", "What the badges mean")).classes(
                "font-semibold"
            )
            ui.label(tr.t(
                "ui.honesty_body",
                "Every figure and animation says what kind of evidence it is. A "
                "simulation is labelled a simulation and never a proof, and every proof "
                "states what it does not establish.",
            )).classes("text-sm")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            for label, value in (
                (tr.t("ui.labs_built", "Labs built"), implemented),
                (tr.t("ui.catalogued", "Catalogued"), catalogued),
                (tr.t("ui.languages", "Languages"), 3),
                (tr.t("ui.proofs", "Proofs"), _proof_count()),
            ):
                with ui.card().classes("vm-surface").props("flat bordered"):
                    ui.label(str(value)).classes("text-2xl font-semibold vm-numeric")
                    ui.label(label).classes("text-xs vm-muted")

        with ui.row().classes("gap-2 flex-wrap"):
            ui.button(
                tr.t("nav.catalog", "Browse the catalogue"),
                on_click=lambda: ui.navigate.to("/catalog"),
            ).props("no-caps").classes("vm-focusable")
            ui.button(
                tr.t("nav.proofs", "Proofs"), on_click=lambda: ui.navigate.to("/proofs")
            ).props("flat no-caps").classes("vm-focusable")
            ui.button(
                tr.t("nav.paths", "Learning paths"),
                on_click=lambda: ui.navigate.to("/paths"),
            ).props("flat no-caps").classes("vm-focusable")
            ui.button(
                tr.t("nav.glossary", "Glossary"),
                on_click=lambda: ui.navigate.to("/glossary"),
            ).props("flat no-caps").classes("vm-focusable")


def _proof_count() -> int:
    from ...proofs import list_proofs

    return len(list_proofs(written_only=True))


def catalog_page(session: Session, rerender: Any) -> None:
    tr = session.translator()
    with page_shell(session, rerender):
        ui.label(tr.t("nav.catalog", "Catalogue")).classes("text-2xl font-semibold")
        render_catalog(session, lambda cid: ui.navigate.to(f"/lab/{cid}"))


def proofs_page(session: Session, rerender: Any) -> None:
    from ...proofs import list_proofs

    tr = session.translator()
    with page_shell(session, rerender):
        ui.label(tr.t("nav.proofs", "Proofs")).classes("text-2xl font-semibold")
        ui.label(tr.t(
            "proofs.ui.checks_note",
            "Each proof states its assumptions, justifies every step and says what it "
            "does not establish.",
        )).classes("text-sm vm-muted")
        for entry in list_proofs():
            with ui.card().classes("w-full vm-surface").props("flat bordered"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.column().classes("gap-0 grow min-w-0"):
                        ui.label(entry.title).classes("font-medium")
                        ui.label(entry.id).classes("text-xs vm-muted vm-ltr")
                    if entry.is_written:
                        ui.button(
                            tr.t("actions.open", "Open"),
                            on_click=lambda _=None, pid=entry.id: ui.navigate.to(
                                f"/proof/{pid}"
                            ),
                        ).props("flat dense no-caps").classes("vm-focusable")
                    else:
                        ui.html(
                            f'<span class="vm-badge" data-proof="false">'
                            f'{tr.t("ui.planned", "Planned")}</span>'
                        )
                if entry.concept_ids:
                    with ui.row().classes("gap-2 flex-wrap"):
                        for concept_id in entry.concept_ids:
                            ui.link(concept_id, f"/lab/{concept_id}").classes(
                                "text-xs vm-ltr"
                            )


def proof_page(proof_id: str, session: Session, rerender: Any) -> None:
    from ...proofs import ProofNotFoundError, proof as load_proof

    tr = session.translator()
    with page_shell(session, rerender):
        try:
            spec = load_proof(proof_id)
        except ProofNotFoundError as error:
            with ui.card().classes("w-full vm-surface").props("flat bordered"):
                ui.label(tr.t("ui.not_available", "This proof is not available")).classes(
                    "text-lg font-semibold"
                )
                ui.label(error.localized(tr)).classes("vm-muted")
                ui.link(tr.t("nav.proofs", "All proofs"), "/proofs").classes("vm-focusable")
            return
        render_proof(spec, session=session)
        if spec.concept_ids:
            ui.separator().classes("my-2")
            ui.label(tr.t("proofs.ui.related_labs", "Explore this in the lab")).classes(
                "font-semibold"
            )
            with ui.row().classes("gap-2 flex-wrap"):
                for concept_id in spec.concept_ids:
                    ui.link(concept_id, f"/lab/{concept_id}").classes("vm-ltr")


def glossary_page(session: Session, rerender: Any) -> None:
    from ...i18n.glossary import glossary_entries

    tr = session.translator()
    with page_shell(session, rerender):
        ui.label(tr.t("nav.glossary", "Glossary")).classes("text-2xl font-semibold")
        ui.label(tr.t(
            "ui.glossary_note",
            "Every term in three languages, so a reader can move between them without "
            "losing the thread.",
        )).classes("text-sm vm-muted")
        rows = [
            {
                "term": entry.terms.get(session.language, entry.terms["en"]),
                "en": entry.terms["en"],
                "ar": entry.terms["ar"],
                "fr": entry.terms["fr"],
                "definition": entry.definitions.get(
                    session.language, entry.definitions["en"]
                ),
            }
            for entry in glossary_entries().values()
        ]
        ui.table(
            rows=sorted(rows, key=lambda r: r["en"]),
            columns=[
                {"name": "en", "label": "English", "field": "en", "align": "left",
                 "sortable": True},
                {"name": "ar", "label": "العربية", "field": "ar", "align": "right"},
                {"name": "fr", "label": "Français", "field": "fr", "align": "left"},
                {"name": "definition", "label": tr.t("ui.definition", "Definition"),
                 "field": "definition", "align": "left"},
            ],
            row_key="en",
        ).classes("w-full").props("flat dense bordered")


def paths_page(session: Session, rerender: Any) -> None:
    from ...catalog.paths import list_paths

    tr = session.translator()
    with page_shell(session, rerender):
        ui.label(tr.t("nav.paths", "Learning paths")).classes("text-2xl font-semibold")
        for path in list_paths():
            with ui.card().classes("w-full vm-surface").props("flat bordered"):
                ui.label(tr.t(f"paths.{path.id}.title", path.id.replace("_", " ").title())
                         ).classes("font-semibold")
                description = tr.t(f"paths.{path.id}.summary", "")
                if description:
                    ui.label(description).classes("text-sm vm-muted")
                with ui.timeline().classes("w-full"):
                    for concept_id in path.concepts:
                        spec = next((s for s in CATALOG if s.id == concept_id), None)
                        title = tr.t(spec.title_key, concept_id) if spec else concept_id
                        built = bool(spec and spec.status.is_implemented)
                        with ui.timeline_entry(
                            title=title,
                            icon="play_circle" if built else "schedule",
                        ):
                            if built:
                                ui.link(
                                    tr.t("actions.open", "Open"), f"/lab/{concept_id}"
                                ).classes("text-sm vm-focusable")
                            else:
                                ui.label(tr.t("ui.planned", "Planned")).classes(
                                    "text-xs vm-muted"
                                )


def doctor_page(session: Session, rerender: Any) -> None:
    """What is installed, what is missing, and what that disables."""
    from ...cli import build_doctor_report

    tr = session.translator()
    report = build_doctor_report()
    with page_shell(session, rerender):
        ui.label(tr.t("nav.doctor", "Environment")).classes("text-2xl font-semibold")
        with ui.card().classes("w-full vm-surface").props("flat bordered"):
            for key in ("visualmetrics", "python", "platform", "languages"):
                if key in report:
                    with ui.row().classes("justify-between w-full"):
                        ui.label(key).classes("text-sm vm-muted")
                        ui.label(str(report[key])).classes("text-sm vm-ltr")
        grouped: dict[str, list[tuple[str, dict]]] = {}
        for name, info in (report.get("capabilities") or {}).items():
            grouped.setdefault(str(info.get("extra", "other")), []).append((name, info))
        for extra, entries in sorted(grouped.items()):
            available = sum(1 for _, info in entries if info.get("available"))
            with ui.expansion(
                f"{extra} - {available}/{len(entries)}", value=available < len(entries)
            ).classes("w-full vm-surface"):
                for name, info in entries:
                    ok = bool(info.get("available"))
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.icon("check" if ok else "close").classes(
                            "vm-positive" if ok else "vm-muted"
                        )
                        ui.label(str(name)).classes("text-sm vm-ltr")
                        ui.label(str(info.get("purpose", ""))).classes(
                            "text-xs vm-muted grow"
                        )
                        if info.get("version"):
                            ui.label(str(info["version"])).classes("text-xs vm-muted vm-ltr")
                    # An installed-but-unusable package is reported as such rather
                    # than silently counted as missing.
                    if not ok and info.get("reason"):
                        ui.label(str(info["reason"])).classes("text-xs vm-warning ms-6")
                if available < len(entries):
                    ui.label(tr.t(
                        "doctor.extra_hint",
                        'Install with: pip install "visualmetrics[{extra}]"', extra=extra,
                    )).classes("text-xs vm-muted vm-ltr")
