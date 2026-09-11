"""The page shell: header, settings and navigation (blueprint section 36.2).

Language, theme, level, terminology mode, motion and presentation mode are
available on every page, because they are not preferences you set once - a
teacher switches language mid-lesson and a reader drops the level while
reading. Changing any of them re-renders the current page in place rather than
sending the visitor back to the start.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from nicegui import ui

from ...i18n.translator import LANGUAGE_NAMES, LANGUAGES, TERMINOLOGY_MODES
from ..accessibility.a11y import keyboard_shortcuts
from ..state.session import LEVELS, THEMES, Session
from ..themes.css import theme_css

__all__ = ["apply_session_style", "page_shell", "render_header", "shortcuts_dialog"]


def apply_session_style(session: Session) -> None:
    """Push this session's theme, direction and motion preference into the page.

    The stylesheet is written into one element and *replaced* on every render.
    Appending instead - which is what ``ui.add_css`` does - left the previous
    language's rules in the document, so switching from Arabic back to English
    kept the whole layout mirrored while the selector read "English".
    """
    css = theme_css(
        session.theme,
        language=session.language,
        reduced_motion=session.reduced_motion,
        presentation=session.presentation,
    )
    direction = "rtl" if session.is_rtl else "ltr"
    script = (
        "(() => {"
        "  let tag = document.getElementById('vm-theme');"
        "  if (!tag) {"
        "    tag = document.createElement('style');"
        "    tag.id = 'vm-theme';"
        "    document.head.appendChild(tag);"
        "  }"
        f" tag.textContent = {json.dumps(css)};"
        f" document.documentElement.setAttribute('dir', {json.dumps(direction)});"
        f" document.documentElement.setAttribute('lang', {json.dumps(session.language)});"
        "})();"
    )

    if not getattr(session, "_style_element_added", False):
        # First build of this page: the client is not connected yet, so the
        # stylesheet goes into the document head directly.
        ui.add_head_html(f'<style id="vm-theme">{css}</style>')
        ui.add_head_html(f"<script>{script}</script>")
        session._style_element_added = True
    else:
        try:
            ui.run_javascript(script)
        except Exception:
            pass

    ui.dark_mode(value=_is_dark(session.theme))


def _is_dark(theme: str) -> bool:
    from ..themes.css import is_dark

    return is_dark(theme)


def render_header(session: Session, on_change: Callable[[], None]) -> None:
    """The application header with every live switch."""
    tr = session.translator()

    def rerender(setter: Callable[[Any], Any]) -> Callable[[Any], None]:
        def handler(event: Any) -> None:
            setter(getattr(event, "value", event))
            on_change()

        return handler

    # A sticky row rather than ui.header(): the header lives inside the
    # container that gets cleared on every re-render, and a NiceGUI header must
    # be a direct child of the page.
    with ui.row().classes(
        "vm-header items-center justify-between px-3 py-2 vm-surface w-full no-wrap"
    ):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.link(
                tr.t("app.name", "VisualMetrics"), "/"
            ).classes("text-lg font-semibold no-underline vm-focusable")
            ui.label(tr.t("app.tagline", "an interactive visual laboratory")).classes(
                "text-xs vm-muted vm-hide-in-presentation"
            )

        with ui.row().classes("items-center gap-2 no-wrap vm-hide-in-presentation"):
            ui.select(
                {code: LANGUAGE_NAMES[code]["native"] for code in LANGUAGES},
                value=session.language,
                on_change=rerender(session.set_language),
            ).props("dense outlined").classes("w-28 vm-focusable").tooltip(
                tr.t("language.label", "Language")
            )
            ui.select(
                {name: tr.t(f"themes.{name}", name.replace("_", " ").title())
                 for name in THEMES},
                value=session.theme,
                on_change=rerender(session.set_theme),
            ).props("dense outlined").classes("w-36 vm-focusable").tooltip(
                tr.t("themes.label", "Theme")
            )
            ui.select(
                {name: tr.t(f"levels.{name}", name.title()) for name in LEVELS},
                value=session.level,
                on_change=rerender(session.set_level),
            ).props("dense outlined").classes("w-36 vm-focusable").tooltip(
                tr.t("levels.label", "Level")
            )
            ui.select(
                {mode: tr.t(f"terminology.{mode}", mode.replace("_", " ").title())
                 for mode in TERMINOLOGY_MODES},
                value=session.terminology,
                on_change=rerender(session.set_terminology),
            ).props("dense outlined").classes("w-40 vm-focusable").tooltip(
                tr.t("terminology.label", "Technical terms")
            )

            def toggle_motion() -> None:
                session.reduced_motion = not session.reduced_motion
                on_change()

            ui.button(
                icon="motion_photos_off" if session.reduced_motion else "motion_photos_on",
                on_click=toggle_motion,
            ).props("flat dense").classes("vm-focusable").tooltip(
                tr.t("actions.reduced_motion", "Reduced motion")
            )

            def toggle_presentation() -> None:
                session.presentation = not session.presentation
                on_change()

            ui.button(icon="slideshow", on_click=toggle_presentation).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(
                tr.t("actions.presentation", "Presentation mode")
            )
            ui.button(
                icon="keyboard", on_click=lambda: shortcuts_dialog(session)
            ).props("flat dense").classes("vm-focusable").tooltip(
                tr.t("actions.shortcuts", "Keyboard shortcuts")
            )

        if session.presentation:
            # Presentation mode hides the switches, so leave one way out.
            def leave() -> None:
                session.presentation = False
                on_change()

            ui.button(icon="close", on_click=leave).props("flat dense").classes(
                "vm-focusable"
            ).tooltip(tr.t("actions.exit_presentation", "Leave presentation mode"))


def shortcuts_dialog(session: Session) -> None:
    """The keyboard map, read from one source so it cannot drift."""
    tr = session.translator()
    with ui.dialog() as dialog, ui.card().classes("vm-surface"):
        ui.label(tr.t("actions.shortcuts", "Keyboard shortcuts")).classes(
            "text-lg font-semibold"
        )
        for keys, key in keyboard_shortcuts().items():
            with ui.row().classes("w-full justify-between gap-6"):
                ui.label(keys).classes("vm-numeric font-medium")
                ui.label(tr.t(key, key.split(".")[-1].replace("_", " "))).classes(
                    "vm-muted"
                )
        ui.button(tr.t("actions.close", "Close"), on_click=dialog.close).props(
            "flat"
        ).classes("self-end vm-focusable")
    dialog.open()


def page_shell(session: Session, on_change: Callable[[], None]):
    """Apply the styling and draw the header; returns the content container."""
    apply_session_style(session)
    render_header(session, on_change)
    return ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4")
