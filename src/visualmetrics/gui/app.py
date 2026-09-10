"""The NiceGUI application (blueprint section 36).

Routes are declared here and each one renders through a single ``page`` helper
that owns the pattern every page shares: build the session from storage and the
URL, draw the page, and re-draw it in place when a setting changes. That is why
switching language mid-lesson keeps you where you were instead of returning you
to the front page.

The GUI is an optional extra. Importing this module without NiceGUI installed
raises :class:`MissingDependencyError`, which names the package and the exact
install command, rather than a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.exceptions import MissingDependencyError

try:  # pragma: no cover - exercised by the missing-dependency path
    from nicegui import app as _nicegui_app
    from nicegui import ui
except ImportError as exc:  # pragma: no cover
    raise MissingDependencyError("nicegui", "gui", feature="the interactive GUI") from exc

from ..config import get_config
from .pages.lab import lab_page
from .pages.simple import (
    catalog_page,
    doctor_page,
    glossary_page,
    home_page,
    paths_page,
    proof_page,
    proofs_page,
)
from .state.session import Session, session_from_query

__all__ = ["build_app", "launch_gui", "render_page", "SESSION_KEY"]

SESSION_KEY = "visualmetrics.session"


def _session_for_request() -> Session:
    """This browser tab's session, restored from storage and refined by the URL.

    Storage keeps the reader's preferences between visits; the URL wins over
    them, so a shared link shows the sender's figure rather than the
    recipient's defaults.
    """
    stored: dict[str, Any] = {}
    try:
        stored = dict(_nicegui_app.storage.tab.get(SESSION_KEY) or {})
    except Exception:
        stored = {}
    session = Session.from_dict(stored) if stored else _session_from_config()
    try:
        query = dict(ui.context.client.request.query_params)
    except Exception:
        query = {}
    if query:
        session = session_from_query(query, base=session)
    return session


def _session_from_config() -> Session:
    config = get_config()
    return Session(
        language=getattr(config, "language", "en"),
        theme=getattr(config, "theme", "light"),
        level=getattr(config, "level", "intermediate"),
        terminology=getattr(config, "terminology", "translated"),
        reduced_motion=bool(getattr(config, "reduced_motion", False)),
        precision=int(getattr(config, "precision", 4)),
        seed=int(getattr(config, "seed", 42)),
    )


def _persist(session: Session) -> None:
    try:
        _nicegui_app.storage.tab[SESSION_KEY] = session.to_dict()
    except Exception:
        pass


def render_page(render: Callable[..., None], *args: Any) -> None:
    """Draw one page, and re-draw it in place whenever a setting changes.

    Every route goes through here, which is what lets the header switches
    (language, theme, level, motion) take effect without navigating away from
    the figure the reader is looking at.
    """
    session = _session_for_request()
    container = ui.column().classes("w-full items-stretch gap-0")

    def draw() -> None:
        _persist(session)
        container.clear()
        with container:
            render(*args, session, draw)

    draw()


def build_app() -> None:
    """Declare every route. Calling twice is harmless.

    The route functions are written out with their real signatures rather than
    generated, because NiceGUI derives the URL parameters from the signature -
    a ``*args`` wrapper would be read as a required query parameter.
    """
    if getattr(build_app, "_done", False):
        return

    @ui.page("/")
    def _home() -> None:
        render_page(home_page)

    @ui.page("/catalog")
    def _catalog() -> None:
        render_page(catalog_page)

    @ui.page("/proofs")
    def _proofs() -> None:
        render_page(proofs_page)

    @ui.page("/glossary")
    def _glossary() -> None:
        render_page(glossary_page)

    @ui.page("/paths")
    def _paths() -> None:
        render_page(paths_page)

    @ui.page("/doctor")
    def _doctor() -> None:
        render_page(doctor_page)

    @ui.page("/lab/{concept_id}")
    def _lab(concept_id: str) -> None:
        render_page(lab_page, concept_id)

    @ui.page("/proof/{proof_id}")
    def _proof(proof_id: str) -> None:
        render_page(proof_page, proof_id)

    build_app._done = True  # type: ignore[attr-defined]


def launch_gui(
    *,
    host: str | None = None,
    port: int | None = None,
    native: bool | None = None,
    open_browser: bool = True,
    reload: bool = False,
    show_welcome: bool = True,
) -> None:
    """Start the local server (``visualmetrics gui`` or :func:`visualmetrics.launch`)."""
    config = get_config()
    build_app()
    ui.run(
        host=host or getattr(config, "host", "127.0.0.1"),
        port=int(port or getattr(config, "port", 8080)),
        native=bool(native) if native is not None else False,
        show=open_browser,
        reload=reload,
        title="VisualMetrics",
        favicon="📐",
        dark=None,
        storage_secret=getattr(config, "storage_secret", "visualmetrics-local"),
        show_welcome_message=show_welcome,
    )
