"""The interactive GUI (blueprint section 36).

Importing this package is cheap: NiceGUI is only imported when the application
is actually built, so ``import visualmetrics`` stays fast for API and notebook
users who never open the GUI.
"""

from __future__ import annotations

from typing import Any

__all__ = ["launch_gui", "build_app", "Session", "build_lab_view"]


def __getattr__(name: str) -> Any:
    if name in {"launch_gui", "build_app"}:
        from .app import build_app, launch_gui

        return {"launch_gui": launch_gui, "build_app": build_app}[name]
    if name == "Session":
        from .state.session import Session

        return Session
    if name == "build_lab_view":
        from .state.viewmodel import build_lab_view

        return build_lab_view
    raise AttributeError(f"module 'visualmetrics.gui' has no attribute {name!r}")
