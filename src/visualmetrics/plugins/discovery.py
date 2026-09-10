"""Plugin discovery via entry points."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

ENTRY_POINT_GROUP = "visualmetrics.concepts"


def discover_plugins() -> Iterator[tuple[str, Any]]:
    """Yield ``(name, plugin)`` for every registered third-party concept pack."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - older API
        eps = entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            yield ep.name, ep.load()
        except Exception:
            continue
