"""Third-party concept plugins.

A plugin adds concepts without touching core VisualMetrics:

.. code-block:: toml

    [project.entry-points."visualmetrics.concepts"]
    my_lab = "my_package.plugin:plugin"
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.concepts import ConceptSpec
from ..core.exceptions import PluginError
from ..version import VERSION_INFO

__all__ = ["ConceptPlugin", "validate_plugin"]


@runtime_checkable
class ConceptPlugin(Protocol):
    """What a concept pack must provide."""

    name: str
    requires_visualmetrics: str

    def concepts(self) -> list[ConceptSpec]:  # pragma: no cover - protocol
        ...


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in text.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts[:3])


def validate_plugin(plugin: Any, known_ids: set[str] | None = None) -> list[ConceptSpec]:
    """Validate a plugin before its concepts enter the registry."""
    known_ids = known_ids or set()
    name = getattr(plugin, "name", type(plugin).__name__)
    if not hasattr(plugin, "concepts"):
        raise PluginError(f"plugin {name!r} does not implement concepts()")

    requirement = getattr(plugin, "requires_visualmetrics", None)
    if requirement:
        required = _parse_version(str(requirement).lstrip(">=~^ "))
        if required > VERSION_INFO:
            raise PluginError(
                f"plugin {name!r} requires visualmetrics >= {requirement}, "
                f"this is {'.'.join(map(str, VERSION_INFO))}"
            )

    specs = list(plugin.concepts())
    seen: set[str] = set()
    for spec in specs:
        if not isinstance(spec, ConceptSpec):
            raise PluginError(f"plugin {name!r} returned a non-ConceptSpec: {spec!r}")
        if spec.id in known_ids or spec.id in seen:
            raise PluginError(f"plugin {name!r} declares a duplicate concept id {spec.id!r}")
        if not spec.title_key or not spec.summary_key:
            raise PluginError(f"concept {spec.id!r} is missing translation keys")
        for control in spec.controls:
            if control.default is None and control.kind.value != "select":
                raise PluginError(f"control {control.id!r} of {spec.id!r} has no default")
        seen.add(spec.id)
    return specs
