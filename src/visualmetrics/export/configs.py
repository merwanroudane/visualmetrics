"""Save and reload lab configurations (blueprint section 66.4).

A configuration is a :class:`~visualmetrics.core.state.LabState`: concept,
scenario, parameters, seed and session settings. Saving one and loading it back
must reproduce the same figures and the same numbers - that is what makes a
result shareable, and it is checked in the test suite.

JSON always works. YAML is used when the file has a ``.yaml`` or ``.yml``
suffix and ``pyyaml`` is installed; otherwise the error says so plainly rather
than writing a JSON file under a YAML name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.exceptions import ExportError, MissingDependencyError
from ..core.state import LabState
from ..version import __version__

__all__ = [
    "save_config",
    "load_config",
    "config_to_dict",
    "config_from_dict",
    "save_configs",
    "load_configs",
]

YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def _yaml():
    try:
        import yaml
    except Exception as exc:
        raise MissingDependencyError("pyyaml", "export", feature="YAML configurations") from exc
    return yaml


def config_to_dict(state: LabState) -> dict[str, Any]:
    """Serialise a state, stamping the version that produced it."""
    data = state.to_dict()
    data.setdefault("visualmetrics_version", __version__)
    return data


def config_from_dict(data: dict[str, Any]) -> LabState:
    """Rebuild a state, warning when it came from a different version.

    An older file still loads: refusing would make saved work disposable. The
    version stamp travels with the state so the GUI can say where it came from.
    """
    if not isinstance(data, dict) or "concept_id" not in data:
        raise ExportError(
            "this file is not a VisualMetrics configuration: no concept_id field",
            key="errors.export",
        )
    return LabState.from_dict(data)


def save_config(state: LabState, path: str | Path) -> Path:
    """Write one configuration to ``path`` (JSON or YAML)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config_to_dict(state)
    if path.suffix.lower() in YAML_SUFFIXES:
        path.write_text(
            _yaml().safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_config(path: str | Path) -> LabState:
    """Read one configuration back."""
    path = Path(path)
    if not path.exists():
        raise ExportError(f"no configuration file at {path}", key="errors.export")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in YAML_SUFFIXES:
        data = _yaml().safe_load(text)
    else:
        data = json.loads(text)
    return config_from_dict(data)


def save_configs(states: list[LabState], path: str | Path, *, name: str = "") -> Path:
    """Write a collection of configurations - a lesson plan, or a comparison set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "visualmetrics_version": __version__,
        "name": name,
        "configs": [config_to_dict(s) for s in states],
    }
    if path.suffix.lower() in YAML_SUFFIXES:
        path.write_text(
            _yaml().safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_configs(path: str | Path) -> list[LabState]:
    """Read a collection of configurations."""
    path = Path(path)
    if not path.exists():
        raise ExportError(f"no configuration file at {path}", key="errors.export")
    text = path.read_text(encoding="utf-8")
    data = _yaml().safe_load(text) if path.suffix.lower() in YAML_SUFFIXES else json.loads(text)
    if isinstance(data, dict) and "configs" in data:
        return [config_from_dict(item) for item in data["configs"]]
    if isinstance(data, list):
        return [config_from_dict(item) for item in data]
    return [config_from_dict(data)]
