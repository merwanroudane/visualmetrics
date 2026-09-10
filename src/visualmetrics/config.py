"""User configuration.

Priority (blueprint section 74):
    explicit function/CLI args > project config > user config > defaults
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .core.exceptions import ConfigurationError

__all__ = ["Config", "get_config", "configure", "reset_config", "config_path", "cache_path"]

_APP = "visualmetrics"
_lock = threading.RLock()
_current: Config | None = None

VALID_THEMES = (
    "light",
    "dark",
    "high_contrast",
    "classroom",
    "publication",
    "colorblind",
    "custom",
)
VALID_LEVELS = ("beginner", "intermediate", "advanced", "phd")


@dataclass
class Config:
    """Everything the user can persist between sessions."""

    language: str = "en"
    terminology: str = "translated"
    level: str = "intermediate"
    theme: str = "light"
    reduced_motion: bool = False
    live_update: bool = True
    precision: int = 4
    seed: int = 42
    show_backend: bool = False
    show_code: bool = False
    colorblind_mode: str = "none"
    classroom_font_scale: float = 1.25
    simulation_preset: str = "teaching"  # fast | teaching | high_precision
    gui_host: str = "127.0.0.1"
    gui_port: int = 8080
    gui_native: bool = False
    telemetry: bool = False  # always off; present so it is auditable

    def __post_init__(self) -> None:
        from .i18n.translator import LANGUAGES, TERMINOLOGY_MODES

        if self.language not in LANGUAGES:
            raise ConfigurationError(
                f"unsupported language {self.language!r}; supported: {list(LANGUAGES)}",
                setting="language",
                value=self.language,
            )
        if self.terminology not in TERMINOLOGY_MODES:
            raise ConfigurationError(
                f"unsupported terminology mode {self.terminology!r}", setting="terminology"
            )
        if self.level not in VALID_LEVELS:
            raise ConfigurationError(f"unsupported level {self.level!r}", setting="level")
        if self.theme not in VALID_THEMES:
            raise ConfigurationError(f"unknown theme {self.theme!r}", setting="theme")
        if self.telemetry:
            raise ConfigurationError(
                "VisualMetrics does not implement telemetry; this flag must stay False",
                setting="telemetry",
            )
        self.precision = max(0, min(12, int(self.precision)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> Config:
        data = self.to_dict()
        unknown = set(changes) - set(data)
        if unknown:
            raise ConfigurationError(
                f"unknown setting(s): {sorted(unknown)}; valid: {sorted(data)}",
                setting=", ".join(sorted(unknown)),
            )
        data.update(changes)
        return Config(**data)

    def save(self, path: Path | None = None) -> Path:
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def _user_dir() -> Path:
    try:
        from platformdirs import user_config_dir

        return Path(user_config_dir(_APP, appauthor=False))
    except Exception:  # pragma: no cover - platformdirs is a core dep
        return Path.home() / ".visualmetrics"


def config_path() -> Path:
    override = os.environ.get("VISUALMETRICS_CONFIG")
    if override:
        return Path(override)
    return _user_dir() / "config.json"


def cache_path() -> Path:
    override = os.environ.get("VISUALMETRICS_CACHE")
    if override:
        return Path(override)
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir(_APP, appauthor=False))
    except Exception:  # pragma: no cover
        return Path.home() / ".visualmetrics" / "cache"


def _load_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_env() -> dict[str, Any]:
    out: dict[str, Any] = {}
    known = {f.name: f.type for f in fields(Config)}
    for name in known:
        raw = os.environ.get(f"VISUALMETRICS_{name.upper()}")
        if raw is None:
            continue
        low = raw.strip().lower()
        if low in {"true", "false"}:
            out[name] = low == "true"
        else:
            try:
                out[name] = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                out[name] = raw
    return out


def _discover() -> Config:
    data: dict[str, Any] = {}
    data.update(_load_file(config_path()))
    project = Path.cwd() / "visualmetrics.json"
    data.update(_load_file(project))
    data.update(_load_env())
    valid = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in data.items() if k in valid}
    try:
        return Config(**filtered)
    except ConfigurationError:
        return Config()


def get_config() -> Config:
    global _current
    with _lock:
        if _current is None:
            _current = _discover()
        return _current


def configure(*, persist: bool = False, **settings: Any) -> Config:
    """Update the active configuration.

    >>> configure(language="ar", terminology="bilingual", theme="classroom")
    """
    global _current
    with _lock:
        cfg = get_config().replace(**settings)
        _current = cfg
    if "language" in settings or "terminology" in settings:
        from .i18n.translator import set_language

        set_language(cfg.language, cfg.terminology)
    if persist:
        cfg.save()
    return cfg


def reset_config() -> Config:
    global _current
    with _lock:
        _current = Config()
        return _current
