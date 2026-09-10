"""Per-browser session state (blueprint section 36.5).

Everything the GUI needs to know about *this* visitor lives here: language,
theme, level, terminology mode, motion preference, and the concept currently
open with its parameters.

This module deliberately contains no NiceGUI. It is plain data with plain
methods, so the whole navigation and parameter model can be tested without a
browser - and so the same state can be serialized, shared as a link, or
replayed from a saved configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ...core.state import LabState
from ...i18n.translator import LANGUAGES, TERMINOLOGY_MODES

__all__ = ["Session", "THEMES", "LEVELS", "session_from_query", "session_to_query"]

THEMES: tuple[str, ...] = (
    "light", "dark", "high_contrast", "classroom", "publication", "colorblind", "custom",
)
LEVELS: tuple[str, ...] = ("beginner", "intermediate", "advanced", "phd")


@dataclass
class Session:
    """One visitor's settings and current position in the catalog."""

    language: str = "en"
    theme: str = "light"
    level: str = "intermediate"
    terminology: str = "translated"
    reduced_motion: bool = False
    precision: int = 4
    seed: int = 42

    concept_id: str | None = None
    scenario: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    compare_with: str | None = None
    """A second concept or scenario shown side by side (blueprint section 36.7)."""

    presentation: bool = False
    """Presentation mode: larger type, controls hidden, one panel at a time."""

    def __post_init__(self) -> None:
        self.language = self._one_of(self.language, LANGUAGES, "en")
        self.theme = self._one_of(self.theme, THEMES, "light")
        self.level = self._one_of(self.level, LEVELS, "intermediate")
        self.terminology = self._one_of(self.terminology, TERMINOLOGY_MODES, "translated")
        self.precision = max(0, min(int(self.precision), 12))

    @staticmethod
    def _one_of(value: Any, allowed: tuple[str, ...], fallback: str) -> str:
        text = str(value or "").strip().lower()
        return text if text in allowed else fallback

    # -- settings ---------------------------------------------------------

    def set_language(self, language: str) -> Session:
        self.language = self._one_of(language, LANGUAGES, self.language)
        return self

    def set_theme(self, theme: str) -> Session:
        self.theme = self._one_of(theme, THEMES, self.theme)
        return self

    def set_level(self, level: str) -> Session:
        self.level = self._one_of(level, LEVELS, self.level)
        return self

    def set_terminology(self, mode: str) -> Session:
        self.terminology = self._one_of(mode, TERMINOLOGY_MODES, self.terminology)
        return self

    @property
    def is_rtl(self) -> bool:
        from ...i18n.rtl import is_rtl

        return is_rtl(self.language)

    # -- navigation -------------------------------------------------------

    def open(self, concept_id: str, *, scenario: str | None = None) -> Session:
        """Move to a concept, discarding parameters that belonged to the old one."""
        if concept_id != self.concept_id:
            self.parameters = {}
            self.compare_with = None
        self.concept_id = concept_id
        self.scenario = scenario
        return self

    def choose_scenario(self, scenario: str | None) -> Session:
        """Pick a preset. Explicit parameter edits are cleared so the preset shows."""
        self.scenario = scenario
        self.parameters = {}
        return self

    def set_parameter(self, name: str, value: Any) -> Session:
        self.parameters[name] = value
        return self

    def reset_parameters(self) -> Session:
        self.parameters = {}
        return self

    # -- handing off to the engine ----------------------------------------

    def lab_state(self, concept_id: str | None = None) -> LabState:
        """The state to run. Raises if no concept is open."""
        target = concept_id or self.concept_id
        if not target:
            raise ValueError("no concept is open in this session")
        return LabState(
            concept_id=target,
            parameters=dict(self.parameters),
            scenario=self.scenario,
            seed=self.seed,
            language=self.language,
            theme=self.theme,
            level=self.level,
            terminology=self.terminology,
            reduced_motion=self.reduced_motion,
            precision=self.precision,
        )

    def translator(self):
        from ...i18n.translator import get_translator

        return get_translator(self.language, self.terminology)

    def copy(self, **changes: Any) -> Session:
        clone = replace(self, parameters=dict(self.parameters))
        for key, value in changes.items():
            setattr(clone, key, value)
        clone.__post_init__()
        return clone

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "theme": self.theme,
            "level": self.level,
            "terminology": self.terminology,
            "reduced_motion": self.reduced_motion,
            "precision": self.precision,
            "seed": self.seed,
            "concept_id": self.concept_id,
            "scenario": self.scenario,
            "parameters": dict(self.parameters),
            "compare_with": self.compare_with,
            "presentation": self.presentation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def session_to_query(session: Session) -> dict[str, str]:
    """The shareable part of a session, as URL query parameters.

    Parameters are included so a colleague opening the link sees the same
    figure, which is the point of sharing it at all.
    """
    query: dict[str, str] = {
        "lang": session.language,
        "theme": session.theme,
        "level": session.level,
    }
    if session.terminology != "translated":
        query["term"] = session.terminology
    if session.scenario:
        query["scenario"] = session.scenario
    if session.seed != 42:
        query["seed"] = str(session.seed)
    if session.reduced_motion:
        query["motion"] = "reduced"
    for name, value in sorted(session.parameters.items()):
        query[f"p.{name}"] = str(value)
    return query


def session_from_query(query: dict[str, Any], *, base: Session | None = None) -> Session:
    """Rebuild a session from URL query parameters.

    Unknown or malformed values fall back to the defaults rather than raising:
    a mistyped link should still open the lab.
    """
    session = base.copy() if base else Session()
    session.set_language(str(query.get("lang", session.language)))
    session.set_theme(str(query.get("theme", session.theme)))
    session.set_level(str(query.get("level", session.level)))
    session.set_terminology(str(query.get("term", session.terminology)))
    if "scenario" in query:
        session.scenario = str(query["scenario"]) or None
    if "motion" in query:
        session.reduced_motion = str(query["motion"]).lower() in {"reduced", "1", "true"}
    if "seed" in query:
        try:
            session.seed = int(query["seed"])
        except (TypeError, ValueError):
            pass
    parameters: dict[str, Any] = {}
    for key, value in query.items():
        if str(key).startswith("p."):
            parameters[str(key)[2:]] = value
    if parameters:
        session.parameters = parameters
    return session
