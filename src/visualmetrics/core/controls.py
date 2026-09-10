"""Declarative control specifications.

A lab never builds widgets by hand. It declares :class:`ControlSpec` objects and
the GUI, the notebook front-end and the CLI all render the same declaration.
This is what makes GUI state and Python state identical (blueprint section 53).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .exceptions import InvalidParameterError

__all__ = [
    "ControlKind",
    "ControlSpec",
    "slider",
    "int_slider",
    "select",
    "toggle",
    "seed_control",
    "number",
    "validate_parameters",
]


class ControlKind(str, Enum):
    SLIDER = "slider"
    INT_SLIDER = "int_slider"
    SELECT = "select"
    TOGGLE = "toggle"
    NUMBER = "number"
    SEED = "seed"
    MULTISELECT = "multiselect"


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """One user-facing parameter of a lab."""

    id: str
    kind: ControlKind
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    values: tuple[Any, ...] = ()
    label_key: str | None = None
    help_key: str | None = None
    unit: str | None = None
    group: str = "main"
    expensive: bool = False
    depends_on: tuple[str, tuple[Any, ...]] | None = None
    advanced: bool = False

    def __post_init__(self) -> None:
        if self.kind in (ControlKind.SLIDER, ControlKind.INT_SLIDER, ControlKind.NUMBER):
            if self.min is None or self.max is None:
                raise InvalidParameterError("control " + self.id + " needs min and max")
            if self.min > self.max:
                raise InvalidParameterError("control " + self.id + " has min > max")
        if self.kind in (ControlKind.SELECT, ControlKind.MULTISELECT) and not self.values:
            raise InvalidParameterError("control " + self.id + " needs a value list")

    @property
    def translation_key(self) -> str:
        return self.label_key or f"controls.{self.id}.label"

    @property
    def help_translation_key(self) -> str:
        return self.help_key or f"controls.{self.id}.help"

    def coerce(self, value: Any) -> Any:
        """Validate and normalize a raw value coming from GUI, API or JSON."""
        if self.kind is ControlKind.TOGGLE:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if self.kind is ControlKind.SELECT:
            if value not in self.values:
                raise InvalidParameterError(
                    f"{value!r} is not a valid option for {self.id!r}; "
                    f"valid options: {list(self.values)}",
                    control=self.id,
                    value=value,
                    options=list(self.values),
                )
            return value
        if self.kind is ControlKind.MULTISELECT:
            seq = tuple(value) if isinstance(value, (list, tuple, set)) else (value,)
            bad = [v for v in seq if v not in self.values]
            if bad:
                raise InvalidParameterError(
                    f"invalid options {bad} for {self.id!r}",
                    control=self.id,
                    value=bad,
                    options=list(self.values),
                )
            return tuple(seq)
        if self.kind in (ControlKind.SEED, ControlKind.INT_SLIDER):
            try:
                iv = int(round(float(value)))
            except (TypeError, ValueError) as exc:
                raise InvalidParameterError(
                    f"{self.id!r} must be an integer, got {value!r}",
                    control=self.id,
                    value=value,
                ) from exc
            return int(self._clip(iv))
        try:
            fv = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError(
                f"{self.id!r} must be numeric, got {value!r}",
                control=self.id,
                value=value,
            ) from exc
        if not math.isfinite(fv):
            raise InvalidParameterError(
                f"{self.id!r} must be a finite number", control=self.id, value=value
            )
        return float(self._clip(fv))

    def _clip(self, value: float) -> float:
        if self.min is not None and value < self.min:
            return self.min
        if self.max is not None and value > self.max:
            return self.max
        return value


def slider(
    id: str, default: float, min: float, max: float, step: float = 0.01, **kw: Any
) -> ControlSpec:
    return ControlSpec(
        id=id, kind=ControlKind.SLIDER, default=default, min=min, max=max, step=step, **kw
    )


def int_slider(
    id: str, default: int, min: int, max: int, step: int = 1, **kw: Any
) -> ControlSpec:
    return ControlSpec(
        id=id, kind=ControlKind.INT_SLIDER, default=default, min=min, max=max, step=step, **kw
    )


def select(id: str, default: Any, values: Any, **kw: Any) -> ControlSpec:
    return ControlSpec(id=id, kind=ControlKind.SELECT, default=default, values=tuple(values), **kw)


def toggle(id: str, default: bool = False, **kw: Any) -> ControlSpec:
    return ControlSpec(id=id, kind=ControlKind.TOGGLE, default=default, **kw)


def number(id: str, default: float, min: float, max: float, **kw: Any) -> ControlSpec:
    return ControlSpec(id=id, kind=ControlKind.NUMBER, default=default, min=min, max=max, **kw)


def seed_control(default: int = 42) -> ControlSpec:
    return ControlSpec(
        id="seed",
        kind=ControlKind.SEED,
        default=default,
        min=0,
        max=2**31 - 1,
        group="reproducibility",
    )


def validate_parameters(
    controls: Any, params: dict[str, Any], *, strict: bool = False
) -> dict[str, Any]:
    """Merge user parameters onto control defaults, validating each one.

    Unknown keys are preserved (labs may accept derived parameters) unless
    ``strict`` is set, which is what the plugin validator uses.
    """
    by_id = {c.id: c for c in controls}
    out: dict[str, Any] = {c.id: c.default for c in controls}
    for key, value in params.items():
        spec = by_id.get(key)
        if spec is None:
            if strict:
                raise InvalidParameterError(
                    f"unknown control {key!r}; known controls: {sorted(by_id)}", control=key
                )
            out[key] = value
            continue
        out[key] = spec.coerce(value)
    return out
