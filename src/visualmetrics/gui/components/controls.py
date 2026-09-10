"""Rendering a :class:`ControlSpec` as a widget (blueprint section 36.4).

The payoff of declaring controls as data: this one function renders every
control in every lab, so a new lab needs no GUI code at all, and the GUI, the
notebook and the CLI can never disagree about what a lab accepts.

Values always pass through ``ControlSpec.coerce`` before they reach the
session, so an out-of-range slider or a mistyped URL parameter is clamped here
rather than blowing up inside a lab.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from ...core.controls import ControlKind, ControlSpec

__all__ = ["render_control", "render_control_panel", "render_scenario_picker"]


def _label_for(control: ControlSpec, translator: Any) -> str:
    return translator.t(control.translation_key, control.id.replace("_", " ").title())


def _help_for(control: ControlSpec, translator: Any) -> str:
    return translator.t(control.help_translation_key, "")


def render_control(
    control: ControlSpec,
    value: Any,
    on_change: Callable[[str, Any], None],
    *,
    translator: Any,
    disabled: bool = False,
) -> Any:
    """Render one control and wire its change handler."""
    label = _label_for(control, translator)
    hint = _help_for(control, translator)

    def commit(event: Any) -> None:
        raw = getattr(event, "value", event)
        try:
            on_change(control.id, control.coerce(raw))
        except Exception as exc:
            ui.notify(str(exc), type="warning")

    element: Any
    if control.kind is ControlKind.TOGGLE:
        element = ui.switch(label, value=bool(value), on_change=commit)
    elif control.kind in (ControlKind.SELECT, ControlKind.MULTISELECT):
        options = {
            option: translator.t(f"{control.translation_key}.option.{option}", str(option))
            for option in control.values
        }
        element = ui.select(
            options,
            label=label,
            value=value,
            multiple=control.kind is ControlKind.MULTISELECT,
            on_change=commit,
        ).classes("w-full")
    elif control.kind is ControlKind.SEED:
        with ui.row().classes("w-full items-center gap-2"):
            element = ui.number(
                label=label, value=int(value), format="%d", on_change=commit
            ).classes("grow")
            ui.button(
                icon="casino",
                on_click=lambda: on_change(control.id, _next_seed(value)),
            ).props("flat dense").tooltip(
                translator.t("actions.new_seed", "Draw a new seed")
            )
    elif control.kind in (ControlKind.SLIDER, ControlKind.INT_SLIDER):
        with ui.column().classes("w-full gap-0"):
            ui.label(f"{label}: {_format(value)}").classes("text-sm vm-muted")
            element = ui.slider(
                min=control.min,
                max=control.max,
                step=control.step or (1 if control.kind is ControlKind.INT_SLIDER else 0.01),
                value=value,
                on_change=commit,
            ).props("label-always")
    else:  # NUMBER and anything new that behaves numerically
        element = ui.number(label=label, value=value, on_change=commit).classes("w-full")

    if hint:
        element.tooltip(hint)
    if disabled:
        element.disable()
    return element


def _next_seed(current: Any) -> int:
    import random

    try:
        return (int(current) + random.randrange(1, 9973)) % 2**31
    except (TypeError, ValueError):
        return random.randrange(1, 2**31)


def _format(value: Any) -> str:
    from ..state.viewmodel import format_value

    return format_value(value, 4)


def render_scenario_picker(
    scenarios: Any,
    current: str | None,
    on_pick: Callable[[str | None], None],
    *,
    translator: Any,
) -> None:
    """Scenarios grouped by what they are for.

    Grouping matters pedagogically: the canonical case and the violation case
    are different kinds of thing, and a flat list of twenty names hides that.
    """
    if not scenarios:
        return
    groups: dict[str, list[Any]] = {}
    for scenario in scenarios:
        groups.setdefault(scenario.category.value, []).append(scenario)

    ui.label(translator.t("ui.scenarios", "Scenarios")).classes("text-sm font-semibold mt-2")
    for category, items in groups.items():
        with ui.expansion(
            translator.t(f"scenarios.category.{category}", category.replace("_", " ").title()),
            value=category == "canonical",
        ).classes("w-full"):
            for scenario in items:
                label = translator.t(scenario.translation_key, scenario.id.replace("_", " "))
                button = ui.button(
                    label, on_click=lambda _=None, s=scenario.id: on_pick(s)
                ).props("flat dense no-caps align=left").classes("w-full vm-focusable")
                if scenario.id == current:
                    button.props("color=primary")
                teaching = translator.t(
                    scenario.teaching_point_key or f"scenarios.{scenario.id}.teaching", ""
                )
                if teaching:
                    button.tooltip(teaching)


def render_control_panel(
    spec: Any,
    session: Any,
    on_change: Callable[[str, Any], None],
    on_scenario: Callable[[str | None], None],
    on_reset: Callable[[], None],
) -> None:
    """The whole left-hand panel: scenarios first, then the individual controls."""
    translator = session.translator()
    resolved = spec.parameters_for(session.scenario, **session.parameters)

    render_scenario_picker(spec.scenarios, session.scenario, on_scenario, translator=translator)

    ui.separator().classes("my-2")
    with ui.row().classes("w-full items-center justify-between"):
        ui.label(translator.t("ui.controls", "Controls")).classes("text-sm font-semibold")
        ui.button(
            translator.t("actions.reset", "Reset"), on_click=lambda: on_reset()
        ).props("flat dense no-caps").classes("vm-focusable")

    for control in spec.controls:
        render_control(
            control,
            resolved.get(control.id, control.default),
            on_change,
            translator=translator,
        )
