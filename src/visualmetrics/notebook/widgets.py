"""Interactive lab controls inside Jupyter (blueprint section 65.2).

The same :class:`ControlSpec` declarations that drive the GUI drive the widgets
here, so a lab exposes one set of controls and the notebook cannot drift out of
step with the application.

``ipywidgets`` is optional. Without it, :func:`interact` raises a
:class:`MissingDependencyError` naming the extra to install, and everything in
:mod:`visualmetrics.notebook.display` keeps working.
"""

from __future__ import annotations

from typing import Any

from ..core.controls import ControlKind, ControlSpec
from ..core.exceptions import MissingDependencyError
from ..core.state import LabState

__all__ = ["interact", "control_widget", "LabWidget"]


def _widgets():
    try:
        import ipywidgets
    except ImportError as exc:
        raise MissingDependencyError(
            "ipywidgets", "notebook", feature="interactive notebook controls"
        ) from exc
    return ipywidgets


def control_widget(control: ControlSpec, value: Any = None, *, translator: Any = None) -> Any:
    """Build the widget for one control, honouring its declared bounds."""
    widgets = _widgets()
    label = (
        translator.t(control.translation_key, control.id.replace("_", " ").title())
        if translator is not None
        else control.id.replace("_", " ").title()
    )
    current = control.default if value is None else value

    if control.kind is ControlKind.TOGGLE:
        return widgets.Checkbox(value=bool(current), description=label, indent=False)
    if control.kind is ControlKind.SELECT:
        return widgets.Dropdown(
            options=list(control.values), value=current, description=label
        )
    if control.kind is ControlKind.MULTISELECT:
        return widgets.SelectMultiple(
            options=list(control.values), value=tuple(current or ()), description=label
        )
    if control.kind in (ControlKind.INT_SLIDER, ControlKind.SEED):
        return widgets.IntSlider(
            value=int(current),
            min=int(control.min if control.min is not None else 0),
            max=int(control.max if control.max is not None else 1000),
            step=int(control.step or 1),
            description=label,
            continuous_update=False,
        )
    if control.kind is ControlKind.SLIDER:
        return widgets.FloatSlider(
            value=float(current),
            min=float(control.min if control.min is not None else 0.0),
            max=float(control.max if control.max is not None else 1.0),
            step=float(control.step or 0.01),
            description=label,
            continuous_update=False,
        )
    return widgets.FloatText(value=float(current), description=label)


class LabWidget:
    """A lab with live controls in a notebook.

    Recomputation is triggered by the controls, but the run button exists for
    the labs that take seconds: the same reason the GUI does not recompute on
    every pixel of a slider drag.
    """

    def __init__(self, concept_id: str, *, language: str | None = None,
                 scenario: str | None = None, seed: int = 42, auto: bool = True) -> None:
        from ..core.registry import registry
        from ..i18n.translator import get_translator

        self.widgets = _widgets()
        self.lab = registry.lab(concept_id)
        self.spec = self.lab.spec
        self.concept_id = concept_id
        self.language = language
        self.translator = get_translator(language)
        self.scenario = scenario
        self.seed = seed
        self.auto = auto
        self.controls: dict[str, Any] = {}
        self.output = self.widgets.Output()

    def build(self) -> Any:
        widgets = self.widgets
        resolved = self.spec.parameters_for(self.scenario)

        children = []
        if self.spec.scenarios:
            picker = widgets.Dropdown(
                options=[("(none)", None)] + [(s.id, s.id) for s in self.spec.scenarios],
                value=self.scenario,
                description=self.translator.t("ui.scenarios", "Scenario"),
            )
            picker.observe(self._on_scenario, names="value")
            self.scenario_picker = picker
            children.append(picker)

        for control in self.spec.controls:
            widget = control_widget(
                control, resolved.get(control.id, control.default),
                translator=self.translator,
            )
            widget.observe(self._on_change, names="value")
            self.controls[control.id] = widget
            children.append(widget)

        run = widgets.Button(
            description=self.translator.t("actions.run", "Run"), button_style="primary"
        )
        run.on_click(lambda _button: self.run())
        children.append(run)

        panel = widgets.VBox(children)
        layout = widgets.HBox([panel, self.output])
        self.run()
        return layout

    def _on_scenario(self, change: Any) -> None:
        self.scenario = change["new"]
        resolved = self.spec.parameters_for(self.scenario)
        for control_id, widget in self.controls.items():
            if control_id in resolved:
                widget.value = resolved[control_id]
        if self.auto:
            self.run()

    def _on_change(self, _change: Any) -> None:
        if self.auto:
            self.run()

    def state(self) -> LabState:
        parameters = {}
        for control in self.spec.controls:
            widget = self.controls.get(control.id)
            if widget is not None:
                parameters[control.id] = control.coerce(widget.value)
        return LabState(
            concept_id=self.concept_id,
            parameters=parameters,
            scenario=self.scenario,
            seed=self.seed,
            language=self.translator.language,
        )

    def run(self) -> None:
        from .display import display_result

        self.output.clear_output(wait=True)
        with self.output:
            try:
                display_result(self.lab.run(self.state()), language=self.language)
            except Exception as exc:  # noqa: BLE001 - a failed run must be visible
                print(f"This run did not complete: {exc}")


def interact(concept_id: str, *, language: str | None = None,
             scenario: str | None = None, seed: int = 42, auto: bool = True) -> Any:
    """Open a lab with live controls in a notebook cell."""
    from IPython.display import display

    widget = LabWidget(
        concept_id, language=language, scenario=scenario, seed=seed, auto=auto
    )
    layout = widget.build()
    display(layout)
    return widget
