"""No-code to code translation.

Blueprint section 53: every GUI state must round-trip into runnable Python that
reproduces it. The generated snippet is executed verbatim in the test suite, so
it can never drift into decoration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..core.concepts import ConceptSpec
    from ..core.state import LabState

__all__ = ["render_lab_code", "render_value", "render_state_script"]


def render_value(value: Any) -> str:
    """Render a Python literal for the generated snippet."""
    import numpy as np

    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return f"{value:.1f}"
        return repr(round(value, 10))
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k!r}: {render_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return repr(value)


def render_lab_code(spec: ConceptSpec, params: dict[str, Any], state: LabState) -> str:
    """Generate the ``vm.lab(...)`` call equivalent to the current GUI state."""
    defaults = spec.default_parameters
    lines = ["import visualmetrics as vm", ""]

    config_bits: list[str] = []
    if state.language != "en":
        config_bits.append(f"language={state.language!r}")
    if state.terminology != "translated":
        config_bits.append(f"terminology={state.terminology!r}")
    if state.theme != "light":
        config_bits.append(f"theme={state.theme!r}")
    if state.level != "intermediate":
        config_bits.append(f"level={state.level!r}")
    if state.reduced_motion:
        config_bits.append("reduced_motion=True")
    if config_bits:
        lines.append("vm.configure(" + ", ".join(config_bits) + ")")
        lines.append("")

    args = [f"{spec.id!r}"]
    if state.scenario:
        args.append(f"scenario={state.scenario!r}")
    changed = {
        key: value
        for key, value in params.items()
        if key not in defaults or _differs(defaults.get(key), value)
    }
    changed.pop("seed", None)
    for key in sorted(changed):
        args.append(f"{key}={render_value(changed[key])}")
    args.append(f"seed={int(state.seed)}")

    call = "result = vm.lab(" + ", ".join(args) + ")"
    if len(call) > 88:
        joined = ",\n    ".join(args)
        call = f"result = vm.lab(\n    {joined},\n)"
    lines.append(call)
    lines.append("result.show()            # open every figure")
    lines.append("print(result.summary())  # numeric read-out")
    return "\n".join(lines)


def _differs(default: Any, value: Any) -> bool:
    if isinstance(default, float) or isinstance(value, float):
        try:
            return abs(float(default) - float(value)) > 1e-12
        except (TypeError, ValueError):
            return default != value
    return default != value


def render_state_script(state: LabState) -> str:
    """A script that reloads an exported JSON state file."""
    return (
        "import json\n"
        "import visualmetrics as vm\n"
        "from visualmetrics.core.state import LabState\n"
        "\n"
        'with open("lab_state.json", encoding="utf-8") as fh:\n'
        "    state = LabState.from_dict(json.load(fh))\n"
        "\n"
        "result = vm.run_state(state)\n"
        "result.show()\n"
    )
