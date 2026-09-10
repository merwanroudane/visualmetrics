# Exporting

Every exporter carries the scientific layer with the figures. There is no
exporter that strips it, and that is not an oversight to be fixed by request.

```python
import visualmetrics as vm
from visualmetrics.export import (
    export_html_report, export_result, export_animation,
    export_animation_frames, save_config, load_config,
)

result = vm.lab("regression.fwl")
```

## A report

```python
export_html_report(result, "fwl.html")
```

A self-contained page: the evidence badge with its caveat, the warnings, every
figure with its caption, the numbers, the assumptions and their status, the
explanations, the animation commentary as a table, the generated Python and the
data-generating process. It needs no build step, and Plotly loads from a CDN.

## Figures

```python
export_result(result, "fwl.html")   # one interactive file per panel
export_result(result, "fwl.png")    # static images (needs kaleido)
export_result(result, "fwl.json")   # the entire result as data
```

A multi-panel result writes one file per panel, with the panel id appended, so
nothing is silently overwritten. The returned list says exactly what was
written.

## Animations

```python
export_animation(result.animations[0], "fwl-animation.html")
```

A standalone page with transport controls and the frame commentary beside the
figure.

Exporting an animation to a static format is **refused**, because the format
cannot carry the commentary. For images:

```python
export_animation_frames(result.animations[0], "frames/")
```

This writes one image per frame *and* a JSON file of the commentary beside
them. The commentary file is not optional.

## Configurations

```python
save_config(state, "lesson-1.json")     # or .yaml
vm.run_state(load_config("lesson-1.json"))
```

A configuration is a `LabState`: concept, scenario, parameters, seed and
session settings, stamped with the version that produced it. Reloading one
reproduces the same figures and the same numbers - a test checks exactly that.
