# Notebooks

```python
import visualmetrics as vm
from visualmetrics import notebook

notebook.setup()
```

After `setup()`, a `LabResult` displays as a small report rather than as a bare
object: the evidence badge with its caveat, any warnings, any violated
assumptions, and the numbers.

That matters because notebook cells get pasted into drafts. A figure that
arrives without its context is exactly the artefact this package exists to
prevent.

```python
vm.lab("econometrics.heteroskedasticity", scenario="severe")
```

## Animations in a notebook

A notebook cannot always play an animation, and a PDF export never can, so the
frame commentary renders as a table:

```python
result = vm.lab("inference.clt")
notebook.display_animation(result.animations[0])
```

## Live controls

```python
notebook.interact("inference.power")
```

This builds widgets from the same `ControlSpec` declarations the GUI uses, so
the notebook cannot drift out of step with the application. Requires the
`notebook` extra; without it, the error names the package to install.

## Without IPython

`result_html()` and `animation_html()` return strings, so the same rendering
works in any tool that displays HTML.
