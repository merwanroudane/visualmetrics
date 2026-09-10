# The Python API

Everything the GUI does is available directly, and the GUI has no privileged
access: it calls the same functions.

## Running a lab

```python
import visualmetrics as vm

result = vm.lab("inference.power", effect_size=0.4, n=60, alpha=0.05)
```

A `LabResult` carries:

| Attribute | What it holds |
|---|---|
| `panels` | The figures, each with a title, caption and evidence type |
| `metrics` | The numbers, with labels and reference values |
| `explanations` | Prose blocks, each tagged with a knowledge tab |
| `assumptions` | Each with its status and, when violated, the consequence |
| `animations` | Each with its per-frame explanation layer |
| `warnings` | What the run wants you to notice |
| `evidence` | What kind of evidence this is |
| `code` | The exact Python that reproduces the run |
| `dgp` | How the data were generated |

## Scenarios

```python
vm.lab("econometrics.heteroskedasticity", scenario="severe")
vm.lab("causal.did", scenario="violated_trends")

vm.concept("causal.did").scenarios      # what a lab offers
```

## Browsing

```python
vm.list_concepts(domain="econometrics")
vm.search("hétéroscédasticité")     # any of the three languages
vm.explain("regression.fwl")
vm.prerequisites("panel.dynamic_gmm")
vm.learning_path("causal.did")
```

## Proofs

```python
vm.list_proofs()
vm.proofs_for("regression.fwl")

rendered = vm.proof("regression.fwl.theorem", language="fr")
print(rendered.to_text())
```

## Teaching material

```python
vm.objectives("inference.power")       # what to take away, and how to check it
vm.misconceptions("inference.power")   # the plausible wrong beliefs
vm.quiz("inference.power", count=3)    # a self-check, never a grade
```

## Session settings

```python
vm.configure(language="ar", theme="dark", level="advanced", seed=7)
vm.lab("inference.clt", language="fr")   # or per call
```

## Reproducibility

Every lab is deterministic given a seed:

```python
a = vm.lab("inference.bootstrap", seed=42).metric_dict()
b = vm.lab("inference.bootstrap", seed=42).metric_dict()
assert a == b
```

State round-trips:

```python
from visualmetrics.export import save_config, load_config

save_config(state, "lesson-1.json")
vm.run_state(load_config("lesson-1.json"))
```
