# Writing a lab

A lab is a module exporting two objects: `SPEC`, a `ConceptSpec`, and `LAB`, an
instance of your `LabBase` subclass. The GUI, the notebook and the CLI all
render from `SPEC`, so a new lab needs no interface code at all.

## The shape

```python
from .._kit import (
    Domain, EvidenceType, LabBase, LabResult, LabState, P,
    make_spec, np, ref, scenario, slider, int_slider,
)

SPEC = make_spec(
    "domain.concept_name",
    domain=Domain.INFERENCE,
    evidence=EvidenceType.SIMULATION,
    controls=(
        int_slider("n", 100, 10, 2000),
        slider("alpha", 0.05, 0.001, 0.2, step=0.001),
    ),
    scenarios=(
        scenario("canonical", "canonical"),
        scenario("underpowered", "weak", n=20),
        scenario("violation", "violation", equal_variance=False),
    ),
    references=(ref("Casella and Berger (2002), section 8.3"),),
)


class MyLab(LabBase):
    def compute(self, p: dict, state: LabState) -> LabResult:
        res = LabResult()
        ...
        return res


LAB = MyLab(SPEC)
```

## What a result must contain

- **panels**: at least one figure, each with a tab and an evidence type;
- **metrics**: at least four numbers, labelled, with a reference value where
  theory supplies one;
- **explanations**: overview, intuition, mathematics, common mistakes and
  warnings at minimum;
- **assumptions**: each with its status and, when violated, the consequence;
- **one animation** whose every frame states what you see, what changed, why
  and how to read it.

## Rules the tests enforce

- at least five scenarios, and at least one where the method fails;
- every scenario overrides only controls that exist;
- reproducible: the same seed gives the same numbers;
- a violation scenario must report a failed assumption or a warning - running
  one silently is a test failure;
- no proof badge without a proof behind it.

## Determinism

Use the seeded generators from the kit rather than global NumPy state:

```python
gen = rng(state.seed, "my_lab", "sampling")
```

Every derived stream is a deterministic function of the session seed, so a
figure is reproducible from its link.
