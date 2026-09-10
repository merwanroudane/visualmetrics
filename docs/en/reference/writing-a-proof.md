# Writing a proof

A proof is data: one module in `proofs/library/` exporting a `PROOF`.

```python
PROOF = ProofSpec(
    id="regression.ols.residual_orthogonality",
    kind=EvidenceType.GEOMETRIC_PROOF,
    title="OLS residuals are orthogonal to every regressor",
    claim="...",
    intuition="One sentence, before any symbol appears.",
    assumptions=(
        assume("full_rank", "X has rank k.", if_violated="..."),
    ),
    steps=(
        step("objective", "...", equation=r"S(b) = \|y - Xb\|^2",
             why="Definition of ordinary least squares.",
             kind=StepKind.SETUP, uses=("euclidean",)),
        ...
    ),
    conclusion="...",
    limitations="What this does NOT establish.",
    references=(...),
    checks=(...),
)
```

## What each part is for

- **`intuition`** - the idea in plain language, before the notation. A reader
  who stops after this sentence should still have learned something.
- **`assumptions`** - named, each with `if_violated` saying what breaks. Steps
  reference them by id, so "where exactly is normality used?" has an exact
  answer.
- **`steps`** - each with a statement, the equation, and `why` *this move is
  legal*. The justification is the part that teaches; without it a proof is a
  list of formulas.
- **`limitations`** - required. The spec will not construct without it.
- **`checks`** - optional numerical checks of the identities proved, reported
  as tests of the code.

## Rules the tests enforce

- the evidence kind must be a proof kind;
- `limitations` must be non-empty;
- no duplicate step ids, and no step depending on something that does not
  exist;
- **every assumption must be used by some step** - an unused hypothesis is
  either decoration or a gap in the argument;
- every step needs both a statement and a justification;
- the chain must reach a `CONCLUSION` step and must not end on scaffolding;
- concept links and prerequisites must resolve;
- every attached check must pass.

## Registering it

Add an entry to `PROOF_INDEX` in `proofs/registry.py` and reference the proof
from the lab's `proof_ids`. A lab wearing a proof badge with no proof behind it
fails the honesty suite.
