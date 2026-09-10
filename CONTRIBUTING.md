# Contributing to VisualMetrics

Thank you for considering a contribution. This document says what the project
will not compromise on, and how to add the things it most needs.

## The rules that are not negotiable

These are enforced by the test suite, not merely requested.

**1. A simulation is never a proof.** Every figure, animation and lab declares
an evidence type. Only `formal_proof`, `symbolic_derivation` and
`geometric_proof` count as proof, a `ProofSpec` refuses any other kind, and a
lab may not wear a proof badge unless it points at a real proof in
`proofs/library/`.

**2. Every proof states what it does not establish.** `ProofSpec` will not
construct without a `limitations` string. Write the real scope: which
assumption carries the argument, what breaks without it, what a reader might
wrongly conclude.

**3. Every animation frame explains itself.** An `AnimationStep` needs
`what_you_see`, `what_changed`, `why` and `interpretation`. A silent animation
is refused by the exporter and by the tests.

**4. The catalogue never overstates itself.** A concept that is not built is
`Status.PLANNED`, is listed, and refuses to open with an explanation. Do not
hide a gap; mark it.

**5. Canonical scientific names only.** Local course labels ("Statistics 3",
"Econometrics 1") must never appear in an id, a title, a domain or the GUI. The
`curriculum_tags` field exists for mapping to a local syllabus; that is the
only place such a label belongs.

**6. Numerics are tested against theory.** A new estimator needs a test against
a closed form, an exact identity, or a known statistical property with a
tolerance derived from the Monte Carlo standard error - never against the
number the code produced yesterday.

## Getting set up

```bash
git clone https://github.com/merwanroudane/visualmetrics
cd visualmetrics
pip install -e ".[dev,gui]"
pytest -m "not slow"          # about four minutes
ruff check src tests
mypy src/visualmetrics/core src/visualmetrics/i18n src/visualmetrics/catalog
```

The full sweep, which runs every scenario of every lab, is marked slow:

```bash
pytest -m slow
```

## Adding a concept

1. **Catalogue it.** Add a `ConceptSpec` to `catalog/builtin.py`, with
   `status=Status.PLANNED` and `module=None` until it works.
2. **Write the lab** in `concepts/<domain>/<name>.py`, exporting `SPEC` and
   `LAB`. Start from a neighbouring lab and from `concepts/_kit.py`.
3. **Declare controls** as `ControlSpec` objects. Do not read parameters any
   other way: the GUI, the notebook and the CLI all render from these
   declarations, and a control that bypasses them exists in only one of them.
4. **Declare at least five scenarios**, and at least one where the method
   fails - a violation, a counterexample or a boundary case. A lab that only
   shows the happy path teaches the wrong lesson, and a test rejects it.
5. **Produce the full result**: panels, at least four metrics, explanations,
   assumptions with their status, and one animation whose every frame explains
   itself.
6. **Add the concept text** to `i18n/{en,ar,fr}/concepts.json` - all three, with
   identical keys.
7. **Flip the status** to `Status.STABLE`, point `module` at your file, run the
   suite.

The GUI needs no changes at all. That is the point of the declarative specs.

## Adding a proof

Write `proofs/library/<name>.py` exporting a `PROOF`, register it in
`proofs/registry.py`, and link it from the lab's `proof_ids`.

Each step needs a statement and a justification - *why this move is legal*, not
just what it says. Name the assumptions and reference them from the steps that
use them; a test rejects an assumption no step invokes, because an unused
hypothesis is either decoration or a gap. Where an identity can be checked
numerically, attach a `ProofCheck`; it will be reported as a test of the code.

## Adding or improving a translation

This is the most valuable contribution available right now.

Bundles live in `src/visualmetrics/i18n/<lang>/`. Keys must match English
exactly - a test compares them and fails on any difference.

The open work is `labs.json`. Lab narrative uses roughly 2,400 keys and only
the shared vocabulary is translated, so Arabic and French currently fall back
to English inside the labs. Translating a single lab's keys is a complete,
useful contribution.

Notes for Arabic: use the `vm-ltr` class or `protect_latin()` for Latin
identifiers inside Arabic prose, and keep technical terms consistent with
`glossary.json`, which drives the three terminology modes.

## Style

- Line length 100, `ruff` for linting and formatting.
- Type hints on public functions; `mypy` runs on `core`, `i18n` and `catalog`.
- Comments explain *why*, not what. If a constant is unusual, say what would
  break with the obvious value instead.
- Docstrings state what a thing is for and what it refuses to do.

## Pull requests

Say what you changed and why. If you touched numerics, include the check you
ran and its output. If a test tolerance moved, explain what the tolerance
represents now. Small, focused pull requests are much easier to review than
large ones.

## Reporting a problem

Open an issue with the version (`visualmetrics doctor`), the concept id, the
scenario, the seed, and what you expected. A wrong number is a serious bug in
this project - please report it even if you are not certain.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
