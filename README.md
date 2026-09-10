# VisualMetrics

**See the theory. Change the assumptions. Understand the model.**

An interactive visual laboratory for statistics, statistical inference,
econometrics, causal inference, machine learning and modern AI - in English,
Arabic and French.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1008%20passing-brightgreen)](tests/)

---

## Screenshots

> Placeholders until the first tagged release. Reproduce any of them locally
> with `visualmetrics gui`.

| | |
|---|---|
| `docs/assets/screenshot-lab.png` — a lab: controls, evidence badge, knowledge tabs | `docs/assets/screenshot-animation.gif` — an animation with its per-frame explanation |
| `docs/assets/screenshot-proof.png` — a proof, step by step, with its limitations | `docs/assets/screenshot-arabic.png` — the same lab in Arabic, right-to-left |

---

## What VisualMetrics is

A laboratory where a statistical idea is something you *operate* rather than
read about. Each concept is a lab: you move the assumptions, watch what the
estimator does, and read the consequence in words and in numbers at the same
time.

Three commitments shape everything:

**1. Every figure says what kind of evidence it is.** A picture that
illustrates a theorem and a Monte Carlo study that suggests a pattern are not
the same thing, so they never wear the same badge. The nine evidence types run
from `formal_proof` through `simulation` to `counterexample`, and each carries
a caveat that travels with the figure into every export.

**2. Every animation explains itself, frame by frame.** What you see, what
changed, why it changed, how to read it, what to conclude, and what to be
careful about - beside the figure, not in a caption underneath. An animation
that arrives without that layer is refused rather than played silently.

**3. The catalogue never overstates itself.** 47 labs are built; 16 more are
catalogued as `planned`. The planned ones are listed, marked, and refuse to
open with an explanation. Nothing is hidden to make the count look better.

## What VisualMetrics is not

- **Not a proof assistant.** The 13 proofs are human-written arguments, checked
  for internal consistency, not machine-verified. Where a numerical check
  accompanies one, it tests this package's code, never the theorem.
- **Not a substitute for a textbook.** It is built to sit beside one.
- **Not an econometrics package for research output.** It uses real estimators
  and reports honest numbers, but its purpose is understanding, not producing
  publication tables. For that, use `statsmodels`, `linearmodels` or R.
- **Not an assessment system.** The self-check quizzes explain their answers
  and grade nothing.
- **Not a claim that a simulation proves anything.** That distinction is
  enforced in code and in the test suite.

## Main domains

| | |
|---|---|
| Mathematical foundations | Probability and random variables |
| Descriptive statistics | Inferential statistics |
| Regression and linear models | Econometrics |
| Time series econometrics | Panel data econometrics |
| Causal inference | Multivariate statistics |
| Spatial econometrics | Machine learning |
| Deep learning | Modern AI |
| Explainable AI | |

## Feature matrix

| Capability | GUI | Python API | Notebook | Arabic | English | French |
|---|---:|---:|---:|---:|---:|---:|
| Statistical inference | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Regression and econometrics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Time series and panel data | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Causal inference | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Machine learning and XAI | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Step-by-step proofs | ✅ | ✅ | Text | ✅ | ✅ | ✅ |
| Interactive scenarios | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Animations with explanations | ✅ | ✅ | Commentary | ✅ | ✅ | ✅ |
| Export (HTML, PNG, SVG, JSON) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Lab narrative text | ✅ | ✅ | ✅ | English fallback | ✅ | English fallback |

The last row is stated honestly: the interface, the catalogue, the glossary and
the proof structure are fully translated, while the per-lab narrative currently
falls back to English in Arabic and French. See
[Adding a translation](#adding-a-translation).

## Installation

```bash
pip install visualmetrics
```

The base install pulls only NumPy, SciPy and `platformdirs`. That gives you the
catalogue, search in three languages, the glossary, the proofs, the learning
objectives, the self-checks and the whole numeric engine - every estimator, test
and covariance matrix.

**Figures need the `viz` extra**, and asking for one without it raises an error
naming the package and the install command rather than crashing. Most people
want the application:

```bash
pip install "visualmetrics[gui]"
```

```bash
pip install "visualmetrics[viz]"      # figures, from the API or a notebook
pip install "visualmetrics[all]"      # every extra that installs from wheels
```

`all` deliberately excludes the `proofs` extra: Manim needs cairo and ffmpeg on
the system, and folding it in would make `pip install visualmetrics[all]` fail
on a clean machine. Install it separately if you want exported proof
animations.

From source:

```bash
git clone https://github.com/merwanroudane/visualmetrics
cd visualmetrics
pip install -e ".[dev,gui]"
```

## Quick start - GUI

```bash
visualmetrics
```

Or explicitly:

```bash
visualmetrics gui --port 8080
```

Then open <http://localhost:8080>. Language, theme, level and terminology can
be changed on every page and take effect without losing your place.

## Quick start - Python API

```python
import visualmetrics as vm

result = vm.lab("inference.power", effect_size=0.4, n=60, alpha=0.05)

result.figure                    # the primary Plotly figure
result.metric_dict()["power"]    # the numbers behind it
result.evidence                  # EvidenceType.SIMULATION - stated, not implied
print(result.code)               # the exact Python that reproduces this run

for step in result.animations[0].steps:
    print(step.title, "->", step.interpretation)
```

Browse and search:

```python
vm.list_concepts(domain="econometrics")
vm.search("قوة الاختبار")          # search works in any of the three languages
vm.explain("regression.fwl")
vm.concept("inference.power").scenarios
```

## Language switching

```python
import visualmetrics as vm

vm.configure(language="ar")                 # for the session
vm.lab("inference.clt", language="fr")      # for one call
```

In the GUI, use the language selector in the header. Arabic renders
right-to-left throughout, with Latin identifiers (`R-squared`, `p`) isolated so
they read correctly inside Arabic prose.

## Terminology modes

Technical vocabulary can be shown three ways, because a class that will read
English papers needs the English term even when working in Arabic:

| Mode | What you see |
|---|---|
| `translated` | The translated term only |
| `bilingual` | The translated term with the English term beside it |
| `english_technical` | Prose in your language, technical terms in English |

```python
vm.configure(language="ar", terminology="bilingual")
```

## GUI map

| Route | What it holds |
|---|---|
| `/` | Overview, counts, what the badges mean |
| `/catalog` | Every concept, searchable in three languages, planned ones marked |
| `/lab/<concept_id>` | Controls, scenarios, knowledge tabs, compare mode, export |
| `/proofs`, `/proof/<id>` | The proof library and the step-by-step viewer |
| `/glossary` | Every term in three languages |
| `/paths` | Curated learning paths |
| `/doctor` | What is installed and what each missing extra disables |

Inside a lab, the knowledge tabs are drawn from what the lab produced:
overview, intuition, visualize, animate, simulation, diagnostics, compare,
counterexample, assumptions, mathematics, proof, interpretation, common
mistakes, warnings, data, code, self-check and references.

## Learning modes

Set the level to change what is shown, not merely how much:

- **beginner** - intuition and pictures first, algebra hidden;
- **intermediate** - the standard treatment;
- **advanced** - full derivations, diagnostics and edge cases;
- **phd** - asymptotics, regularity conditions and the failure modes.

## Scenario presets

Every lab exposes named scenarios, grouped by what they are for, because the
interesting cases are the ones where a method struggles:

`canonical` · `positive` · `negative` · `null` · `weak` · `strong` ·
`boundary` · `violation` · `counterexample` · `sensitivity` · `small_sample` ·
`large_sample` · `high_noise` · `low_noise` · `robustness` ·
`misspecification` · `compare_methods`

```python
vm.lab("econometrics.heteroskedasticity", scenario="severe")
vm.lab("causal.did", scenario="violated_trends")
```

## Using your own data

**Not supported yet.** Every lab currently generates its own data, and the
loader module is a stub. This is listed here rather than omitted because the
feature is planned and the catalogue's own rule applies to the README too.

The reason it is not first in the queue: most labs exist to compare an estimate
against a truth you control, which your data cannot provide. Where real data
earns its place - diagnostics, model comparison, exploratory work - it is
tracked as an open item, and the `real_data` scenario category is already
reserved for it.

What you can do today is drive a lab from the API and use its output:

```python
import visualmetrics as vm

result = vm.lab("regression.simple_linear", n=200, noise=1.5, seed=7)
frame = result.data          # the simulated sample, as produced
result.metric_dict()         # the numbers computed from it
```

## Exporting

```python
from visualmetrics.export import export_html_report, export_result, export_animation

result = vm.lab("regression.fwl")

export_html_report(result, "fwl-report.html")   # figures + the whole scientific layer
export_result(result, "fwl.png")                # static images (needs kaleido)
export_result(result, "fwl.json")               # everything as data
export_animation(result.animations[0], "fwl-animation.html")
```

Every exporter carries the evidence badge, the assumptions, the warnings and
the frame commentary with the figures. There is no exporter that strips them.

## Reproducing GUI work in Python

Every lab generates the code that reproduces it. In the GUI, open the **Code**
tab; from the API, read `result.code`. Configurations round-trip:

```python
from visualmetrics.export import save_config, load_config

save_config(state, "lesson-1.json")
vm.run_state(load_config("lesson-1.json"))   # same figures, same numbers
```

## Optional extras

| Extra | Enables |
|---|---|
| `gui` | The NiceGUI application |
| `viz` | Plotly and Matplotlib rendering |
| `symbolic` | SymPy derivations |
| `econometrics` | statsmodels, linearmodels, arch cross-checks |
| `causal` | networkx DAG tooling |
| `ai` | scikit-learn, SHAP |
| `proofs` | Manim, for high-quality exported proof animations |
| `notebook` | ipywidgets controls |
| `data` | openpyxl, pyarrow loaders |
| `export` | kaleido, for static image export |
| `all` | everything above |

A missing extra disables only what needs it, and the error says which package
and which install command.

## Troubleshooting

```bash
visualmetrics doctor
```

Reports the Python version, every optional backend with its status and version,
which extra provides it, the available languages, the catalogue counts and
whether the config and cache directories are writable. A package that is
installed but unusable - a NumPy ABI mismatch, say - is reported as such rather
than as missing.

## Notebook usage

```python
import visualmetrics as vm
from visualmetrics import notebook

notebook.setup()                     # results now render as small reports
vm.lab("inference.bootstrap")

notebook.interact("inference.power") # live controls (needs the notebook extra)
```

## Visual proof policy

A picture can carry a proof, but only under conditions:

- the argument must be complete in the general case, not only in the two or
  three dimensions that can be drawn;
- what the picture adds must be *the argument*, not a decoration of it;
- the limits of the picture must be stated - `regression.ols_geometry` says
  plainly that the drawing is exact only in low dimensions while `X'e = 0`
  holds in any.

Anything that fails those conditions is labelled `visual_intuition` or
`visual_derivation`, which are not proof badges.

## Scientific disclaimer

**A simulation is never a proof.** A finite number of finite runs cannot
establish a statement about all sample sizes or all distributions. This is
enforced, not merely intended:

- `ProofSpec` refuses any evidence type that is not `formal_proof`,
  `symbolic_derivation` or `geometric_proof`;
- every proof must state what it does **not** establish, or it will not
  construct;
- numerical checks attached to a proof are reported as
  `numerical_demonstration` and described as tests of the code;
- a lab may not wear a proof badge unless it points at a real proof - a test
  enforces this across the whole catalogue.

The evidence types: `formal_proof`, `symbolic_derivation`, `geometric_proof`,
`visual_derivation`, `visual_intuition`, `simulation`,
`numerical_demonstration`, `counterexample`, `empirical_example`.

## Accessibility

- Nothing is encoded by colour alone: every series carries a dash pattern or a
  marker shape as well, and an assumption's status is shown by icon, word and
  colour together.
- All seven themes meet WCAG AA contrast for body text; `high_contrast` reaches
  21:1. A colourblind-safe palette is included.
- Reduced motion is obeyed as a rule: transitions are disabled outright and
  autoplay is refused, with every frame reachable by hand.
- Every figure carries a text alternative; assumptions expose ARIA status.
- Full keyboard navigation with a visible focus ring; the shortcut map is in
  the header.
- Arabic is right-to-left by architecture, with proper Naskh font stacks.

## Supported Python versions

Python 3.11, 3.12 and 3.13. Developed against 3.11.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: the scientific-honesty rules
are not negotiable, every lab needs violation scenarios as well as a canonical
one, every animation frame needs its explanation, and new numerics need a test
against a closed form or a known property rather than against last week's
output.

## Adding a new concept

1. Add a `ConceptSpec` stub to `catalog/builtin.py` (`status=Status.PLANNED`
   until it is built).
2. Write `concepts/<domain>/<name>.py` exporting `SPEC` and `LAB`, using the
   helpers in `concepts/_kit.py`.
3. Declare controls and at least five scenarios, including one where the
   method fails.
4. Produce panels, metrics, explanations, assumptions and one animation whose
   every frame explains itself.
5. Add the concept text to `i18n/{en,ar,fr}/concepts.json`.
6. Flip the status to `Status.STABLE` and run `pytest`.

The GUI needs no changes: it renders whatever the spec declares.

## Adding a translation

The bundles live in `src/visualmetrics/i18n/<lang>/`: `common.json`,
`concepts.json`, `labs.json`, `proofs.json`, `glossary.json`, `errors.json`.
Keys must match English exactly - a test enforces it.

The open task is `labs.json`: the lab narrative uses about 2,400 keys and only
the shared vocabulary is translated so far, so Arabic and French fall back to
English inside the labs. Contributions there are the most valuable thing anyone
can add.

For a new language, add its directory, register it in
`i18n/translator.py`, and add RTL handling in `i18n/rtl.py` if it is
right-to-left.

## Citation

```bibtex
@software{roudane_visualmetrics,
  author  = {Roudane, Merwan},
  title   = {VisualMetrics: an interactive visual laboratory for statistics,
             econometrics, causal inference and machine learning},
  year     = {2026},
  url      = {https://github.com/merwanroudane/visualmetrics},
  license  = {MIT}
}
```

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).

## Contact

**Dr Merwan Roudane** — <merwanroudane920@gmail.com>
Repository: <https://github.com/merwanroudane/visualmetrics>
Issues: <https://github.com/merwanroudane/visualmetrics/issues>
