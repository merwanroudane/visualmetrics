# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-11

### Fixed

- The README used repository-relative paths for its screenshots and file links.
  PyPI renders the description outside the repository, so those resolved to
  nothing: the walkthrough images did not load and the links to `LICENSE`,
  `CONTRIBUTING.md` and `CITATION.cff` returned 404. Every link is now absolute,
  which renders correctly both on the project page and on GitHub.

### Added

- The project page now states the GitHub repository, the documentation site,
  the changelog and the issue tracker in the description itself, not only in
  the sidebar metadata.
- A recorded walkthrough of the application, and a step-by-step guide built
  around it.
- Plotly's own play/pause buttons are removed from any animation shown beside
  its explanation layer: using them advanced the figure while the commentary
  stayed on the previous frame.

## [0.1.0] - 2026-09-11

First release.

### Added

- **Scientific core.** Evidence taxonomy, concept and lab protocol, declarative
  controls and scenarios, serializable lab state, lazy registry, and errors
  that carry a translation key so they can be shown in any language.
- **Numeric engine.** A numpy-native linear-model backend: OLS, WLS, restricted
  least squares and 2SLS, with HC0-HC3, HAC and cluster covariance, plus
  Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, Jarque-Bera and RESET.
  Monte Carlo studies therefore run fast and the base install needs no
  statsmodels.
- **47 labs** across 15 canonical domains, each with 7 to 27 scenarios covering
  violation, counterexample and boundary cases as well as the canonical one.
- **Proof engine and 13 proofs**, each with per-step justifications, named
  assumptions, an explicit statement of what it does not establish, and
  numerical checks of the identities it proves.
- **GUI** (NiceGUI): catalogue, labs with live controls, knowledge tabs,
  animation player with the per-frame explanation layer, proof navigator,
  glossary, learning paths and an environment page. Language, theme, level and
  terminology switch on any page without losing your place.
- **Exporters** for figures, self-contained HTML reports, standalone animations
  and reloadable configurations - each carrying the evidence badge, the
  assumptions and the frame commentary.
- **Notebook integration**: lab results render as small reports in Jupyter,
  with optional ipywidgets controls.
- **Teaching layer**: 189 learning objectives covering every implemented
  concept, 13 misconception entries, and a self-check quiz engine that always
  explains its answers and grades nothing.
- **Trilingual interface** in English, Arabic and French, with true
  right-to-left support, Arabic-aware search normalisation and three
  terminology modes.
- **1008 tests** across unit, integration, scientific-identity,
  translation-alignment, GUI and honesty-regression suites.

### Fixed during development

Recorded because each was a real defect in the science, not only in the code:

- The heteroskedasticity generator used a linear variance ramp that inflated
  the slope variance by about 3% at full strength - far too weak to teach with.
  Replaced with an exponential form; classical test size is now 13.8% at a
  nominal 5%, while HC1 stays near 4.7%.
- ARMA order selection compared information criteria computed on different
  effective samples, so white noise selected ARMA(0,2). Every candidate is now
  scored on identical observations.
- PCA reconstruction error was off by a factor of (n-1)/n against the discarded
  eigenvalues; the identity now holds exactly.
- Gradient descent could diverge without being flagged when the loss grew
  smoothly below the magnitude cutoff.
- Three labs wore a `symbolic_derivation` badge with no proof behind them. The
  three missing proofs were written rather than the badges downgraded.
- A GARCH scenario simulating an integrated process reported nothing unusual,
  because the lab judged only the estimate - which is biased downward. It now
  judges the simulated process as well.
- An exploratory lab ran its bimodal counterexample without reporting that a
  single-centre summary had failed.
- Assumption status could carry a numpy boolean, which is not a `bool` and does
  not survive JSON serialisation.
- The proof renderer looked for `translator.locale`, which does not exist, so
  Arabic prose never received its Latin-isolation marks.

### Known gaps

Stated here rather than omitted:

- Lab narrative text falls back to English in Arabic and French; about 2,400
  `labs.*` keys remain untranslated.
- Loading your own data is not supported yet.
- 16 catalogued concepts are marked `planned` and refuse to open.

[Unreleased]: https://github.com/merwanroudane/visualmetrics/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/merwanroudane/visualmetrics/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/merwanroudane/visualmetrics/releases/tag/v0.1.0
