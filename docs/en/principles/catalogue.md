# What the catalogue promises

The catalogue lists 63 concepts. 47 are built.
16 are not.

Both numbers are visible in the application, and the unbuilt ones are listed
rather than hidden.

## Why list what does not exist

Hiding them would make the package look more complete than it is. Someone
choosing a tool for a course needs to know what is there, and a catalogue that
quietly omits the gaps makes that judgement impossible.

So a planned concept:

- appears in the catalogue with a **Planned** badge;
- is not clickable;
- refuses to open if you reach its URL directly, with a message saying it is
  catalogued but not built;
- raises `ConceptNotFoundError` with the key `errors.concept_planned` from the
  API.

## What is planned

- `probability.inequalities`
- `probability.order_statistics`
- `inference.convergence_modes`
- `regression.gauss_markov`
- `econometrics.measurement_error`
- `econometrics.quantile_regression`
- `econometrics.gmm`
- `timeseries.ardl`
- `panel.dynamic_gmm`
- `causal.synthetic_control`
- `causal.dml`
- `ml.cross_validation`
- `ai.tokenization`
- `xai.partial_dependence`
- `spatial.autocorrelation`
- `math.optimization`

## Canonical names only

Concepts are named for the science, never for a local syllabus. There is no
"Statistics 3" or "Econometrics 1" in an id, a title, a domain or the
interface - a test enforces it across the whole catalogue.

Mapping to a local curriculum happens through the `curriculum_tags` field,
which is metadata and never appears as a name.

## Every lab shows failure

A lab that only demonstrates the case where a method works teaches the wrong
lesson. Each one therefore declares at least five scenarios, including at least
one where the method breaks - a violation, a counterexample or a boundary case
- and the test suite rejects a lab that offers only the happy path.
