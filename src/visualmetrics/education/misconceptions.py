"""The misconception library (blueprint section 63).

A misconception entry is not a scolding. It records the *plausible* wrong
belief, says precisely where the reasoning breaks, states the correct claim,
and - where the package can - names the lab and the scenario that makes the
error visible instead of merely asserting it.

Every id referenced by a concept spec must exist here. A dangling reference is
a test failure, not a silent omission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Misconception", "MISCONCEPTIONS", "get", "for_concept", "all_ids"]


@dataclass(frozen=True)
class Misconception:
    """One common wrong belief, with the correction and where to see it."""

    id: str
    claim: str
    """The mistaken belief, stated the way a learner would actually say it."""

    why_wrong: str
    """Where the reasoning goes off - the mechanism, not just 'this is false'."""

    correct: str
    """The accurate statement that replaces it."""

    concept_ids: tuple[str, ...] = ()
    demo: tuple[str, str] | None = None
    """``(concept_id, scenario_id)`` that exhibits the error, when one exists."""

    probe: str = ""
    """A question whose wrong answer reveals the misconception."""

    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label_key(self) -> str:
        return f"education.misconception.{self.id}.claim"


def _m(**kwargs) -> Misconception:
    return Misconception(**kwargs)


MISCONCEPTIONS: dict[str, Misconception] = {
    m.id: m
    for m in (
        _m(
            id="p_is_prob_h0",
            claim="A p-value of 0.03 means there is a 3% chance the null hypothesis is true.",
            why_wrong=(
                "The p-value is computed *assuming* the null is true: it is "
                "P(data at least this extreme | H0), a probability about data. Turning it "
                "into P(H0 | data) reverses the conditioning, which requires a prior and "
                "Bayes' rule. With a plausible prior the posterior probability of the null "
                "is often many times larger than the p-value."
            ),
            correct=(
                "A p-value of 0.03 means: if the null were true, data this extreme or more "
                "would arise 3% of the time."
            ),
            concept_ids=("inference.hypothesis_testing", "inference.bayesian_updating"),
            demo=("inference.hypothesis_testing", "null_true"),
            probe=(
                "Under a true null, what fraction of experiments produce p < 0.05 - and what "
                "does that say about whether a p of 0.04 makes the null unlikely?"
            ),
            tags=("inference", "p_value"),
        ),
        _m(
            id="p_measures_effect",
            claim="A smaller p-value means a bigger effect.",
            why_wrong=(
                "The p-value depends on the effect size *and* on the sample size and the "
                "noise. A trivial effect becomes highly significant with enough data, and a "
                "large effect can fail to reach significance in a small sample. The p-value "
                "answers 'could this be noise?', not 'how big is it?'."
            ),
            correct=(
                "Effect size and precision must be reported separately; a confidence "
                "interval shows both, a p-value shows neither."
            ),
            concept_ids=("inference.hypothesis_testing", "inference.power"),
            demo=("inference.power", "large_sample"),
            probe="Hold the effect fixed and raise n. What happens to the p-value, and why?",
            tags=("inference", "p_value", "effect_size"),
        ),
        _m(
            id="accept_h0",
            claim="A non-significant result means the null hypothesis is true.",
            why_wrong=(
                "Failing to reject is a statement about the evidence, not about the world. "
                "It is exactly what an underpowered study produces when a real effect "
                "exists, so absence of evidence is confounded with evidence of absence "
                "unless power was high."
            ),
            correct=(
                "Say the data are consistent with the null and report the interval: if it "
                "still contains large effects, nothing has been ruled out. Ruling an effect "
                "out requires an equivalence test."
            ),
            concept_ids=("inference.hypothesis_testing", "inference.power"),
            demo=("inference.power", "underpowered"),
            probe=(
                "At 20% power, how often does a real effect produce a non-significant "
                "result? Does that mean the effect is absent?"
            ),
            tags=("inference", "power"),
        ),
        _m(
            id="power_is_one_minus_p",
            claim="Power is one minus the p-value.",
            why_wrong=(
                "Power is a property of the design fixed before the data arrive: the "
                "probability of rejecting when a *specified* alternative is true. The "
                "p-value is a property of the data actually observed. They are computed "
                "under different hypotheses and answer different questions."
            ),
            correct=(
                "Power = P(reject | a specific alternative). It depends on the effect size "
                "you care about, the sample size, the noise and alpha - not on the data you "
                "happened to get."
            ),
            concept_ids=("inference.power",),
            demo=("inference.power", "canonical"),
            probe="Which of these can be computed before collecting any data at all?",
            tags=("inference", "power"),
        ),
        _m(
            id="observed_power",
            claim=(
                "After a non-significant result, computing power at the observed effect "
                "size tells you whether the study was big enough."
            ),
            why_wrong=(
                "Observed power is a deterministic function of the p-value: a p just above "
                "0.05 always yields low observed power, whatever the design. It therefore "
                "adds no information beyond the p-value and cannot explain the "
                "non-significance it was computed from - the reasoning is circular."
            ),
            correct=(
                "Power must be computed for the smallest effect worth detecting, decided "
                "before the study. Afterwards, report the confidence interval instead."
            ),
            concept_ids=("inference.power",),
            demo=("inference.power", "underpowered"),
            probe="If two studies have the same p-value, can their observed power differ?",
            tags=("inference", "power", "post_hoc"),
        ),
        _m(
            id="alpha_free_lunch",
            claim="Lowering alpha to 0.01 simply makes the analysis more rigorous.",
            why_wrong=(
                "Alpha and power trade off at a fixed design. Lowering alpha shrinks false "
                "positives and, holding everything else constant, shrinks power as well - "
                "more real effects get missed. Rigour is a choice about which error is more "
                "costly here, not a dial that only improves things."
            ),
            correct=(
                "Alpha buys protection against Type I error at the price of Type II error. "
                "Choosing it means weighing the two costs; keeping power requires more data."
            ),
            concept_ids=("inference.power", "inference.hypothesis_testing"),
            demo=("inference.power", "strict_alpha"),
            probe="Halve alpha and hold n fixed. What happens to the probability of missing a real effect?",
            tags=("inference", "power", "alpha"),
        ),
        _m(
            id="ci_probability_statement",
            claim="There is a 95% probability that the true parameter lies in this interval.",
            why_wrong=(
                "Once computed, the interval is a fixed pair of numbers and the parameter is "
                "a fixed constant: the parameter either is inside or is not, so no "
                "probability remains. The 95% describes the *procedure* across repeated "
                "samples, not this particular interval."
            ),
            correct=(
                "95% of intervals built this way, over repeated samples, contain the true "
                "value. A probability statement about the parameter itself requires a "
                "Bayesian credible interval and a prior."
            ),
            concept_ids=("inference.confidence_intervals", "inference.bayesian_updating"),
            demo=("inference.confidence_intervals", "coverage_check"),
            probe=(
                "Draw 100 samples and build 100 intervals. What varies from sample to "
                "sample - the interval, or the parameter?"
            ),
            tags=("inference", "interval_estimation"),
        ),
        _m(
            id="clt_makes_data_normal",
            claim="The central limit theorem says that with enough data, the data become normal.",
            why_wrong=(
                "The theorem is about the sampling distribution of an average, not about the "
                "observations. A large sample from a skewed distribution is a large skewed "
                "sample; what becomes approximately normal is the distribution of its mean "
                "across repeated samples."
            ),
            correct=(
                "The standardised sample mean converges to a normal law. The data keep "
                "whatever shape they had, which is why a histogram of the raw data is not a "
                "check of the theorem."
            ),
            concept_ids=("inference.clt", "inference.sampling_distributions"),
            demo=("inference.clt", "skewed_population"),
            probe=(
                "Increase n and watch two histograms: the raw observations and the sample "
                "means. Which one changes shape?"
            ),
            tags=("inference", "asymptotics"),
        ),
        _m(
            id="n30_rule",
            claim="n = 30 is enough for the central limit theorem to apply.",
            why_wrong=(
                "There is no threshold in the theorem, which is a limit statement. How large "
                "n must be depends on the skewness and tail weight of the population: for a "
                "mildly skewed distribution 20 may do, for a lognormal with heavy tails "
                "several hundred may not, and with infinite variance no n suffices."
            ),
            correct=(
                "The required sample size depends on the shape of the population. Check the "
                "approximation for the distribution at hand instead of applying a rule of thumb."
            ),
            concept_ids=("inference.clt",),
            demo=("inference.clt", "heavy_tailed"),
            probe="At n = 30, how close to normal is the mean of a heavily skewed variable?",
            tags=("inference", "asymptotics", "rules_of_thumb"),
        ),
        _m(
            id="gamblers_fallacy",
            claim=(
                "After a run of heads, tails is more likely, because the law of large "
                "numbers has to balance things out."
            ),
            why_wrong=(
                "Independent tosses have no memory: the next toss is still 50/50. The law of "
                "large numbers works by *dilution*, not compensation - the early imbalance "
                "stays, and simply becomes a smaller share of a growing total."
            ),
            correct=(
                "The proportion converges while the absolute difference in counts typically "
                "grows. Nothing pushes the process back toward an even split."
            ),
            concept_ids=("inference.lln",),
            demo=("inference.lln", "long_run"),
            probe=(
                "Watch the running proportion and the running count difference in the same "
                "simulation. Which one settles down?"
            ),
            tags=("probability", "asymptotics"),
        ),
        _m(
            id="lln_is_proof",
            claim="Watching the simulated average settle down proves the law of large numbers.",
            why_wrong=(
                "A simulation shows finitely many paths of finite length. The theorem is a "
                "statement about a limit over all sample sizes for every distribution with a "
                "finite mean; no number of runs can establish it, and a lucky-looking run "
                "cannot refute it either."
            ),
            correct=(
                "The simulation illustrates the theorem; the proof comes from Chebyshev's "
                "inequality or from the strong law. This package labels the two differently "
                "for exactly this reason."
            ),
            concept_ids=("inference.lln", "inference.clt"),
            demo=("inference.lln", "canonical"),
            probe="What would a simulation have to show to *prove* a statement about every n?",
            tags=("epistemology", "simulation"),
        ),
        _m(
            id="high_r2_means_correct",
            claim="A high R-squared means the model is correct.",
            why_wrong=(
                "R-squared measures how much variation the fitted values track in this "
                "sample. It rises mechanically with every added regressor, is high in "
                "spurious regressions between unrelated trending series, and says nothing "
                "about omitted variables, functional form or causality."
            ),
            correct=(
                "R-squared describes fit, not correctness. Specification is judged by "
                "residual diagnostics, out-of-sample performance and the identification "
                "argument - none of which is an R-squared."
            ),
            concept_ids=("regression.simple_linear", "timeseries.stationarity", "ml.bias_variance"),
            demo=("timeseries.cointegration", "spurious"),
            probe=(
                "Two independent random walks are regressed on each other. What R-squared "
                "would you expect, and what does it mean?"
            ),
            tags=("regression", "model_fit"),
        ),
        _m(
            id="extrapolation_is_estimation",
            claim="A fitted line can be used to predict outside the range of the data.",
            why_wrong=(
                "The fit is only supported where there are observations. Outside that range "
                "nothing in the data distinguishes the straight line from any curve that "
                "agrees with it inside - the prediction rests entirely on an untested "
                "assumption of linearity, while the reported standard errors keep behaving "
                "as though it were verified."
            ),
            correct=(
                "Predictions outside the observed range are assumptions, not estimates. "
                "State the assumed functional form explicitly, and widen the uncertainty."
            ),
            concept_ids=("regression.simple_linear", "ml.bias_variance"),
            demo=("regression.simple_linear", "nonlinear_truth"),
            probe=(
                "Fit a line to data on [0, 10] generated by a curve. How wrong is the "
                "prediction at x = 20, and does the standard error warn you?"
            ),
            tags=("regression", "prediction"),
        ),
    )
}


def get(misconception_id: str) -> Misconception:
    """Look one up, failing loudly on an unknown id."""
    try:
        return MISCONCEPTIONS[misconception_id]
    except KeyError as exc:
        raise KeyError(
            f"no misconception is registered under {misconception_id!r}"
        ) from exc


def for_concept(concept_id: str) -> tuple[Misconception, ...]:
    """Every misconception attached to a concept, from both directions.

    A concept spec lists ids it wants shown; a misconception lists concepts it
    belongs to. Both are honoured, so neither side has to be kept in sync by hand.
    """
    from ..core.registry import registry

    ids: list[str] = []
    try:
        spec = registry.lab(concept_id).spec
        ids.extend(spec.misconceptions)
    except Exception:  # noqa: BLE001 - planned or unknown concepts simply have none
        pass
    for entry in MISCONCEPTIONS.values():
        if concept_id in entry.concept_ids and entry.id not in ids:
            ids.append(entry.id)
    return tuple(MISCONCEPTIONS[i] for i in ids if i in MISCONCEPTIONS)


def all_ids() -> tuple[str, ...]:
    return tuple(MISCONCEPTIONS)
