"""Sampling distribution builder.

Population -> sampling scheme -> statistic -> repeated sampling, with the
theoretical sampling distribution overlaid and the finite-population correction
applied when sampling without replacement.
"""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep,
    Domain,
    EvidenceType,
    LabBase,
    LabResult,
    LabState,
    P,
    animation,
    build_frames,
    context,
    fmt,
    int_slider,
    make_spec,
    np,
    ref,
    scenario,
    seed_control,
    select,
    slider,
    toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "STATISTICS"]

STATISTICS = (
    "mean",
    "proportion",
    "variance",
    "sd",
    "median",
    "max",
    "diff_means",
    "diff_proportions",
    "var_ratio",
    "t_statistic",
)
POPULATIONS = ("normal", "uniform", "exponential", "bernoulli", "lognormal", "bimodal")


def _draw_population(gen, population: str, size, mu: float, sigma: float, p: float):
    if population == "normal":
        return gen.normal(mu, sigma, size)
    if population == "uniform":
        half = sigma * np.sqrt(3.0)
        return gen.uniform(mu - half, mu + half, size)
    if population == "exponential":
        return gen.exponential(sigma, size) + (mu - sigma)
    if population == "bernoulli":
        return (gen.random(size) < p).astype(float)
    if population == "lognormal":
        return gen.lognormal(np.log(max(mu, 0.1)), 0.8, size)
    if population == "bimodal":
        pick = gen.random(size) < 0.5
        return np.where(pick, gen.normal(mu - sigma, sigma * 0.4, size),
                        gen.normal(mu + sigma, sigma * 0.4, size))
    return gen.normal(mu, sigma, size)


SPEC = make_spec(
    "inference.sampling_distributions",
    Domain.INFERENCE,
    "sampling_theory",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "code", "quiz", "data", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("population", "normal", POPULATIONS, group="population"),
        slider("mu", 10.0, -20.0, 50.0, 0.1, group="population"),
        slider("sigma", 3.0, 0.1, 20.0, 0.1, group="population"),
        slider("p", 0.4, 0.01, 0.99, 0.01, group="population",
               depends_on=("population", ("bernoulli",))),
        select("statistic", "mean", STATISTICS, group="statistic"),
        int_slider("n", 25, 2, 500, 1, group="design"),
        int_slider("n2", 25, 2, 500, 1, group="design", advanced=True,
                   depends_on=("statistic", ("diff_means", "diff_proportions", "var_ratio"))),
        int_slider("reps", 3000, 100, 40000, 100, group="design", expensive=True),
        toggle("with_replacement", True, group="scheme"),
        int_slider("population_size", 200, 10, 5000, 10, group="scheme",
                   depends_on=("with_replacement", (False,))),
        toggle("show_theory", True, group="views"),
        toggle("show_one_sample", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", statistic="mean", n=25),
        scenario("proportion", "positive", population="bernoulli", statistic="proportion", n=50),
        scenario("variance", "compare_methods", statistic="variance", n=20),
        scenario("difference_of_means", "compare_methods", statistic="diff_means",
                 n=30, n2=30),
        scenario("difference_of_proportions", "compare_methods", population="bernoulli",
                 statistic="diff_proportions", n=60, n2=60),
        scenario("variance_ratio", "compare_methods", statistic="var_ratio", n=20, n2=20),
        scenario("t_statistic", "compare_methods", statistic="t_statistic", n=8),
        scenario("median_vs_mean", "robustness", statistic="median",
                 population="lognormal", n=30),
        scenario("max_order_statistic", "boundary", statistic="max",
                 population="uniform", n=20),
        scenario("small_sample", "small_sample", n=4),
        scenario("large_sample", "large_sample", n=400),
        scenario("finite_population", "sensitivity", with_replacement=False,
                 population_size=60, n=30),
        scenario("skewed_population", "high_noise", population="lognormal", n=15),
    ),
    prerequisites=("probability.distributions",),
    related=("inference.clt", "inference.confidence_intervals"),
    next_concepts=("inference.confidence_intervals", "inference.hypothesis_testing"),
    tags=("sampling distribution", "standard error", "finite population correction",
          "order statistics"),
    aliases=("sampling distribution builder", "distribution d'echantillonnage",
             "توزيع المعاينة", "standard error"),
    backends=("numpy", "scipy"),
    references=(
        ref("Cochran, W. G. (1977). Sampling Techniques.", kind="book"),
        ref("Casella, G. and Berger, R. L. (2002). Statistical Inference.", kind="book"),
    ),
    curriculum_tags=("dz.stat4",),
)


class SamplingLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        stat = str(p["statistic"])
        n, n2 = int(p["n"]), int(p["n2"])
        reps = int(p["reps"])
        pop = str(p["population"])
        mu, sigma, prob = float(p["mu"]), float(p["sigma"]), float(p["p"])
        with_repl = bool(p["with_replacement"])
        N = int(p["population_size"])

        gen = rng(state.seed, "sampling", stat, n)
        finite_pop = None
        if not with_repl:
            finite_pop = _draw_population(rng(state.seed, "finitepop"), pop, N, mu, sigma, prob)

        draws = self._replicate(gen, pop, stat, n, n2, reps, mu, sigma, prob, finite_pop)
        theory, se_theory, target = self._theory(stat, pop, n, n2, mu, sigma, prob,
                                                 with_repl, N)

        res.dgp = ctx.t(
            "labs.sampling.dgp",
            "{reps} samples of size n = {n} drawn {scheme} from a {pop} population; the "
            "{stat} of each sample is one point of the histogram.",
            reps=reps, n=n, pop=pop, stat=stat,
            scheme=("with replacement" if with_repl else f"without replacement from N = {N}"),
        )

        res.add_panel(ctx.panel(
            "sampling", self._sampling_figure(ctx, draws, theory, stat, p, target),
            "labs.sampling.figure.sampling", evidence=EvidenceType.SIMULATION,
        ))
        if p["show_one_sample"]:
            res.add_panel(ctx.panel(
                "one_sample",
                self._one_sample_figure(ctx, gen, pop, n, mu, sigma, prob, finite_pop),
                "labs.sampling.figure.one_sample", tab="data",
                evidence=EvidenceType.SIMULATION,
            ))
        res.add_panel(ctx.panel(
            "se_curve", self._se_curve_figure(ctx, pop, stat, mu, sigma, prob, reps,
                                              state.seed, with_repl, N),
            "labs.sampling.figure.se_curve", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))

        emp_mean = float(np.mean(draws))
        emp_se = float(np.std(draws, ddof=1))
        res.metric("emp_mean", ctx.t("labs.sampling.metric.emp_mean",
                                     "Mean of the statistic across samples"), emp_mean,
                   reference=target)
        res.metric("emp_se", ctx.t("glossary.standard_error.term", "Standard error"), emp_se,
                   reference=se_theory,
                   note=ctx.t("labs.sampling.metric.se_note",
                              "the spread of the statistic, not the spread of the data"))
        res.metric("population_sd", ctx.t("labs.sampling.metric.pop_sd",
                                          "Population standard deviation"),
                   sigma if pop != "bernoulli" else float(np.sqrt(prob * (1 - prob))))
        if np.isfinite(target):
            res.metric("bias", ctx.t("labs.sampling.metric.bias", "Simulated bias"),
                       emp_mean - target, reference=0.0)
        if not with_repl:
            fpc = np.sqrt(max((N - n) / max(N - 1, 1), 0.0))
            res.metric("fpc", ctx.t("labs.sampling.metric.fpc",
                                    "Finite-population correction"), float(fpc),
                       note=ctx.t("labs.sampling.metric.fpc_note",
                                  "multiplies the standard error; it reaches zero at a census"))

        res.assume("independence", ctx.t("assumptions.independence"), with_repl,
                   detail=ctx.t("labs.sampling.assume.independence",
                                "Sampling without replacement makes draws dependent; the "
                                "finite-population correction is exactly that dependence."))
        res.assume("random_sampling", ctx.t("assumptions.random_sampling"), True)
        if stat in ("variance", "sd", "t_statistic"):
            res.assume("normal_errors", ctx.t("assumptions.normal_errors"), pop == "normal",
                       detail=ctx.t("labs.sampling.assume.normal",
                                    "The chi-square, t and F sampling results are exact only "
                                    "for a normal population."))

        res.animations.append(self._animation(ctx, gen, pop, stat, n, n2, mu, sigma, prob,
                                              draws, theory, state.seed, finite_pop))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.sampling_distributions.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.sampling_distributions.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.inference.sampling_distributions.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.sampling_distributions.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.sampling_distributions.warning"), kind="warning")

        if stat == "sd":
            res.warnings.append(ctx.t(
                "labs.sampling.warn.sd_bias",
                "The sample standard deviation is a biased estimator of sigma even though "
                "the sample variance is unbiased: the square root is a concave function, so "
                "Jensen's inequality pulls its expectation down.",
            ))
        if stat == "max":
            res.warnings.append(ctx.t(
                "labs.sampling.warn.max",
                "The sample maximum is never normal, however large n becomes. Extreme order "
                "statistics obey extreme-value theory, not the central limit theorem.",
            ))
        return res

    # -- replication ------------------------------------------------------
    def _replicate(self, gen, pop, stat, n, n2, reps, mu, sigma, prob, finite_pop):
        def sample(size):
            if finite_pop is not None:
                idx = np.array([gen.choice(finite_pop.size, size=size, replace=False)
                                for _ in range(1)])[0]
                return finite_pop[idx]
            return _draw_population(gen, pop, size, mu, sigma, prob)

        out = np.empty(reps)
        for i in range(reps):
            if stat in ("diff_means", "diff_proportions", "var_ratio"):
                a, b = sample(n), sample(n2)
                if stat == "var_ratio":
                    out[i] = np.var(a, ddof=1) / max(np.var(b, ddof=1), 1e-12)
                else:
                    out[i] = a.mean() - b.mean()
                continue
            x = sample(n)
            if stat in ("mean", "proportion"):
                out[i] = x.mean()
            elif stat == "variance":
                out[i] = np.var(x, ddof=1)
            elif stat == "sd":
                out[i] = np.std(x, ddof=1)
            elif stat == "median":
                out[i] = np.median(x)
            elif stat == "max":
                out[i] = x.max()
            elif stat == "t_statistic":
                s = np.std(x, ddof=1)
                out[i] = (x.mean() - mu) / (s / np.sqrt(n)) if s > 0 else 0.0
            else:
                out[i] = x.mean()
        return out

    def _theory(self, stat, pop, n, n2, mu, sigma, prob, with_repl, N):
        """Return (frozen distribution or None, theoretical SE, theoretical centre)."""
        pop_sd = float(np.sqrt(prob * (1 - prob))) if pop == "bernoulli" else sigma
        pop_mean = prob if pop == "bernoulli" else mu
        fpc = 1.0 if with_repl else float(np.sqrt(max((N - n) / max(N - 1, 1), 0.0)))

        if stat in ("mean", "proportion"):
            se = pop_sd / np.sqrt(n) * fpc
            return stats.norm(pop_mean, se), se, pop_mean
        if stat == "diff_means" or stat == "diff_proportions":
            se = np.sqrt(pop_sd**2 / n + pop_sd**2 / n2)
            return stats.norm(0.0, se), se, 0.0
        if stat == "variance" and pop == "normal":
            scale = sigma**2 / (n - 1)
            return stats.chi2(n - 1, scale=scale), float(sigma**2 * np.sqrt(2 / (n - 1))), sigma**2
        if stat == "sd" and pop == "normal":
            import math

            c4 = float(np.sqrt(2 / (n - 1))
                       * np.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2)))
            return None, float(sigma * np.sqrt(max(1 - c4**2, 0.0))), float(sigma * c4)
        if stat == "var_ratio" and pop == "normal":
            return stats.f(n - 1, n2 - 1), float("nan"), 1.0
        if stat == "t_statistic" and pop == "normal":
            return stats.t(n - 1), float(np.sqrt((n - 1) / max(n - 3, 1))), 0.0
        if stat == "median" and pop == "normal":
            se = float(np.sqrt(np.pi / 2) * sigma / np.sqrt(n))
            return stats.norm(pop_mean, se), se, pop_mean
        return None, float("nan"), float("nan")

    # -- figures -----------------------------------------------------------
    def _sampling_figure(self, ctx, draws, theory, stat, p, target):
        fig = ctx.figure(
            "labs.sampling.figure.sampling",
            xaxis_title=ctx.t("labs.sampling.axis.statistic", "Value of the statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=430,
        )
        P.add_histogram(fig, draws,
                        ctx.t("labs.sampling.trace.empirical",
                              "empirical sampling distribution"),
                        "primary", theme=ctx.theme, nbins=60, opacity=0.65)
        if theory is not None and p["show_theory"]:
            lo, hi = np.percentile(draws, [0.2, 99.8])
            pad = 0.3 * (hi - lo + 1e-9)
            x = np.linspace(lo - pad, hi + pad, 400)
            P.add_curve(fig, x, theory.pdf(x),
                        ctx.t("labs.sampling.trace.theory",
                              "theoretical sampling distribution"),
                        "truth", theme=ctx.theme, dash="solid")
        if np.isfinite(target):
            P.add_vline(fig, target, ctx.t("labs.common.trace.truth"), "truth",
                        theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.sampling.legend",
            "Every bar counts whole samples, not observations. The width of this histogram "
            "is the standard error.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _one_sample_figure(self, ctx, gen, pop, n, mu, sigma, prob, finite_pop):
        x = (finite_pop[gen.choice(finite_pop.size, size=min(n, finite_pop.size),
                                   replace=False)]
             if finite_pop is not None else _draw_population(gen, pop, n, mu, sigma, prob))
        fig = ctx.figure(
            "labs.sampling.figure.one_sample",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.observations"),
            height=300,
        )
        P.add_points(fig, x, np.zeros_like(x) + gen.normal(0, 0.03, x.size),
                     ctx.t("labs.sampling.trace.one_sample", "one realized sample"),
                     "secondary", theme=ctx.theme, size=8)
        P.add_vline(fig, float(np.mean(x)),
                    ctx.t("labs.sampling.trace.sample_mean", "this sample's mean"),
                    "primary", theme=ctx.theme, dash="solid")
        fig.update_yaxes(visible=False)
        P.add_legend_note(fig, ctx.t(
            "labs.sampling.legend_one_sample",
            "This is what a researcher actually sees: one sample, one number. The sampling "
            "distribution above describes all the numbers this procedure could have produced.",
        ), theme=ctx.theme)
        return fig

    def _se_curve_figure(self, ctx, pop, stat, mu, sigma, prob, reps, seed, with_repl, N):
        sizes = np.unique(np.round(np.geomspace(2, 400, 14)).astype(int))
        sub = min(reps, 1200)
        emp, theo = [], []
        for ns in sizes:
            gen = rng(seed, "se_curve", ns)
            d = self._replicate(gen, pop, stat, int(ns), int(ns), sub, mu, sigma, prob, None)
            emp.append(float(np.std(d, ddof=1)))
            theo.append(self._theory(stat, pop, int(ns), int(ns), mu, sigma, prob,
                                     with_repl, N)[1])
        fig = ctx.figure(
            "labs.sampling.figure.se_curve",
            xaxis_title=ctx.t("labs.common.axis.sample_size"),
            yaxis_title=ctx.t("glossary.standard_error.term", "Standard error"),
            height=350,
        )
        P.add_curve(fig, sizes, emp, ctx.t("labs.common.trace.empirical"),
                    "primary", theme=ctx.theme)
        if np.any(np.isfinite(theo)):
            P.add_curve(fig, sizes, theo, ctx.t("labs.common.trace.theoretical"),
                        "truth", theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.sampling.legend_se",
            "On log-log axes the classic 1/sqrt(n) rate is a straight line of slope -1/2. "
            "Quadrupling the sample halves the standard error, and no faster.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, gen, pop, stat, n, n2, mu, sigma, prob, draws, theory,
                   seed, finite_pop):
        go = P.require_plotly()
        counts = np.unique(np.round(np.geomspace(5, max(len(draws), 10), 20)).astype(int))
        lo, hi = np.percentile(draws, [0.5, 99.5])
        edges = np.linspace(lo, hi, 46)
        centres = 0.5 * (edges[1:] + edges[:-1])
        frames, steps = [], []
        for i, c in enumerate(counts):
            hist, _ = np.histogram(draws[:c], bins=edges, density=True)
            frames.append(go.Frame(name=str(c), data=[go.Bar(x=centres, y=hist)]))
            partial = draws[:c]
            steps.append(AnimationStep(
                id=f"reps_{c}",
                frame=i,
                title=ctx.t("labs.sampling.anim.title",
                            "{k} samples drawn so far", k=int(c)),
                what_you_see=ctx.t(
                    "labs.sampling.anim.see",
                    "One bar per group of repeated samples: the histogram of the {stat} "
                    "computed from each of them.", stat=stat),
                what_changed=ctx.t("labs.sampling.anim.changed",
                                   "{k} repeated samples have now been drawn.", k=int(c)),
                why=ctx.t(
                    "labs.sampling.anim.why",
                    "Each repetition draws a fresh sample of the same size from the same "
                    "population, so the differences between the bars come from sampling "
                    "variability alone."),
                interpretation=ctx.t(
                    "labs.sampling.anim.interpret",
                    "The histogram's centre is {m} and its spread is {s} - that spread is "
                    "the standard error.",
                    m=fmt(float(np.mean(partial)), 4), s=fmt(float(np.std(partial, ddof=1)), 4)
                    if partial.size > 1 else "0"),
                conclusion=ctx.t(
                    "labs.sampling.anim.conclude",
                    "More repetitions sharpen our picture of the sampling distribution; they "
                    "do NOT make any individual estimate more precise."),
                warning=ctx.t(
                    "labs.sampling.anim.warn",
                    "Repetitions and sample size do different things. Only n moves the "
                    "standard error."),
                math="SE = sd(statistic across repeated samples)",
                outputs={"repetitions": int(c),
                         "mean": round(float(np.mean(partial)), 5),
                         "se": round(float(np.std(partial, ddof=1)), 5)
                         if partial.size > 1 else 0.0},
                active_assumptions=("random_sampling",),
                highlighted=("histogram",),
            ))
        fig = ctx.figure(
            "labs.sampling.figure.animation",
            xaxis_title=ctx.t("labs.sampling.axis.statistic", "Value of the statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=390,
        )
        fig.add_trace(go.Bar(x=centres, y=np.zeros_like(centres),
                             marker={"color": ctx.color("primary")}, opacity=0.7,
                             name=ctx.t("labs.sampling.trace.empirical",
                                        "empirical sampling distribution")))
        if theory is not None:
            x = np.linspace(lo, hi, 300)
            P.add_curve(fig, x, theory.pdf(x), ctx.t("labs.common.trace.theoretical"),
                        "truth", theme=ctx.theme)
        build_frames(fig, frames, duration=380, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.sampling.slider", "samples"))
        return animation(
            "repeated_sampling",
            fig,
            steps,
            purpose=ctx.t("labs.sampling.anim.purpose",
                          "Build a sampling distribution one repeated sample at a time."),
            summary=ctx.t(
                "labs.sampling.anim.summary",
                "A statistic is a random variable because the sample is random. Repeating "
                "the study reveals the distribution of that random variable; a real study "
                "only ever draws one point from it."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = SamplingLab(SPEC)
