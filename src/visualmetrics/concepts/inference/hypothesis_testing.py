"""Hypothesis testing lab: null distribution, rejection region, p-value, CI duality."""

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
    pct,
    ref,
    scenario,
    seed_control,
    select,
    slider,
    toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "TESTS", "run_test"]

TESTS = (
    "z_mean",
    "t_one_sample",
    "t_paired",
    "t_two_sample",
    "welch",
    "proportion",
    "variance_chi2",
    "f_variance_ratio",
    "wilcoxon_signed_rank",
    "mann_whitney",
    "ks_goodness_of_fit",
    "permutation_mean",
)
ALTERNATIVES = ("two_sided", "larger", "smaller")


def run_test(x, y, test: str, mu0: float, sigma0: float, alternative: str, seed: int):
    """Run one test and return (statistic, p-value, null distribution or None)."""
    side = {"two_sided": "two-sided", "larger": "greater", "smaller": "less"}[alternative]
    n = x.size
    if test == "z_mean":
        z = (x.mean() - mu0) / (sigma0 / np.sqrt(n))
        return float(z), _norm_p(z, alternative), stats.norm(0, 1)
    if test in ("t_one_sample", "t_paired"):
        r = stats.ttest_1samp(x if test == "t_one_sample" else x - y, mu0, alternative=side)
        return float(r.statistic), float(r.pvalue), stats.t(n - 1)
    if test == "t_two_sample":
        r = stats.ttest_ind(x, y, equal_var=True, alternative=side)
        return float(r.statistic), float(r.pvalue), stats.t(n + y.size - 2)
    if test == "welch":
        r = stats.ttest_ind(x, y, equal_var=False, alternative=side)
        return float(r.statistic), float(r.pvalue), stats.t(float(r.df))
    if test == "proportion":
        ph = float(x.mean())
        se = np.sqrt(mu0 * (1 - mu0) / n)
        z = (ph - mu0) / se
        return float(z), _norm_p(z, alternative), stats.norm(0, 1)
    if test == "variance_chi2":
        stat = (n - 1) * float(np.var(x, ddof=1)) / sigma0**2
        dist = stats.chi2(n - 1)
        p = (2 * min(dist.cdf(stat), dist.sf(stat)) if alternative == "two_sided"
             else (dist.sf(stat) if alternative == "larger" else dist.cdf(stat)))
        return float(stat), float(min(p, 1.0)), dist
    if test == "f_variance_ratio":
        stat = float(np.var(x, ddof=1) / max(np.var(y, ddof=1), 1e-12))
        dist = stats.f(n - 1, y.size - 1)
        p = (2 * min(dist.cdf(stat), dist.sf(stat)) if alternative == "two_sided"
             else (dist.sf(stat) if alternative == "larger" else dist.cdf(stat)))
        return float(stat), float(min(p, 1.0)), dist
    if test == "wilcoxon_signed_rank":
        r = stats.wilcoxon(x - mu0, alternative=side)
        return float(r.statistic), float(r.pvalue), None
    if test == "mann_whitney":
        r = stats.mannwhitneyu(x, y, alternative=side)
        return float(r.statistic), float(r.pvalue), None
    if test == "ks_goodness_of_fit":
        r = stats.kstest((x - mu0) / max(np.std(x, ddof=1), 1e-12), "norm")
        return float(r.statistic), float(r.pvalue), None
    if test == "permutation_mean":
        gen = rng(seed, "perm")
        obs = float(x.mean() - y.mean())
        pooled = np.concatenate([x, y])
        draws = np.empty(2000)
        for i in range(2000):
            gen.shuffle(pooled)
            draws[i] = pooled[: x.size].mean() - pooled[x.size:].mean()
        if alternative == "two_sided":
            p = float(np.mean(np.abs(draws) >= abs(obs)))
        elif alternative == "larger":
            p = float(np.mean(draws >= obs))
        else:
            p = float(np.mean(draws <= obs))
        return obs, p, draws
    raise ValueError(test)


def _norm_p(z: float, alternative: str) -> float:
    if alternative == "two_sided":
        return float(2 * stats.norm.sf(abs(z)))
    return float(stats.norm.sf(z) if alternative == "larger" else stats.norm.cdf(z))


SPEC = make_spec(
    "inference.hypothesis_testing",
    Domain.INFERENCE,
    "hypothesis_testing",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("test", "t_one_sample", TESTS, group="test"),
        select("alternative", "two_sided", ALTERNATIVES, group="test"),
        slider("alpha", 0.05, 0.001, 0.20, 0.001, group="test"),
        slider("mu0", 0.0, -10.0, 10.0, 0.1, group="hypotheses"),
        slider("true_mu", 0.4, -10.0, 10.0, 0.05, group="hypotheses"),
        slider("sigma0", 1.0, 0.1, 10.0, 0.1, group="hypotheses"),
        slider("true_sigma", 1.0, 0.1, 10.0, 0.1, group="hypotheses"),
        int_slider("n", 30, 3, 1000, 1, group="design"),
        int_slider("n2", 30, 3, 1000, 1, group="design", advanced=True),
        select("population", "normal", ("normal", "uniform", "exponential", "lognormal",
                                        "bernoulli", "contaminated"), group="design"),
        int_slider("reps", 3000, 200, 30000, 100, group="simulation", expensive=True),
        toggle("show_pvalue_distribution", True, group="views"),
        toggle("show_ci_duality", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", test="t_one_sample", true_mu=0.4, n=30),
        scenario("null_true", "null", true_mu=0.0, n=30),
        scenario("clear_effect", "strong", true_mu=1.2, n=40),
        scenario("weak_effect", "weak", true_mu=0.1, n=30),
        scenario("tiny_effect_big_n", "boundary", true_mu=0.05, n=2000),
        scenario("small_sample", "small_sample", n=5, true_mu=0.8),
        scenario("one_sided", "compare_methods", alternative="larger", true_mu=0.4),
        scenario("two_sample", "compare_methods", test="t_two_sample", n=30, n2=30,
                 true_mu=0.5),
        scenario("welch_unequal", "robustness", test="welch", n=10, n2=60,
                 true_sigma=3.0, true_mu=0.5),
        scenario("variance_test", "compare_methods", test="variance_chi2",
                 true_sigma=1.6, n=25),
        scenario("nonparametric", "robustness", test="wilcoxon_signed_rank",
                 population="lognormal", n=25),
        scenario("permutation", "robustness", test="permutation_mean", n=20, n2=20,
                 true_mu=0.6),
        scenario("heavy_contamination", "violation", population="contaminated",
                 test="t_one_sample", n=30, true_mu=0.4),
        scenario("skewed_population", "violation", population="lognormal",
                 test="t_one_sample", n=15),
    ),
    prerequisites=("inference.sampling_distributions",),
    related=("inference.power", "inference.confidence_intervals",
             "inference.neyman_pearson"),
    next_concepts=("inference.power", "inference.multiple_testing"),
    tags=("p-value", "type i error", "critical region", "null distribution", "duality"),
    aliases=("hypothesis test", "test d'hypothese", "اختبار الفرضيات", "p value"),
    backends=("scipy",),
    misconceptions=("p_is_prob_h0", "p_measures_effect", "accept_h0"),
    references=(
        ref("Lehmann, E. L. and Romano, J. P. (2005). Testing Statistical Hypotheses.",
            kind="book"),
        ref("Wasserstein, R. L. and Lazar, N. A. (2016). The ASA statement on p-values. "
            "The American Statistician 70(2).", kind="paper",
            doi="10.1080/00031305.2016.1154108"),
    ),
    curriculum_tags=("dz.stat4",),
)


class TestingLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        test = str(p["test"])
        alt = str(p["alternative"])
        alpha = float(p["alpha"])
        mu0, sigma0 = float(p["mu0"]), float(p["sigma0"])
        true_mu, true_sigma = float(p["true_mu"]), float(p["true_sigma"])
        n, n2 = int(p["n"]), int(p["n2"])
        pop = str(p["population"])

        gen = rng(state.seed, "test", test)
        x = self._draw(gen, pop, n, true_mu, true_sigma)
        y = self._draw(gen, pop, n2, mu0, sigma0)
        stat, pval, null = run_test(x, y, test, mu0, sigma0, alt, state.seed)
        reject = pval < alpha

        res.dgp = ctx.t(
            "labs.test.dgp",
            "n = {n} drawn from a {pop} population centred at {mu} with sd {sd}; "
            "H0 states the parameter equals {mu0}.",
            n=n, pop=pop, mu=fmt(true_mu, 2), sd=fmt(true_sigma, 2), mu0=fmt(mu0, 2),
        )

        res.add_panel(ctx.panel(
            "null", self._null_figure(ctx, null, stat, alpha, alt, pval, test),
            "labs.test.figure.null", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "sample", self._sample_figure(ctx, x, y, mu0, test),
            "labs.test.figure.sample", tab="data",
            evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        if p["show_pvalue_distribution"]:
            emp = self._pvalue_distribution(p, state, pop)
            res.add_panel(ctx.panel(
                "pvalues", self._pvalue_figure(ctx, emp, alpha, true_mu, mu0),
                "labs.test.figure.pvalues", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            res.metric("empirical_rejection",
                       ctx.t("labs.test.metric.rejection", "Empirical rejection rate"),
                       emp["rejection_rate"],
                       reference=alpha if abs(true_mu - mu0) < 1e-9 else None,
                       note=ctx.t("labs.test.metric.rejection_note",
                                  "this is the size under H0 and the power under H1"))
        if p["show_ci_duality"] and test in ("z_mean", "t_one_sample"):
            res.add_panel(ctx.panel(
                "duality", self._duality_figure(ctx, x, mu0, alpha, sigma0, test),
                "labs.test.figure.duality", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("statistic", ctx.t("labs.test.metric.statistic", "Test statistic"), stat)
        res.metric("p_value", ctx.term("glossary.p_value"), pval)
        res.metric("alpha", ctx.term("glossary.type_i_error"), alpha)
        res.metric("decision", ctx.t("labs.test.metric.decision", "Decision at this alpha"),
                   ctx.t("labs.test.reject", "reject H0") if reject
                   else ctx.t("labs.test.fail", "fail to reject H0"))
        if null is not None and not isinstance(null, np.ndarray):
            crit = self._critical(null, alpha, alt)
            for name, value in crit.items():
                if np.isfinite(value):
                    res.metric(name, ctx.t(f"labs.test.metric.{name}",
                                           "Critical value ({name})", name=name), value)

        res.assume("random_sampling", ctx.t("assumptions.random_sampling"), True)
        parametric = test not in ("wilcoxon_signed_rank", "mann_whitney",
                                  "ks_goodness_of_fit", "permutation_mean")
        res.assume("normal_errors", ctx.t("assumptions.normal_errors"),
                   pop == "normal" or not parametric or n >= 30,
                   detail=ctx.t("labs.test.assume.normal",
                                "The t and chi-square null distributions are exact under "
                                "normality; otherwise they are large-sample approximations."))
        if test == "t_two_sample":
            res.assume("equal_variance",
                       ctx.t("labs.test.assume.equal_var_label", "Equal group variances"),
                       abs(np.var(x, ddof=1) - np.var(y, ddof=1)) /
                       max(np.var(y, ddof=1), 1e-9) < 1.0,
                       detail=ctx.t("labs.test.assume.equal_var",
                                    "The pooled t test assumes both groups share a variance; "
                                    "Welch's test does not."),
                       consequence=ctx.t("labs.test.assume.equal_var_consequence",
                                         "With unequal variances and unequal group sizes the "
                                         "pooled test's actual size drifts away from alpha."))
        if test == "variance_chi2":
            res.assume("normal_errors", ctx.t("assumptions.normal_errors"), pop == "normal",
                       detail=ctx.t("labs.test.assume.variance_normal",
                                    "The chi-square variance test is notoriously "
                                    "non-robust: with non-normal data its size is wrong "
                                    "even in large samples."))

        res.animations.append(self._alpha_animation(ctx, null, stat, alt, test))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.hypothesis_testing.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.hypothesis_testing.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.inference.hypothesis_testing.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.hypothesis_testing.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.hypothesis_testing.warning"), kind="warning")
        res.explain("interpretation", ctx.t("ui.conclusion"), ctx.t(
            "labs.test.verdict",
            "p = {p}. Under H0, a statistic at least this extreme occurs {p} of the time. "
            "At alpha = {a} the decision is to {d}. The p-value is not the probability that "
            "H0 is true, and it does not measure how large the effect is.",
            p=fmt(pval, 4), a=fmt(alpha, 3),
            d=ctx.t("labs.test.reject", "reject H0") if reject
            else ctx.t("labs.test.fail", "fail to reject H0"),
        ))
        if not reject:
            res.warnings.append(ctx.t(
                "labs.test.warn.no_evidence",
                "Failing to reject is not evidence for H0. Open the power lab with these "
                "settings to see how large an effect this design could actually detect.",
            ))
        return res

    @staticmethod
    def _draw(gen, pop, size, mu, sigma):
        if pop == "normal":
            return gen.normal(mu, sigma, size)
        if pop == "uniform":
            h = sigma * np.sqrt(3.0)
            return gen.uniform(mu - h, mu + h, size)
        if pop == "exponential":
            return gen.exponential(sigma, size) + (mu - sigma)
        if pop == "lognormal":
            s = 0.9
            return gen.lognormal(np.log(max(abs(mu) + 1.0, 0.1)) - s**2 / 2, s, size) + mu - 1.0
        if pop == "bernoulli":
            return (gen.random(size) < min(max(mu, 0.01), 0.99)).astype(float)
        contaminated = gen.random(size) < 0.08
        return np.where(contaminated, gen.normal(mu, sigma * 8, size),
                        gen.normal(mu, sigma, size))

    @staticmethod
    def _critical(null, alpha, alt):
        if alt == "two_sided":
            return {"crit_low": float(null.ppf(alpha / 2)),
                    "crit_high": float(null.ppf(1 - alpha / 2))}
        if alt == "larger":
            return {"crit_high": float(null.ppf(1 - alpha))}
        return {"crit_low": float(null.ppf(alpha))}

    def _null_figure(self, ctx, null, stat, alpha, alt, pval, test):
        fig = ctx.figure(
            "labs.test.figure.null",
            xaxis_title=ctx.t("labs.common.axis.statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=430,
        )
        if null is None:
            return P.empty_figure(ctx.t(
                "labs.test.no_null",
                "This test uses an exact discrete or rank-based null distribution that is "
                "not drawn here; the p-value in the read-out is exact.",
            ), theme=ctx.theme)
        if isinstance(null, np.ndarray):
            P.add_histogram(fig, null,
                            ctx.t("labs.test.trace.permutation",
                                  "permutation null distribution"),
                            "null", theme=ctx.theme, nbins=60, opacity=0.6)
            P.add_vline(fig, stat, ctx.t("labs.test.trace.observed", "observed"),
                        "highlight", theme=ctx.theme, dash="solid")
            P.add_legend_note(fig, ctx.t(
                "labs.test.legend_permutation",
                "The null distribution here is built by relabelling the data, so it needs no "
                "distributional assumption at all - only exchangeability under H0.",
            ), theme=ctx.theme)
            return fig

        lo, hi = float(null.ppf(0.0005)), float(null.ppf(0.9995))
        lo = min(lo, stat - 0.5)
        hi = max(hi, stat + 0.5)
        x = np.linspace(lo, hi, 800)
        y = null.pdf(x)
        P.add_curve(fig, x, y, ctx.t("labs.test.trace.null", "null distribution"),
                    "null", theme=ctx.theme)
        crit = self._critical(null, alpha, alt)
        reject = np.zeros_like(x, dtype=bool)
        if "crit_high" in crit:
            reject |= x >= crit["crit_high"]
        if "crit_low" in crit:
            reject |= x <= crit["crit_low"]
        P.shade_tail(fig, x, y, reject,
                     ctx.t("labs.test.trace.rejection",
                           "rejection region (area = alpha = {a})", a=fmt(alpha, 3)),
                     "type_i", theme=ctx.theme, alpha=0.4)
        if alt == "two_sided":
            pmask = np.abs(x) >= abs(stat)
        elif alt == "larger":
            pmask = x >= stat
        else:
            pmask = x <= stat
        P.shade_tail(fig, x, y, pmask,
                     ctx.t("labs.test.trace.pvalue", "p-value area = {p}", p=fmt(pval, 4)),
                     "secondary", theme=ctx.theme, alpha=0.25)
        for value in crit.values():
            P.add_vline(fig, value, ctx.t("labs.test.trace.critical", "critical value"),
                        "warning", theme=ctx.theme)
        P.add_vline(fig, stat, ctx.t("labs.test.trace.observed", "observed = {s}",
                                     s=fmt(stat, 3)),
                    "highlight", theme=ctx.theme, dash="solid",
                    annotation_position="bottom")
        P.add_legend_note(fig, ctx.t(
            "labs.test.legend_null",
            "This curve is the behaviour of the statistic IF H0 were true. The red area is "
            "fixed by alpha before seeing data; the purple area is the p-value, read off "
            "after seeing it.",
        ), theme=ctx.theme)
        return fig

    def _sample_figure(self, ctx, x, y, mu0, test):
        two_sample = test in ("t_two_sample", "welch", "mann_whitney", "f_variance_ratio",
                              "permutation_mean")
        fig = ctx.figure(
            "labs.test.figure.sample",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.group"),
            height=300,
        )
        gen = rng(1, "jitter")
        P.add_points(fig, x, 1 + gen.normal(0, 0.05, x.size),
                     ctx.t("labs.test.trace.group1", "group 1"), "primary",
                     theme=ctx.theme, size=8)
        P.add_vline(fig, float(x.mean()),
                    ctx.t("labs.test.trace.mean1", "mean 1 = {m}", m=fmt(float(x.mean()), 3)),
                    "primary", theme=ctx.theme, dash="solid")
        if two_sample:
            P.add_points(fig, y, 2 + gen.normal(0, 0.05, y.size),
                         ctx.t("labs.test.trace.group2", "group 2"), "secondary",
                         theme=ctx.theme, size=8)
            P.add_vline(fig, float(y.mean()),
                        ctx.t("labs.test.trace.mean2", "mean 2"), "secondary",
                        theme=ctx.theme, dash="dash")
        else:
            P.add_vline(fig, mu0, ctx.t("labs.test.trace.mu0", "H0 value = {m}",
                                        m=fmt(mu0, 3)),
                        "null", theme=ctx.theme)
        fig.update_yaxes(visible=False, range=[0.4, 2.6 if two_sample else 1.6])
        P.add_legend_note(fig, ctx.t(
            "labs.test.legend_sample",
            "The whole test compresses this picture into one number. Whether that "
            "compression keeps what matters is a modelling decision, not a computation.",
        ), theme=ctx.theme)
        return fig

    def _pvalue_distribution(self, p, state, pop):
        from ...simulation.monte_carlo import rejection_study

        test, alt = str(p["test"]), str(p["alternative"])
        mu0, sigma0 = float(p["mu0"]), float(p["sigma0"])
        true_mu, true_sigma = float(p["true_mu"]), float(p["true_sigma"])
        n, n2 = int(p["n"]), int(p["n2"])
        reps = min(int(p["reps"]), 3000 if test == "permutation_mean" else 30000)

        def experiment(gen):
            x = self._draw(gen, pop, n, true_mu, true_sigma)
            y = self._draw(gen, pop, n2, mu0, sigma0)
            return run_test(x, y, test, mu0, sigma0, alt, state.seed)[1]

        return rejection_study(experiment, reps, state.seed, float(p["alpha"]))

    def _pvalue_figure(self, ctx, emp, alpha, true_mu, mu0):
        under_null = abs(true_mu - mu0) < 1e-9
        fig = ctx.figure(
            "labs.test.figure.pvalues",
            xaxis_title=ctx.term("glossary.p_value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=350,
        )
        P.add_histogram(fig, emp["p_values"],
                        ctx.t("labs.test.trace.pdist", "p-values across repeated studies"),
                        "primary", theme=ctx.theme, nbins=40)
        P.add_vline(fig, alpha, f"alpha = {fmt(alpha, 3)}", "type_i", theme=ctx.theme)
        if under_null:
            P.add_hline(fig, 1.0, ctx.t("labs.test.trace.uniform",
                                        "uniform density under a valid H0"),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, (
            ctx.t("labs.test.legend_p_null",
                  "When H0 is true and the test is valid, p-values are uniform on [0, 1]. "
                  "A histogram that is not flat means the test's size is wrong.")
            if under_null else
            ctx.t("labs.test.legend_p_alt",
                  "Under a real effect, p-values pile up near zero. The share below alpha "
                  "({r}) is the power of this design.", r=pct(emp["rejection_rate"]))
        ), theme=ctx.theme)
        return fig

    def _duality_figure(self, ctx, x, mu0, alpha, sigma0, test):
        n = x.size
        if test == "z_mean":
            crit = stats.norm.ppf(1 - alpha / 2)
            se = sigma0 / np.sqrt(n)
        else:
            crit = stats.t.ppf(1 - alpha / 2, n - 1)
            se = float(np.std(x, ddof=1)) / np.sqrt(n)
        centre = float(x.mean())
        lo, hi = centre - crit * se, centre + crit * se
        grid = np.linspace(centre - 4 * se, centre + 4 * se, 400)
        pvals = np.array([
            2 * (1 - stats.norm.cdf(abs(centre - g) / se)) if test == "z_mean"
            else 2 * stats.t.sf(abs(centre - g) / se, n - 1)
            for g in grid
        ])
        fig = ctx.figure(
            "labs.test.figure.duality",
            xaxis_title=ctx.t("labs.test.axis.hypothesised", "Hypothesised value"),
            yaxis_title=ctx.term("glossary.p_value"),
            height=350,
        )
        P.add_curve(fig, grid, pvals,
                    ctx.t("labs.test.trace.pcurve", "p-value against each candidate H0"),
                    "primary", theme=ctx.theme)
        P.shade_tail(fig, grid, pvals, pvals >= alpha,
                     ctx.t("labs.test.trace.nonreject",
                           "values NOT rejected = the confidence interval"),
                     "positive", theme=ctx.theme, alpha=0.22)
        P.add_hline(fig, alpha, f"alpha = {fmt(alpha, 3)}", "type_i", theme=ctx.theme)
        P.add_vline(fig, mu0, ctx.t("labs.test.trace.mu0", "H0 value"), "null",
                    theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.test.legend_duality",
            "The confidence interval is exactly the set of null values this test would not "
            "reject: [{lo}, {hi}]. Test and interval are two readings of one calculation.",
            lo=fmt(lo, 3), hi=fmt(hi, 3),
        ), theme=ctx.theme)
        return fig

    def _alpha_animation(self, ctx, null, stat, alt, test):
        go = P.require_plotly()
        if null is None or isinstance(null, np.ndarray):
            null = stats.norm(0, 1)
        alphas = [0.20, 0.15, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001]
        lo, hi = float(null.ppf(0.0005)), float(null.ppf(0.9995))
        x = np.linspace(lo, hi, 500)
        y = null.pdf(x)
        frames, steps = [], []
        for i, a in enumerate(alphas):
            crit = self._critical(null, a, alt)
            mask = np.zeros_like(x, dtype=bool)
            if "crit_high" in crit:
                mask |= x >= crit["crit_high"]
            if "crit_low" in crit:
                mask |= x <= crit["crit_low"]
            xs = np.concatenate([x[mask], x[mask][::-1]]) if mask.any() else np.array([])
            ys = (np.concatenate([y[mask], np.zeros(mask.sum())]) if mask.any()
                  else np.array([]))
            frames.append(go.Frame(name=f"{a:g}", data=[go.Scatter(x=xs, y=ys)]))
            steps.append(AnimationStep(
                id=f"alpha_{a:g}",
                frame=i,
                title=ctx.t("labs.test.anim.title", "alpha = {a}", a=fmt(a, 3)),
                what_you_see=ctx.t("labs.test.anim.see",
                                   "The null distribution with its rejection region shaded."),
                what_changed=ctx.t("labs.test.anim.changed",
                                   "The significance level moved to {a}.", a=fmt(a, 3)),
                why=ctx.t("labs.test.anim.why",
                          "alpha is the area you are willing to leave in the tail; the "
                          "critical value is wherever that area starts."),
                interpretation=ctx.t(
                    "labs.test.anim.interpret",
                    "The observed statistic {s} is {inside} the rejection region at this "
                    "level.", s=fmt(stat, 3),
                    inside=("inside" if (("crit_high" in crit and stat >= crit["crit_high"])
                                         or ("crit_low" in crit and stat <= crit["crit_low"]))
                            else "outside")),
                conclusion=ctx.t(
                    "labs.test.anim.conclude",
                    "A stricter alpha shrinks the rejection region, so fewer results are "
                    "called significant - including some true effects."),
                warning=ctx.t(
                    "labs.test.anim.warn",
                    "alpha must be chosen before looking at the data. Sliding it after the "
                    "fact until the result changes is not inference."),
                math="reject when |T| >= critical value; area of that region under H0 = alpha",
                outputs={"alpha": a, **{k: round(v, 4) for k, v in crit.items()}},
                active_assumptions=("random_sampling",),
                highlighted=("rejection_region", "critical_value"),
            ))
        fig = ctx.figure(
            "labs.test.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=[], y=[], fill="toself",
                                 fillcolor=P.rgba("type_i", 0.4, ctx.theme),
                                 line={"width": 0},
                                 name=ctx.t("labs.test.trace.rejection_short",
                                            "rejection region")))
        P.add_curve(fig, x, y, ctx.t("labs.test.trace.null", "null distribution"),
                    "null", theme=ctx.theme)
        P.add_vline(fig, stat, ctx.t("labs.test.trace.observed", "observed"),
                    "highlight", theme=ctx.theme, dash="solid")
        build_frames(fig, frames, duration=650, reduced_motion=ctx.reduced_motion,
                     slider_label="alpha")
        return animation(
            "alpha_sweep",
            fig,
            steps,
            purpose=ctx.t("labs.test.anim.purpose",
                          "Show that the rejection region is a decision, not a discovery."),
            summary=ctx.t(
                "labs.test.anim.summary",
                "Nothing about the data changed while alpha swept from 0.2 to 0.001 - only "
                "the threshold did. Significance is a property of the pair (evidence, "
                "threshold), and the threshold belongs to you."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = TestingLab(SPEC)
