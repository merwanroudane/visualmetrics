"""Size and power lab.

Two overlapping distributions, four coloured areas, one power curve and one
sample-size curve - all driven by the same handful of parameters. Exact
calculations come from ``scipy``; ``statsmodels`` is used as a cross-check when
available.
"""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep,
    EvidenceType,
    LabBase,
    LabResult,
    LabState,
    P,
    Domain,
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

__all__ = ["LAB", "SPEC", "power_of_test", "required_n"]


TEST_FAMILIES = ("z_one_sample", "t_one_sample", "t_two_sample", "proportion")
ALTERNATIVES = ("two_sided", "larger", "smaller")


# ---------------------------------------------------------------------------
# scientific core
# ---------------------------------------------------------------------------
def _effective_n(n: float, family: str, allocation: float) -> float:
    """Effective sample size behind the standard error of the statistic."""
    if family == "t_two_sample":
        n1 = n * allocation / (1.0 + allocation)
        n2 = n - n1
        return 1.0 / (1.0 / max(n1, 1e-9) + 1.0 / max(n2, 1e-9))
    return n


def power_of_test(
    effect_size: float,
    n: float,
    alpha: float = 0.05,
    *,
    family: str = "t_one_sample",
    alternative: str = "two_sided",
    allocation: float = 1.0,
) -> dict[str, float]:
    """Exact power, critical value(s) and non-centrality for the chosen test.

    ``effect_size`` is standardized (Cohen's d for the t tests, a difference in
    proportions for the proportion test).
    """
    n_eff = _effective_n(n, family, allocation)
    ncp = float(effect_size) * np.sqrt(max(n_eff, 1e-9))

    if family in ("z_one_sample", "proportion"):
        null = stats.norm(0.0, 1.0)
        alt = stats.norm(ncp, 1.0)
        df = float("inf")
    else:
        df = max(n - (2 if family == "t_two_sample" else 1), 1.0)
        null = stats.t(df)
        alt = stats.nct(df, ncp)

    if alternative == "two_sided":
        crit_hi = float(null.ppf(1 - alpha / 2))
        crit_lo = -crit_hi
        power = float(alt.sf(crit_hi) + alt.cdf(crit_lo))
    elif alternative == "larger":
        crit_hi = float(null.ppf(1 - alpha))
        crit_lo = float("-inf")
        power = float(alt.sf(crit_hi))
    else:
        crit_lo = float(null.ppf(alpha))
        crit_hi = float("inf")
        power = float(alt.cdf(crit_lo))

    return {
        "power": power,
        "beta": 1.0 - power,
        "crit_low": crit_lo,
        "crit_high": crit_hi,
        "ncp": ncp,
        "df": df,
        "n_eff": n_eff,
    }


def required_n(
    effect_size: float,
    target_power: float = 0.80,
    alpha: float = 0.05,
    *,
    family: str = "t_one_sample",
    alternative: str = "two_sided",
    allocation: float = 1.0,
    n_max: int = 100_000,
) -> float:
    """Smallest n reaching ``target_power`` (bisection on the exact power curve)."""
    if abs(effect_size) < 1e-9:
        return float("inf")
    lo, hi = 4.0, 64.0
    while hi < n_max:
        got = power_of_test(effect_size, hi, alpha, family=family,
                            alternative=alternative, allocation=allocation)["power"]
        if got >= target_power:
            break
        lo, hi = hi, hi * 2
    else:
        return float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        got = power_of_test(effect_size, mid, alpha, family=family,
                            alternative=alternative, allocation=allocation)["power"]
        if got >= target_power:
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi))


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------
SPEC = make_spec(
    "inference.power",
    Domain.INFERENCE,
    "hypothesis_testing",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        slider("alpha", 0.05, 0.001, 0.20, 0.001,
               help_key="controls.alpha.help", group="test"),
        slider("effect_size", 0.5, 0.0, 2.0, 0.01, group="alternative"),
        int_slider("n", 50, 4, 2000, 1, group="design"),
        select("family", "t_one_sample", TEST_FAMILIES, group="test"),
        select("alternative", "two_sided", ALTERNATIVES, group="test"),
        slider("allocation", 1.0, 0.2, 5.0, 0.05, group="design", advanced=True,
               depends_on=("family", ("t_two_sample",))),
        slider("target_power", 0.80, 0.50, 0.99, 0.01, group="design"),
        slider("observed_statistic", 0.0, -6.0, 6.0, 0.05, group="observed",
               advanced=True),
        toggle("show_observed", False, group="observed"),
        toggle("show_power_curve", True, group="views"),
        toggle("show_sample_size_curve", True, group="views"),
        int_slider("reps", 2000, 200, 40000, 100, group="simulation", expensive=True),
        toggle("run_simulation", True, group="simulation"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", alpha=0.05, effect_size=0.5, n=50),
        scenario("low_power", "weak", alpha=0.05, effect_size=0.2, n=30),
        scenario("high_power", "strong", alpha=0.05, effect_size=0.8, n=120),
        scenario("null_effect", "null", effect_size=0.0, n=60),
        scenario("tiny_effect_huge_n", "boundary", effect_size=0.05, n=2000),
        scenario("small_sample", "small_sample", n=8, effect_size=0.6),
        scenario("large_sample", "large_sample", n=1000, effect_size=0.15),
        scenario("strict_alpha", "sensitivity", alpha=0.001, effect_size=0.5, n=50),
        scenario("one_sided_gain", "compare_methods", alternative="larger",
                 effect_size=0.4, n=60),
        scenario("unbalanced_groups", "sensitivity", family="t_two_sample",
                 allocation=4.0, n=100, effect_size=0.5),
        scenario("two_sample", "compare_methods", family="t_two_sample", n=100,
                 effect_size=0.5),
        scenario("proportion_test", "compare_methods", family="proportion", n=200,
                 effect_size=0.2),
    ),
    prerequisites=("inference.hypothesis_testing",),
    related=("inference.confidence_intervals", "inference.neyman_pearson"),
    next_concepts=("inference.neyman_pearson", "inference.multiple_testing"),
    confused_with=("inference.hypothesis_testing",),
    tags=("power", "effect size", "sample size", "beta", "type ii error", "size"),
    aliases=("statistical power", "puissance statistique", "القوة الإحصائية",
             "القدرة الإحصائية", "power analysis", "sample size calculation"),
    backends=("scipy", "statsmodels"),
    learning_objectives=("understand_power", "distinguish_alpha_beta", "see_effect_of_n"),
    misconceptions=("power_is_one_minus_p", "observed_power", "alpha_free_lunch"),
    references=(
        ref("Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.",
            kind="book"),
        ref("Lehmann, E. L. and Romano, J. P. (2005). Testing Statistical Hypotheses.",
            kind="book"),
        ref("statsmodels power and sample-size documentation", kind="docs",
            url="https://www.statsmodels.org/dev/stats.html#power-and-sample-size-calculations"),
    ),
    curriculum_tags=("dz.stat4", "mit.14381", "ksu.econ416"),
)


# ---------------------------------------------------------------------------
# lab
# ---------------------------------------------------------------------------
class PowerLab(LabBase):
    """The flagship size/power laboratory."""

    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()

        alpha = float(p["alpha"])
        d = float(p["effect_size"])
        n = int(p["n"])
        family = str(p["family"])
        alternative = str(p["alternative"])
        allocation = float(p["allocation"])

        out = power_of_test(d, n, alpha, family=family, alternative=alternative,
                            allocation=allocation)
        power, beta = out["power"], out["beta"]
        null, alt = self._distributions(out, family)

        res.dgp = ctx.t(
            "labs.power.dgp",
            "Test: {family}; H0 effect = 0 against effect = {d}; alpha = {alpha}; n = {n}.",
            family=family, d=fmt(d, 3), alpha=fmt(alpha, 3), n=n,
        )

        # -- main panel: the four areas ------------------------------------
        res.add_panel(ctx.panel(
            "regions",
            self._region_figure(ctx, null, alt, out, alpha, d, n, p),
            "labs.power.figure.regions",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))

        # -- power curve ----------------------------------------------------
        if p["show_power_curve"]:
            res.add_panel(ctx.panel(
                "power_curve",
                self._power_curve_figure(ctx, alpha, d, n, family, alternative, allocation),
                "labs.power.figure.power_curve",
                tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        # -- sample-size curve ----------------------------------------------
        if p["show_sample_size_curve"]:
            res.add_panel(ctx.panel(
                "sample_size",
                self._sample_size_figure(ctx, alpha, d, family, alternative, allocation,
                                         float(p["target_power"])),
                "labs.power.figure.sample_size",
                tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        # -- metrics ---------------------------------------------------------
        need = required_n(d, float(p["target_power"]), alpha, family=family,
                          alternative=alternative, allocation=allocation)
        res.metric("power", ctx.term("glossary.statistical_power"), power)
        res.metric("beta", ctx.term("glossary.type_ii_error"), beta)
        res.metric("alpha", ctx.term("glossary.type_i_error"), alpha)
        res.metric("effect_size", ctx.t("controls.effect_size.label"), d)
        res.metric("ncp", ctx.t("labs.power.metric.ncp", "Non-centrality"), out["ncp"])
        if np.isfinite(out["crit_low"]):
            res.metric("crit_low", ctx.t("labs.power.metric.crit_low", "Lower critical value"),
                       out["crit_low"])
        if np.isfinite(out["crit_high"]):
            res.metric("crit_high", ctx.t("labs.power.metric.crit_high", "Upper critical value"),
                       out["crit_high"])
        res.metric(
            "required_n",
            ctx.t("labs.power.metric.required_n", "n needed for power {target}",
                  target=fmt(p["target_power"], 2)),
            need if np.isfinite(need) else "not attainable",
        )
        if p["show_observed"]:
            t_obs = float(p["observed_statistic"])
            pval = self._p_value(null, t_obs, alternative)
            res.metric("observed", ctx.t("labs.power.metric.observed", "Observed statistic"),
                       t_obs)
            res.metric("p_value", ctx.term("glossary.p_value"), pval,
                       note=ctx.t("labs.power.metric.p_note",
                                  "computed under H0, not a probability that H0 is true"))

        # -- simulation cross-check ------------------------------------------
        if p["run_simulation"]:
            emp = self._simulate(d, n, alpha, family, alternative, allocation,
                                 int(p["reps"]), int(state.seed))
            res.metric(
                "empirical_power",
                ctx.t("labs.power.metric.empirical", "Empirical rejection rate"),
                emp["rejection_rate"],
                reference=power,
                note=ctx.t("ui.simulation_note", reps=emp["reps"]) + f"  +/- {fmt(1.96 * emp['mc_se'], 4)}",
            )
            res.add_panel(ctx.panel(
                "simulation",
                self._simulation_figure(ctx, emp, alpha, power),
                "labs.power.figure.simulation",
                tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))

        # -- animation --------------------------------------------------------
        res.animations.append(self._sample_size_animation(ctx, p, alpha, d, family,
                                                          alternative, allocation))

        # -- assumptions -------------------------------------------------------
        res.assume("independence", ctx.t("assumptions.independence"), True,
                   detail=ctx.t("labs.power.assume.independence",
                                "Observations are independent draws."))
        res.assume("normal_errors", ctx.t("assumptions.normal_errors"),
                   family != "proportion",
                   detail=ctx.t("labs.power.assume.normality",
                                "The t and z calculations assume normal sampling or a "
                                "sample large enough for the central limit theorem."))
        res.assume("known_effect", ctx.t("labs.power.assume.known_effect_label",
                                         "The alternative effect size is assumed, not estimated"),
                   True,
                   detail=ctx.t("labs.power.assume.known_effect",
                                "Power is always power against a specific alternative you chose."))
        if family == "proportion":
            res.warnings.append(ctx.t(
                "labs.power.warn.proportion",
                "The proportion test uses a normal approximation; with a small n or a "
                "probability near 0 or 1 the exact binomial test behaves differently.",
            ))

        # -- explanations --------------------------------------------------------
        self._explain(ctx, res, alpha, beta, power, d, n, need, alternative)
        return res

    # -- pieces ----------------------------------------------------------------
    @staticmethod
    def _distributions(out: dict[str, float], family: str):
        if family in ("z_one_sample", "proportion"):
            return stats.norm(0.0, 1.0), stats.norm(out["ncp"], 1.0)
        return stats.t(out["df"]), stats.nct(out["df"], out["ncp"])

    @staticmethod
    def _p_value(null, t_obs: float, alternative: str) -> float:
        if alternative == "two_sided":
            return float(2 * null.sf(abs(t_obs)))
        if alternative == "larger":
            return float(null.sf(t_obs))
        return float(null.cdf(t_obs))

    def _region_figure(self, ctx, null, alt, out, alpha, d, n, p):
        lo = float(min(null.ppf(0.0005), alt.ppf(0.0005)))
        hi = float(max(null.ppf(0.9995), alt.ppf(0.9995)))
        x = np.linspace(lo, hi, 900)
        y0 = null.pdf(x)
        y1 = alt.pdf(x)

        fig = ctx.figure(
            "labs.power.figure.regions",
            xaxis_title=ctx.t("labs.power.axis.statistic", "Test statistic"),
            yaxis_title=ctx.t("labs.power.axis.density", "Density"),
            height=460,
        )
        P.add_curve(fig, x, y0, ctx.t("labs.power.trace.h0", "H0 distribution"),
                    "null", theme=ctx.theme)
        P.add_curve(fig, x, y1,
                    ctx.t("labs.power.trace.h1", "H1 distribution (effect = {d})", d=fmt(d, 2)),
                    "alternative", theme=ctx.theme)

        crit_lo, crit_hi = out["crit_low"], out["crit_high"]
        reject = np.zeros_like(x, dtype=bool)
        if np.isfinite(crit_hi):
            reject |= x >= crit_hi
        if np.isfinite(crit_lo):
            reject |= x <= crit_lo

        P.shade_tail(fig, x, y0, reject,
                     ctx.t("labs.power.trace.alpha", "Type I error (alpha = {a})",
                           a=fmt(alpha, 3)),
                     "type_i", theme=ctx.theme, alpha=0.45)
        P.shade_tail(fig, x, y1, ~reject,
                     ctx.t("labs.power.trace.beta", "Type II error (beta = {b})",
                           b=fmt(out["beta"], 3)),
                     "type_ii", theme=ctx.theme, alpha=0.35)
        P.shade_tail(fig, x, y1, reject,
                     ctx.t("labs.power.trace.power", "Power = {p}", p=fmt(out["power"], 3)),
                     "power", theme=ctx.theme, alpha=0.35)

        for crit in (crit_lo, crit_hi):
            if np.isfinite(crit):
                P.add_vline(fig, crit,
                            ctx.t("labs.power.trace.critical", "critical value"),
                            "warning", theme=ctx.theme)
        if p["show_observed"]:
            P.add_vline(fig, float(p["observed_statistic"]),
                        ctx.t("labs.power.trace.observed", "observed"),
                        "highlight", theme=ctx.theme, dash="solid",
                        annotation_position="bottom")

        P.add_legend_note(fig, ctx.t(
            "labs.power.legend",
            "Dashed grey = H0; solid blue = H1. Red area = alpha (false alarm), "
            "orange area = beta (missed effect), green area = power.",
        ), theme=ctx.theme)
        return fig

    def _power_curve_figure(self, ctx, alpha, d, n, family, alternative, allocation):
        effects = np.linspace(0.0, max(2.0, abs(d) * 1.5), 80)
        fig = ctx.figure(
            "labs.power.figure.power_curve",
            xaxis_title=ctx.t("controls.effect_size.label"),
            yaxis_title=ctx.term("glossary.statistical_power"),
            height=380,
        )
        for scale, role, label in (
            (0.5, "muted", "n/2"),
            (1.0, "primary", "n"),
            (2.0, "secondary", "2n"),
        ):
            ns = max(4, int(round(n * scale)))
            curve = [
                power_of_test(e, ns, alpha, family=family, alternative=alternative,
                              allocation=allocation)["power"]
                for e in effects
            ]
            P.add_curve(fig, effects, curve,
                        f"{label} = {ns}", role, theme=ctx.theme,
                        dash="solid" if scale == 1.0 else "dash")
        P.add_hline(fig, alpha, ctx.t("labs.power.trace.alpha_floor",
                                      "power = alpha when the effect is zero"),
                    "type_i", theme=ctx.theme, dash="dot")
        P.add_vline(fig, abs(d), ctx.t("labs.power.trace.current", "current effect"),
                    "highlight", theme=ctx.theme)
        fig.update_yaxes(range=[0, 1.02])
        P.add_legend_note(fig, ctx.t(
            "labs.power.legend_power_curve",
            "Every curve starts at alpha: with no effect, a valid test rejects exactly "
            "alpha of the time. Doubling n shifts the curve left, it does not lift its floor.",
        ), theme=ctx.theme)
        return fig

    def _sample_size_figure(self, ctx, alpha, d, family, alternative, allocation, target):
        effects = np.linspace(0.08, 1.6, 45)
        needed = [
            required_n(e, target, alpha, family=family, alternative=alternative,
                       allocation=allocation)
            for e in effects
        ]
        fig = ctx.figure(
            "labs.power.figure.sample_size",
            xaxis_title=ctx.t("controls.effect_size.label"),
            yaxis_title=ctx.t("labs.power.axis.required_n", "n required"),
            height=380,
        )
        P.add_curve(fig, effects, needed,
                    ctx.t("labs.power.trace.required", "n for power {t}", t=fmt(target, 2)),
                    "primary", theme=ctx.theme)
        if abs(d) > 0.05:
            need = required_n(d, target, alpha, family=family, alternative=alternative,
                              allocation=allocation)
            if np.isfinite(need):
                P.add_points(fig, [abs(d)], [need],
                             ctx.t("labs.power.trace.current", "current effect"),
                             "highlight", theme=ctx.theme, size=13)
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.power.legend_sample_size",
            "The axis is logarithmic: required n grows like 1/effect^2, so halving the "
            "effect you want to detect roughly quadruples the sample you need.",
        ), theme=ctx.theme)
        return fig

    def _simulate(self, d, n, alpha, family, alternative, allocation, reps, seed):
        from ...simulation.monte_carlo import rejection_study

        n = int(n)
        if family == "t_two_sample":
            n1 = max(2, int(round(n * allocation / (1 + allocation))))
            n2 = max(2, n - n1)

            def experiment(gen):
                a = gen.standard_normal(n1) + d
                b = gen.standard_normal(n2)
                t, pv = stats.ttest_ind(a, b, equal_var=True)
                return self._one_sided(pv, t, alternative)
        elif family == "proportion":
            p0, p1 = 0.5, min(max(0.5 + d / 2, 0.01), 0.99)

            def experiment(gen):
                x = gen.random(n) < p1
                phat = x.mean()
                se = np.sqrt(p0 * (1 - p0) / n)
                z = (phat - p0) / se
                pv = 2 * stats.norm.sf(abs(z))
                return self._one_sided(pv, z, alternative)
        else:

            def experiment(gen):
                x = gen.standard_normal(n) + d
                t, pv = stats.ttest_1samp(x, 0.0)
                return self._one_sided(pv, t, alternative)

        return rejection_study(experiment, reps, seed, alpha)

    @staticmethod
    def _one_sided(pv: float, stat: float, alternative: str) -> float:
        if alternative == "two_sided":
            return float(pv)
        half = float(pv) / 2
        if alternative == "larger":
            return half if stat > 0 else 1.0 - half
        return half if stat < 0 else 1.0 - half

    def _simulation_figure(self, ctx, emp, alpha, theoretical):
        fig = ctx.figure(
            "labs.power.figure.simulation",
            xaxis_title=ctx.term("glossary.p_value"),
            yaxis_title=ctx.t("labs.power.axis.frequency", "Frequency"),
            height=360,
        )
        pvals = emp["p_values"]
        P.add_histogram(fig, pvals, ctx.t("labs.power.trace.pvalues", "simulated p-values"),
                        "primary", theme=ctx.theme, nbins=40, density=True)
        P.add_vline(fig, alpha, f"alpha = {fmt(alpha, 3)}", "type_i", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.power.legend_simulation",
            "Share of p-values below alpha = {emp} (theoretical power {theory}). "
            "Under a true null this histogram would be flat.",
            emp=pct(emp["rejection_rate"]), theory=pct(theoretical),
        ), theme=ctx.theme)
        return fig

    def _sample_size_animation(self, ctx, p, alpha, d, family, alternative, allocation):
        go = P.require_plotly()
        base_n = int(p["n"])
        grid = np.unique(np.round(np.geomspace(max(4, base_n // 8),
                                               max(12, base_n * 6), 22)).astype(int))
        frames, steps = [], []
        x = None
        for i, ns in enumerate(grid):
            out = power_of_test(d, ns, alpha, family=family, alternative=alternative,
                                allocation=allocation)
            null, alt = self._distributions(out, family)
            if x is None:
                lo = float(min(null.ppf(0.0005), -4))
                hi = float(max(alt.ppf(0.9995) + 1.0, 6))
                x = np.linspace(lo, hi, 500)
            frames.append(
                go.Frame(
                    name=str(ns),
                    data=[
                        go.Scatter(x=x, y=null.pdf(x)),
                        go.Scatter(x=x, y=alt.pdf(x)),
                    ],
                )
            )
            steps.append(
                AnimationStep(
                    id=f"n_{ns}",
                    frame=i,
                    title=ctx.t("labs.power.anim.title", "Sample size n = {n}", n=ns),
                    what_you_see=ctx.t(
                        "labs.power.anim.see",
                        "Two sampling distributions of the same test statistic: under the "
                        "null (no effect) and under the alternative (effect = {d}).",
                        d=fmt(d, 2),
                    ),
                    what_changed=ctx.t(
                        "labs.power.anim.changed",
                        "n moved to {n}, so the non-centrality is now {ncp}.",
                        n=ns, ncp=fmt(out["ncp"], 2),
                    ),
                    why=ctx.t(
                        "labs.power.anim.why",
                        "The statistic is standardized by the standard error, which shrinks "
                        "like 1/sqrt(n). The same real effect therefore sits further from "
                        "zero in standardized units.",
                    ),
                    interpretation=ctx.t(
                        "labs.power.anim.interpret",
                        "The alternative curve slides right while the critical value barely "
                        "moves, so more of its mass falls in the rejection region.",
                    ),
                    conclusion=ctx.t(
                        "labs.power.anim.conclude",
                        "Power is now {power}; the probability of missing this effect is {beta}.",
                        power=pct(out["power"]), beta=pct(out["beta"]),
                    ),
                    warning=ctx.t(
                        "labs.power.anim.warn",
                        "alpha did not change: a larger sample buys power, never a lower "
                        "false-alarm rate.",
                    ),
                    math="power = P(reject | effect = d) with non-centrality d*sqrt(n)",
                    outputs={"n": int(ns), "power": round(out["power"], 4),
                             "beta": round(out["beta"], 4), "alpha": alpha},
                    active_assumptions=("independence", "normal_errors"),
                    highlighted=("h1_curve", "power_region"),
                )
            )

        out0 = power_of_test(d, int(grid[0]), alpha, family=family,
                             alternative=alternative, allocation=allocation)
        null0, alt0 = self._distributions(out0, family)
        fig = ctx.figure(
            "labs.power.figure.animation",
            xaxis_title=ctx.t("labs.power.axis.statistic", "Test statistic"),
            yaxis_title=ctx.t("labs.power.axis.density", "Density"),
            height=420,
        )
        P.add_curve(fig, x, null0.pdf(x), ctx.t("labs.power.trace.h0", "H0 distribution"),
                    "null", theme=ctx.theme)
        P.add_curve(fig, x, alt0.pdf(x), ctx.t("labs.power.trace.h1", "H1 distribution"),
                    "alternative", theme=ctx.theme)
        crit = out0["crit_high"] if np.isfinite(out0["crit_high"]) else out0["crit_low"]
        if np.isfinite(crit):
            P.add_vline(fig, crit, ctx.t("labs.power.trace.critical", "critical value"),
                        "warning", theme=ctx.theme)
        fig.update_yaxes(range=[0, 0.55])
        build_frames(fig, frames, duration=520, reduced_motion=ctx.reduced_motion,
                     slider_label="n")

        return animation(
            "sample_size_effect",
            fig,
            steps,
            purpose=ctx.t(
                "labs.power.anim.purpose",
                "Show exactly what a larger sample does - and does not do - to a test.",
            ),
            summary=ctx.t(
                "labs.power.anim.summary",
                "Increasing n concentrates both sampling distributions. The null keeps its "
                "shape in standardized units, the alternative moves away from it, and power "
                "rises towards one. alpha never moves: it is a choice, not a consequence.",
            ),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )

    def _explain(self, ctx, res, alpha, beta, power, d, n, need, alternative):
        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.inference.power.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.power.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.inference.power.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.power.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.power.warning"), kind="warning")

        verdict_key, default = (
            ("labs.power.verdict.low",
             "With alpha = {alpha}, an effect of {d} and n = {n}, power is only {power}. "
             "You would miss this effect {beta} of the time. Reaching power {target} needs "
             "n = {need}.")
            if power < 0.5 else
            ("labs.power.verdict.ok",
             "With alpha = {alpha}, an effect of {d} and n = {n}, power is {power}: you "
             "would still miss the effect {beta} of the time.")
        )
        res.explain(
            "interpretation", ctx.t("ui.conclusion"),
            ctx.t(verdict_key, default, alpha=fmt(alpha, 3), d=fmt(d, 2), n=n,
                  power=pct(power), beta=pct(beta), target="0.80",
                  need=("not attainable" if not np.isfinite(need) else int(need))),
        )
        if alternative != "two_sided":
            res.explain(
                "assumptions", ctx.t("ui.warning"),
                ctx.t("labs.power.note.one_sided",
                      "A one-sided test buys power only in the direction you chose. If the "
                      "true effect goes the other way, this test can never detect it - and "
                      "choosing the direction after seeing the data invalidates alpha."),
                kind="warning",
            )


LAB = PowerLab(SPEC)
