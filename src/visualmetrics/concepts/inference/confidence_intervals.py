"""Confidence interval coverage lab."""

from __future__ import annotations

from typing import Any

from scipy import stats

from ...simulation.random import rng
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

__all__ = ["LAB", "SPEC", "TARGETS", "interval"]

TARGETS = (
    "mean_known_sigma",
    "mean_unknown_sigma",
    "proportion_wald",
    "proportion_wilson",
    "variance",
    "diff_means_equal_var",
    "diff_means_welch",
    "diff_proportions",
    "var_ratio",
)
POPULATIONS = ("normal", "uniform", "exponential", "lognormal", "bernoulli")


def _draw(gen, pop, size, mu, sigma, p):
    if pop == "normal":
        return gen.normal(mu, sigma, size)
    if pop == "uniform":
        h = sigma * np.sqrt(3.0)
        return gen.uniform(mu - h, mu + h, size)
    if pop == "exponential":
        return gen.exponential(sigma, size) + (mu - sigma)
    if pop == "lognormal":
        s = 0.9
        return gen.lognormal(np.log(max(mu, 0.1)) - s**2 / 2, s, size)
    return (gen.random(size) < p).astype(float)


def interval(x, target: str, level: float, sigma: float = 1.0, y=None) -> tuple[float, float]:
    """Return the interval a given procedure produces for this sample."""
    a = 1.0 - level
    n = x.size
    if target == "mean_known_sigma":
        z = stats.norm.ppf(1 - a / 2)
        half = z * sigma / np.sqrt(n)
        return float(x.mean() - half), float(x.mean() + half)
    if target == "mean_unknown_sigma":
        t = stats.t.ppf(1 - a / 2, n - 1)
        half = t * np.std(x, ddof=1) / np.sqrt(n)
        return float(x.mean() - half), float(x.mean() + half)
    if target == "proportion_wald":
        ph = float(x.mean())
        z = stats.norm.ppf(1 - a / 2)
        half = z * np.sqrt(max(ph * (1 - ph), 0.0) / n)
        return ph - half, ph + half
    if target == "proportion_wilson":
        ph = float(x.mean())
        z = stats.norm.ppf(1 - a / 2)
        denom = 1 + z**2 / n
        centre = (ph + z**2 / (2 * n)) / denom
        half = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / denom
        return centre - half, centre + half
    if target == "variance":
        s2 = float(np.var(x, ddof=1))
        lo = (n - 1) * s2 / stats.chi2.ppf(1 - a / 2, n - 1)
        hi = (n - 1) * s2 / stats.chi2.ppf(a / 2, n - 1)
        return float(lo), float(hi)
    if target in ("diff_means_equal_var", "diff_means_welch"):
        n1, n2 = x.size, y.size
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        diff = float(x.mean() - y.mean())
        if target == "diff_means_equal_var":
            sp2 = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
            se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            se = np.sqrt(s1 / n1 + s2 / n2)
            df = (s1 / n1 + s2 / n2) ** 2 / (
                (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1))
        t = stats.t.ppf(1 - a / 2, df)
        return diff - t * se, diff + t * se
    if target == "diff_proportions":
        p1, p2 = float(x.mean()), float(y.mean())
        se = np.sqrt(p1 * (1 - p1) / x.size + p2 * (1 - p2) / y.size)
        z = stats.norm.ppf(1 - a / 2)
        return (p1 - p2) - z * se, (p1 - p2) + z * se
    if target == "var_ratio":
        r = float(np.var(x, ddof=1) / max(np.var(y, ddof=1), 1e-12))
        d1, d2 = x.size - 1, y.size - 1
        return r / stats.f.ppf(1 - a / 2, d1, d2), r / stats.f.ppf(a / 2, d1, d2)
    raise ValueError(target)


SPEC = make_spec(
    "inference.confidence_intervals",
    Domain.INFERENCE,
    "interval_estimation",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("target", "mean_unknown_sigma", TARGETS, group="procedure"),
        select("population", "normal", POPULATIONS, group="population"),
        slider("mu", 10.0, -20.0, 50.0, 0.1, group="population"),
        slider("sigma", 3.0, 0.1, 20.0, 0.1, group="population"),
        slider("p", 0.5, 0.01, 0.99, 0.01, group="population"),
        slider("p2", 0.5, 0.01, 0.99, 0.01, group="population", advanced=True),
        slider("level", 0.95, 0.50, 0.999, 0.005, group="procedure"),
        int_slider("n", 25, 2, 500, 1, group="design"),
        int_slider("n2", 25, 2, 500, 1, group="design", advanced=True),
        int_slider("intervals", 100, 10, 600, 10, group="design"),
        int_slider("coverage_reps", 4000, 200, 40000, 100, group="simulation",
                   expensive=True),
        toggle("show_width_curve", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", target="mean_unknown_sigma", n=25, level=0.95),
        scenario("known_sigma", "compare_methods", target="mean_known_sigma", n=25),
        scenario("tiny_sample", "small_sample", n=3, level=0.95),
        scenario("large_sample", "large_sample", n=400),
        scenario("level_99", "sensitivity", level=0.99),
        scenario("level_80", "sensitivity", level=0.80),
        scenario("wald_undercovers", "counterexample", target="proportion_wald",
                 population="bernoulli", p=0.05, n=30),
        scenario("wilson_repairs", "robustness", target="proportion_wilson",
                 population="bernoulli", p=0.05, n=30),
        scenario("variance_interval", "compare_methods", target="variance", n=20),
        scenario("skewed_population", "violation", population="lognormal", n=15),
        scenario("welch_vs_pooled", "compare_methods", target="diff_means_welch",
                 n=15, n2=45),
        scenario("variance_ratio", "compare_methods", target="var_ratio", n=20, n2=20),
    ),
    prerequisites=("inference.sampling_distributions",),
    related=("inference.hypothesis_testing", "inference.bootstrap"),
    next_concepts=("inference.hypothesis_testing",),
    confused_with=("inference.bayesian_updating",),
    tags=("confidence interval", "coverage", "margin of error", "wilson", "welch"),
    aliases=("ci", "intervalle de confiance", "مجال الثقة", "coverage probability"),
    backends=("scipy",),
    misconceptions=("ci_probability_statement",),
    references=(
        ref("Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a "
            "binomial proportion. Statistical Science 16(2).", kind="paper",
            doi="10.1214/ss/1009213286"),
        ref("Casella, G. and Berger, R. L. (2002). Statistical Inference.", kind="book"),
    ),
    curriculum_tags=("dz.stat4", "ksu.econ416"),
)


class CILab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        target = str(p["target"])
        pop = str(p["population"])
        level = float(p["level"])
        n, n2 = int(p["n"]), int(p["n2"])
        mu, sigma = float(p["mu"]), float(p["sigma"])
        prob, prob2 = float(p["p"]), float(p["p2"])
        m = int(p["intervals"])

        if target in ("proportion_wald", "proportion_wilson", "diff_proportions"):
            pop = "bernoulli"
        true_value = self._true_value(target, mu, sigma, prob, prob2)

        gen = rng(state.seed, "ci", target)
        lows, highs = [], []
        for _ in range(m):
            x = _draw(gen, pop, n, mu, sigma, prob)
            y = _draw(gen, pop, n2, mu, sigma, prob2) if "diff" in target or "ratio" in target else None
            lo, hi = interval(x, target, level, sigma, y)
            lows.append(lo)
            highs.append(hi)
        lows, highs = np.asarray(lows), np.asarray(highs)
        covered = (lows <= true_value) & (true_value <= highs)

        cov = self._coverage(target, pop, level, n, n2, mu, sigma, prob, prob2,
                             int(p["coverage_reps"]), state.seed, true_value)

        res.dgp = ctx.t(
            "labs.ci.dgp",
            "{m} independent samples of size n = {n} from a {pop} population; each sample "
            "produces one {target} interval at level {level}.",
            m=m, n=n, pop=pop, target=target, level=fmt(level, 3),
        )

        res.add_panel(ctx.panel(
            "coverage", self._coverage_figure(ctx, lows, highs, covered, true_value, level),
            "labs.ci.figure.coverage", evidence=EvidenceType.SIMULATION,
        ))
        if p["show_width_curve"]:
            res.add_panel(ctx.panel(
                "width", self._width_figure(ctx, target, pop, level, mu, sigma, prob,
                                            prob2, n2, state.seed),
                "labs.ci.figure.width", tab="compare",
                evidence=EvidenceType.SIMULATION,
            ))
        res.add_panel(ctx.panel(
            "coverage_curve",
            self._coverage_curve_figure(ctx, target, pop, level, mu, sigma, prob, prob2,
                                        n2, state.seed),
            "labs.ci.figure.coverage_curve", tab="diagnostics",
            evidence=EvidenceType.SIMULATION,
        ))

        res.metric("nominal", ctx.t("labs.ci.metric.nominal", "Nominal confidence level"),
                   level)
        res.metric("realized", ctx.t("labs.ci.metric.realized",
                                     "Realized coverage in {k} repetitions", k=cov["reps"]),
                   cov["coverage"], reference=level,
                   note=f"+/- {fmt(1.96 * cov['mc_se'], 4)} (simulation)")
        res.metric("displayed", ctx.t("labs.ci.metric.displayed",
                                      "Covered among the {m} intervals drawn", m=m),
                   f"{int(covered.sum())} / {m}")
        res.metric("width", ctx.t("labs.ci.metric.width", "Average interval width"),
                   cov["width"])
        res.metric("true_value", ctx.t("labs.common.trace.truth"), true_value)

        exact = pop == "normal" and target in ("mean_known_sigma", "mean_unknown_sigma",
                                               "variance", "diff_means_equal_var",
                                               "var_ratio")
        res.assume("random_sampling", ctx.t("assumptions.random_sampling"), True)
        res.assume("normal_errors", ctx.t("assumptions.normal_errors"), pop == "normal",
                   detail=ctx.t("labs.ci.assume.normal",
                                "The t, chi-square and F intervals are exact only for a "
                                "normal population; otherwise their coverage is approximate."))
        if target == "proportion_wald":
            res.assume("regularity", ctx.t("labs.ci.assume.wald_label",
                                           "Normal approximation adequate for this p and n"),
                       not (prob < 0.15 or prob > 0.85) or n > 200,
                       detail=ctx.t("labs.ci.assume.wald",
                                    "The Wald interval uses the estimated proportion in its "
                                    "own standard error, which collapses when p is near 0 or 1."))

        res.animations.append(self._animation(ctx, lows, highs, covered, true_value, level))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.confidence_intervals.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.confidence_intervals.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.inference.confidence_intervals.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.confidence_intervals.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.confidence_intervals.warning"), kind="warning")
        res.explain("interpretation", ctx.t("ui.conclusion"), ctx.t(
            "labs.ci.verdict",
            "This procedure captured the true value {realized} of the time against a "
            "nominal {nominal}. {judgement}",
            realized=pct(cov["coverage"]), nominal=pct(level),
            judgement=(
                ctx.t("labs.ci.verdict_ok", "Coverage matches the promise.")
                if abs(cov["coverage"] - level) < 3 * cov["mc_se"] + 0.005 else
                ctx.t("labs.ci.verdict_bad",
                      "Coverage misses the promise: the interval is not delivering the "
                      "confidence level printed on it.")),
        ))
        if not exact and abs(cov["coverage"] - level) > 0.02:
            res.warnings.append(ctx.t(
                "labs.ci.warn.undercover",
                "Realized coverage is {realized} against a nominal {nominal}. An interval "
                "that under-covers reports more certainty than it has.",
                realized=pct(cov["coverage"]), nominal=pct(level),
            ))
        return res

    @staticmethod
    def _true_value(target, mu, sigma, prob, prob2):
        return {
            "mean_known_sigma": mu,
            "mean_unknown_sigma": mu,
            "proportion_wald": prob,
            "proportion_wilson": prob,
            "variance": sigma**2,
            "diff_means_equal_var": 0.0,
            "diff_means_welch": 0.0,
            "diff_proportions": prob - prob2,
            "var_ratio": 1.0,
        }[target]

    def _coverage(self, target, pop, level, n, n2, mu, sigma, prob, prob2, reps, seed,
                  true_value):
        from ...simulation.monte_carlo import coverage_study

        two_sample = "diff" in target or "ratio" in target

        def experiment(gen):
            x = _draw(gen, pop, n, mu, sigma, prob)
            y = _draw(gen, pop, n2, mu, sigma, prob2) if two_sample else None
            return interval(x, target, level, sigma, y)

        return coverage_study(experiment, reps, seed, true_value)

    def _coverage_figure(self, ctx, lows, highs, covered, true_value, level):
        fig = ctx.figure(
            "labs.ci.figure.coverage",
            xaxis_title=ctx.t("labs.common.axis.replication"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=460,
        )
        go = P.require_plotly()
        idx = np.arange(1, lows.size + 1)
        for mask, role, label in (
            (covered, "positive",
             ctx.t("labs.ci.trace.covered", "covers the true value")),
            (~covered, "negative",
             ctx.t("labs.ci.trace.missed", "misses the true value")),
        ):
            if not mask.any():
                continue
            xs, ys = [], []
            for i in idx[mask]:
                xs += [i, i, None]
                ys += [lows[i - 1], highs[i - 1], None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=label,
                line={"color": ctx.color(role), "width": 2.0},
            ))
            P.add_points(fig, idx[mask], 0.5 * (lows[mask] + highs[mask]),
                         label + " " + ctx.t("labs.ci.trace.centre", "(centre)"),
                         role, theme=ctx.theme, size=4, showlegend=False)
        P.add_hline(fig, true_value,
                    ctx.t("labs.ci.trace.true", "true value = {v}", v=fmt(true_value, 3)),
                    "truth", theme=ctx.theme, dash="solid", width=2.4)
        P.add_legend_note(fig, ctx.t(
            "labs.ci.legend",
            "Each vertical bar is one interval from one sample. The horizontal line is the "
            "fixed truth. About {level} of the bars should cross it - the randomness is in "
            "the bars, never in the line.", level=pct(level),
        ), theme=ctx.theme)
        return fig

    def _width_figure(self, ctx, target, pop, level, mu, sigma, prob, prob2, n2, seed):
        sizes = np.unique(np.round(np.geomspace(3, 400, 14)).astype(int))
        widths = []
        two_sample = "diff" in target or "ratio" in target
        for ns in sizes:
            gen = rng(seed, "ci_width", ns)
            w = []
            for _ in range(200):
                x = _draw(gen, pop, int(ns), mu, sigma, prob)
                y = _draw(gen, pop, n2, mu, sigma, prob2) if two_sample else None
                lo, hi = interval(x, target, level, sigma, y)
                w.append(hi - lo)
            widths.append(float(np.mean(w)))
        fig = ctx.figure(
            "labs.ci.figure.width",
            xaxis_title=ctx.t("labs.common.axis.sample_size"),
            yaxis_title=ctx.t("labs.ci.axis.width", "Average interval width"),
            height=340,
        )
        P.add_curve(fig, sizes, widths,
                    ctx.t("labs.ci.trace.width", "width at level {l}", l=fmt(level, 3)),
                    "primary", theme=ctx.theme)
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.ci.legend_width",
            "Precision is bought with sample size at a rate of 1/sqrt(n): to halve the "
            "margin of error you need roughly four times the data.",
        ), theme=ctx.theme)
        return fig

    def _coverage_curve_figure(self, ctx, target, pop, level, mu, sigma, prob, prob2, n2, seed):
        if target in ("proportion_wald", "proportion_wilson"):
            grid = np.linspace(0.01, 0.5, 40)
            covs = []
            for pv in grid:
                res = self._coverage(target, "bernoulli", level, 30, n2, mu, sigma,
                                     float(pv), prob2, 1200, seed, float(pv))
                covs.append(res["coverage"])
            fig = ctx.figure(
                "labs.ci.figure.coverage_curve",
                xaxis_title=ctx.t("labs.ci.axis.true_p", "True proportion p"),
                yaxis_title=ctx.t("labs.common.axis.coverage"),
                height=340,
            )
            P.add_curve(fig, grid, covs,
                        ctx.t("labs.ci.trace.realized", "realized coverage (n = 30)"),
                        "primary", theme=ctx.theme)
            note = ctx.t(
                "labs.ci.legend_coverage_p",
                "Coverage of a proportion interval oscillates with p because the data are "
                "discrete. The Wald interval dips far below the nominal level near 0 and 1.",
            )
        else:
            sizes = np.unique(np.round(np.geomspace(3, 200, 12)).astype(int))
            covs = []
            true_value = self._true_value(target, mu, sigma, prob, prob2)
            for ns in sizes:
                res = self._coverage(target, pop, level, int(ns), n2, mu, sigma, prob,
                                     prob2, 1200, seed, true_value)
                covs.append(res["coverage"])
            fig = ctx.figure(
                "labs.ci.figure.coverage_curve",
                xaxis_title=ctx.t("labs.common.axis.sample_size"),
                yaxis_title=ctx.t("labs.common.axis.coverage"),
                height=340,
            )
            P.add_curve(fig, sizes, covs,
                        ctx.t("labs.ci.trace.realized_n", "realized coverage"),
                        "primary", theme=ctx.theme)
            fig.update_xaxes(type="log")
            note = ctx.t(
                "labs.ci.legend_coverage_n",
                "A procedure whose coverage sits on the nominal line at every n is exact. "
                "One that only reaches it for large n is asymptotically valid - which is a "
                "weaker promise.",
            )
        P.add_hline(fig, level, ctx.t("labs.ci.trace.nominal", "nominal level"),
                    "truth", theme=ctx.theme, dash="dash")
        fig.update_yaxes(range=[max(0.4, level - 0.35), 1.01])
        P.add_legend_note(fig, note, theme=ctx.theme)
        return fig

    def _animation(self, ctx, lows, highs, covered, true_value, level):
        go = P.require_plotly()
        steps_at = np.unique(np.round(np.linspace(5, lows.size, 20)).astype(int))
        frames, steps = [], []
        for i, k in enumerate(steps_at):
            xs, ys, colors = [], [], []
            for j in range(k):
                xs += [j + 1, j + 1, None]
                ys += [lows[j], highs[j], None]
                colors.append(ctx.color("positive" if covered[j] else "negative"))
            frames.append(go.Frame(name=str(k), data=[
                go.Scatter(x=xs, y=ys, mode="lines",
                           line={"color": ctx.color("primary"), "width": 1.6}),
                go.Scatter(x=np.arange(1, k + 1)[~covered[:k]],
                           y=0.5 * (lows[:k] + highs[:k])[~covered[:k]],
                           mode="markers",
                           marker={"color": ctx.color("negative"), "size": 8,
                                   "symbol": "x"}),
            ]))
            rate = float(covered[:k].mean())
            steps.append(AnimationStep(
                id=f"draw_{k}",
                frame=i,
                title=ctx.t("labs.ci.anim.title", "{k} samples drawn", k=int(k)),
                what_you_see=ctx.t(
                    "labs.ci.anim.see",
                    "One vertical interval per sample, plus a fixed horizontal line at the "
                    "true parameter value."),
                what_changed=ctx.t("labs.ci.anim.changed",
                                   "Samples {a} to {b} were drawn and produced their own "
                                   "intervals.", a=max(1, int(k) - 4), b=int(k)),
                why=ctx.t(
                    "labs.ci.anim.why",
                    "Every sample gives a different estimate and a different standard error, "
                    "so both the centre and the length of the interval move."),
                interpretation=ctx.t(
                    "labs.ci.anim.interpret",
                    "So far {r} of the intervals cross the true line; the target is {l}.",
                    r=pct(rate), l=pct(level)),
                conclusion=ctx.t(
                    "labs.ci.anim.conclude",
                    "The confidence level describes this long-run hit rate of the procedure."),
                warning=ctx.t(
                    "labs.ci.anim.warn",
                    "Pick any single bar: it either crosses the line or it does not. There "
                    "is no probability left in it once it has been computed."),
                math="P(L(X) <= theta <= U(X)) = level, over repeated samples X",
                outputs={"drawn": int(k), "covered": int(covered[:k].sum()),
                         "rate": round(rate, 4)},
                active_assumptions=("random_sampling",),
                highlighted=("intervals", "true_line"),
            ))
        fig = ctx.figure(
            "labs.ci.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.replication"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=400,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line={"color": ctx.color("primary"), "width": 1.6},
                                 name=ctx.t("labs.ci.trace.intervals", "intervals")))
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("negative"), "size": 8,
                                         "symbol": "x"},
                                 name=ctx.t("labs.ci.trace.missed",
                                            "misses the true value")))
        P.add_hline(fig, true_value, ctx.t("labs.common.trace.truth"), "truth",
                    theme=ctx.theme, dash="solid")
        fig.update_xaxes(range=[0, lows.size + 1])
        span = float(np.percentile(highs, 99) - np.percentile(lows, 1))
        fig.update_yaxes(range=[float(np.percentile(lows, 1)) - 0.1 * span,
                                float(np.percentile(highs, 99)) + 0.1 * span])
        build_frames(fig, frames, duration=430, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.ci.slider", "samples"))
        return animation(
            "repeated_sampling_coverage",
            fig,
            steps,
            purpose=ctx.t("labs.ci.anim.purpose",
                          "Make the repeated-sampling meaning of 'confidence' literal."),
            summary=ctx.t(
                "labs.ci.anim.summary",
                "The parameter never moved. The intervals did. A confidence level is a "
                "property of the procedure that generates intervals, and the coverage you "
                "just counted is the only thing the number 95% ever promised."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = CILab(SPEC)
