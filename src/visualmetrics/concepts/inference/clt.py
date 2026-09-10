"""Central limit theorem lab."""

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
from .lln import PARENT_MOMENTS, PARENTS, draw_parent, parent_mean

__all__ = ["LAB", "SPEC"]


def parent_sd(parent: str, p: float = 0.3) -> float:
    return {
        "normal": 1.0,
        "uniform": float(np.sqrt(1 / 12)),
        "bernoulli": float(np.sqrt(p * (1 - p))),
        "exponential": 1.0,
        "lognormal": float(np.sqrt((np.e - 1) * np.e)),
        "student_t3": float("inf"),
        "pareto": float("inf"),
        "cauchy": float("nan"),
    }[parent]


SPEC = make_spec(
    "inference.clt",
    Domain.INFERENCE,
    "limit_theory",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("parent", "exponential", PARENTS, group="population"),
        slider("bernoulli_p", 0.1, 0.01, 0.99, 0.01, group="population",
               depends_on=("parent", ("bernoulli",))),
        int_slider("n", 30, 1, 1000, 1, group="design"),
        int_slider("reps", 4000, 200, 50000, 100, group="design", expensive=True),
        toggle("standardize", True, group="views"),
        toggle("show_parent", True, group="views"),
        toggle("show_normal", True, group="views"),
        toggle("show_qq", True, group="views"),
        select("statistic", "mean", ("mean", "sum"), group="design"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", parent="exponential", n=30),
        scenario("already_normal", "low_noise", parent="normal", n=5),
        scenario("uniform_fast", "positive", parent="uniform", n=5),
        scenario("skewed_slow", "high_noise", parent="lognormal", n=30),
        scenario("rare_binary", "boundary", parent="bernoulli", bernoulli_p=0.02, n=30),
        scenario("n_equals_one", "boundary", n=1),
        scenario("small_n", "small_sample", n=5, parent="exponential"),
        scenario("large_n", "large_sample", n=500, parent="lognormal"),
        scenario("heavy_tail", "violation", parent="student_t3", n=100),
        scenario("cauchy_counterexample", "counterexample", parent="cauchy", n=200),
        scenario("sum_not_mean", "compare_methods", statistic="sum", parent="uniform", n=12),
    ),
    prerequisites=("inference.lln", "probability.distributions"),
    related=("inference.sampling_distributions", "probability.approximations"),
    next_concepts=("inference.confidence_intervals",),
    tags=("central limit theorem", "asymptotic normality", "sampling distribution"),
    aliases=("clt", "tcl", "theoreme central limite", "نظرية النهاية المركزية"),
    backends=("numpy", "scipy"),
    misconceptions=("clt_makes_data_normal", "n30_rule"),
    references=(
        ref("Billingsley, P. (1995). Probability and Measure.", kind="book"),
        ref("MIT 14.381 Statistical Method in Economics", kind="course",
            url="https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/pages/syllabus/"),
    ),
    curriculum_tags=("dz.stat4", "mit.14381"),
)


class CLTLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        parent = str(p["parent"])
        n = int(p["n"])
        reps = int(p["reps"])
        bp = float(p["bernoulli_p"])
        standardize = bool(p["standardize"])
        use_sum = p["statistic"] == "sum"

        finite_mean, finite_var = PARENT_MOMENTS[parent]
        gen = rng(state.seed, "clt", parent, n)
        draws = draw_parent(gen, parent, (reps, n), bp)
        raw = draws.sum(axis=1) if use_sum else draws.mean(axis=1)

        mu, sd = parent_mean(parent, bp), parent_sd(parent, bp)
        if use_sum:
            centre, scale = mu * n, sd * np.sqrt(n)
        else:
            centre, scale = mu, sd / np.sqrt(n)
        if standardize and np.isfinite(scale) and scale > 0:
            values = (raw - centre) / scale
            theory = stats.norm(0, 1)
            axis_label = ctx.t("labs.clt.axis.standardized", "Standardized statistic")
        else:
            values = raw
            theory = (stats.norm(centre, scale)
                      if np.isfinite(centre) and np.isfinite(scale) and scale > 0 else None)
            axis_label = ctx.t("labs.common.axis.estimate")

        res.dgp = ctx.t(
            "labs.clt.dgp",
            "{reps} independent samples of size n = {n} from a {parent} population; the "
            "{stat} of each sample is one point of the histogram.",
            reps=reps, n=n, parent=parent, stat=p["statistic"],
        )

        res.add_panel(ctx.panel(
            "sampling", self._sampling_figure(ctx, values, theory, axis_label, p, parent, n),
            "labs.clt.figure.sampling", evidence=EvidenceType.SIMULATION,
        ))
        if p["show_parent"]:
            res.add_panel(ctx.panel(
                "parent", self._parent_figure(ctx, draws, parent),
                "labs.clt.figure.parent", tab="data",
                evidence=EvidenceType.SIMULATION,
            ))
        if p["show_qq"]:
            res.add_panel(ctx.panel(
                "qq", self._qq_figure(ctx, values),
                "labs.clt.figure.qq", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        res.add_panel(ctx.panel(
            "ladder", self._ladder_figure(ctx, parent, bp, reps, state.seed, use_sum),
            "labs.clt.figure.ladder", tab="compare", evidence=EvidenceType.SIMULATION,
        ))

        finite_vals = values[np.isfinite(values)]
        skew = float(stats.skew(finite_vals)) if finite_vals.size > 2 else float("nan")
        kurt = float(stats.kurtosis(finite_vals, fisher=True)) if finite_vals.size > 3 else float("nan")
        ks = stats.kstest(finite_vals, "norm") if standardize and finite_vals.size > 5 else None

        res.metric("mean", ctx.t("labs.clt.metric.mean", "Mean of the statistic"),
                   float(np.mean(finite_vals)),
                   reference=0.0 if standardize else centre)
        res.metric("sd", ctx.t("labs.clt.metric.sd", "SD of the statistic"),
                   float(np.std(finite_vals, ddof=1)),
                   reference=1.0 if standardize else scale)
        res.metric("skewness", ctx.t("labs.clt.metric.skewness", "Skewness"), skew,
                   reference=0.0,
                   note=ctx.t("labs.clt.metric.skew_note",
                              "shrinks roughly like 1/sqrt(n) for a mean"))
        res.metric("excess_kurtosis", ctx.t("labs.clt.metric.kurtosis", "Excess kurtosis"),
                   kurt, reference=0.0)
        if ks is not None:
            res.metric("ks_stat", ctx.t("labs.clt.metric.ks",
                                        "Kolmogorov-Smirnov distance to the normal"),
                       float(ks.statistic))

        res.assume("independence", ctx.t("assumptions.independence"), True)
        res.assume("identical_distribution", ctx.t("assumptions.identical_distribution"), True)
        res.assume("finite_variance", ctx.t("assumptions.finite_variance"), finite_var,
                   detail=ctx.t("labs.clt.assume.variance",
                                "The classical central limit theorem needs a finite "
                                "population variance. Without it the limit is a stable law, "
                                "not a normal law."))

        res.animations.append(self._animation(ctx, parent, bp, reps, state.seed, use_sum))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.inference.clt.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"), ctx.t("concepts.inference.clt.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.inference.clt.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.clt.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.inference.clt.warning"),
                    kind="warning")

        if not finite_var:
            res.warnings.append(ctx.t(
                "labs.clt.warn.infinite_variance",
                "This population has infinite variance, so the classical central limit "
                "theorem does not apply. The histogram keeps heavy tails however large n "
                "becomes - this is a counterexample, not slow convergence.",
            ))
        elif abs(skew) > 0.3:
            res.warnings.append(ctx.t(
                "labs.clt.warn.slow",
                "The sampling distribution is still visibly skewed (skewness {s}) at "
                "n = {n}. The '30 is enough' rule of thumb clearly fails for this parent.",
                s=fmt(skew, 2), n=n,
            ))
        return res

    def _sampling_figure(self, ctx, values, theory, axis_label, p, parent, n):
        fig = ctx.figure(
            "labs.clt.figure.sampling",
            xaxis_title=axis_label,
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=430,
        )
        finite = values[np.isfinite(values)]
        lo, hi = np.percentile(finite, [0.5, 99.5]) if finite.size else (-1, 1)
        pad = 0.25 * (hi - lo + 1e-9)
        clipped = finite[(finite >= lo - pad) & (finite <= hi + pad)]
        P.add_histogram(fig, clipped,
                        ctx.t("labs.clt.trace.sampling",
                              "sampling distribution (n = {n})", n=n),
                        "primary", theme=ctx.theme, nbins=60, opacity=0.65)
        if theory is not None and p["show_normal"]:
            x = np.linspace(lo - pad, hi + pad, 400)
            P.add_curve(fig, x, theory.pdf(x),
                        ctx.t("labs.common.trace.normal_ref"), "truth", theme=ctx.theme,
                        dash="solid")
        P.add_legend_note(fig, ctx.t(
            "labs.clt.legend",
            "Each bar counts sample statistics, not observations. The black curve is the "
            "normal law the theorem points at.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _parent_figure(self, ctx, draws, parent):
        fig = ctx.figure(
            "labs.clt.figure.parent",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=310,
        )
        flat = draws.ravel()
        finite = flat[np.isfinite(flat)]
        lo, hi = np.percentile(finite, [0.5, 99.5])
        P.add_histogram(fig, finite[(finite >= lo) & (finite <= hi)],
                        ctx.t("labs.clt.trace.parent", "{parent} population", parent=parent),
                        "secondary", theme=ctx.theme, nbins=60, opacity=0.7)
        P.add_legend_note(fig, ctx.t(
            "labs.clt.legend_parent",
            "The population itself does not change and does not become normal. Only the "
            "distribution of the sample statistic does.",
        ), theme=ctx.theme)
        return fig

    def _qq_figure(self, ctx, values):
        finite = np.sort(values[np.isfinite(values)])
        if finite.size < 5:
            return P.empty_figure(ctx.t("labs.clt.no_qq", "Too few finite values to plot."),
                                  theme=ctx.theme)
        m = finite.size
        probs = (np.arange(1, m + 1) - 0.5) / m
        theo = stats.norm.ppf(probs)
        z = (finite - finite.mean()) / (finite.std(ddof=1) or 1.0)
        fig = ctx.figure(
            "labs.clt.figure.qq",
            xaxis_title=ctx.t("labs.clt.axis.theoretical_q", "Normal quantile"),
            yaxis_title=ctx.t("labs.clt.axis.sample_q", "Sample quantile (standardized)"),
            height=340,
        )
        P.add_points(fig, theo, z, ctx.t("labs.clt.trace.qq", "quantile pairs"),
                     "primary", theme=ctx.theme, size=4, opacity=0.6)
        lim = float(max(abs(theo[0]), abs(theo[-1])))
        P.add_curve(fig, [-lim, lim], [-lim, lim],
                    ctx.t("labs.clt.trace.identity", "perfect normality"),
                    "truth", theme=ctx.theme, dash="solid")
        P.add_legend_note(fig, ctx.t(
            "labs.clt.legend_qq",
            "Points on the diagonal mean the sampling distribution is normal. Curvature at "
            "the ends is exactly where a confidence interval or a p-value would go wrong.",
        ), theme=ctx.theme)
        return fig

    def _ladder_figure(self, ctx, parent, bp, reps, seed, use_sum):
        sizes = (1, 2, 5, 10, 30, 100)
        fig = ctx.figure(
            "labs.clt.figure.ladder",
            xaxis_title=ctx.t("labs.clt.axis.standardized", "Standardized statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=380,
        )
        mu, sd = parent_mean(parent, bp), parent_sd(parent, bp)
        sub = min(reps, 3000)
        for i, ns in enumerate(sizes):
            gen = rng(seed, "clt_ladder", parent, ns)
            draws = draw_parent(gen, parent, (sub, ns), bp)
            raw = draws.sum(axis=1) if use_sum else draws.mean(axis=1)
            if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
                centre = mu * ns if use_sum else mu
                scale = sd * np.sqrt(ns) if use_sum else sd / np.sqrt(ns)
                z = (raw - centre) / scale
            else:
                z = (raw - np.median(raw)) / (stats.iqr(raw) or 1.0)
            z = z[np.isfinite(z)]
            z = z[(z > -5) & (z < 5)]
            hist, edges = np.histogram(z, bins=70, range=(-5, 5), density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])
            P.add_curve(fig, centres, hist, f"n = {ns}",
                        ["muted", "baseline", "info", "secondary", "primary", "positive"][i],
                        theme=ctx.theme, dash="solid", opacity=0.9)
        x = np.linspace(-5, 5, 300)
        P.add_curve(fig, x, stats.norm.pdf(x), ctx.t("labs.common.trace.normal_ref"),
                    "truth", theme=ctx.theme, width=3.0)
        P.add_legend_note(fig, ctx.t(
            "labs.clt.legend_ladder",
            "The same experiment at six sample sizes, all standardized onto one axis. If "
            "the curves stop approaching the black one, a condition of the theorem is failing.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, parent, bp, reps, seed, use_sum):
        go = P.require_plotly()
        sizes = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400]
        mu, sd = parent_mean(parent, bp), parent_sd(parent, bp)
        sub = min(reps, 4000)
        frames, steps = [], []
        edges = np.linspace(-5, 5, 71)
        centres = 0.5 * (edges[1:] + edges[:-1])
        for i, ns in enumerate(sizes):
            gen = rng(seed, "clt_anim", parent, ns)
            draws = draw_parent(gen, parent, (sub, ns), bp)
            raw = draws.sum(axis=1) if use_sum else draws.mean(axis=1)
            if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
                centre = mu * ns if use_sum else mu
                scale = sd * np.sqrt(ns) if use_sum else sd / np.sqrt(ns)
                z = (raw - centre) / scale
            else:
                z = (raw - np.median(raw)) / (stats.iqr(raw) or 1.0)
            z = z[np.isfinite(z)]
            hist, _ = np.histogram(z, bins=edges, density=True)
            core = z[(z > -5) & (z < 5)]
            skew = (float(stats.skew(core))
                    if core.size > 3 and float(np.std(core)) > 1e-9 else float("nan"))
            frames.append(go.Frame(name=str(ns), data=[go.Bar(x=centres, y=hist)]))
            steps.append(AnimationStep(
                id=f"n_{ns}",
                frame=i,
                title=ctx.t("labs.clt.anim.title", "Sample size n = {n}", n=ns),
                what_you_see=ctx.t(
                    "labs.clt.anim.see",
                    "The histogram of {reps} standardized sample {stat}s, with the standard "
                    "normal density drawn on top.", reps=sub,
                    stat="sum" if use_sum else "mean"),
                what_changed=ctx.t("labs.clt.anim.changed",
                                   "Each statistic now averages {n} observations.", n=ns),
                why=ctx.t(
                    "labs.clt.anim.why",
                    "Summing independent draws adds their variances while the standardization "
                    "divides by sqrt(n). The parent's asymmetry contributes a term of order "
                    "1/sqrt(n), which therefore fades."),
                interpretation=ctx.t(
                    "labs.clt.anim.interpret",
                    "Remaining skewness is {s}; the closer to zero, the better a normal "
                    "approximation would work here.", s=fmt(skew, 3)),
                conclusion=ctx.t(
                    "labs.clt.anim.conclude",
                    "The shape of the sampling distribution stops depending on the parent - "
                    "only its mean and variance survive."),
                warning=ctx.t(
                    "labs.clt.anim.warn",
                    "The population has not changed at all. Only the distribution of the "
                    "statistic is approaching normality."),
                math="sqrt(n) * (mean_n - mu) / sigma  ->  N(0, 1) in distribution",
                outputs={"n": ns, "skewness": None if not np.isfinite(skew) else round(skew, 4)},
                active_assumptions=("independence", "identical_distribution"),
                violated_assumptions=() if PARENT_MOMENTS[parent][1] else ("finite_variance",),
                highlighted=("histogram", "normal_curve"),
            ))

        fig = ctx.figure(
            "labs.clt.figure.animation",
            xaxis_title=ctx.t("labs.clt.axis.standardized", "Standardized statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=400,
        )
        fig.add_trace(go.Bar(x=centres, y=np.zeros_like(centres),
                             name=ctx.t("labs.clt.trace.sampling",
                                        "sampling distribution (n = {n})", n=sizes[0]),
                             marker={"color": ctx.color("primary")}, opacity=0.7))
        x = np.linspace(-5, 5, 300)
        P.add_curve(fig, x, stats.norm.pdf(x), ctx.t("labs.common.trace.normal_ref"),
                    "truth", theme=ctx.theme, width=3.0)
        fig.update_yaxes(range=[0, 0.62])
        fig.update_xaxes(range=[-5, 5])
        build_frames(fig, frames, duration=520, reduced_motion=ctx.reduced_motion,
                     slider_label="n")
        return animation(
            "convergence_to_normal",
            fig,
            steps,
            purpose=ctx.t("labs.clt.anim.purpose",
                          "Watch the sampling distribution lose every trace of its parent."),
            summary=ctx.t(
                "labs.clt.anim.summary",
                "As n grows the standardized statistic settles onto the same bell curve "
                "whatever the parent was, provided the parent has a finite variance. What "
                "you watched is Monte Carlo evidence for a limit statement - it is not a "
                "proof, and it says nothing about any single realized sample."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = CLTLab(SPEC)
