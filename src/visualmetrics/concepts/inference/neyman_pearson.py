"""Neyman-Pearson lemma lab: the likelihood-ratio ordering and why it is optimal."""

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

__all__ = ["LAB", "SPEC"]

FAMILIES = ("normal_shift", "normal_scale", "exponential_rate", "poisson_rate")

SPEC = make_spec(
    "inference.neyman_pearson",
    Domain.INFERENCE,
    "test_theory",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.SYMBOLIC_DERIVATION,
    controls=(
        select("family", "normal_shift", FAMILIES, group="model"),
        slider("theta0", 0.0, -3.0, 3.0, 0.05, group="hypotheses"),
        slider("theta1", 1.0, -3.0, 6.0, 0.05, group="hypotheses"),
        slider("alpha", 0.05, 0.001, 0.30, 0.001, group="test"),
        int_slider("n", 1, 1, 100, 1, group="design"),
        slider("threshold", 1.0, 0.05, 20.0, 0.05, group="test", advanced=True),
        toggle("use_alpha_threshold", True, group="test"),
        toggle("show_competitor", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", theta0=0.0, theta1=1.0, alpha=0.05, n=1),
        scenario("well_separated", "strong", theta1=3.0, n=1),
        scenario("barely_separated", "weak", theta1=0.25, n=1),
        scenario("larger_sample", "large_sample", n=25, theta1=0.5),
        scenario("strict_alpha", "sensitivity", alpha=0.005),
        scenario("loose_alpha", "sensitivity", alpha=0.20),
        scenario("scale_alternative", "compare_methods", family="normal_scale",
                 theta0=1.0, theta1=2.0),
        scenario("exponential", "compare_methods", family="exponential_rate",
                 theta0=1.0, theta1=0.5),
        scenario("poisson", "compare_methods", family="poisson_rate", theta0=2.0,
                 theta1=4.0, n=5),
        scenario("bad_competitor", "counterexample", show_competitor=True, alpha=0.10),
    ),
    prerequisites=("inference.hypothesis_testing", "inference.power"),
    related=("inference.power", "inference.mle"),
    tags=("neyman-pearson", "likelihood ratio", "most powerful test", "rejection region"),
    aliases=("np lemma", "lemme de neyman-pearson", "نيمان-بيرسون"),
    backends=("scipy",),
    proof_ids=("inference.neyman_pearson.lemma",),
    references=(
        ref("Neyman, J. and Pearson, E. S. (1933). On the problem of the most efficient "
            "tests of statistical hypotheses. Phil. Trans. R. Soc. A 231.", kind="paper",
            doi="10.1098/rsta.1933.0009"),
        ref("Lehmann, E. L. and Romano, J. P. (2005). Testing Statistical Hypotheses.",
            kind="book"),
    ),
    curriculum_tags=("mit.14381",),
)


class NeymanPearsonLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        family = str(p["family"])
        t0, t1 = float(p["theta0"]), float(p["theta1"])
        alpha = float(p["alpha"])
        n = int(p["n"])
        f0, f1, support, discrete = self._models(family, t0, t1, n)

        x = (np.arange(0, int(max(f0.ppf(0.9999), f1.ppf(0.9999)) + 5))
             if discrete else np.linspace(support[0], support[1], 1200))
        d0 = f0.pmf(x) if discrete else f0.pdf(x)
        d1 = f1.pmf(x) if discrete else f1.pdf(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(d0 > 0, d1 / d0, np.inf)

        if p["use_alpha_threshold"]:
            k, crit = self._threshold_for_alpha(x, d0, lr, alpha, discrete)
        else:
            k = float(p["threshold"])
            crit = lr >= k
        size = float(np.sum(d0[crit]) if discrete else np.trapezoid(d0 * crit, x))
        power = float(np.sum(d1[crit]) if discrete else np.trapezoid(d1 * crit, x))

        comp_crit, comp_size, comp_power = self._competitor(x, d0, d1, size, discrete)

        res.dgp = ctx.t(
            "labs.np.dgp",
            "Simple H0 (parameter = {t0}) against simple H1 (parameter = {t1}), "
            "sample size n = {n}, size constrained to alpha = {a}.",
            t0=fmt(t0, 2), t1=fmt(t1, 2), n=n, a=fmt(alpha, 3),
        )

        res.add_panel(ctx.panel(
            "densities", self._density_figure(ctx, x, d0, d1, crit, comp_crit, p,
                                              size, power, comp_power, discrete),
            "labs.np.figure.densities", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "ratio", self._ratio_figure(ctx, x, lr, k, crit),
            "labs.np.figure.ratio", tab="math",
            evidence=EvidenceType.SYMBOLIC_DERIVATION,
        ))
        res.add_panel(ctx.panel(
            "roc", self._roc_figure(ctx, x, d0, d1, lr, size, power, comp_size, comp_power,
                                    discrete),
            "labs.np.figure.roc", tab="compare",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))

        res.metric("threshold", ctx.t("labs.np.metric.k", "Likelihood-ratio threshold k"), k)
        res.metric("size", ctx.t("labs.np.metric.size", "Size of the test"), size,
                   reference=alpha)
        res.metric("power", ctx.term("glossary.statistical_power"), power)
        if p["show_competitor"]:
            res.metric("competitor_size", ctx.t("labs.np.metric.comp_size",
                                                "Size of the competing region"), comp_size,
                       reference=size)
            res.metric("competitor_power", ctx.t("labs.np.metric.comp_power",
                                                 "Power of the competing region"), comp_power)
            res.metric("power_gap", ctx.t("labs.np.metric.gap",
                                          "Power sacrificed by the competitor"),
                       power - comp_power,
                       note=ctx.t("labs.np.metric.gap_note",
                                  "never negative - that is the lemma"))

        res.assume("simple_hypotheses", ctx.t("labs.np.assume.simple_label",
                                              "Both hypotheses are simple"), True,
                   detail=ctx.t("labs.np.assume.simple",
                                "The lemma is stated for two fully specified distributions. "
                                "For composite alternatives a uniformly most powerful test "
                                "needs additional structure such as a monotone likelihood ratio."))
        res.assume("same_size", ctx.t("labs.np.assume.size_label",
                                      "Competing tests are compared at equal size"),
                   abs(comp_size - size) < 0.01,
                   detail=ctx.t("labs.np.assume.size",
                                "Comparing power is only meaningful between tests with the "
                                "same false-alarm rate."))

        res.animations.append(self._animation(ctx, x, d0, d1, lr, discrete))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.neyman_pearson.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.neyman_pearson.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.inference.neyman_pearson.math"), kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.np.proof",
            "Let R be the likelihood-ratio region {x: f1(x) >= k f0(x)} with size alpha, and "
            "let S be any other region with size at most alpha. Consider the difference "
            "(indicator of R minus indicator of S) times (f1 - k f0). Inside R the second "
            "factor is non-negative and the first is non-negative; outside R the second "
            "factor is negative and the first is non-positive. The product is therefore "
            "non-negative everywhere, so its integral is non-negative: "
            "power(R) - power(S) >= k (size(R) - size(S)) >= 0. Hence no region of the same "
            "size has more power.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.neyman_pearson.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.neyman_pearson.warning"), kind="warning")
        res.explain("interpretation", ctx.t("ui.conclusion"), ctx.t(
            "labs.np.verdict",
            "At size {s}, the likelihood-ratio region achieves power {p}; the competing "
            "region of the same size achieves {c}. The gap of {g} is not a numerical "
            "accident - the lemma guarantees it can never be negative.",
            s=pct(size), p=pct(power), c=pct(comp_power), g=pct(power - comp_power),
        ))
        return res

    @staticmethod
    def _models(family, t0, t1, n):
        if family == "normal_shift":
            se = 1.0 / np.sqrt(n)
            f0, f1 = stats.norm(t0, se), stats.norm(t1, se)
            lo = min(t0, t1) - 5 * se
            hi = max(t0, t1) + 5 * se
            return f0, f1, (lo, hi), False
        if family == "normal_scale":
            s0, s1 = max(abs(t0), 0.2), max(abs(t1), 0.2)
            f0, f1 = stats.norm(0, s0 / np.sqrt(n)), stats.norm(0, s1 / np.sqrt(n))
            hi = 4 * max(s0, s1)
            return f0, f1, (-hi, hi), False
        if family == "exponential_rate":
            r0, r1 = max(abs(t0), 0.05), max(abs(t1), 0.05)
            f0 = stats.gamma(n, scale=1.0 / r0)
            f1 = stats.gamma(n, scale=1.0 / r1)
            return f0, f1, (1e-4, float(max(f0.ppf(0.999), f1.ppf(0.999)))), False
        f0 = stats.poisson(max(t0, 0.05) * n)
        f1 = stats.poisson(max(t1, 0.05) * n)
        return f0, f1, (0, 0), True

    @staticmethod
    def _threshold_for_alpha(x, d0, lr, alpha, discrete):
        order = np.argsort(-lr)
        mass = d0[order] if discrete else None
        if discrete:
            cum = np.cumsum(mass)
            idx = int(np.searchsorted(cum, alpha))
            idx = min(idx, order.size - 1)
            k = float(lr[order[idx]])
        else:
            dx = np.gradient(x)
            cum = np.cumsum(d0[order] * dx[order])
            idx = int(np.searchsorted(cum, alpha))
            idx = min(idx, order.size - 1)
            k = float(lr[order[idx]])
        return k, lr >= k

    @staticmethod
    def _competitor(x, d0, d1, size, discrete):
        """A same-size region built from the WRONG ordering (the left tail)."""
        order = np.argsort(x)
        if discrete:
            cum = np.cumsum(d0[order])
        else:
            dx = np.gradient(x)
            cum = np.cumsum(d0[order] * dx[order])
        idx = int(np.searchsorted(cum, size))
        idx = min(max(idx, 0), order.size - 1)
        crit = np.zeros_like(x, dtype=bool)
        crit[order[: idx + 1]] = True
        if discrete:
            return crit, float(d0[crit].sum()), float(d1[crit].sum())
        return (crit, float(np.trapezoid(d0 * crit, x)), float(np.trapezoid(d1 * crit, x)))

    def _density_figure(self, ctx, x, d0, d1, crit, comp_crit, p, size, power, comp_power,
                        discrete):
        fig = ctx.figure(
            "labs.np.figure.densities",
            xaxis_title=ctx.t("labs.np.axis.data", "Sufficient statistic"),
            yaxis_title=(ctx.t("labs.common.axis.probability") if discrete
                         else ctx.t("labs.common.axis.density")),
            height=430,
        )
        if discrete:
            P.add_bar(fig, x, d0, ctx.t("labs.np.trace.f0", "density under H0"),
                      "null", theme=ctx.theme, opacity=0.6)
            P.add_bar(fig, x, d1, ctx.t("labs.np.trace.f1", "density under H1"),
                      "alternative", theme=ctx.theme, opacity=0.6)
        else:
            P.add_curve(fig, x, d0, ctx.t("labs.np.trace.f0", "density under H0"),
                        "null", theme=ctx.theme)
            P.add_curve(fig, x, d1, ctx.t("labs.np.trace.f1", "density under H1"),
                        "alternative", theme=ctx.theme)
            P.shade_tail(fig, x, d0, crit,
                         ctx.t("labs.np.trace.alpha", "size = {s}", s=pct(size)),
                         "type_i", theme=ctx.theme, alpha=0.42)
            P.shade_tail(fig, x, d1, crit,
                         ctx.t("labs.np.trace.power", "power = {p}", p=pct(power)),
                         "power", theme=ctx.theme, alpha=0.3)
            if p["show_competitor"]:
                P.shade_tail(fig, x, d1, comp_crit,
                             ctx.t("labs.np.trace.comp",
                                   "competing region of the same size, power = {p}",
                                   p=pct(comp_power)),
                             "warning", theme=ctx.theme, alpha=0.22)
        P.add_legend_note(fig, ctx.t(
            "labs.np.legend",
            "Both shaded regions spend exactly the same alpha. The likelihood-ratio region "
            "spends it where H1 is relatively most likely, which is why it buys more power.",
        ), theme=ctx.theme)
        return fig

    def _ratio_figure(self, ctx, x, lr, k, crit):
        fig = ctx.figure(
            "labs.np.figure.ratio",
            xaxis_title=ctx.t("labs.np.axis.data", "Sufficient statistic"),
            yaxis_title=ctx.t("labs.np.axis.lr", "Likelihood ratio f1 / f0"),
            height=350,
        )
        finite = np.isfinite(lr)
        P.add_curve(fig, x[finite], lr[finite],
                    ctx.t("labs.np.trace.lr", "likelihood ratio"), "primary",
                    theme=ctx.theme)
        P.add_hline(fig, k, ctx.t("labs.np.trace.k", "threshold k = {k}", k=fmt(k, 3)),
                    "warning", theme=ctx.theme)
        if crit.any():
            edges = x[crit]
            P.add_annotation(fig, float(np.median(edges)), float(k * 1.6),
                             ctx.t("labs.np.trace.reject_here", "reject here"),
                             role="type_i", theme=ctx.theme)
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.np.legend_ratio",
            "The lemma says: sort the sample space by this ratio and fill the rejection "
            "region from the top down until the size budget is spent. Nothing else to choose.",
        ), theme=ctx.theme)
        return fig

    def _roc_figure(self, ctx, x, d0, d1, lr, size, power, comp_size, comp_power, discrete):
        order = np.argsort(-lr)
        if discrete:
            sizes = np.cumsum(d0[order])
            powers = np.cumsum(d1[order])
        else:
            dx = np.gradient(x)
            sizes = np.cumsum(d0[order] * dx[order])
            powers = np.cumsum(d1[order] * dx[order])
        fig = ctx.figure(
            "labs.np.figure.roc",
            xaxis_title=ctx.t("labs.common.axis.false_positive"),
            yaxis_title=ctx.t("labs.common.axis.true_positive"),
            height=380,
        )
        P.add_curve(fig, sizes, powers,
                    ctx.t("labs.np.trace.envelope",
                          "likelihood-ratio tests (the power envelope)"),
                    "primary", theme=ctx.theme)
        P.add_curve(fig, [0, 1], [0, 1], ctx.t("labs.np.trace.chance", "coin flip"),
                    "baseline", theme=ctx.theme, dash="dash")
        P.add_points(fig, [size], [power], ctx.t("labs.np.trace.current", "current test"),
                     "power", theme=ctx.theme, size=12)
        P.add_points(fig, [comp_size], [comp_power],
                     ctx.t("labs.np.trace.comp_point", "competing region"),
                     "warning", theme=ctx.theme, size=11)
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1.02])
        P.add_legend_note(fig, ctx.t(
            "labs.np.legend_roc",
            "Every possible test is a point in this square. The curve is the best achievable "
            "power at each size, and the lemma proves no test can sit above it.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, x, d0, d1, lr, discrete):
        go = P.require_plotly()
        order = np.argsort(-lr)
        if discrete:
            cum0 = np.cumsum(d0[order])
            cum1 = np.cumsum(d1[order])
        else:
            dx = np.gradient(x)
            cum0 = np.cumsum(d0[order] * dx[order])
            cum1 = np.cumsum(d1[order] * dx[order])
        targets = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        frames, steps = [], []
        for i, a in enumerate(targets):
            idx = int(min(np.searchsorted(cum0, a), order.size - 1))
            mask = np.zeros_like(x, dtype=bool)
            mask[order[: idx + 1]] = True
            xs = x[mask]
            ys = d1[mask]
            if xs.size:
                poly_x = np.concatenate([xs, xs[::-1]])
                poly_y = np.concatenate([ys, np.zeros_like(ys)])
            else:
                poly_x = poly_y = np.array([])
            frames.append(go.Frame(name=f"{a:g}", data=[go.Scatter(x=poly_x, y=poly_y)]))
            steps.append(AnimationStep(
                id=f"alpha_{a:g}",
                frame=i,
                title=ctx.t("labs.np.anim.title", "Size budget alpha = {a}", a=fmt(a, 3)),
                what_you_see=ctx.t(
                    "labs.np.anim.see",
                    "The rejection region, filled in decreasing order of the likelihood ratio."),
                what_changed=ctx.t("labs.np.anim.changed",
                                   "The size budget rose to {a}, so more sample points are "
                                   "admitted.", a=fmt(a, 3)),
                why=ctx.t("labs.np.anim.why",
                          "The greedy rule admits whichever remaining point offers the most "
                          "H1 probability per unit of H0 probability - which is precisely "
                          "the likelihood ratio."),
                interpretation=ctx.t("labs.np.anim.interpret",
                                     "Power is now {p} for a false-alarm rate of {s}.",
                                     p=pct(float(cum1[idx])), s=pct(float(cum0[idx]))),
                conclusion=ctx.t("labs.np.anim.conclude",
                                 "This greedy filling traces the power envelope: no other "
                                 "region of this size can do better."),
                warning=ctx.t("labs.np.anim.warn",
                              "Optimality here is for these two simple hypotheses only."),
                math="reject when f1(x) / f0(x) >= k, with k fixed by size(R) = alpha",
                outputs={"alpha": a, "size": round(float(cum0[idx]), 4),
                         "power": round(float(cum1[idx]), 4)},
                active_assumptions=("simple_hypotheses",),
                highlighted=("rejection_region",),
            ))
        fig = ctx.figure(
            "labs.np.figure.animation",
            xaxis_title=ctx.t("labs.np.axis.data", "Sufficient statistic"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=[], y=[], fill="toself",
                                 fillcolor=P.rgba("power", 0.35, ctx.theme),
                                 line={"width": 0},
                                 name=ctx.t("labs.np.trace.region", "rejection region")))
        P.add_curve(fig, x, d0, ctx.t("labs.np.trace.f0", "density under H0"), "null",
                    theme=ctx.theme)
        P.add_curve(fig, x, d1, ctx.t("labs.np.trace.f1", "density under H1"),
                    "alternative", theme=ctx.theme)
        build_frames(fig, frames, duration=620, reduced_motion=ctx.reduced_motion,
                     slider_label="alpha")
        return animation(
            "greedy_region", fig, steps,
            purpose=ctx.t("labs.np.anim.purpose",
                          "Show the lemma as a greedy filling rule rather than a formula."),
            summary=ctx.t(
                "labs.np.anim.summary",
                "The optimal rejection region is not a shape you guess; it is whatever the "
                "likelihood ratio ranks highest, cut off wherever the size budget runs out. "
                "That single rule is the entire content of the lemma."),
            evidence=EvidenceType.SYMBOLIC_DERIVATION,
        )


LAB = NeymanPearsonLab(SPEC)
