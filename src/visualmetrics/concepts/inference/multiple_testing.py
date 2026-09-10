"""Multiple testing: family-wise error, false discovery rate and p-hacking."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "adjust"]

METHODS = ("none", "bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli")


def adjust(pvals: np.ndarray, method: str) -> np.ndarray:
    """Return adjusted p-values (compare with alpha directly)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return p
    if method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p)
    ranked = p[order]
    out = np.empty(m)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method in ("benjamini_hochberg", "benjamini_yekutieli"):
        factor = 1.0
        if method == "benjamini_yekutieli":
            factor = float(np.sum(1.0 / np.arange(1, m + 1)))
        scaled = ranked * m * factor / np.arange(1, m + 1)
        adj = np.minimum.accumulate(scaled[::-1])[::-1]
    else:
        adj = ranked
    out[order] = np.minimum(adj, 1.0)
    return out


SPEC = make_spec(
    "inference.multiple_testing",
    Domain.INFERENCE,
    "multiple_testing",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "experiment", "compare", "simulate", "diagnose",
           "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        int_slider("m", 100, 2, 5000, 1, group="design"),
        int_slider("n_true_effects", 10, 0, 500, 1, group="design"),
        slider("effect_size", 0.6, 0.0, 2.0, 0.01, group="design"),
        int_slider("n", 30, 3, 500, 1, group="design"),
        slider("alpha", 0.05, 0.001, 0.20, 0.001, group="test"),
        select("method", "benjamini_hochberg", METHODS, group="test"),
        int_slider("reps", 400, 50, 5000, 50, group="simulation", expensive=True),
        toggle("show_all_methods", True, group="views"),
        toggle("show_phacking", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", m=100, n_true_effects=10, method="benjamini_hochberg"),
        scenario("all_null", "null", n_true_effects=0, method="none"),
        scenario("all_null_bonferroni", "robustness", n_true_effects=0, method="bonferroni"),
        scenario("no_correction", "counterexample", method="none", n_true_effects=5),
        scenario("bonferroni_conservative", "compare_methods", method="bonferroni",
                 m=1000, n_true_effects=50, effect_size=0.4),
        scenario("holm_gains", "compare_methods", method="holm", m=1000,
                 n_true_effects=50, effect_size=0.4),
        scenario("fdr_gains", "compare_methods", method="benjamini_hochberg", m=1000,
                 n_true_effects=50, effect_size=0.4),
        scenario("genomics_scale", "large_sample", m=5000, n_true_effects=100,
                 effect_size=0.5),
        scenario("weak_effects", "weak", effect_size=0.15, n_true_effects=20),
        scenario("strong_effects", "strong", effect_size=1.2, n_true_effects=20),
    ),
    prerequisites=("inference.hypothesis_testing",),
    related=("inference.power",),
    tags=("bonferroni", "holm", "fdr", "benjamini-hochberg", "p-hacking",
          "family-wise error"),
    aliases=("multiple comparisons", "tests multiples", "الاختبارات المتعددة"),
    backends=("scipy",),
    references=(
        ref("Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. "
            "JRSS-B 57(1).", kind="paper", doi="10.1111/j.2517-6161.1995.tb02031.x"),
        ref("Holm, S. (1979). A simple sequentially rejective multiple test procedure. "
            "Scandinavian Journal of Statistics 6(2).", kind="paper"),
    ),
)


class MultipleTestingLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        m = int(p["m"])
        k = min(int(p["n_true_effects"]), m)
        d = float(p["effect_size"])
        n = int(p["n"])
        alpha = float(p["alpha"])
        method = str(p["method"])

        gen = rng(state.seed, "mt")
        truth = np.zeros(m, dtype=bool)
        truth[:k] = True
        means = np.where(truth, d, 0.0)
        pvals = self._pvalues(gen, means, n)
        adjusted = adjust(pvals, method)
        rejected = adjusted < alpha

        res.dgp = ctx.t(
            "labs.mt.dgp",
            "{m} independent one-sample tests at n = {n}; {k} of them have a real effect "
            "of {d} standard deviations and {z} have no effect at all.",
            m=m, n=n, k=k, d=fmt(d, 2), z=m - k,
        )

        res.add_panel(ctx.panel(
            "pvalues", self._pvalue_figure(ctx, pvals, truth, alpha, method, m),
            "labs.mt.figure.pvalues", evidence=EvidenceType.SIMULATION,
        ))
        if p["show_all_methods"]:
            res.add_panel(ctx.panel(
                "methods", self._methods_figure(ctx, pvals, truth, alpha),
                "labs.mt.figure.methods", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        res.add_panel(ctx.panel(
            "error_rates", self._error_figure(ctx, p, state),
            "labs.mt.figure.error_rates", tab="simulation",
            evidence=EvidenceType.SIMULATION,
        ))
        if p["show_phacking"]:
            res.add_panel(ctx.panel(
                "phacking", self._phacking_figure(ctx, state.seed, n, alpha),
                "labs.mt.figure.phacking", tab="misconceptions",
                evidence=EvidenceType.COUNTEREXAMPLE,
            ))

        tp = int(np.sum(rejected & truth))
        fp = int(np.sum(rejected & ~truth))
        fn = int(np.sum(~rejected & truth))
        res.metric("discoveries", ctx.t("labs.mt.metric.discoveries", "Discoveries"),
                   int(rejected.sum()))
        res.metric("true_positives", ctx.t("labs.mt.metric.tp", "True discoveries"), tp)
        res.metric("false_positives", ctx.t("labs.mt.metric.fp", "False discoveries"), fp)
        res.metric("missed", ctx.t("labs.mt.metric.fn", "Real effects missed"), fn)
        res.metric("fdp", ctx.t("labs.mt.metric.fdp",
                                "False discovery proportion in this run"),
                   float(fp / max(rejected.sum(), 1)))
        res.metric("uncorrected_expected",
                   ctx.t("labs.mt.metric.expected_fp",
                         "False positives expected with no correction"),
                   float(alpha * (m - k)))

        res.assume("independence", ctx.t("assumptions.independence"), True,
                   detail=ctx.t("labs.mt.assume.independence",
                                "Benjamini-Hochberg controls the FDR under independence or "
                                "positive dependence; Benjamini-Yekutieli holds under any "
                                "dependence at the price of power."))
        res.assume("prespecified", ctx.t("labs.mt.assume.prespecified_label",
                                         "The set of tests was fixed in advance"), True,
                   detail=ctx.t("labs.mt.assume.prespecified",
                                "Every correction here assumes you decided which tests to "
                                "run before seeing any of them."))

        res.animations.append(self._animation(ctx, pvals, truth, alpha, m))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.multiple_testing.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.multiple_testing.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.inference.multiple_testing.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.multiple_testing.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.multiple_testing.warning"), kind="warning")

        if method == "none" and m > 10:
            res.warnings.append(ctx.t(
                "labs.mt.warn.none",
                "With no correction and {z} true nulls, this design manufactures about {e} "
                "false discoveries per run purely by chance.",
                z=m - k, e=fmt(alpha * (m - k), 1),
            ))
        return res

    @staticmethod
    def _pvalues(gen, means, n):
        x = gen.normal(means[:, None], 1.0, (means.size, n))
        t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n))
        return 2 * stats.t.sf(np.abs(t), n - 1)

    def _pvalue_figure(self, ctx, pvals, truth, alpha, method, m):
        order = np.argsort(pvals)
        ranks = np.arange(1, m + 1)
        fig = ctx.figure(
            "labs.mt.figure.pvalues",
            xaxis_title=ctx.t("labs.common.axis.rank"),
            yaxis_title=ctx.term("glossary.p_value"),
            height=420,
        )
        sorted_p = pvals[order]
        sorted_truth = truth[order]
        P.add_points(fig, ranks[~sorted_truth], sorted_p[~sorted_truth],
                     ctx.t("labs.mt.trace.null", "no real effect"), "baseline",
                     theme=ctx.theme, size=6)
        P.add_points(fig, ranks[sorted_truth], sorted_p[sorted_truth],
                     ctx.t("labs.mt.trace.true", "real effect"), "positive",
                     theme=ctx.theme, size=7)
        P.add_hline(fig, alpha, ctx.t("labs.mt.trace.uncorrected",
                                      "uncorrected threshold alpha"),
                    "type_i", theme=ctx.theme)
        P.add_hline(fig, alpha / m, ctx.t("labs.mt.trace.bonferroni",
                                          "Bonferroni threshold alpha/m"),
                    "warning", theme=ctx.theme, dash="dot")
        P.add_curve(fig, ranks, alpha * ranks / m,
                    ctx.t("labs.mt.trace.bh", "Benjamini-Hochberg line k*alpha/m"),
                    "secondary", theme=ctx.theme, dash="dash")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.mt.legend",
            "Sorted p-values. Bonferroni draws one flat line; Benjamini-Hochberg draws a "
            "sloping one and rejects everything below its last crossing - which is why it "
            "finds more when there really is something to find.",
        ), theme=ctx.theme)
        return fig

    def _methods_figure(self, ctx, pvals, truth, alpha):
        labels, tps, fps = [], [], []
        for method in METHODS:
            rej = adjust(pvals, method) < alpha
            labels.append(method.replace("_", " "))
            tps.append(int(np.sum(rej & truth)))
            fps.append(int(np.sum(rej & ~truth)))
        fig = ctx.figure(
            "labs.mt.figure.methods",
            xaxis_title=ctx.t("labs.mt.axis.method", "Correction"),
            yaxis_title=ctx.t("labs.mt.axis.count", "Number of discoveries"),
            height=360,
            barmode="stack",
        )
        P.add_bar(fig, labels, tps, ctx.t("labs.mt.trace.tp", "true discoveries"),
                  "positive", theme=ctx.theme)
        P.add_bar(fig, labels, fps, ctx.t("labs.mt.trace.fp", "false discoveries"),
                  "negative", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.mt.legend_methods",
            "Reading left to right you trade discoveries for certainty. There is no method "
            "that gives you both; the choice depends on the cost of a false alarm.",
        ), theme=ctx.theme)
        return fig

    def _error_figure(self, ctx, p, state):
        m, k = int(p["m"]), min(int(p["n_true_effects"]), int(p["m"]))
        d, n, alpha = float(p["effect_size"]), int(p["n"]), float(p["alpha"])
        reps = min(int(p["reps"]), 400)
        truth = np.zeros(m, dtype=bool)
        truth[:k] = True
        means = np.where(truth, d, 0.0)
        fwer = {meth: 0 for meth in METHODS}
        fdr = {meth: [] for meth in METHODS}
        power = {meth: [] for meth in METHODS}
        for r in range(reps):
            gen = rng(state.seed, "mt_sim", r)
            pv = self._pvalues(gen, means, n)
            for meth in METHODS:
                rej = adjust(pv, meth) < alpha
                fp = int(np.sum(rej & ~truth))
                fwer[meth] += int(fp > 0)
                fdr[meth].append(fp / max(rej.sum(), 1))
                power[meth].append(np.sum(rej & truth) / max(k, 1))
        labels = [meth.replace("_", " ") for meth in METHODS]
        fig = ctx.figure(
            "labs.mt.figure.error_rates",
            xaxis_title=ctx.t("labs.mt.axis.method", "Correction"),
            yaxis_title=ctx.t("labs.mt.axis.rate", "Rate across {r} simulated studies",
                              r=reps),
            height=370,
        )
        P.add_bar(fig, labels, [fwer[meth] / reps for meth in METHODS],
                  ctx.t("labs.mt.trace.fwer", "at least one false discovery (FWER)"),
                  "negative", theme=ctx.theme)
        P.add_bar(fig, labels, [float(np.mean(fdr[meth])) for meth in METHODS],
                  ctx.t("labs.mt.trace.fdr", "false discovery rate"),
                  "warning", theme=ctx.theme)
        P.add_bar(fig, labels, [float(np.mean(power[meth])) for meth in METHODS],
                  ctx.t("labs.mt.trace.power", "share of real effects found"),
                  "positive", theme=ctx.theme)
        P.add_hline(fig, alpha, f"alpha = {fmt(alpha, 3)}", "truth", theme=ctx.theme,
                    dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.mt.legend_rates",
            "Bonferroni and Holm hold the red bar (any false discovery) at alpha; "
            "Benjamini-Hochberg only holds the orange bar (the share of discoveries that "
            "are false) - and gets a taller green bar in exchange.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _phacking_figure(self, ctx, seed, n, alpha):
        gen = rng(seed, "phack")
        reps = 2000
        best = np.empty(reps)
        single = np.empty(reps)
        for i in range(reps):
            x = gen.normal(0.0, 1.0, (20, n))
            t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n))
            pv = 2 * stats.t.sf(np.abs(t), n - 1)
            best[i] = pv.min()
            single[i] = pv[0]
        fig = ctx.figure(
            "labs.mt.figure.phacking",
            xaxis_title=ctx.term("glossary.p_value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=350,
        )
        P.add_histogram(fig, single,
                        ctx.t("labs.mt.trace.one_test", "one pre-specified test"),
                        "primary", theme=ctx.theme, nbins=40, opacity=0.55)
        P.add_histogram(fig, best,
                        ctx.t("labs.mt.trace.best_of", "smallest of 20 tests, reported alone"),
                        "negative", theme=ctx.theme, nbins=40, opacity=0.55)
        P.add_vline(fig, alpha, f"alpha = {fmt(alpha, 3)}", "type_i", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.mt.legend_phacking",
            "All 40000 tests here are on pure noise. Reporting only the best of twenty makes "
            "{r} of them 'significant' at alpha - selective reporting is a correction "
            "problem you cannot fix after the fact.",
            r=pct(float(np.mean(best < alpha))),
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, pvals, truth, alpha, m):
        go = P.require_plotly()
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        ranks = np.arange(1, m + 1)
        frames, steps = [], []
        levels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20]
        for i, a in enumerate(levels):
            rej_bh = adjust(pvals, "benjamini_hochberg") < a
            frames.append(go.Frame(name=f"{a:g}", data=[
                go.Scatter(x=ranks, y=a * ranks / m),
                go.Scatter(x=[1, m], y=[a / m, a / m]),
            ]))
            fp = int(np.sum(rej_bh & ~truth))
            steps.append(AnimationStep(
                id=f"alpha_{a:g}",
                frame=i,
                title=ctx.t("labs.mt.anim.title", "alpha = {a}", a=fmt(a, 3)),
                what_you_see=ctx.t("labs.mt.anim.see",
                                   "Sorted p-values with the Bonferroni line and the "
                                   "Benjamini-Hochberg line."),
                what_changed=ctx.t("labs.mt.anim.changed",
                                   "The error budget moved to {a}.", a=fmt(a, 3)),
                why=ctx.t("labs.mt.anim.why",
                          "Both thresholds scale with alpha, but the Bonferroni line stays "
                          "flat at alpha/m while the BH line rises with the rank."),
                interpretation=ctx.t("labs.mt.anim.interpret",
                                     "BH now makes {d} discoveries, {f} of them false.",
                                     d=int(rej_bh.sum()), f=fp),
                conclusion=ctx.t("labs.mt.anim.conclude",
                                 "The two procedures answer different questions, so they "
                                 "cross the data at different points."),
                warning=ctx.t("labs.mt.anim.warn",
                              "Choosing the correction after seeing which one gives the "
                              "answer you like defeats both."),
                math="Bonferroni: reject p <= alpha/m.  BH: reject the k smallest with "
                     "p_(k) <= k*alpha/m.",
                outputs={"alpha": a, "bh_discoveries": int(rej_bh.sum()),
                         "bh_false": fp},
                active_assumptions=("independence", "prespecified"),
                highlighted=("bh_line", "bonferroni_line"),
            ))
        fig = ctx.figure(
            "labs.mt.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.rank"),
            yaxis_title=ctx.term("glossary.p_value"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=ranks, y=alpha * ranks / m, mode="lines",
                                 line={"color": ctx.color("secondary"), "dash": "dash"},
                                 name=ctx.t("labs.mt.trace.bh",
                                            "Benjamini-Hochberg line k*alpha/m")))
        fig.add_trace(go.Scatter(x=[1, m], y=[alpha / m, alpha / m], mode="lines",
                                 line={"color": ctx.color("warning"), "dash": "dot"},
                                 name=ctx.t("labs.mt.trace.bonferroni",
                                            "Bonferroni threshold alpha/m")))
        P.add_points(fig, ranks, sorted_p, ctx.t("labs.mt.trace.sorted", "sorted p-values"),
                     "primary", theme=ctx.theme, size=5)
        fig.update_yaxes(type="log")
        build_frames(fig, frames, duration=620, reduced_motion=ctx.reduced_motion,
                     slider_label="alpha")
        return animation(
            "thresholds", fig, steps,
            purpose=ctx.t("labs.mt.anim.purpose",
                          "Compare two corrections as two lines drawn across the same data."),
            summary=ctx.t(
                "labs.mt.anim.summary",
                "Controlling the chance of ANY false discovery and controlling the SHARE of "
                "discoveries that are false are different goals. Neither is a stricter "
                "version of the other, and the right choice depends on what a false alarm "
                "actually costs you."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = MultipleTestingLab(SPEC)
