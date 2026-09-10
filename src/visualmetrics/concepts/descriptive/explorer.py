"""Exploratory data explorer: five views of one sample, and what each one hides."""

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
    ref,
    scenario,
    seed_control,
    select,
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC", "SHAPES"]

SHAPES = ("normal", "right_skewed", "left_skewed", "bimodal", "uniform", "heavy_tailed",
          "discrete_counts", "constant")


SPEC = make_spec(
    "descriptive.explorer",
    Domain.DESCRIPTIVE,
    "exploratory_data_analysis",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "experiment", "compare", "diagnose", "code", "quiz",
           "data", "references"),
    evidence=EvidenceType.EMPIRICAL_EXAMPLE,
    controls=(
        select("shape", "right_skewed", SHAPES, group="population"),
        int_slider("n", 200, 5, 20000, 5, group="design"),
        int_slider("n_outliers", 0, 0, 30, 1, group="contamination"),
        slider("outlier_magnitude", 8.0, 1.0, 60.0, 0.5, group="contamination"),
        int_slider("bins", 30, 5, 150, 1, group="views"),
        slider("missing_rate", 0.0, 0.0, 0.5, 0.01, group="contamination"),
        toggle("show_boxplot", True, group="views"),
        toggle("show_ecdf", True, group="views"),
        toggle("show_robust", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", shape="right_skewed", n=200),
        scenario("symmetric", "positive", shape="normal", n=300),
        scenario("bimodal_hidden", "counterexample", shape="bimodal", n=400, bins=8),
        scenario("bimodal_revealed", "compare_methods", shape="bimodal", n=400, bins=60),
        scenario("one_outlier", "violation", shape="normal", n=60, n_outliers=1,
                 outlier_magnitude=25.0),
        scenario("many_outliers", "violation", shape="normal", n=200, n_outliers=15),
        scenario("heavy_tails", "high_noise", shape="heavy_tailed", n=300),
        scenario("small_sample", "small_sample", n=8),
        scenario("large_sample", "large_sample", n=10000),
        scenario("counts", "compare_methods", shape="discrete_counts", n=300),
        scenario("missing_data", "misspecification", missing_rate=0.25, n=300),
        scenario("no_variation", "boundary", shape="constant", n=100),
    ),
    related=("probability.distributions", "inference.sampling_distributions"),
    tags=("histogram", "boxplot", "ecdf", "outliers", "skewness", "kurtosis", "robust"),
    aliases=("eda", "descriptive statistics", "الإحصاء الوصفي",
             "statistique descriptive", "exploratory data analysis"),
    backends=("numpy", "scipy"),
    references=(
        ref("Tukey, J. W. (1977). Exploratory Data Analysis.", kind="book"),
        ref("Cleveland, W. S. (1993). Visualizing Data.", kind="book"),
    ),
)


class DescriptiveLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        shape = str(p["shape"])
        n = int(p["n"])
        gen = rng(state.seed, "eda", shape)
        x = self._draw(gen, shape, n)

        n_out = int(p["n_outliers"])
        if n_out and x.size:
            idx = gen.choice(x.size, size=min(n_out, x.size), replace=False)
            x[idx] = x[idx] + gen.choice([-1.0, 1.0], size=idx.size) * \
                float(p["outlier_magnitude"]) * max(float(np.std(x)), 1e-9)

        missing_rate = float(p["missing_rate"])
        n_missing = 0
        if missing_rate > 0:
            drop = gen.random(x.size) < missing_rate
            n_missing = int(drop.sum())
            x = x[~drop]

        res.dgp = ctx.t(
            "labs.eda.dgp",
            "n = {n} observations from a {shape} population"
            "{out}{miss}.",
            n=x.size, shape=shape,
            out=(f"; {n_out} contaminated by an outlier of {fmt(p['outlier_magnitude'], 1)} "
                 "standard deviations" if n_out else ""),
            miss=(f"; {n_missing} values dropped at random" if n_missing else ""),
        )

        res.add_panel(ctx.panel(
            "histogram", self._hist_figure(ctx, x, p),
            "labs.eda.figure.histogram", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        if p["show_boxplot"]:
            res.add_panel(ctx.panel(
                "distributional", self._box_violin_figure(ctx, x),
                "labs.eda.figure.box", tab="compare",
                evidence=EvidenceType.EMPIRICAL_EXAMPLE,
            ))
        if p["show_ecdf"]:
            res.add_panel(ctx.panel(
                "ecdf", self._ecdf_figure(ctx, x),
                "labs.eda.figure.ecdf", tab="diagnostics",
                evidence=EvidenceType.EMPIRICAL_EXAMPLE,
            ))
        if p["show_robust"]:
            res.add_panel(ctx.panel(
                "sensitivity", self._sensitivity_figure(ctx, x, gen),
                "labs.eda.figure.sensitivity", tab="misconceptions",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        if x.size < 2:
            res.warnings.append(ctx.t("labs.eda.warn.too_few",
                                      "Fewer than two observations remain; most summaries "
                                      "are undefined."))
            return res

        summary = self._summaries(x)
        for key, label, value in (
            ("mean", ctx.t("labs.eda.metric.mean", "Mean"), summary["mean"]),
            ("median", ctx.t("labs.eda.metric.median", "Median"), summary["median"]),
            ("trimmed", ctx.t("labs.eda.metric.trimmed", "20% trimmed mean"),
             summary["trimmed"]),
            ("sd", ctx.t("labs.eda.metric.sd", "Standard deviation"), summary["sd"]),
            ("iqr", ctx.t("labs.eda.metric.iqr", "Interquartile range"), summary["iqr"]),
            ("mad", ctx.t("labs.eda.metric.mad", "Median absolute deviation (scaled)"),
             summary["mad"]),
            ("skewness", ctx.t("labs.eda.metric.skewness", "Skewness"), summary["skewness"]),
            ("kurtosis", ctx.t("labs.eda.metric.kurtosis", "Excess kurtosis"),
             summary["kurtosis"]),
            ("range", ctx.t("labs.eda.metric.range", "Range"), summary["range"]),
        ):
            res.metric(key, label, value)
        res.metric("outliers_flagged", ctx.t("labs.eda.metric.outliers",
                                             "Points beyond 1.5 IQR from the quartiles"),
                   int(summary["n_outliers"]))
        res.metric("mean_minus_median", ctx.t("labs.eda.metric.gap",
                                              "Mean minus median"),
                   summary["mean"] - summary["median"],
                   note=ctx.t("labs.eda.metric.gap_note",
                              "a quick and dirty read on skewness"))
        if n_missing:
            res.metric("missing", ctx.t("labs.eda.metric.missing", "Values dropped"),
                       n_missing)

        res.assume("representative", ctx.t("labs.eda.assume.representative_label",
                                           "The remaining data represent the population"),
                   missing_rate == 0,
                   detail=ctx.t("labs.eda.assume.missing",
                                "Values were removed completely at random here. Real "
                                "missingness is rarely random, and then every summary below "
                                "describes only the observed subset."))
        res.assume("no_contamination", ctx.t("labs.eda.assume.contamination_label",
                                             "No contaminated observations"), n_out == 0,
                   detail=ctx.t("labs.eda.assume.contamination",
                                "Non-robust summaries have a breakdown point of zero: one "
                                "arbitrarily large value moves the mean arbitrarily far."))

        n_modes = self._count_modes(x)
        res.metric("modes_detected", ctx.t("labs.eda.metric.modes",
                                           "Clearly separated peaks in the density"),
                   n_modes,
                   note=ctx.t("labs.eda.metric.modes_note",
                              "counted from a kernel density estimate, not from the "
                              "histogram, so it does not depend on the bin width"))
        res.assume("single_centre", ctx.t("labs.eda.assume.unimodal_label",
                                          "One centre summarises the data"),
                   n_modes <= 1,
                   detail=ctx.t("labs.eda.assume.unimodal",
                                "The mean, the median and the standard deviation all "
                                "describe a distribution as one location plus one spread."),
                   consequence="" if n_modes <= 1 else ctx.t(
                       "labs.eda.assume.multimodal",
                       "With {k} separated peaks the mean falls in the valley between "
                       "them, where almost no observation lies. Every one-number summary "
                       "below describes a group that does not exist.",
                       k=n_modes))
        counts = np.histogram(x, bins=int(p["bins"]))[0].astype(float)
        if n_modes > 1 and self._count_peaks(counts) < n_modes:
            res.warnings.append(ctx.t(
                "labs.eda.warn.bins_hide_modes",
                "The data contain {k} separated peaks, but {b} bins are too few to "
                "resolve them: the histogram looks single-peaked while the data are "
                "not. Increase the bin count before trusting the shape you see.",
                k=n_modes, b=int(p["bins"]),
            ))

        res.animations.append(self._animation(ctx, x, p))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.descriptive.explorer.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.descriptive.explorer.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.descriptive.explorer.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.descriptive.explorer.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.descriptive.explorer.warning"), kind="warning")

        if n_out:
            gap = abs(summary["mean"] - summary["median"])
            res.explain("diagnostics", ctx.t("labs.eda.contamination.title",
                                             "What the contamination did"), ctx.t(
                "labs.eda.contamination.body",
                "With {k} contaminated points the mean and the median differ by {g}, and the "
                "standard deviation is {r} times the scaled median absolute deviation. Both "
                "gaps are diagnostics, not errors.",
                k=n_out, g=fmt(gap, 3),
                r=fmt(summary["sd"] / max(summary["mad"], 1e-9), 2),
            ))
        return res

    @staticmethod
    def _count_modes(x):
        """Count clearly separated peaks in a kernel density estimate.

        A peak counts only if it rises at least 15% of the tallest peak above the
        deepest valley separating it from its neighbour, so ordinary sampling
        wobble in the tails is not mistaken for structure.
        """
        if x.size < 20 or float(np.std(x)) < 1e-12:
            return 1
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 512)
        try:
            density = stats.gaussian_kde(x)(grid)
        except Exception:
            return 1
        peaks = [
            i for i in range(1, len(grid) - 1)
            if density[i] > density[i - 1] and density[i] >= density[i + 1]
        ]
        if len(peaks) < 2:
            return len(peaks) or 1
        tallest = float(density.max())
        kept = [peaks[0]]
        for current in peaks[1:]:
            valley = float(np.min(density[kept[-1]:current + 1]))
            rise = min(density[kept[-1]], density[current]) - valley
            if rise >= 0.15 * tallest:
                kept.append(current)
            elif density[current] > density[kept[-1]]:
                kept[-1] = current
        return len(kept)

    @staticmethod
    def _count_peaks(heights):
        """Peaks in a bar profile, using the same prominence rule as _count_modes."""
        heights = np.asarray(heights, dtype=float)
        if heights.size < 3 or float(heights.max()) <= 0:
            return 1
        peaks = [
            i for i in range(1, len(heights) - 1)
            if heights[i] > heights[i - 1] and heights[i] >= heights[i + 1]
        ]
        if len(peaks) < 2:
            return len(peaks) or 1
        tallest = float(heights.max())
        kept = [peaks[0]]
        for current in peaks[1:]:
            valley = float(np.min(heights[kept[-1]:current + 1]))
            if min(heights[kept[-1]], heights[current]) - valley >= 0.15 * tallest:
                kept.append(current)
            elif heights[current] > heights[kept[-1]]:
                kept[-1] = current
        return len(kept)

    @staticmethod
    def _draw(gen, shape, n):
        if shape == "normal":
            return gen.normal(10.0, 2.0, n)
        if shape == "right_skewed":
            return gen.lognormal(2.0, 0.7, n)
        if shape == "left_skewed":
            return 40.0 - gen.lognormal(2.0, 0.7, n)
        if shape == "bimodal":
            pick = gen.random(n) < 0.5
            return np.where(pick, gen.normal(5.0, 1.0, n), gen.normal(14.0, 1.2, n))
        if shape == "uniform":
            return gen.uniform(0.0, 20.0, n)
        if shape == "heavy_tailed":
            return 10.0 + 2.0 * gen.standard_t(2.5, n)
        if shape == "discrete_counts":
            return gen.poisson(4.0, n).astype(float)
        return np.full(n, 7.0)

    @staticmethod
    def _summaries(x):
        q1, q3 = np.percentile(x, [25, 75])
        iqr = float(q3 - q1)
        mad = float(1.4826 * np.median(np.abs(x - np.median(x))))
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        constant = float(np.std(x)) < 1e-12
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "trimmed": float(stats.trim_mean(x, 0.2)),
            "sd": float(np.std(x, ddof=1)),
            "iqr": iqr,
            "mad": mad,
            "skewness": float("nan") if constant else float(stats.skew(x)),
            "kurtosis": float("nan") if constant else float(stats.kurtosis(x, fisher=True)),
            "range": float(np.max(x) - np.min(x)),
            "n_outliers": int(np.sum((x < lo) | (x > hi))),
        }

    def _hist_figure(self, ctx, x, p):
        fig = ctx.figure(
            "labs.eda.figure.histogram",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=420,
        )
        if x.size < 2:
            return P.empty_figure(ctx.t("labs.eda.no_data", "Not enough data to plot."),
                                  theme=ctx.theme)
        P.add_histogram(fig, x, ctx.t("labs.eda.trace.hist", "histogram ({b} bins)",
                                      b=int(p["bins"])),
                        "primary", theme=ctx.theme, nbins=int(p["bins"]), opacity=0.55)
        if float(np.std(x)) > 1e-12:
            kde = stats.gaussian_kde(x)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 400)
            P.add_curve(fig, grid, kde(grid),
                        ctx.t("labs.eda.trace.kde", "kernel density estimate"),
                        "secondary", theme=ctx.theme)
        P.add_vline(fig, float(np.mean(x)), ctx.t("labs.eda.metric.mean", "Mean"),
                    "negative", theme=ctx.theme, dash="solid")
        P.add_vline(fig, float(np.median(x)), ctx.t("labs.eda.metric.median", "Median"),
                    "positive", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.eda.legend_hist",
            "Bin width is your choice, not a property of the data. Slide the bin count and "
            "watch how much of the 'shape' was really a rendering decision.",
        ), theme=ctx.theme)
        return fig

    def _box_violin_figure(self, ctx, x):
        go = P.require_plotly()
        fig = ctx.figure(
            "labs.eda.figure.box",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title="", height=330,
        )
        fig.add_trace(go.Violin(x=x, name=ctx.t("labs.eda.trace.violin", "violin"),
                                side="positive", line={"color": ctx.color("secondary")},
                                fillcolor=P.rgba("secondary", 0.3, ctx.theme),
                                points=False, orientation="h", y0=1))
        fig.add_trace(go.Box(x=x, name=ctx.t("labs.eda.trace.box", "boxplot"),
                             boxpoints="outliers", marker={"color": ctx.color("primary")},
                             line={"color": ctx.color("primary")}, orientation="h", y0=0))
        P.add_legend_note(fig, ctx.t(
            "labs.eda.legend_box",
            "The boxplot shows five numbers; the violin shows the whole shape. A symmetric "
            "box above a two-humped violin is a warning that the box discarded the story.",
        ), theme=ctx.theme)
        return fig

    def _ecdf_figure(self, ctx, x):
        s = np.sort(x)
        ecdf = np.arange(1, s.size + 1) / s.size
        fig = ctx.figure(
            "labs.eda.figure.ecdf",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.cumulative"),
            height=340,
        )
        P.add_curve(fig, s, ecdf, ctx.t("labs.eda.trace.ecdf", "empirical CDF"),
                    "primary", theme=ctx.theme, mode="lines")
        for q, label in ((0.25, "Q1"), (0.5, "median"), (0.75, "Q3")):
            P.add_vline(fig, float(np.quantile(x, q)), label, "baseline", theme=ctx.theme,
                        dash="dot")
        fig.update_yaxes(range=[0, 1.02])
        P.add_legend_note(fig, ctx.t(
            "labs.eda.legend_ecdf",
            "The empirical CDF uses every observation and needs no bin choice at all, which "
            "makes it the honest default when two samples must be compared.",
        ), theme=ctx.theme)
        return fig

    def _sensitivity_figure(self, ctx, x, gen):
        magnitudes = np.linspace(0, 40, 30)
        means, medians, trimmed, sds, mads = [], [], [], [], []
        base = x.copy()
        sd0 = max(float(np.std(base)), 1e-9)
        for m in magnitudes:
            y = base.copy()
            y[0] = base[0] + m * sd0
            means.append(float(np.mean(y)))
            medians.append(float(np.median(y)))
            trimmed.append(float(stats.trim_mean(y, 0.2)))
            sds.append(float(np.std(y, ddof=1)))
            mads.append(float(1.4826 * np.median(np.abs(y - np.median(y)))))
        fig = ctx.figure(
            "labs.eda.figure.sensitivity",
            xaxis_title=ctx.t("labs.eda.axis.contamination",
                              "Size of one contaminated point (standard deviations)"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=360,
        )
        P.add_curve(fig, magnitudes, means, ctx.t("labs.eda.metric.mean", "Mean"),
                    "negative", theme=ctx.theme)
        P.add_curve(fig, magnitudes, medians, ctx.t("labs.eda.metric.median", "Median"),
                    "positive", theme=ctx.theme)
        P.add_curve(fig, magnitudes, trimmed,
                    ctx.t("labs.eda.metric.trimmed", "20% trimmed mean"),
                    "info", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.eda.legend_sensitivity",
            "Moving ONE observation drags the mean without limit while the median barely "
            "moves. That is the difference between a breakdown point of 0% and of 50%.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, x, p):
        go = P.require_plotly()
        bins = np.unique(np.round(np.geomspace(3, 120, 18)).astype(int))
        lo, hi = float(np.min(x)), float(np.max(x))
        frames, steps = [], []
        for i, b in enumerate(bins):
            hist, edges = np.histogram(x, bins=int(b), range=(lo, hi), density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])
            frames.append(go.Frame(name=str(b), data=[go.Bar(x=centres, y=hist,
                                                             width=(hi - lo) / b)]))
            modes = int(np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))) if b > 3 else 1
            steps.append(AnimationStep(
                id=f"bins_{b}", frame=i,
                title=ctx.t("labs.eda.anim.title", "{b} bins", b=int(b)),
                what_you_see=ctx.t("labs.eda.anim.see",
                                   "The same {n} observations, rendered with a different "
                                   "bin width.", n=x.size),
                what_changed=ctx.t("labs.eda.anim.changed",
                                   "The number of bins moved to {b}.", b=int(b)),
                why=ctx.t("labs.eda.anim.why",
                          "Binning is a smoothing choice: few wide bins average detail away, "
                          "many narrow bins amplify sampling noise into apparent structure."),
                interpretation=ctx.t("labs.eda.anim.interpret",
                                     "This rendering shows {m} local peak(s).", m=modes),
                conclusion=ctx.t("labs.eda.anim.conclude",
                                 "Any claim about 'the shape' that survives only one bin "
                                 "width is a claim about the rendering, not the data."),
                warning=ctx.t("labs.eda.anim.warn",
                              "The data never changed during this animation. Not one value "
                              "moved."),
                math="", outputs={"bins": int(b), "visible_modes": modes},
                highlighted=("histogram",),
            ))
        fig = ctx.figure(
            "labs.eda.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=380,
        )
        fig.add_trace(go.Bar(x=[], y=[], marker={"color": ctx.color("primary")},
                             opacity=0.7,
                             name=ctx.t("labs.eda.trace.hist_short", "histogram")))
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.eda.slider", "bins"))
        return animation(
            "bin_width", fig, steps,
            purpose=ctx.t("labs.eda.anim.purpose",
                          "Show how much of a histogram's 'shape' is a rendering choice."),
            summary=ctx.t(
                "labs.eda.anim.summary",
                "One sample, many pictures. Before believing a feature in a histogram, check "
                "whether it survives a different bin width - and prefer the empirical CDF or "
                "a density estimate when the answer matters."),
            evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        )


LAB = DescriptiveLab(SPEC)
