"""Bootstrap: resampling the sample to approximate the sampling distribution."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "STATISTICS"]

STATISTICS = ("mean", "median", "trimmed_mean", "sd", "correlation", "ratio", "max", "quantile90")
POPULATIONS = ("normal", "exponential", "lognormal", "uniform", "contaminated")
INTERVALS = ("percentile", "basic", "normal_approx")


def _statistic(x, y, kind):
    if kind == "mean":
        return float(np.mean(x))
    if kind == "median":
        return float(np.median(x))
    if kind == "trimmed_mean":
        return float(stats.trim_mean(x, 0.2))
    if kind == "sd":
        return float(np.std(x, ddof=1))
    if kind == "correlation":
        return float(np.corrcoef(x, y)[0, 1])
    if kind == "ratio":
        return float(np.mean(x) / max(np.mean(y), 1e-9))
    if kind == "max":
        return float(np.max(x))
    return float(np.quantile(x, 0.9))


def _draw(gen, pop, n):
    if pop == "normal":
        return gen.normal(5.0, 2.0, n)
    if pop == "exponential":
        return gen.exponential(3.0, n)
    if pop == "lognormal":
        return gen.lognormal(1.0, 0.9, n)
    if pop == "uniform":
        return gen.uniform(0.0, 10.0, n)
    contaminated = gen.random(n) < 0.08
    return np.where(contaminated, gen.normal(5.0, 20.0, n), gen.normal(5.0, 2.0, n))


SPEC = make_spec(
    "inference.bootstrap",
    Domain.INFERENCE,
    "resampling",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("statistic", "median", STATISTICS, group="statistic"),
        select("population", "lognormal", POPULATIONS, group="population"),
        select("interval", "percentile", INTERVALS, group="interval"),
        int_slider("n", 30, 4, 500, 1, group="design"),
        int_slider("boot", 2000, 100, 20000, 100, group="design", expensive=True),
        slider("level", 0.95, 0.5, 0.999, 0.005, group="interval"),
        int_slider("coverage_reps", 200, 30, 5000, 10, group="simulation", expensive=True),
        toggle("show_truth", True, group="views"),
        toggle("show_coverage", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", statistic="median", population="lognormal", n=30),
        scenario("mean_easy", "positive", statistic="mean", population="normal", n=30),
        scenario("skewed_mean", "high_noise", statistic="mean", population="lognormal", n=20),
        scenario("robust_vs_not", "robustness", statistic="trimmed_mean",
                 population="contaminated", n=40),
        scenario("correlation", "compare_methods", statistic="correlation", n=40),
        scenario("ratio", "compare_methods", statistic="ratio", n=40),
        scenario("tiny_sample", "small_sample", n=6),
        scenario("large_sample", "large_sample", n=400),
        scenario("bootstrap_fails", "counterexample", statistic="max",
                 population="uniform", n=30),
        scenario("basic_interval", "compare_methods", interval="basic"),
        scenario("normal_interval", "compare_methods", interval="normal_approx"),
    ),
    prerequisites=("inference.sampling_distributions",),
    related=("inference.confidence_intervals",),
    tags=("bootstrap", "resampling", "percentile interval", "robust"),
    aliases=("bootstrap", "reechantillonnage", "إعادة المعاينة"),
    backends=("numpy", "scipy"),
    references=(
        ref("Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap.",
            kind="book"),
        ref("Davison, A. C. and Hinkley, D. V. (1997). Bootstrap Methods and their "
            "Application.", kind="book"),
    ),
)


class BootstrapLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        kind = str(p["statistic"])
        pop = str(p["population"])
        n, boot = int(p["n"]), int(p["boot"])
        level = float(p["level"])
        interval_kind = str(p["interval"])

        gen = rng(state.seed, "bootstrap", kind)
        x = _draw(gen, pop, n)
        y = _draw(gen, pop, n) if kind in ("correlation", "ratio") else None
        if kind == "correlation":
            y = 0.6 * x + np.sqrt(1 - 0.36) * _draw(gen, pop, n)
        theta_hat = _statistic(x, y, kind)

        reps = self._bootstrap(gen, x, y, kind, boot)
        lo, hi = self._interval(reps, theta_hat, level, interval_kind)
        truth = self._truth(pop, kind, state.seed)
        exact = self._exact_sampling(pop, kind, n, state.seed)

        res.dgp = ctx.t(
            "labs.boot.dgp",
            "One sample of n = {n} from a {pop} population; {b} bootstrap resamples of the "
            "{stat}.", n=n, pop=pop, b=boot, stat=kind,
        )

        res.add_panel(ctx.panel(
            "bootstrap", self._bootstrap_figure(ctx, reps, exact, theta_hat, lo, hi,
                                                truth, level, p),
            "labs.boot.figure.bootstrap", evidence=EvidenceType.SIMULATION,
        ))
        res.add_panel(ctx.panel(
            "sample", self._sample_figure(ctx, x, theta_hat, gen),
            "labs.boot.figure.sample", tab="data",
            evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        if p["show_coverage"]:
            cov = self._coverage(pop, kind, n, boot, level, interval_kind,
                                 int(p["coverage_reps"]), state.seed, truth)
            res.add_panel(ctx.panel(
                "coverage", self._coverage_figure(ctx, cov, level, interval_kind),
                "labs.boot.figure.coverage", tab="diagnostics",
                evidence=EvidenceType.SIMULATION,
            ))
            res.metric("coverage", ctx.t("labs.boot.metric.coverage",
                                         "Realized coverage of the bootstrap interval"),
                       cov["coverage"], reference=level,
                       note=f"+/- {fmt(1.96 * cov['mc_se'], 4)}")

        res.metric("estimate", ctx.t("labs.boot.metric.estimate",
                                     "Estimate from the observed sample"), theta_hat,
                   reference=truth)
        res.metric("boot_se", ctx.t("labs.boot.metric.se", "Bootstrap standard error"),
                   float(np.std(reps, ddof=1)),
                   reference=float(np.std(exact, ddof=1)) if exact is not None else None,
                   note=ctx.t("labs.boot.metric.se_note",
                              "compared with the true sampling standard error"))
        res.metric("boot_bias", ctx.t("labs.boot.metric.bias",
                                      "Bootstrap bias estimate"),
                   float(np.mean(reps) - theta_hat))
        res.metric("interval", ctx.t("labs.boot.metric.interval",
                                     "{l} {k} interval", l=pct(level), k=interval_kind),
                   f"[{fmt(lo, 4)}, {fmt(hi, 4)}]")
        res.metric("distinct", ctx.t("labs.boot.metric.distinct",
                                     "Distinct bootstrap values"),
                   int(np.unique(np.round(reps, 10)).size),
                   note=ctx.t("labs.boot.metric.distinct_note",
                              "a small number means the resamples are exhausting the data"))

        smooth = kind not in ("max", "quantile90")
        res.assume("independence", ctx.t("assumptions.independence"), True)
        res.assume("smooth_functional",
                   ctx.t("labs.boot.assume.smooth_label",
                         "The statistic is a smooth function of the sample"), smooth,
                   detail=ctx.t("labs.boot.assume.smooth",
                                "The bootstrap approximates the sampling distribution well "
                                "for smooth functionals; it fails for statistics driven by "
                                "extreme order statistics such as the maximum."),
                   consequence="" if smooth else ctx.t(
                       "labs.boot.assume.smooth_consequence",
                       "The bootstrap distribution of the sample maximum is a lump of atoms "
                       "that cannot approach the true continuous limit."))
        res.assume("representative",
                   ctx.t("labs.boot.assume.representative_label",
                         "The sample represents the population"), True,
                   detail=ctx.t("labs.boot.assume.representative",
                                "Resampling can only reproduce what is already in the "
                                "sample. It cannot recover a tail you never observed."))

        res.animations.append(self._animation(ctx, x, y, kind, gen, boot, theta_hat))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.bootstrap.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.bootstrap.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.inference.bootstrap.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.bootstrap.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.bootstrap.warning"), kind="warning")

        if not smooth:
            res.warnings.append(ctx.t(
                "labs.boot.warn.max",
                "The bootstrap distribution of the maximum can never exceed the observed "
                "maximum, so it is systematically shifted and cannot be repaired by more "
                "resamples. This is a genuine failure case, included on purpose.",
            ))
        return res

    @staticmethod
    def _bootstrap(gen, x, y, kind, boot):
        n = x.size
        out = np.empty(boot)
        for i in range(boot):
            idx = gen.integers(0, n, n)
            out[i] = _statistic(x[idx], None if y is None else y[idx], kind)
        return out

    @staticmethod
    def _interval(reps, theta_hat, level, kind):
        a = 1 - level
        if kind == "percentile":
            return (float(np.quantile(reps, a / 2)), float(np.quantile(reps, 1 - a / 2)))
        if kind == "basic":
            return (float(2 * theta_hat - np.quantile(reps, 1 - a / 2)),
                    float(2 * theta_hat - np.quantile(reps, a / 2)))
        z = stats.norm.ppf(1 - a / 2)
        se = float(np.std(reps, ddof=1))
        return theta_hat - z * se, theta_hat + z * se

    def _truth(self, pop, kind, seed):
        gen = rng(seed, "truth_pop")
        big = _draw(gen, pop, 400_000)
        other = (0.6 * big + np.sqrt(1 - 0.36) * _draw(gen, pop, 400_000)
                 if kind == "correlation" else _draw(gen, pop, 400_000))
        return _statistic(big, other, kind)

    def _exact_sampling(self, pop, kind, n, seed, reps: int = 800):
        gen = rng(seed, "exact_sampling", n)
        out = np.empty(reps)
        for i in range(reps):
            x = _draw(gen, pop, n)
            y = (0.6 * x + np.sqrt(1 - 0.36) * _draw(gen, pop, n)
                 if kind == "correlation" else
                 (_draw(gen, pop, n) if kind == "ratio" else None))
            out[i] = _statistic(x, y, kind)
        return out

    def _coverage(self, pop, kind, n, boot, level, interval_kind, reps, seed, truth):
        reps = min(reps, 400)
        boot_small = min(boot, 400)
        covered = np.zeros(reps, dtype=bool)
        widths = np.zeros(reps)
        for r in range(reps):
            gen = rng(seed, "boot_cov", r)
            x = _draw(gen, pop, n)
            y = (0.6 * x + np.sqrt(1 - 0.36) * _draw(gen, pop, n)
                 if kind == "correlation" else
                 (_draw(gen, pop, n) if kind == "ratio" else None))
            hat = _statistic(x, y, kind)
            b = self._bootstrap(gen, x, y, kind, boot_small)
            lo, hi = self._interval(b, hat, level, interval_kind)
            covered[r] = lo <= truth <= hi
            widths[r] = hi - lo
        p = float(covered.mean())
        return {"coverage": p, "mc_se": float(np.sqrt(max(p * (1 - p), 0) / reps)),
                "reps": reps, "width": float(np.mean(widths)), "covered": covered}

    def _bootstrap_figure(self, ctx, reps, exact, theta_hat, lo, hi, truth, level, p):
        fig = ctx.figure(
            "labs.boot.figure.bootstrap",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=430,
        )
        P.add_histogram(fig, reps,
                        ctx.t("labs.boot.trace.boot", "bootstrap distribution"),
                        "primary", theme=ctx.theme, nbins=60, opacity=0.6)
        if exact is not None:
            P.add_histogram(fig, exact,
                            ctx.t("labs.boot.trace.exact",
                                  "true sampling distribution (only a simulation can show it)"),
                            "secondary", theme=ctx.theme, nbins=60, opacity=0.35)
        P.add_vline(fig, theta_hat, ctx.t("labs.boot.trace.estimate",
                                          "estimate from the observed sample"),
                    "estimate", theme=ctx.theme, dash="solid")
        if p["show_truth"]:
            P.add_vline(fig, truth, ctx.t("labs.common.trace.truth"), "truth",
                        theme=ctx.theme, dash="dot")
        P.add_vline(fig, lo, "", "warning", theme=ctx.theme)
        P.add_vline(fig, hi, ctx.t("labs.boot.trace.interval", "{l} interval",
                                   l=pct(level)),
                    "warning", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.boot.legend",
            "The blue histogram was built from ONE sample by resampling it. The faint one is "
            "the truth, visible only because this is a simulation. Where they disagree, the "
            "bootstrap is inheriting a limitation of the sample.",
        ), theme=ctx.theme)
        return fig

    def _sample_figure(self, ctx, x, theta_hat, gen):
        fig = ctx.figure(
            "labs.boot.figure.sample",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title="",
            height=280,
        )
        P.add_points(fig, x, gen.normal(0, 0.03, x.size),
                     ctx.t("labs.boot.trace.observed", "the one observed sample"),
                     "secondary", theme=ctx.theme, size=9)
        P.add_vline(fig, theta_hat, ctx.t("labs.boot.trace.estimate", "estimate"),
                    "estimate", theme=ctx.theme, dash="solid")
        fig.update_yaxes(visible=False)
        P.add_legend_note(fig, ctx.t(
            "labs.boot.legend_sample",
            "This is all the information the bootstrap has. Resampling reuses these exact "
            "values with repetition - it never invents a new one.",
        ), theme=ctx.theme)
        return fig

    def _coverage_figure(self, ctx, cov, level, interval_kind):
        fig = ctx.figure(
            "labs.boot.figure.coverage",
            xaxis_title=ctx.t("labs.boot.axis.study", "Simulated study"),
            yaxis_title=ctx.t("labs.common.axis.coverage"),
            height=330,
        )
        running = np.cumsum(cov["covered"]) / np.arange(1, cov["covered"].size + 1)
        P.add_curve(fig, np.arange(1, running.size + 1), running,
                    ctx.t("labs.boot.trace.running", "running coverage"),
                    "primary", theme=ctx.theme)
        P.add_hline(fig, level, ctx.t("labs.ci.trace.nominal", "nominal level"),
                    "truth", theme=ctx.theme, dash="dash")
        fig.update_yaxes(range=[max(0.3, level - 0.4), 1.02])
        P.add_legend_note(fig, ctx.t(
            "labs.boot.legend_coverage",
            "Bootstrap intervals are asymptotically valid, not exact. The {k} interval "
            "reached {c} against a nominal {l} at this sample size.",
            k=interval_kind, c=pct(cov["coverage"]), l=pct(level),
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, x, y, kind, gen, boot, theta_hat):
        go = P.require_plotly()
        n = x.size
        counts = np.unique(np.round(np.geomspace(5, min(boot, 5000), 20)).astype(int))
        reps = self._bootstrap(rng(1, "anim"), x, y, kind, int(counts[-1]))
        lo, hi = np.percentile(reps, [0.5, 99.5])
        edges = np.linspace(lo, hi, 46)
        centres = 0.5 * (edges[1:] + edges[:-1])
        frames, steps = [], []
        for i, c in enumerate(counts):
            hist, _ = np.histogram(reps[:c], bins=edges, density=True)
            frames.append(go.Frame(name=str(c), data=[go.Bar(x=centres, y=hist)]))
            steps.append(AnimationStep(
                id=f"b_{c}",
                frame=i,
                title=ctx.t("labs.boot.anim.title", "{b} resamples", b=int(c)),
                what_you_see=ctx.t("labs.boot.anim.see",
                                   "The distribution of the statistic across bootstrap "
                                   "resamples of the SAME {n} observations.", n=n),
                what_changed=ctx.t("labs.boot.anim.changed",
                                   "{b} resamples have been drawn with replacement.",
                                   b=int(c)),
                why=ctx.t("labs.boot.anim.why",
                          "Each resample reweights the observed data at random, mimicking "
                          "the variation a fresh sample would have shown."),
                interpretation=ctx.t("labs.boot.anim.interpret",
                                     "Bootstrap standard error is {s}.",
                                     s=fmt(float(np.std(reps[:c], ddof=1)), 4)
                                     if c > 1 else "0"),
                conclusion=ctx.t("labs.boot.anim.conclude",
                                 "More resamples reduce Monte Carlo noise in the estimate of "
                                 "the sampling distribution; they add no new information."),
                warning=ctx.t("labs.boot.anim.warn",
                              "Increasing the number of resamples is not the same as "
                              "increasing n. Only n reduces the standard error itself."),
                math="theta*_b = T(x* sampled with replacement from x);  SE_boot = sd(theta*)",
                outputs={"resamples": int(c),
                         "se": round(float(np.std(reps[:c], ddof=1)), 5) if c > 1 else 0.0},
                active_assumptions=("independence", "representative"),
                highlighted=("bootstrap_histogram",),
            ))
        fig = ctx.figure(
            "labs.boot.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=380,
        )
        fig.add_trace(go.Bar(x=centres, y=np.zeros_like(centres),
                             marker={"color": ctx.color("primary")}, opacity=0.7,
                             name=ctx.t("labs.boot.trace.boot", "bootstrap distribution")))
        P.add_vline(fig, theta_hat, ctx.t("labs.boot.trace.estimate", "estimate"),
                    "estimate", theme=ctx.theme, dash="solid")
        build_frames(fig, frames, duration=380, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.boot.slider", "resamples"))
        return animation(
            "resampling", fig, steps,
            purpose=ctx.t("labs.boot.anim.purpose",
                          "Watch a sampling distribution being rebuilt from a single sample."),
            summary=ctx.t(
                "labs.boot.anim.summary",
                "The bootstrap substitutes the empirical distribution for the unknown "
                "population. That substitution is what makes it work for smooth statistics - "
                "and exactly what makes it fail when the statistic depends on the tail you "
                "never saw."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = BootstrapLab(SPEC)
