"""Law of large numbers lab.

Running averages converging - and, in the Cauchy preset, conspicuously not
converging, because the theorem's finite-expectation condition fails.
"""

from __future__ import annotations

from typing import Any

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

__all__ = ["LAB", "SPEC", "PARENTS", "draw_parent"]

PARENTS = (
    "normal",
    "uniform",
    "bernoulli",
    "exponential",
    "lognormal",
    "student_t3",
    "pareto",
    "cauchy",
)

#: (has finite mean, has finite variance) - the conditions the theorems need
PARENT_MOMENTS = {
    "normal": (True, True),
    "uniform": (True, True),
    "bernoulli": (True, True),
    "exponential": (True, True),
    "lognormal": (True, True),
    "student_t3": (True, False),
    "pareto": (True, False),
    "cauchy": (False, False),
}


def draw_parent(gen, parent: str, size, p: float = 0.3) -> np.ndarray:
    """Draw from the chosen population."""
    if parent == "normal":
        return gen.normal(0.0, 1.0, size)
    if parent == "uniform":
        return gen.uniform(0.0, 1.0, size)
    if parent == "bernoulli":
        return (gen.random(size) < p).astype(float)
    if parent == "exponential":
        return gen.exponential(1.0, size)
    if parent == "lognormal":
        return gen.lognormal(0.0, 1.0, size)
    if parent == "student_t3":
        return gen.standard_t(3.0, size)
    if parent == "pareto":
        return (gen.pareto(1.5, size) + 1.0)
    if parent == "cauchy":
        return gen.standard_cauchy(size)
    return gen.standard_normal(size)


def parent_tail_probability(parent: str, threshold: float = 0.5, p: float = 0.3) -> float:
    """P(X > threshold) for each population, computed exactly.

    The law of large numbers is a statement about convergence to a *population*
    constant, so the target line has to be that constant. Estimating it from the
    same draws would quietly compare the sample with itself.
    """
    from scipy import stats

    if parent == "normal":
        return float(stats.norm.sf(threshold))
    if parent == "uniform":
        return float(min(max(1.0 - threshold, 0.0), 1.0))
    if parent == "bernoulli":
        # The draws are 0 or 1, so exceeding 0.5 means drawing a one.
        return float(p) if threshold < 1.0 else 0.0
    if parent == "exponential":
        return float(stats.expon.sf(threshold))
    if parent == "lognormal":
        return float(stats.lognorm.sf(threshold, s=1.0, scale=1.0))
    if parent == "student_t3":
        return float(stats.t.sf(threshold, df=3.0))
    if parent == "pareto":
        # draw_parent returns gen.pareto(1.5) + 1, which is a Pareto with
        # shape 1.5 and scale 1 supported on [1, inf).
        return float(stats.pareto.sf(threshold, b=1.5)) if threshold >= 1.0 else 1.0
    if parent == "cauchy":
        return float(stats.cauchy.sf(threshold))
    return float(stats.norm.sf(threshold))


def parent_mean(parent: str, p: float = 0.3) -> float:
    return {
        "normal": 0.0,
        "uniform": 0.5,
        "bernoulli": p,
        "exponential": 1.0,
        "lognormal": float(np.exp(0.5)),
        "student_t3": 0.0,
        "pareto": 1.5 / 0.5,
        "cauchy": float("nan"),
    }[parent]


SPEC = make_spec(
    "inference.lln",
    Domain.INFERENCE,
    "limit_theory",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("parent", "exponential", PARENTS, group="population"),
        slider("bernoulli_p", 0.3, 0.01, 0.99, 0.01, group="population",
               depends_on=("parent", ("bernoulli",))),
        int_slider("n_max", 2000, 50, 50000, 50, group="design", expensive=True),
        int_slider("n_paths", 8, 1, 60, 1, group="design"),
        select("statistic", "mean", ("mean", "proportion", "variance"), group="design"),
        toggle("log_x", True, group="views"),
        toggle("show_band", True, group="views"),
        toggle("show_distance", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", parent="exponential", n_max=2000, n_paths=8),
        scenario("coin_flips", "positive", parent="bernoulli", statistic="proportion",
                 bernoulli_p=0.3, n_max=3000),
        scenario("fast_convergence", "low_noise", parent="uniform", n_max=1000),
        scenario("slow_convergence", "high_noise", parent="lognormal", n_max=20000),
        scenario("heavy_tail_finite_mean", "boundary", parent="pareto", n_max=20000),
        scenario("infinite_variance", "violation", parent="student_t3", n_max=20000),
        scenario("cauchy_counterexample", "counterexample", parent="cauchy",
                 n_max=20000, n_paths=12),
        scenario("many_paths", "sensitivity", n_paths=40, n_max=5000),
        scenario("small_sample", "small_sample", n_max=100, n_paths=20),
        scenario("large_sample", "large_sample", n_max=50000, n_paths=4),
    ),
    prerequisites=("probability.distributions",),
    related=("inference.clt", "inference.sampling_distributions"),
    next_concepts=("inference.clt",),
    tags=("law of large numbers", "convergence", "consistency", "average"),
    aliases=("lln", "loi des grands nombres", "قانون الأعداد الكبيرة",
             "weak law", "strong law"),
    backends=("numpy", "scipy"),
    misconceptions=("gamblers_fallacy", "lln_is_proof"),
    references=(
        ref("Billingsley, P. (1995). Probability and Measure.", kind="book"),
        ref("MIT 14.381 Statistical Method in Economics", kind="course",
            url="https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/pages/syllabus/"),
    ),
    curriculum_tags=("dz.stat4", "mit.14381"),
)


class LLNLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        parent = str(p["parent"])
        n_max = int(p["n_max"])
        n_paths = int(p["n_paths"])
        stat = str(p["statistic"])
        bp = float(p["bernoulli_p"])

        finite_mean, finite_var = PARENT_MOMENTS[parent]
        gen = rng(state.seed, "lln", parent)
        draws = draw_parent(gen, parent, (n_paths, n_max), bp)

        if stat == "variance":
            running = self._running_variance(draws)
            target = self._target_variance(parent, bp)
        elif stat == "proportion":
            running = np.cumsum(draws > 0.5, axis=1) / np.arange(1, n_max + 1)
            target = parent_tail_probability(parent, 0.5, bp)
        else:
            running = np.cumsum(draws, axis=1) / np.arange(1, n_max + 1)
            target = parent_mean(parent, bp)

        idx = np.arange(1, n_max + 1)
        res.dgp = ctx.t(
            "labs.lln.dgp",
            "{paths} independent paths from a {parent} population; running {stat} of the "
            "first n observations, n up to {n}.",
            paths=n_paths, parent=parent, stat=stat, n=n_max,
        )

        res.add_panel(ctx.panel(
            "paths", self._paths_figure(ctx, idx, running, target, p, parent, finite_var),
            "labs.lln.figure.paths", evidence=EvidenceType.SIMULATION,
        ))
        if p["show_distance"]:
            res.add_panel(ctx.panel(
                "distance", self._distance_figure(ctx, idx, running, target, p),
                "labs.lln.figure.distance", tab="diagnostics",
                evidence=EvidenceType.SIMULATION,
            ))
        res.add_panel(ctx.panel(
            "spread", self._spread_figure(ctx, running, target, n_max),
            "labs.lln.figure.spread", tab="compare", evidence=EvidenceType.SIMULATION,
        ))

        final = running[:, -1]
        res.metric("target", ctx.t("labs.common.trace.expectation"),
                   target if np.isfinite(target) else "does not exist")
        res.metric("mean_final", ctx.t("labs.lln.metric.mean_final",
                                       "Average of the {k} final values", k=n_paths),
                   float(np.mean(final)))
        res.metric("spread_final", ctx.t("labs.lln.metric.spread",
                                         "Spread of the final values (sd)"),
                   float(np.std(final, ddof=1)) if n_paths > 1 else 0.0)
        if np.isfinite(target):
            res.metric("max_error", ctx.t("labs.lln.metric.max_error",
                                          "Largest remaining distance to the target"),
                       float(np.max(np.abs(final - target))))
        for checkpoint in (10, 100, 1000):
            if checkpoint <= n_max and np.isfinite(target):
                res.metric(
                    f"error_at_{checkpoint}",
                    ctx.t("labs.lln.metric.error_at", "Mean |error| at n = {n}", n=checkpoint),
                    float(np.mean(np.abs(running[:, checkpoint - 1] - target))),
                )

        res.assume("independence", ctx.t("assumptions.independence"), True)
        res.assume("identical_distribution", ctx.t("assumptions.identical_distribution"), True)
        res.assume("finite_mean", ctx.t("labs.lln.assume.finite_mean",
                                        "The population expectation exists"), finite_mean,
                   detail=ctx.t("labs.lln.assume.finite_mean_detail",
                                "Without a finite expectation there is nothing for the "
                                "average to converge to."),
                   consequence="" if finite_mean else ctx.t(
                       "labs.lln.assume.no_mean_consequence",
                       "The running average keeps jumping forever; it never settles."))
        res.assume("finite_variance", ctx.t("assumptions.finite_variance"), finite_var,
                   detail=ctx.t("labs.lln.assume.finite_var_detail",
                                "A finite variance is not needed for the law of large "
                                "numbers, only for the usual 1/sqrt(n) rate."))

        res.animations.append(self._animation(ctx, idx, running, target, parent, n_max))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.inference.lln.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"), ctx.t("concepts.inference.lln.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.inference.lln.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.lln.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.inference.lln.warning"),
                    kind="warning")

        if not finite_mean:
            res.warnings.append(ctx.t(
                "labs.lln.warn.cauchy",
                "The Cauchy distribution has no expectation. The average of n Cauchy draws "
                "is itself Cauchy with the same scale, so it never stabilizes no matter how "
                "large n becomes. This is a genuine counterexample, not slow convergence.",
            ))
            res.explain("counterexample", ctx.t("modes.counterexample"),
                        res.warnings[-1], kind="warning")
        elif not finite_var:
            res.warnings.append(ctx.t(
                "labs.lln.warn.heavy_tail",
                "The expectation exists, so the average does converge - but with infinite "
                "variance the usual 1/sqrt(n) rate does not apply and convergence is much "
                "slower and lumpier than the normal case.",
            ))
        return res

    @staticmethod
    def _running_variance(draws: np.ndarray) -> np.ndarray:
        n = np.arange(1, draws.shape[1] + 1)
        csum = np.cumsum(draws, axis=1)
        csum2 = np.cumsum(draws**2, axis=1)
        mean = csum / n
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (csum2 - n * mean**2) / np.maximum(n - 1, 1)
        return var

    @staticmethod
    def _target_variance(parent: str, p: float) -> float:
        return {
            "normal": 1.0,
            "uniform": 1.0 / 12.0,
            "bernoulli": p * (1 - p),
            "exponential": 1.0,
            "lognormal": float((np.e - 1) * np.e),
            "student_t3": float("inf"),
            "pareto": float("inf"),
            "cauchy": float("nan"),
        }[parent]

    def _paths_figure(self, ctx, idx, running, target, p, parent, finite_var):
        fig = ctx.figure(
            "labs.lln.figure.paths",
            xaxis_title=ctx.t("labs.common.axis.observations"),
            yaxis_title=ctx.t("labs.common.axis.running_mean"),
            height=440,
        )
        if p["show_band"] and np.isfinite(target) and finite_var:
            sd = float(np.nanstd(running[:, min(200, running.shape[1] - 1)], ddof=1)) or 1.0
            scale = sd * np.sqrt(min(200, running.shape[1]))
            band = 1.96 * scale / np.sqrt(idx)
            P.shade_between(fig, idx, target - band, target + band,
                            ctx.t("labs.lln.trace.band",
                                  "approximate 95% band, width shrinking like 1/sqrt(n)"),
                            "info", theme=ctx.theme, alpha=0.14)
        for i in range(running.shape[0]):
            P.add_curve(fig, idx, running[i],
                        ctx.t("labs.lln.trace.path", "path {i}", i=i + 1),
                        "primary" if i == 0 else "muted", theme=ctx.theme,
                        width=2.2 if i == 0 else 1.0,
                        opacity=1.0 if i == 0 else 0.55,
                        showlegend=i < 3, dash="solid")
        if np.isfinite(target):
            P.add_hline(fig, target,
                        ctx.t("labs.lln.trace.target", "E[X] = {v}", v=fmt(target, 3)),
                        "truth", theme=ctx.theme, dash="solid", width=2.4)
        if p["log_x"]:
            fig.update_xaxes(type="log")
        note = (ctx.t("labs.lln.legend",
                      "Each grey line is one sequence of running averages. They squeeze "
                      "towards the black target line as n grows.")
                if np.isfinite(target) else
                ctx.t("labs.lln.legend_no_mean",
                      "There is no target line to draw: this population has no expectation, "
                      "so the running averages have nothing to converge to."))
        P.add_legend_note(fig, note + "  " + ctx.t("labs.common.note.simulation"),
                          theme=ctx.theme)
        return fig

    def _distance_figure(self, ctx, idx, running, target, p):
        fig = ctx.figure(
            "labs.lln.figure.distance",
            xaxis_title=ctx.t("labs.common.axis.observations"),
            yaxis_title=ctx.t("labs.lln.axis.distance", "|running mean - target|"),
            height=340,
        )
        if not np.isfinite(target):
            return P.empty_figure(
                ctx.t("labs.lln.no_distance",
                      "No target exists for this population, so no distance can be plotted."),
                theme=ctx.theme,
            )
        dist = np.abs(running - target)
        P.add_curve(fig, idx, np.mean(dist, axis=0),
                    ctx.t("labs.lln.trace.mean_distance", "average distance across paths"),
                    "primary", theme=ctx.theme)
        P.add_curve(fig, idx, np.max(dist, axis=0),
                    ctx.t("labs.lln.trace.max_distance", "worst path"),
                    "warning", theme=ctx.theme, dash="dot")
        ref_curve = dist[:, min(9, dist.shape[1] - 1)].mean() * np.sqrt(10) / np.sqrt(idx)
        P.add_curve(fig, idx, ref_curve,
                    ctx.t("labs.lln.trace.rate", "1/sqrt(n) reference"),
                    "baseline", theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.lln.legend_distance",
            "On log-log axes a 1/sqrt(n) rate is a straight line of slope -1/2. Departures "
            "from that line tell you the convergence rate is not the textbook one.",
        ), theme=ctx.theme)
        return fig

    def _spread_figure(self, ctx, running, target, n_max):
        checkpoints = [c for c in (10, 30, 100, 300, 1000, 3000, 10000, 30000) if c <= n_max]
        if not checkpoints:
            checkpoints = [n_max]
        fig = ctx.figure(
            "labs.lln.figure.spread",
            xaxis_title=ctx.t("labs.common.axis.sample_size"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=340,
        )
        go = P.require_plotly()
        for c in checkpoints:
            vals = running[:, c - 1]
            fig.add_trace(go.Box(y=vals, name=str(c), boxpoints="all", jitter=0.4,
                                 marker={"color": ctx.color("primary")},
                                 line={"color": ctx.color("primary")}))
        if np.isfinite(target):
            P.add_hline(fig, target, ctx.t("labs.common.trace.truth"), "truth",
                        theme=ctx.theme, dash="solid")
        fig.update_layout(showlegend=False)
        P.add_legend_note(fig, ctx.t(
            "labs.lln.legend_spread",
            "Each box collects the running averages of every path at that sample size. The "
            "boxes narrow around the target; that narrowing is what convergence looks like.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, idx, running, target, parent, n_max):
        go = P.require_plotly()
        checkpoints = np.unique(np.round(np.geomspace(5, n_max, 24)).astype(int))
        frames, steps = [], []
        for i, c in enumerate(checkpoints):
            frames.append(
                go.Frame(
                    name=str(c),
                    data=[go.Scatter(x=idx[:c], y=running[j, :c])
                          for j in range(min(running.shape[0], 6))],
                )
            )
            window = running[:, c - 1]
            err = (float(np.mean(np.abs(window - target))) if np.isfinite(target)
                   else float("nan"))
            steps.append(AnimationStep(
                id=f"n_{c}",
                frame=i,
                title=ctx.t("labs.lln.anim.title", "First {n} observations", n=int(c)),
                what_you_see=ctx.t(
                    "labs.lln.anim.see",
                    "Running averages of several independent sequences drawn from the same "
                    "{parent} population.", parent=parent),
                what_changed=ctx.t(
                    "labs.lln.anim.changed",
                    "The averages now use {n} observations each.", n=int(c)),
                why=ctx.t(
                    "labs.lln.anim.why",
                    "Each new observation enters the average with weight 1/n, so its power "
                    "to move the average keeps shrinking while the accumulated centre stays."),
                interpretation=(
                    ctx.t("labs.lln.anim.interpret",
                          "The paths are now within {e} of the target on average.",
                          e=fmt(err, 4))
                    if np.isfinite(err) else
                    ctx.t("labs.lln.anim.interpret_none",
                          "The paths are still wandering: with no expectation there is no "
                          "value for them to approach.")),
                conclusion=(
                    ctx.t("labs.lln.anim.conclude",
                          "Larger n concentrates the sample average around the expectation.")
                    if np.isfinite(err) else
                    ctx.t("labs.lln.anim.conclude_none",
                          "The law of large numbers simply does not apply here.")),
                warning=ctx.t("labs.lln.anim.warn",
                              "Watching finitely many finite paths is evidence, not proof."),
                math="mean_n = (1/n) * sum(X_i);  P(|mean_n - mu| > eps) -> 0",
                outputs={"n": int(c), "mean_abs_error": None if not np.isfinite(err)
                         else round(err, 5)},
                active_assumptions=("independence", "identical_distribution"),
                violated_assumptions=() if np.isfinite(target) else ("finite_mean",),
                highlighted=("paths", "target_line"),
            ))

        fig = ctx.figure(
            "labs.lln.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.observations"),
            yaxis_title=ctx.t("labs.common.axis.running_mean"),
            height=400,
        )
        for j in range(min(running.shape[0], 6)):
            P.add_curve(fig, idx[:5], running[j, :5],
                        ctx.t("labs.lln.trace.path", "path {i}", i=j + 1),
                        "primary" if j == 0 else "muted", theme=ctx.theme,
                        showlegend=j < 2)
        if np.isfinite(target):
            P.add_hline(fig, target, ctx.t("labs.common.trace.expectation"), "truth",
                        theme=ctx.theme, dash="solid")
        fig.update_xaxes(type="log", range=[np.log10(1), np.log10(n_max)])
        finite = running[np.isfinite(running)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            pad = 0.4 * (hi - lo + 1e-9)
            fig.update_yaxes(range=[lo - pad, hi + pad])
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label="n")
        return animation(
            "convergence",
            fig,
            steps,
            purpose=ctx.t("labs.lln.anim.purpose",
                          "Watch independent sample averages settle - or fail to."),
            summary=(
                ctx.t("labs.lln.anim.summary",
                      "The paths converge because averaging dilutes each individual "
                      "observation. Note what did NOT happen: no unlucky stretch was ever "
                      "compensated by a lucky one. Deviations were diluted, not corrected.")
                if np.isfinite(target) else
                ctx.t("labs.lln.anim.summary_none",
                      "The paths never settle. With no finite expectation the average of n "
                      "draws has exactly the same distribution as a single draw, so "
                      "collecting more data changes nothing at all.")),
            evidence=EvidenceType.SIMULATION,
        )


LAB = LLNLab(SPEC)
