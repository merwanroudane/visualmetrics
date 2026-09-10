"""Fisher information and the Cramer-Rao lower bound."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...simulation.random import rng
from .mle import MODEL_SPECS

__all__ = ["LAB", "SPEC"]

ESTIMATORS = ("mle", "method_of_moments", "shrunk", "first_observation")

SPEC = make_spec(
    "inference.cramer_rao",
    Domain.INFERENCE,
    "estimation_theory",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "experiment", "compare", "derive", "simulate",
           "code", "quiz", "references"),
    evidence=EvidenceType.SYMBOLIC_DERIVATION,
    controls=(
        select("model", "normal_mean", ("normal_mean", "bernoulli_p", "poisson_lambda",
                                        "exponential_rate", "uniform_upper"), group="model"),
        slider("theta", 1.0, 0.05, 5.0, 0.01, group="model"),
        int_slider("n", 30, 2, 500, 1, group="design"),
        int_slider("reps", 3000, 200, 30000, 100, group="simulation", expensive=True),
        slider("shrink", 0.8, 0.0, 1.0, 0.01, group="estimators"),
        toggle("show_bound_curve", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", model="normal_mean", theta=1.0, n=30),
        scenario("efficient_mle", "compare_methods", model="poisson_lambda", theta=2.0),
        scenario("biased_but_better", "counterexample", model="normal_mean", shrink=0.7,
                 n=10),
        scenario("bound_fails", "boundary", model="uniform_upper", theta=2.0, n=20),
        scenario("small_sample", "small_sample", n=4),
        scenario("large_sample", "large_sample", n=400),
        scenario("bernoulli", "compare_methods", model="bernoulli_p", theta=0.3, n=60),
    ),
    prerequisites=("inference.mle",),
    related=("inference.mle",),
    tags=("cramer-rao", "efficiency", "information bound", "shrinkage"),
    aliases=("crlb", "borne de cramer-rao", "حد كرامر-راو", "efficiency bound"),
    backends=("scipy",),
    proof_ids=("inference.cramer_rao.bound",),
    references=(
        ref("Rao, C. R. (1945). Information and the accuracy attainable in the estimation "
            "of statistical parameters. Bulletin of the Calcutta Mathematical Society 37.",
            kind="paper"),
        ref("Lehmann, E. L. and Casella, G. (1998). Theory of Point Estimation.", kind="book"),
    ),
    curriculum_tags=("mit.14381",),
)


class CramerRaoLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        model = MODEL_SPECS[str(p["model"])]
        theta = float(p["theta"])
        if model.name == "bernoulli_p":
            theta = float(np.clip(theta, 0.02, 0.98))
        n, reps = int(p["n"]), int(p["reps"])
        shrink = float(p["shrink"])

        draws = self._simulate(model, theta, n, reps, state.seed, shrink)
        info_n = model.fisher(np.zeros(n), theta)
        bound = 1.0 / info_n if np.isfinite(info_n) and info_n > 0 else float("nan")

        res.dgp = ctx.t(
            "labs.crlb.dgp",
            "{reps} samples of size n = {n} from {label}, true parameter {theta}; four "
            "estimators are computed from each sample.",
            reps=reps, n=n, label=model.label, theta=fmt(theta, 3),
        )

        res.add_panel(ctx.panel(
            "distributions", self._dist_figure(ctx, draws, theta, bound, model),
            "labs.crlb.figure.distributions", evidence=EvidenceType.SIMULATION,
        ))
        res.add_panel(ctx.panel(
            "decomposition", self._decomposition_figure(ctx, draws, theta, bound),
            "labs.crlb.figure.decomposition", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))
        if p["show_bound_curve"]:
            res.add_panel(ctx.panel(
                "bound", self._bound_figure(ctx, model, theta, reps, state.seed, shrink),
                "labs.crlb.figure.bound", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("information", ctx.term("glossary.fisher_information"), info_n,
                   note=ctx.t("labs.crlb.metric.info_note",
                              "for the whole sample: n times the information in one observation"))
        res.metric("bound", ctx.term("glossary.cramer_rao_bound"), bound)
        for name in ESTIMATORS:
            vals = draws[name]
            bias = float(np.mean(vals) - theta)
            var = float(np.var(vals, ddof=1))
            res.metric(f"{name}_bias", ctx.t("labs.crlb.metric.bias", "{est}: bias",
                                             est=name), bias, reference=0.0)
            res.metric(f"{name}_var", ctx.t("labs.crlb.metric.var", "{est}: variance",
                                            est=name), var,
                       reference=bound if name in ("mle", "method_of_moments") else None)
            res.metric(f"{name}_mse", ctx.t("labs.crlb.metric.mse", "{est}: MSE",
                                            est=name), var + bias**2)

        unbiased_var = float(np.var(draws["mle"], ddof=1))
        res.assume("regularity", ctx.t("assumptions.regularity"), model.regular,
                   detail=ctx.t("labs.crlb.assume.regularity",
                                "The bound requires that the support does not depend on the "
                                "parameter and that differentiation under the integral sign "
                                "is valid."),
                   consequence="" if model.regular else ctx.t(
                       "labs.crlb.assume.violated",
                       "For Uniform(0, theta) the support moves with the parameter, the "
                       "bound does not apply, and the sample maximum beats it."))
        res.assume("unbiasedness", ctx.t("labs.crlb.assume.unbiased_label",
                                         "The bound constrains unbiased estimators only"),
                   True,
                   detail=ctx.t("labs.crlb.assume.unbiased",
                                "A biased estimator may legitimately have a smaller mean "
                                "squared error than the bound."))

        res.animations.append(self._animation(ctx, model, theta, state.seed, n, reps))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.cramer_rao.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.cramer_rao.intuition"))
        res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.inference.cramer_rao.math"),
                    kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.cramer_rao.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.cramer_rao.warning"), kind="warning")

        shrunk_mse = float(np.var(draws["shrunk"], ddof=1)
                           + (np.mean(draws["shrunk"]) - theta) ** 2)
        if np.isfinite(bound) and shrunk_mse < bound:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.crlb.counterexample",
                "The shrunk estimator has mean squared error {m}, below the Cramer-Rao "
                "bound of {b}. Nothing is broken: the bound restricts unbiased estimators, "
                "and this one trades bias for variance on purpose.",
                m=fmt(shrunk_mse, 5), b=fmt(bound, 5),
            ), kind="warning")
        if not model.regular:
            res.warnings.append(ctx.t(
                "labs.crlb.warn.irregular",
                "This family violates the regularity conditions, so the plotted bound is "
                "not a valid lower bound here at all.",
            ))
        return res

    def _simulate(self, model, theta, n, reps, seed, shrink):
        from ...simulation.monte_carlo import monte_carlo

        def experiment(gen):
            x = model.draw(gen, theta, n)
            mle = model.mle(x)
            return {
                "mle": mle,
                "method_of_moments": model.mom(x),
                "shrunk": shrink * mle,
                "first_observation": float(x[0]),
            }

        return monte_carlo(experiment, reps, seed).draws

    def _dist_figure(self, ctx, draws, theta, bound, model):
        fig = ctx.figure(
            "labs.crlb.figure.distributions",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=400,
        )
        roles = {"mle": "primary", "method_of_moments": "secondary",
                 "shrunk": "warning", "first_observation": "muted"}
        for name, role in roles.items():
            P.add_histogram(fig, draws[name], name.replace("_", " "), role,
                            theme=ctx.theme, nbins=50, opacity=0.42)
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        if np.isfinite(bound):
            se = np.sqrt(bound)
            x = np.linspace(theta - 4 * se, theta + 4 * se, 300)
            P.add_curve(fig, x, stats.norm.pdf(x, theta, se),
                        ctx.t("labs.crlb.trace.bound_curve",
                              "the tightest an unbiased estimator may be"),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.crlb.legend",
            "Four estimators from identical data. A single observation is unbiased and "
            "useless; the shrunk estimator is biased and often excellent.",
        ), theme=ctx.theme)
        return fig

    def _decomposition_figure(self, ctx, draws, theta, bound):
        names = list(ESTIMATORS)
        bias2 = [float((np.mean(draws[k]) - theta) ** 2) for k in names]
        var = [float(np.var(draws[k], ddof=1)) for k in names]
        fig = ctx.figure(
            "labs.crlb.figure.decomposition",
            xaxis_title=ctx.t("labs.crlb.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.crlb.axis.mse", "Mean squared error"),
            height=350,
            barmode="stack",
        )
        labels = [n.replace("_", " ") for n in names]
        P.add_bar(fig, labels, var, ctx.t("labs.crlb.trace.variance", "variance"),
                  "primary", theme=ctx.theme)
        P.add_bar(fig, labels, bias2, ctx.t("labs.crlb.trace.bias2", "bias squared"),
                  "warning", theme=ctx.theme)
        if np.isfinite(bound):
            P.add_hline(fig, bound, ctx.t("labs.crlb.trace.bound", "Cramer-Rao bound"),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.crlb.legend_decomposition",
            "MSE = variance + bias squared. The dashed line binds the variance of unbiased "
            "estimators only, which is why a stacked bar can legitimately sit below it.",
        ), theme=ctx.theme)
        return fig

    def _bound_figure(self, ctx, model, theta, reps, seed, shrink):
        sizes = np.unique(np.round(np.geomspace(2, 300, 12)).astype(int))
        sub = min(reps, 1500)
        var_mle, bounds = [], []
        for ns in sizes:
            d = self._simulate(model, theta, int(ns), sub, seed, shrink)
            var_mle.append(float(np.var(d["mle"], ddof=1)))
            info = model.fisher(np.zeros(int(ns)), theta)
            bounds.append(1.0 / info if np.isfinite(info) and info > 0 else np.nan)
        fig = ctx.figure(
            "labs.crlb.figure.bound",
            xaxis_title=ctx.t("labs.common.axis.sample_size"),
            yaxis_title=ctx.t("labs.common.axis.variance"),
            height=350,
        )
        P.add_curve(fig, sizes, var_mle,
                    ctx.t("labs.crlb.trace.mle_var", "simulated variance of the MLE"),
                    "primary", theme=ctx.theme)
        if np.any(np.isfinite(bounds)):
            P.add_curve(fig, sizes, bounds, ctx.t("labs.crlb.trace.bound",
                                                  "Cramer-Rao bound"),
                        "truth", theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.crlb.legend_bound",
            "Both curves fall like 1/n. The MLE meets the bound asymptotically; the gap at "
            "small n is exactly the finite-sample price of estimation.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, model, theta, seed, n_final, reps):
        go = P.require_plotly()
        sizes = np.unique(np.round(np.geomspace(2, max(n_final, 20), 16)).astype(int))
        frames, steps = [], []
        sub = min(reps, 1200)
        for i, ns in enumerate(sizes):
            d = self._simulate(model, theta, int(ns), sub, seed, 0.8)
            info = model.fisher(np.zeros(int(ns)), theta)
            bound = 1.0 / info if np.isfinite(info) and info > 0 else np.nan
            vals = d["mle"]
            lo, hi = theta - 3, theta + 3
            hist, edges = np.histogram(vals, bins=np.linspace(lo, hi, 61), density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])
            frames.append(go.Frame(name=str(ns), data=[go.Bar(x=centres, y=hist)]))
            steps.append(AnimationStep(
                id=f"n_{ns}",
                frame=i,
                title=ctx.t("labs.crlb.anim.title", "n = {n}", n=int(ns)),
                what_you_see=ctx.t("labs.crlb.anim.see",
                                   "The sampling distribution of the maximum-likelihood "
                                   "estimator at this sample size."),
                what_changed=ctx.t("labs.crlb.anim.changed",
                                   "Information rose to {i}, so the floor on variance fell "
                                   "to {b}.", i=fmt(info, 3), b=fmt(bound, 5)),
                why=ctx.t("labs.crlb.anim.why",
                          "Information is additive across independent observations, so it "
                          "grows linearly in n and the bound falls like 1/n."),
                interpretation=ctx.t("labs.crlb.anim.interpret",
                                     "Simulated variance is {v}, against a bound of {b}.",
                                     v=fmt(float(np.var(vals, ddof=1)), 5), b=fmt(bound, 5)),
                conclusion=ctx.t("labs.crlb.anim.conclude",
                                 "The MLE approaches the bound: it is asymptotically efficient."),
                warning=ctx.t("labs.crlb.anim.warn",
                              "Approaching the bound is an asymptotic statement. At small n "
                              "the estimator can sit well above it."),
                math="Var(theta_hat) >= 1 / I_n(theta),  I_n(theta) = n * I_1(theta)",
                outputs={"n": int(ns), "information": round(float(info), 4),
                         "bound": None if not np.isfinite(bound) else round(bound, 6)},
                active_assumptions=("independence", "regularity"),
                violated_assumptions=() if model.regular else ("regularity",),
                highlighted=("sampling_distribution",),
            ))
        fig = ctx.figure(
            "labs.crlb.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=380,
        )
        centres = np.linspace(theta - 3, theta + 3, 60)
        fig.add_trace(go.Bar(x=centres, y=np.zeros_like(centres),
                             marker={"color": ctx.color("primary")}, opacity=0.7,
                             name=ctx.t("labs.crlb.trace.mle_dist", "MLE")))
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        build_frames(fig, frames, duration=480, reduced_motion=ctx.reduced_motion,
                     slider_label="n")
        return animation(
            "information_accumulation", fig, steps,
            purpose=ctx.t("labs.crlb.anim.purpose",
                          "Watch information accumulate and the variance floor descend."),
            summary=ctx.t(
                "labs.crlb.anim.summary",
                "Information is what data buy you, and 1/information is the best precision "
                "an unbiased estimator can convert it into. The MLE reaches that ceiling "
                "asymptotically; nothing unbiased can beat it."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = CramerRaoLab(SPEC)
