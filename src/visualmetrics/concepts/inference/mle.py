"""Maximum likelihood lab: likelihood surface, score, curvature, MoM comparison."""

from __future__ import annotations

from typing import Any

from scipy import optimize, stats

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

__all__ = ["LAB", "SPEC", "MODELS", "MODEL_SPECS"]

MODELS = ("normal_mean", "bernoulli_p", "poisson_lambda", "exponential_rate",
          "uniform_upper", "normal_variance")


class ModelSpec:
    """Everything the lab needs about one one-parameter family."""

    def __init__(self, name, label, grid, draw, loglik, mle, mom, fisher, regular=True):
        self.name = name
        self.label = label
        self.grid = grid
        self.draw = draw
        self.loglik = loglik
        self.mle = mle
        self.mom = mom
        self.fisher = fisher
        self.regular = regular


def _normal_mean():
    return ModelSpec(
        "normal_mean", "mu of N(mu, 1)",
        lambda theta: np.linspace(theta - 3, theta + 3, 400),
        lambda gen, theta, n: gen.normal(theta, 1.0, n),
        lambda x, t: -0.5 * np.sum((x[:, None] - np.atleast_1d(t)[None, :]) ** 2, axis=0)
        - x.size / 2 * np.log(2 * np.pi),
        lambda x: float(x.mean()),
        lambda x: float(x.mean()),
        lambda x, t: float(x.size),
    )


def _bernoulli_p():
    def loglik(x, t):
        t = np.clip(np.atleast_1d(t), 1e-9, 1 - 1e-9)
        s = x.sum()
        return s * np.log(t) + (x.size - s) * np.log(1 - t)

    return ModelSpec(
        "bernoulli_p", "p of Bernoulli(p)",
        lambda theta: np.linspace(0.001, 0.999, 400),
        lambda gen, theta, n: (gen.random(n) < theta).astype(float),
        loglik,
        lambda x: float(x.mean()),
        lambda x: float(x.mean()),
        lambda x, t: float(x.size / max(t * (1 - t), 1e-9)),
    )


def _poisson_lambda():
    from scipy.special import gammaln

    def loglik(x, t):
        t = np.clip(np.atleast_1d(t), 1e-9, None)
        return x.sum() * np.log(t) - x.size * t - float(gammaln(x + 1).sum())

    return ModelSpec(
        "poisson_lambda", "lambda of Poisson(lambda)",
        lambda theta: np.linspace(max(0.05, theta - 3), theta + 3, 400),
        lambda gen, theta, n: gen.poisson(theta, n).astype(float),
        loglik,
        lambda x: float(x.mean()),
        lambda x: float(x.mean()),
        lambda x, t: float(x.size / max(t, 1e-9)),
    )


def _exponential_rate():
    def loglik(x, t):
        t = np.clip(np.atleast_1d(t), 1e-9, None)
        return x.size * np.log(t) - t * x.sum()

    return ModelSpec(
        "exponential_rate", "rate of Exponential(rate)",
        lambda theta: np.linspace(max(0.02, theta - 2), theta + 2, 400),
        lambda gen, theta, n: gen.exponential(1.0 / theta, n),
        loglik,
        lambda x: float(1.0 / max(x.mean(), 1e-9)),
        lambda x: float(1.0 / max(x.mean(), 1e-9)),
        lambda x, t: float(x.size / max(t**2, 1e-9)),
    )


def _uniform_upper():
    def loglik(x, t):
        t = np.atleast_1d(t)
        out = np.where(t >= x.max(), -x.size * np.log(np.clip(t, 1e-9, None)), -np.inf)
        return out

    return ModelSpec(
        "uniform_upper", "theta of Uniform(0, theta)",
        lambda theta: np.linspace(0.05, theta * 2, 500),
        lambda gen, theta, n: gen.uniform(0.0, theta, n),
        loglik,
        lambda x: float(x.max()),
        lambda x: float(2 * x.mean()),
        lambda x, t: float("nan"),
        regular=False,
    )


def _normal_variance():
    def loglik(x, t):
        t = np.clip(np.atleast_1d(t), 1e-9, None)
        return -0.5 * x.size * np.log(2 * np.pi * t) - np.sum(x**2) / (2 * t)

    return ModelSpec(
        "normal_variance", "sigma^2 of N(0, sigma^2)",
        lambda theta: np.linspace(max(0.05, theta * 0.2), theta * 3, 400),
        lambda gen, theta, n: gen.normal(0.0, np.sqrt(theta), n),
        loglik,
        lambda x: float(np.mean(x**2)),
        lambda x: float(np.mean(x**2)),
        lambda x, t: float(x.size / (2 * max(t**2, 1e-9))),
    )


MODEL_SPECS = {
    "normal_mean": _normal_mean(),
    "bernoulli_p": _bernoulli_p(),
    "poisson_lambda": _poisson_lambda(),
    "exponential_rate": _exponential_rate(),
    "uniform_upper": _uniform_upper(),
    "normal_variance": _normal_variance(),
}


SPEC = make_spec(
    "inference.mle",
    Domain.INFERENCE,
    "estimation_theory",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive",
           "simulate", "code", "quiz", "references"),
    evidence=EvidenceType.VISUAL_DERIVATION,
    controls=(
        select("model", "bernoulli_p", MODELS, group="model"),
        slider("theta", 0.35, 0.05, 5.0, 0.01, group="model"),
        int_slider("n", 40, 2, 2000, 1, group="design"),
        int_slider("reps", 2000, 200, 20000, 100, group="simulation", expensive=True),
        toggle("show_score", True, group="views"),
        toggle("show_sampling", True, group="views"),
        toggle("compare_mom", True, group="views"),
        toggle("log_scale", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", model="bernoulli_p", theta=0.35, n=40),
        scenario("small_sample", "small_sample", n=5),
        scenario("large_sample", "large_sample", n=1000),
        scenario("normal_mean", "compare_methods", model="normal_mean", theta=1.5, n=40),
        scenario("poisson", "compare_methods", model="poisson_lambda", theta=2.5, n=50),
        scenario("exponential", "compare_methods", model="exponential_rate",
                 theta=1.2, n=50),
        scenario("variance", "compare_methods", model="normal_variance", theta=2.0, n=40),
        scenario("mle_beats_mom", "compare_methods", model="uniform_upper",
                 theta=3.0, n=20),
        scenario("boundary_irregular", "boundary", model="uniform_upper", theta=3.0, n=8),
        scenario("rare_event", "weak", model="bernoulli_p", theta=0.03, n=40),
        scenario("flat_likelihood", "weak", model="bernoulli_p", theta=0.5, n=4),
    ),
    prerequisites=("probability.distributions",),
    related=("inference.cramer_rao", "inference.hypothesis_testing"),
    next_concepts=("inference.cramer_rao",),
    tags=("maximum likelihood", "score", "fisher information", "method of moments"),
    aliases=("mle", "maximum de vraisemblance", "الإمكان الأعظم", "likelihood"),
    backends=("scipy",),
    references=(
        ref("Casella, G. and Berger, R. L. (2002). Statistical Inference.", kind="book"),
        ref("MIT 14.381 Statistical Method in Economics", kind="course",
            url="https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/pages/syllabus/"),
    ),
    curriculum_tags=("dz.stat4", "mit.14381"),
)


class MLELab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        model = MODEL_SPECS[str(p["model"])]
        theta = float(p["theta"])
        if model.name == "bernoulli_p":
            theta = float(np.clip(theta, 0.01, 0.99))
        n = int(p["n"])

        gen = rng(state.seed, "mle", model.name, n)
        x = model.draw(gen, theta, n)
        grid = model.grid(theta)
        ll = np.asarray(model.loglik(x, grid), dtype=float)
        hat = model.mle(x)
        mom = model.mom(x)
        info = model.fisher(x, hat)

        res.dgp = ctx.t(
            "labs.mle.dgp",
            "n = {n} observations from {label} with the true parameter set to {theta}.",
            n=n, label=model.label, theta=fmt(theta, 3),
        )

        res.add_panel(ctx.panel(
            "likelihood", self._likelihood_figure(ctx, grid, ll, hat, mom, theta, p, model),
            "labs.mle.figure.likelihood", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_score"] and model.regular:
            res.add_panel(ctx.panel(
                "score", self._score_figure(ctx, grid, ll, hat, theta),
                "labs.mle.figure.score", tab="math",
                evidence=EvidenceType.VISUAL_DERIVATION,
            ))
        if p["show_sampling"]:
            samp = self._sampling(model, theta, n, int(p["reps"]), state.seed)
            res.add_panel(ctx.panel(
                "sampling", self._sampling_figure(ctx, samp, theta, info, model, p),
                "labs.mle.figure.sampling", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            res.metric("mle_bias", ctx.t("labs.mle.metric.bias", "Bias of the MLE"),
                       float(np.mean(samp["mle"]) - theta), reference=0.0)
            res.metric("mle_sd", ctx.t("labs.mle.metric.sd", "SD of the MLE"),
                       float(np.std(samp["mle"], ddof=1)),
                       reference=float(1 / np.sqrt(info)) if np.isfinite(info) and info > 0
                       else None,
                       note=ctx.t("labs.mle.metric.sd_note",
                                  "compared with 1/sqrt(information)"))
            if p["compare_mom"]:
                res.metric("mom_bias", ctx.t("labs.mle.metric.mom_bias",
                                             "Bias of the method-of-moments estimator"),
                           float(np.mean(samp["mom"]) - theta), reference=0.0)
                res.metric("mom_sd", ctx.t("labs.mle.metric.mom_sd",
                                           "SD of the method-of-moments estimator"),
                           float(np.std(samp["mom"], ddof=1)))
                eff = (np.var(samp["mle"], ddof=1) / max(np.var(samp["mom"], ddof=1), 1e-12))
                res.metric("relative_efficiency",
                           ctx.t("labs.mle.metric.efficiency",
                                 "Variance ratio MLE / MoM"), float(eff),
                           note=ctx.t("labs.mle.metric.efficiency_note",
                                      "below 1 means the MLE is more precise here"))

        res.metric("mle", ctx.t("labs.mle.metric.mle", "MLE from this sample"), hat,
                   reference=theta)
        res.metric("mom", ctx.t("labs.mle.metric.mom", "Method of moments from this sample"),
                   mom, reference=theta)
        if np.isfinite(info):
            res.metric("information", ctx.term("glossary.fisher_information"), info)
            res.metric("se_asymptotic", ctx.t("labs.mle.metric.se",
                                              "Asymptotic standard error"),
                       float(1 / np.sqrt(max(info, 1e-12))))
        res.metric("curvature", ctx.t("labs.mle.metric.curvature",
                                      "Observed curvature at the optimum"),
                   self._curvature(grid, ll, hat))

        res.assume("independence", ctx.t("assumptions.independence"), True)
        res.assume("regularity", ctx.t("assumptions.regularity"), model.regular,
                   detail=ctx.t("labs.mle.assume.regularity",
                                "The asymptotic normality and information results require "
                                "that the support does not depend on the parameter and that "
                                "the optimum is interior."),
                   consequence="" if model.regular else ctx.t(
                       "labs.mle.assume.irregular",
                       "For Uniform(0, theta) the maximum sits on the boundary of the "
                       "support, the score is undefined there, and the estimator converges "
                       "at rate n rather than sqrt(n)."))

        res.animations.append(self._animation(ctx, model, theta, state.seed, n))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.inference.mle.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"), ctx.t("concepts.inference.mle.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.inference.mle.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.mle.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.inference.mle.warning"),
                    kind="warning")
        if not model.regular:
            res.warnings.append(ctx.t(
                "labs.mle.warn.boundary",
                "The likelihood for Uniform(0, theta) is zero below the sample maximum and "
                "decreasing above it, so the maximum is a corner, not a stationary point. "
                "Setting the derivative to zero would find nothing at all here.",
            ))
        return res

    @staticmethod
    def _curvature(grid, ll, hat):
        i = int(np.argmin(np.abs(grid - hat)))
        i = int(np.clip(i, 1, len(grid) - 2))
        h = grid[1] - grid[0]
        finite = np.isfinite(ll[i - 1: i + 2])
        if not finite.all():
            return float("nan")
        return float(-(ll[i + 1] - 2 * ll[i] + ll[i - 1]) / h**2)

    def _likelihood_figure(self, ctx, grid, ll, hat, mom, theta, p, model):
        fig = ctx.figure(
            "labs.mle.figure.likelihood",
            xaxis_title=ctx.t("labs.common.axis.parameter"),
            yaxis_title=(ctx.t("labs.common.axis.loglik") if p["log_scale"]
                         else ctx.t("labs.mle.axis.likelihood", "Likelihood")),
            height=430,
        )
        finite = np.isfinite(ll)
        if p["log_scale"]:
            y = np.where(finite, ll, np.nan)
        else:
            shifted = ll - np.nanmax(ll[finite])
            y = np.where(finite, np.exp(shifted), np.nan)
        P.add_curve(fig, grid, y,
                    ctx.t("labs.mle.trace.loglik", "log-likelihood") if p["log_scale"]
                    else ctx.t("labs.mle.trace.lik", "likelihood"),
                    "primary", theme=ctx.theme)
        P.add_vline(fig, hat, ctx.t("labs.mle.trace.mle", "MLE = {v}", v=fmt(hat, 4)),
                    "estimate", theme=ctx.theme, dash="solid")
        if p["compare_mom"] and abs(mom - hat) > 1e-9:
            P.add_vline(fig, mom, ctx.t("labs.mle.trace.mom", "MoM = {v}", v=fmt(mom, 4)),
                        "secondary", theme=ctx.theme, dash="dashdot")
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme,
                    dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.mle.legend",
            "The likelihood is a function of the parameter for FIXED data. Its peak is the "
            "estimate; how sharp that peak is becomes the standard error.",
        ), theme=ctx.theme)
        return fig

    def _score_figure(self, ctx, grid, ll, hat, theta):
        with np.errstate(invalid="ignore"):
            score = np.gradient(ll, grid)
        fig = ctx.figure(
            "labs.mle.figure.score",
            xaxis_title=ctx.t("labs.common.axis.parameter"),
            yaxis_title=ctx.t("labs.mle.axis.score", "Score (d log L / d theta)"),
            height=340,
        )
        P.add_curve(fig, grid, score, ctx.t("labs.mle.trace.score", "score function"),
                    "secondary", theme=ctx.theme)
        P.add_hline(fig, 0.0, ctx.t("labs.mle.trace.zero", "score = 0"), "baseline",
                    theme=ctx.theme)
        P.add_vline(fig, hat, ctx.t("labs.mle.trace.mle", "MLE"), "estimate",
                    theme=ctx.theme, dash="solid")
        P.add_legend_note(fig, ctx.t(
            "labs.mle.legend_score",
            "The estimate is where the score crosses zero. How steeply it crosses is the "
            "observed information - a steep crossing pins the parameter down tightly.",
        ), theme=ctx.theme)
        return fig

    def _sampling(self, model, theta, n, reps, seed):
        from ...simulation.monte_carlo import monte_carlo

        def experiment(gen):
            x = model.draw(gen, theta, n)
            return {"mle": model.mle(x), "mom": model.mom(x)}

        mc = monte_carlo(experiment, reps, seed, truth={"mle": theta, "mom": theta})
        return mc.draws

    def _sampling_figure(self, ctx, samp, theta, info, model, p):
        fig = ctx.figure(
            "labs.mle.figure.sampling",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=360,
        )
        P.add_histogram(fig, samp["mle"], ctx.t("labs.mle.trace.mle_dist", "MLE"),
                        "primary", theme=ctx.theme, nbins=50, opacity=0.6)
        if p["compare_mom"] and np.std(samp["mom"]) > 1e-12:
            P.add_histogram(fig, samp["mom"],
                            ctx.t("labs.mle.trace.mom_dist", "method of moments"),
                            "secondary", theme=ctx.theme, nbins=50, opacity=0.45)
        if np.isfinite(info) and info > 0:
            se = 1 / np.sqrt(info)
            x = np.linspace(theta - 4 * se, theta + 4 * se, 300)
            P.add_curve(fig, x, stats.norm.pdf(x, theta, se),
                        ctx.t("labs.mle.trace.asymptotic",
                              "asymptotic N(theta, 1/I) approximation"),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.mle.legend_sampling",
            "The asymptotic normal curve is what the theory promises for large n. Where the "
            "histogram departs from it, the promise has not arrived yet.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, model, theta, seed, n_final):
        go = P.require_plotly()
        sizes = np.unique(np.round(np.geomspace(2, max(n_final, 20), 18)).astype(int))
        grid = model.grid(theta)
        frames, steps = [], []
        for i, ns in enumerate(sizes):
            gen = rng(seed, "mle_anim", int(ns))
            x = model.draw(gen, theta, int(ns))
            ll = np.asarray(model.loglik(x, grid), dtype=float)
            finite = np.isfinite(ll)
            norm = np.where(finite, np.exp(ll - np.nanmax(ll[finite])), np.nan)
            hat = model.mle(x)
            info = model.fisher(x, hat)
            frames.append(go.Frame(name=str(ns), data=[go.Scatter(x=grid, y=norm)]))
            steps.append(AnimationStep(
                id=f"n_{ns}",
                frame=i,
                title=ctx.t("labs.mle.anim.title", "n = {n} observations", n=int(ns)),
                what_you_see=ctx.t(
                    "labs.mle.anim.see",
                    "The likelihood rescaled to a maximum of one, so only its shape matters."),
                what_changed=ctx.t("labs.mle.anim.changed",
                                   "The sample now holds {n} observations; the estimate is {h}.",
                                   n=int(ns), h=fmt(hat, 4)),
                why=ctx.t(
                    "labs.mle.anim.why",
                    "Each observation contributes an additive term to the log-likelihood, so "
                    "information accumulates and the curve becomes proportionally sharper."),
                interpretation=ctx.t(
                    "labs.mle.anim.interpret",
                    "A narrower peak means a smaller standard error: currently about {se}.",
                    se=fmt(1 / np.sqrt(info), 4) if np.isfinite(info) and info > 0 else "n/a"),
                conclusion=ctx.t(
                    "labs.mle.anim.conclude",
                    "Consistency is this concentration: eventually all the likelihood mass "
                    "sits arbitrarily close to the true value."),
                warning=ctx.t(
                    "labs.mle.anim.warn",
                    "The rescaled height is always one. Do not read the peak's height as "
                    "confidence - read its width."),
                math="log L(theta) = sum log f(x_i; theta);  I_n(theta) = n * I_1(theta)",
                outputs={"n": int(ns), "mle": round(hat, 5),
                         "information": None if not np.isfinite(info) else round(info, 4)},
                active_assumptions=("independence",),
                violated_assumptions=() if model.regular else ("regularity",),
                highlighted=("likelihood_curve", "peak"),
            ))
        fig = ctx.figure(
            "labs.mle.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.parameter"),
            yaxis_title=ctx.t("labs.mle.axis.relative_lik", "Relative likelihood"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=grid, y=np.zeros_like(grid), mode="lines",
                                 line={"color": ctx.color("primary"), "width": 2.6},
                                 name=ctx.t("labs.mle.trace.lik", "likelihood")))
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        fig.update_yaxes(range=[0, 1.05])
        build_frames(fig, frames, duration=480, reduced_motion=ctx.reduced_motion,
                     slider_label="n")
        return animation(
            "likelihood_concentration",
            fig,
            steps,
            purpose=ctx.t("labs.mle.anim.purpose",
                          "Watch information accumulate and the likelihood concentrate."),
            summary=ctx.t(
                "labs.mle.anim.summary",
                "Adding data does not move the true parameter; it sharpens the likelihood "
                "around it. Consistency, asymptotic normality and the information bound are "
                "three descriptions of that sharpening."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = MLELab(SPEC)
