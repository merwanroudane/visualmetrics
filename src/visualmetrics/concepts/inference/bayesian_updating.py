"""Bayesian updating: prior x likelihood -> posterior, one observation at a time."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC"]

MODELS = ("beta_binomial", "normal_normal", "gamma_poisson")


SPEC = make_spec(
    "inference.bayesian_updating",
    Domain.INFERENCE,
    "bayesian_statistics",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive",
           "simulate", "code", "quiz", "references"),
    evidence=EvidenceType.VISUAL_DERIVATION,
    controls=(
        select("model", "beta_binomial", MODELS, group="model"),
        slider("true_parameter", 0.35, 0.01, 0.99, 0.01, group="model"),
        int_slider("n", 20, 0, 500, 1, group="data"),
        slider("prior_a", 2.0, 0.1, 50.0, 0.1, group="prior"),
        slider("prior_b", 2.0, 0.1, 50.0, 0.1, group="prior"),
        slider("prior_mean", 0.0, -10.0, 10.0, 0.1, group="prior",
               depends_on=("model", ("normal_normal",))),
        slider("prior_sd", 2.0, 0.05, 20.0, 0.05, group="prior",
               depends_on=("model", ("normal_normal",))),
        slider("credible_level", 0.95, 0.5, 0.999, 0.005, group="summary"),
        toggle("show_prior_sensitivity", True, group="views"),
        toggle("show_predictive", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", model="beta_binomial", n=20, prior_a=2, prior_b=2),
        scenario("no_data", "boundary", n=0),
        scenario("flat_prior", "sensitivity", prior_a=1.0, prior_b=1.0),
        scenario("strong_wrong_prior", "violation", prior_a=30.0, prior_b=3.0,
                 true_parameter=0.2, n=20),
        scenario("prior_overwhelmed", "large_sample", prior_a=30.0, prior_b=3.0,
                 true_parameter=0.2, n=500),
        scenario("small_sample", "small_sample", n=3),
        scenario("rare_event", "weak", true_parameter=0.03, n=40),
        scenario("normal_normal", "compare_methods", model="normal_normal",
                 true_parameter=0.5, prior_mean=0.0, prior_sd=1.0, n=25),
        scenario("gamma_poisson", "compare_methods", model="gamma_poisson",
                 true_parameter=0.6, prior_a=2.0, prior_b=1.0, n=30),
    ),
    prerequisites=("probability.distributions",),
    related=("inference.confidence_intervals", "inference.mle"),
    confused_with=("inference.confidence_intervals",),
    tags=("prior", "posterior", "conjugate", "credible interval", "bayes"),
    aliases=("bayes", "bayesien", "بايز", "posterior"),
    backends=("scipy",),
    references=(
        ref("Gelman, A. et al. (2013). Bayesian Data Analysis.", kind="book"),
        ref("Robert, C. P. (2007). The Bayesian Choice.", kind="book"),
    ),
)


class BayesLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        model = str(p["model"])
        theta = float(p["true_parameter"])
        n = int(p["n"])
        level = float(p["credible_level"])
        gen = rng(state.seed, "bayes", model)

        prior, posterior, data, grid, likelihood, summary = self._update(
            model, p, theta, n, gen)

        res.dgp = ctx.t(
            "labs.bayes.dgp",
            "{model} conjugate pair; {n} observations generated with the true parameter "
            "set to {theta}.", model=model, n=n, theta=fmt(theta, 3),
        )

        res.add_panel(ctx.panel(
            "update", self._update_figure(ctx, grid, prior, posterior, likelihood, theta,
                                          level, summary),
            "labs.bayes.figure.update", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_prior_sensitivity"]:
            res.add_panel(ctx.panel(
                "sensitivity", self._sensitivity_figure(ctx, model, p, data, grid, theta),
                "labs.bayes.figure.sensitivity", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_predictive"] and model == "beta_binomial":
            res.add_panel(ctx.panel(
                "predictive", self._predictive_figure(ctx, summary["post_dist"]),
                "labs.bayes.figure.predictive", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        lo, hi = summary["credible"]
        res.metric("posterior_mean", ctx.t("labs.bayes.metric.mean", "Posterior mean"),
                   summary["mean"], reference=theta)
        res.metric("posterior_mode", ctx.t("labs.bayes.metric.mode",
                                           "Posterior mode (MAP)"), summary["mode"])
        res.metric("posterior_sd", ctx.t("labs.bayes.metric.sd", "Posterior sd"),
                   summary["sd"])
        res.metric("credible", ctx.t("labs.bayes.metric.credible",
                                     "{l} credible interval", l=pct(level)),
                   f"[{fmt(lo, 4)}, {fmt(hi, 4)}]")
        res.metric("mle", ctx.t("labs.bayes.metric.mle",
                                "Maximum-likelihood estimate for comparison"),
                   summary["mle"], reference=theta)
        res.metric("prior_weight", ctx.t("labs.bayes.metric.prior_weight",
                                         "Prior weight in the posterior mean"),
                   summary["prior_weight"],
                   note=ctx.t("labs.bayes.metric.prior_weight_note",
                              "the prior acts like this many pseudo-observations"))

        res.assume("independence", ctx.t("assumptions.independence"), True)
        res.assume("model_correct", ctx.t("labs.bayes.assume.model_label",
                                          "The likelihood model is correct"), True,
                   detail=ctx.t("labs.bayes.assume.model",
                                "A posterior is only as good as the model that produced the "
                                "likelihood; conjugacy is convenience, not evidence."))
        res.assume("prior_honest", ctx.t("labs.bayes.assume.prior_label",
                                         "The prior was chosen before seeing the data"),
                   True,
                   detail=ctx.t("labs.bayes.assume.prior",
                                "Tuning the prior after seeing the data and then reporting "
                                "the posterior is circular."))

        res.animations.append(self._animation(ctx, model, p, data, grid, theta, n))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.inference.bayesian_updating.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.inference.bayesian_updating.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.inference.bayesian_updating.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.inference.bayesian_updating.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.inference.bayesian_updating.warning"), kind="warning")

        if summary["prior_weight"] > max(n, 1):
            res.warnings.append(ctx.t(
                "labs.bayes.warn.prior_dominates",
                "The prior carries the weight of about {w} observations against {n} real "
                "ones, so the posterior is mostly reporting the prior back to you.",
                w=fmt(summary["prior_weight"], 1), n=n,
            ))
        return res

    def _update(self, model, p, theta, n, gen):
        a, b = float(p["prior_a"]), float(p["prior_b"])
        if model == "beta_binomial":
            data = (gen.random(n) < theta).astype(float) if n else np.array([])
            s = float(data.sum())
            prior_d = stats.beta(a, b)
            post_d = stats.beta(a + s, b + n - s)
            grid = np.linspace(0.001, 0.999, 600)
            like = grid**s * (1 - grid) ** (n - s) if n else np.ones_like(grid)
            mle = s / n if n else float("nan")
            prior_weight = a + b
        elif model == "normal_normal":
            sigma = 1.0
            mu0, tau = float(p["prior_mean"]), float(p["prior_sd"])
            data = gen.normal(theta, sigma, n) if n else np.array([])
            prior_d = stats.norm(mu0, tau)
            if n:
                prec = 1 / tau**2 + n / sigma**2
                mean = (mu0 / tau**2 + data.sum() / sigma**2) / prec
                post_d = stats.norm(mean, np.sqrt(1 / prec))
            else:
                post_d = prior_d
            span = max(tau * 4, 4.0)
            grid = np.linspace(mu0 - span, mu0 + span, 600)
            like = (np.exp(-0.5 * np.sum((data[:, None] - grid[None, :]) ** 2, axis=0))
                    if n else np.ones_like(grid))
            mle = float(data.mean()) if n else float("nan")
            prior_weight = sigma**2 / tau**2
        else:
            data = gen.poisson(theta, n).astype(float) if n else np.array([])
            prior_d = stats.gamma(a, scale=1.0 / b)
            post_d = stats.gamma(a + data.sum(), scale=1.0 / (b + n))
            grid = np.linspace(0.001, max(theta * 4, 4.0), 600)
            like = (grid ** data.sum() * np.exp(-n * grid) if n else np.ones_like(grid))
            mle = float(data.mean()) if n else float("nan")
            prior_weight = b

        prior = prior_d.pdf(grid)
        posterior = post_d.pdf(grid)
        like = like / (np.trapezoid(like, grid) or 1.0)
        alpha = 1 - float(p["credible_level"])
        summary = {
            "mean": float(post_d.mean()),
            "sd": float(post_d.std()),
            "mode": float(grid[int(np.argmax(posterior))]),
            "credible": (float(post_d.ppf(alpha / 2)), float(post_d.ppf(1 - alpha / 2))),
            "mle": mle,
            "prior_weight": float(prior_weight),
            "post_dist": post_d,
        }
        return prior, posterior, data, grid, like, summary

    def _update_figure(self, ctx, grid, prior, posterior, like, theta, level, summary):
        fig = ctx.figure(
            "labs.bayes.figure.update",
            xaxis_title=ctx.t("labs.common.axis.parameter"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=430,
        )
        P.add_curve(fig, grid, prior, ctx.t("labs.bayes.trace.prior", "prior"),
                    "baseline", theme=ctx.theme, dash="dash")
        P.add_curve(fig, grid, like,
                    ctx.t("labs.bayes.trace.likelihood", "likelihood (rescaled)"),
                    "secondary", theme=ctx.theme, dash="dot")
        P.add_curve(fig, grid, posterior, ctx.t("labs.bayes.trace.posterior", "posterior"),
                    "primary", theme=ctx.theme)
        lo, hi = summary["credible"]
        mask = (grid >= lo) & (grid <= hi)
        P.shade_tail(fig, grid, posterior, mask,
                     ctx.t("labs.bayes.trace.credible", "{l} credible interval",
                           l=pct(level)),
                     "power", theme=ctx.theme, alpha=0.22)
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.bayes.legend",
            "The posterior is literally the pointwise product of the other two curves, "
            "rescaled to integrate to one. Where they disagree, the product collapses.",
        ), theme=ctx.theme)
        return fig

    def _sensitivity_figure(self, ctx, model, p, data, grid, theta):
        fig = ctx.figure(
            "labs.bayes.figure.sensitivity",
            xaxis_title=ctx.t("labs.common.axis.parameter"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=350,
        )
        priors = ((1.0, 1.0, "flat"), (2.0, 2.0, "weak"), (10.0, 10.0, "sceptical"),
                  (30.0, 3.0, "confident and wrong"))
        for i, (a, b, label) in enumerate(priors):
            q = dict(p)
            q["prior_a"], q["prior_b"] = a, b
            if model == "normal_normal":
                q["prior_sd"] = [4.0, 2.0, 0.5, 0.2][i]
                q["prior_mean"] = [0.0, 0.0, 0.0, 3.0][i]
            _, post, _, g, _, summ = self._update(
                model, q, theta, data.size,
                rng(int(p["seed"]) if "seed" in p else 0, "sens"))
            P.add_curve(fig, g, post,
                        ctx.t("labs.bayes.trace.prior_variant",
                              "posterior under a {label} prior", label=label),
                        ["muted", "info", "secondary", "warning"][i], theme=ctx.theme)
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.bayes.legend_sensitivity",
            "If these posteriors sit on top of each other, the data are speaking. If they "
            "are far apart, your conclusion is a statement about your prior.",
        ), theme=ctx.theme)
        return fig

    def _predictive_figure(self, ctx, posterior_dist):
        from scipy.special import betaln

        m = 20
        a, b = posterior_dist.args
        k = np.arange(m + 1)
        import math

        logp = (np.log(np.array([float(math.comb(m, int(i))) for i in k]))
                + betaln(a + k, b + m - k) - betaln(a, b))
        probs = np.exp(logp)
        fig = ctx.figure(
            "labs.bayes.figure.predictive",
            xaxis_title=ctx.t("labs.bayes.axis.successes",
                              "Successes in the next {m} trials", m=m),
            yaxis_title=ctx.t("labs.common.axis.probability"),
            height=340,
        )
        P.add_bar(fig, k, probs,
                  ctx.t("labs.bayes.trace.predictive", "posterior predictive"),
                  "primary", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.bayes.legend_predictive",
            "The predictive distribution is wider than plugging the posterior mean into a "
            "binomial, because it carries the remaining uncertainty about the parameter "
            "instead of pretending it away.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, model, p, data, grid, theta, n):
        go = P.require_plotly()
        counts = list(range(0, min(n, 40) + 1)) or [0]
        if n > 40:
            counts = sorted(set(list(range(0, 41)) + list(
                np.round(np.linspace(41, n, 10)).astype(int))))
        frames, steps = [], []
        for i, k in enumerate(counts):
            q = dict(p)
            gen = rng(int(p.get("seed", 42)), "bayes", model)
            _, post, dsub, g, _, summ = self._update(model, q, theta, int(k), gen)
            frames.append(go.Frame(name=str(k), data=[go.Scatter(x=g, y=post)]))
            lo, hi = summ["credible"]
            steps.append(AnimationStep(
                id=f"obs_{k}",
                frame=i,
                title=ctx.t("labs.bayes.anim.title", "{k} observations used", k=int(k)),
                what_you_see=ctx.t("labs.bayes.anim.see",
                                   "The posterior density of the unknown parameter."),
                what_changed=ctx.t("labs.bayes.anim.changed",
                                   "The first {k} observations have been folded in.", k=int(k)),
                why=ctx.t("labs.bayes.anim.why",
                          "Each observation multiplies the current density by its likelihood "
                          "contribution, then everything is renormalized."),
                interpretation=ctx.t("labs.bayes.anim.interpret",
                                     "Posterior mean {m}, {l} credible interval "
                                     "[{lo}, {hi}].",
                                     m=fmt(summ["mean"], 4), l=pct(float(p["credible_level"])),
                                     lo=fmt(lo, 4), hi=fmt(hi, 4)),
                conclusion=ctx.t("labs.bayes.anim.conclude",
                                 "The posterior concentrates and the prior's influence fades "
                                 "as evidence accumulates."),
                warning=ctx.t("labs.bayes.anim.warn",
                              "A credible interval is a probability statement about the "
                              "parameter GIVEN this prior - it is not a confidence interval."),
                math="p(theta | x) proportional to p(x | theta) * p(theta)",
                outputs={"observations": int(k), "posterior_mean": round(summ["mean"], 5),
                         "posterior_sd": round(summ["sd"], 5)},
                active_assumptions=("independence", "model_correct"),
                highlighted=("posterior_curve", "credible_interval"),
            ))
        fig = ctx.figure(
            "labs.bayes.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.parameter"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=grid, y=np.zeros_like(grid), mode="lines",
                                 line={"color": ctx.color("primary"), "width": 2.8},
                                 name=ctx.t("labs.bayes.trace.posterior", "posterior")))
        P.add_vline(fig, theta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        build_frames(fig, frames, duration=280, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.bayes.slider", "observations"))
        return animation(
            "sequential_update", fig, steps,
            purpose=ctx.t("labs.bayes.anim.purpose",
                          "See belief updating as repeated multiplication, one datum at a time."),
            summary=ctx.t(
                "labs.bayes.anim.summary",
                "Bayesian updating is order-independent: the same observations in any order "
                "give the same final posterior. What you watched was not a search or an "
                "optimization but the accumulation of a product."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = BayesLab(SPEC)
