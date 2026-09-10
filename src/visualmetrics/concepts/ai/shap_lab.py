"""Shapley additive explanations, computed exactly on a small model."""

from __future__ import annotations

import itertools
import math
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

__all__ = ["LAB", "SPEC", "exact_shapley", "MODELS"]

MODELS = ("linear", "with_interaction", "nonlinear", "tree_like")


def _predict(x, kind, betas, interaction):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if kind == "linear":
        out = x @ betas
    elif kind == "with_interaction":
        out = x @ betas + interaction * x[:, 0] * x[:, 1]
    elif kind == "nonlinear":
        out = x @ betas + interaction * np.tanh(2.0 * x[:, 0]) * x[:, 1]
    else:  # tree_like: piecewise constant in x0, linear elsewhere
        out = (np.where(x[:, 0] > 0, 2.0, -1.0) * betas[0]
               + x[:, 1:] @ betas[1:])
    return out


def exact_shapley(x, background, kind, betas, interaction):
    """Exact Shapley values by enumerating every coalition (marginal expectations)."""
    x = np.asarray(x, dtype=float)
    p = x.size
    features = list(range(p))
    baseline = float(np.mean(_predict(background, kind, betas, interaction)))

    def value(subset):
        """E[f(X)] with the features in `subset` fixed at their observed values."""
        data = background.copy()
        for j in subset:
            data[:, j] = x[j]
        return float(np.mean(_predict(data, kind, betas, interaction)))

    cache = {}
    for size in range(p + 1):
        for subset in itertools.combinations(features, size):
            cache[subset] = value(subset)

    phi = np.zeros(p)
    for j in features:
        others = [f for f in features if f != j]
        for size in range(p):
            weight = (math.factorial(size) * math.factorial(p - size - 1)
                      / math.factorial(p))
            for subset in itertools.combinations(others, size):
                phi[j] += weight * (cache[tuple(sorted(subset + (j,)))] - cache[subset])
    return {"phi": phi, "baseline": baseline,
            "prediction": float(_predict(x[None, :], kind, betas, interaction)[0]),
            "cache": cache}


SPEC = make_spec(
    "xai.shap",
    Domain.XAI,
    "local_explanations",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "diagnose",
           "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("model", "linear", MODELS, group="model"),
        int_slider("n_features", 4, 2, 8, 1, group="model"),
        slider("interaction", 0.0, -3.0, 3.0, 0.05, group="model"),
        slider("x0", 1.0, -3.0, 3.0, 0.05, group="observation"),
        slider("x1", -0.5, -3.0, 3.0, 0.05, group="observation"),
        slider("correlation", 0.0, 0.0, 0.95, 0.01, group="data"),
        int_slider("n_background", 300, 30, 3000, 10, group="data"),
        toggle("show_orderings", True, group="views"),
        toggle("show_global", True, group="views"),
        toggle("proxy_feature", False, group="violations"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", model="linear", n_features=4),
        scenario("interaction", "compare_methods", model="with_interaction",
                 interaction=2.0),
        scenario("nonlinear", "compare_methods", model="nonlinear", interaction=2.0),
        scenario("tree_like", "compare_methods", model="tree_like"),
        scenario("zero_contribution", "null", x0=0.0, x1=0.0),
        scenario("extreme_observation", "boundary", x0=3.0, x1=3.0),
        scenario("correlated_features", "violation", correlation=0.9),
        scenario("proxy_feature", "counterexample", proxy_feature=True,
                 correlation=0.95),
        scenario("few_features", "small_sample", n_features=2),
        scenario("many_features", "large_sample", n_features=7),
    ),
    related=("ml.classification_threshold", "econometrics.omitted_variable_bias"),
    tags=("shap", "shapley", "feature attribution", "additive explanation",
          "local explanation", "interaction"),
    aliases=("shap values", "valeurs de shapley", "قيم شابلي",
             "feature attribution", "explainability"),
    backends=("numpy", "shap"),
    required_extras=("ai",),
    references=(
        ref("Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting "
            "model predictions. NeurIPS 30.", kind="paper",
            url="https://arxiv.org/abs/1705.07874"),
        ref("Shapley, L. S. (1953). A value for n-person games. Contributions to the "
            "Theory of Games II.", kind="paper"),
        ref("Chen, H., Janizek, J. D., Lundberg, S. and Lee, S.-I. (2020). True to the "
            "model or true to the data?", kind="paper",
            url="https://arxiv.org/abs/2006.16234"),
    ),
)


class SHAPLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        k = int(p["n_features"])
        kind = str(p["model"])
        gen = rng(state.seed, "shap")
        betas = np.round(gen.uniform(-2.0, 2.0, k), 2)
        background = self._background(gen, k, int(p["n_background"]),
                                      float(p["correlation"]),
                                      bool(p["proxy_feature"]))
        x = np.round(gen.uniform(-1.5, 1.5, k), 2)
        x[0] = float(p["x0"])
        if k > 1:
            x[1] = float(p["x1"])
        if bool(p["proxy_feature"]) and k > 1:
            betas[1] = 0.0
            x[1] = float(p["x0"]) * float(p["correlation"])

        result = exact_shapley(x, background, kind, betas, float(p["interaction"]))
        phi = result["phi"]

        res.dgp = ctx.t(
            "labs.shap.dgp",
            "A {kind} model with {k} features, explained at one observation against a "
            "background of {n} points. Shapley values are computed exactly by enumerating "
            "all {c} coalitions.",
            kind=kind, k=k, n=int(p["n_background"]), c=2**k,
        )

        res.add_panel(ctx.panel(
            "waterfall", self._waterfall_figure(ctx, phi, result, k),
            "labs.shap.figure.waterfall", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_orderings"]:
            res.add_panel(ctx.panel(
                "orderings", self._ordering_figure(ctx, result, k),
                "labs.shap.figure.orderings", tab="math",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_global"]:
            res.add_panel(ctx.panel(
                "global", self._global_figure(ctx, background, kind, betas,
                                              float(p["interaction"]), k),
                "labs.shap.figure.global", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        res.add_panel(ctx.panel(
            "comparison", self._comparison_figure(ctx, phi, betas, x, kind),
            "labs.shap.figure.comparison", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))

        total = float(result["baseline"] + np.sum(phi))
        res.metric("baseline", ctx.t("labs.shap.metric.baseline",
                                     "Baseline prediction (average over the background)"),
                   result["baseline"])
        res.metric("prediction", ctx.t("labs.shap.metric.prediction",
                                       "Prediction for this observation"),
                   result["prediction"])
        res.metric("sum_of_contributions", ctx.t("labs.shap.metric.sum",
                                                 "Baseline plus all contributions"),
                   total, reference=result["prediction"],
                   note=ctx.t("labs.shap.metric.sum_note",
                              "efficiency: these must agree exactly"))
        res.metric("efficiency_error", ctx.t("labs.shap.metric.efficiency",
                                             "|prediction - baseline - sum(phi)|"),
                   abs(total - result["prediction"]), reference=0.0)
        order = np.argsort(-np.abs(phi))
        for rank, j in enumerate(order[: min(k, 5)], start=1):
            res.metric(f"phi_{j}", ctx.t("labs.shap.metric.phi",
                                         "Contribution of feature {j} (rank {r})",
                                         j=j, r=rank), float(phi[j]))
        res.metric("largest_feature", ctx.t("labs.shap.metric.largest",
                                            "Largest contribution"),
                   f"feature {int(order[0])} ({fmt(float(phi[order[0]]), 3)})")
        res.metric("total_magnitude", ctx.t("labs.shap.metric.magnitude",
                                            "Total absolute contribution"),
                   float(np.sum(np.abs(phi))))
        if kind == "linear":
            expected = betas * (x - background.mean(axis=0))
            res.metric("linear_check", ctx.t("labs.shap.metric.linear_check",
                                             "Largest gap from beta_j (x_j - E[x_j])"),
                       float(np.max(np.abs(phi - expected))), reference=0.0,
                       note=ctx.t("labs.shap.metric.linear_check_note",
                                  "for a linear model with independent features the "
                                  "Shapley value has this closed form"))

        correlated = float(p["correlation"]) > 0.3
        res.assume("independence", ctx.t("labs.shap.assume.independence_label",
                                         "Features are independent"), not correlated,
                   detail=ctx.t("labs.shap.assume.independence",
                                "Marginal-expectation Shapley values evaluate the model on "
                                "feature combinations that never occur when features are "
                                "correlated."),
                   consequence="" if not correlated else ctx.t(
                       "labs.shap.assume.correlated",
                       "The attributions below describe how the model behaves on "
                       "impossible inputs as much as on realistic ones."))
        res.assume("not_causal", ctx.t("labs.shap.assume.causal_label",
                                       "Attribution is not a causal effect"), False,
                   detail=ctx.t("labs.shap.assume.causal",
                                "A Shapley value explains the MODEL's output. A feature "
                                "receives credit for being useful to the model, which can "
                                "happen purely because it proxies for something else."))
        res.assume("baseline_chosen", ctx.t("labs.shap.assume.baseline_label",
                                            "The baseline is a deliberate choice"), True,
                   detail=ctx.t("labs.shap.assume.baseline",
                                "Every contribution is measured relative to the background "
                                "distribution. Change the background and every number "
                                "changes with it."))

        res.animations.append(self._animation(ctx, result, phi, k))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.xai.shap.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.xai.shap.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.xai.shap.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.xai.shap.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.xai.shap.warning"),
                    kind="warning")

        if bool(p["proxy_feature"]):
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.shap.counterexample",
                "Feature 1 has a coefficient of exactly zero - it does not enter the model "
                "at all - yet it receives a contribution of {v} because it is correlated "
                "with feature 0 at {c}. This is not a bug in the method: the attribution is "
                "faithful to the model on the perturbed inputs it was asked about. It is "
                "precisely why a SHAP value must not be read as an effect.",
                v=fmt(float(phi[1]), 4), c=fmt(p["correlation"], 2),
            ), kind="warning")
            res.warnings.append(ctx.t(
                "labs.shap.warn.proxy",
                "A feature the model never uses is receiving non-zero credit here because "
                "of correlation with a feature it does use.",
            ))
        if correlated and not bool(p["proxy_feature"]):
            res.warnings.append(ctx.t(
                "labs.shap.warn.correlation",
                "Features are correlated at {c}. Coalition values are computed by holding "
                "some features at their observed value while others vary marginally, which "
                "creates input combinations the data never contain.",
                c=fmt(p["correlation"], 2),
            ))
        return res

    @staticmethod
    def _background(gen, k, n, correlation, proxy):
        base = gen.standard_normal((n, k))
        if correlation > 0 and k > 1:
            base[:, 1] = (correlation * base[:, 0]
                          + np.sqrt(max(1 - correlation**2, 1e-9)) * base[:, 1])
        if proxy and k > 1:
            base[:, 1] = correlation * base[:, 0] + 0.05 * gen.standard_normal(n)
        return base

    def _waterfall_figure(self, ctx, phi, result, k):
        go = P.require_plotly()
        order = np.argsort(-np.abs(phi))
        labels = ([ctx.t("labs.shap.trace.baseline", "baseline")]
                  + [f"feature {j}" for j in order]
                  + [ctx.t("labs.shap.trace.prediction", "prediction")])
        measures = ["absolute"] + ["relative"] * k + ["total"]
        values = [result["baseline"]] + [float(phi[j]) for j in order] + [0.0]
        fig = ctx.figure(
            "labs.shap.figure.waterfall",
            xaxis_title=ctx.t("labs.shap.axis.step", "Contribution"),
            yaxis_title=ctx.t("labs.shap.axis.value", "Model output"),
            height=430,
        )
        fig.add_trace(go.Waterfall(
            x=labels, y=values, measure=measures,
            text=[fmt(v, 3) for v in values[:-1]] + [fmt(result["prediction"], 3)],
            textposition="outside",
            increasing={"marker": {"color": ctx.color("positive")}},
            decreasing={"marker": {"color": ctx.color("negative")}},
            totals={"marker": {"color": ctx.color("primary")}},
            connector={"line": {"color": ctx.color("muted")}},
        ))
        P.add_legend_note(fig, ctx.t(
            "labs.shap.legend_waterfall",
            "The bars start at the average prediction and end exactly at this "
            "observation's prediction. That exactness is the efficiency property, and it is "
            "what makes the decomposition additive.",
        ), theme=ctx.theme)
        return fig

    def _ordering_figure(self, ctx, result, k):
        """Marginal contribution of feature 0 across every arrival order."""
        cache = result["cache"]
        features = list(range(k))
        contributions = []
        for perm in itertools.permutations(features):
            before = []
            for f in perm:
                if f == 0:
                    contributions.append(cache[tuple(sorted(before + [0]))]
                                         - cache[tuple(sorted(before))])
                    break
                before.append(f)
        contributions = np.asarray(contributions)
        fig = ctx.figure(
            "labs.shap.figure.orderings",
            xaxis_title=ctx.t("labs.shap.axis.contribution",
                              "Marginal contribution of feature 0"),
            yaxis_title=ctx.t("labs.common.axis.frequency"),
            height=360,
        )
        P.add_histogram(fig, contributions,
                        ctx.t("labs.shap.trace.orderings",
                              "one value per arrival order ({n} orders)",
                              n=contributions.size),
                        "primary", theme=ctx.theme, nbins=30, density=False)
        P.add_vline(fig, float(np.mean(contributions)),
                    ctx.t("labs.shap.trace.shapley",
                          "their average = the Shapley value"),
                    "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.shap.legend_orderings",
            "A feature's contribution depends on which other features are already present. "
            "The Shapley value averages over every possible order - and when the histogram "
            "is wide, that single average is hiding real interaction.",
        ), theme=ctx.theme)
        return fig

    def _global_figure(self, ctx, background, kind, betas, interaction, k):
        sub = background[: min(background.shape[0], 60)]
        phis = []
        for row in sub:
            phis.append(exact_shapley(row, background, kind, betas, interaction)["phi"])
        phis = np.asarray(phis)
        go = P.require_plotly()
        fig = ctx.figure(
            "labs.shap.figure.global",
            xaxis_title=ctx.t("labs.shap.axis.contribution_global",
                              "Contribution to the prediction"),
            yaxis_title=ctx.t("labs.shap.axis.feature", "Feature"),
            height=400,
        )
        for j in range(k):
            fig.add_trace(go.Box(x=phis[:, j], name=f"feature {j}",
                                 marker={"color": ctx.color("primary")},
                                 line={"color": ctx.color("primary")},
                                 boxpoints="outliers", orientation="h",
                                 showlegend=False))
        P.add_vline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.shap.legend_global",
            "Local explanations aggregated across the background. A feature whose box is "
            "wide matters a lot in some cases and not at all in others - which a single "
            "global importance number would hide entirely.",
        ), theme=ctx.theme)
        return fig

    def _comparison_figure(self, ctx, phi, betas, x, kind):
        go = P.require_plotly()
        idx = np.arange(phi.size)
        fig = ctx.figure(
            "labs.shap.figure.comparison",
            xaxis_title=ctx.t("labs.shap.axis.feature", "Feature"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=370,
        )
        fig.add_trace(go.Bar(x=idx, y=phi, marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.shap.trace.phi", "Shapley contribution")))
        fig.add_trace(go.Bar(x=idx, y=betas,
                             marker={"color": ctx.color("muted")},
                             name=ctx.t("labs.shap.trace.beta",
                                        "model coefficient")))
        fig.add_trace(go.Bar(x=idx, y=x, marker={"color": ctx.color("secondary")},
                             name=ctx.t("labs.shap.trace.x",
                                        "this observation's values"),
                             opacity=0.55))
        P.add_legend_note(fig, ctx.t(
            "labs.shap.legend_comparison",
            "A coefficient describes the model; a Shapley value describes this "
            "observation's prediction. A feature with a large coefficient contributes "
            "nothing when its value sits at the background average.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, result, phi, k):
        go = P.require_plotly()
        order = list(np.argsort(-np.abs(phi)))
        frames, steps = [], []
        running = result["baseline"]
        for i in range(k + 1):
            xs = ([ctx.t("labs.shap.trace.baseline", "baseline")]
                  + [f"feature {j}" for j in order[:i]])
            ys = [result["baseline"]] + [float(phi[j]) for j in order[:i]]
            measures = ["absolute"] + ["relative"] * i
            frames.append(go.Frame(name=str(i), data=[
                go.Waterfall(x=xs, y=ys, measure=measures),
            ]))
            running = result["baseline"] + float(np.sum([phi[j] for j in order[:i]]))
            remaining = result["prediction"] - running
            steps.append(AnimationStep(
                id=f"add_{i}", frame=i,
                title=(ctx.t("labs.shap.anim.title_start", "starting at the baseline")
                       if i == 0 else
                       ctx.t("labs.shap.anim.title", "after adding feature {j}",
                             j=int(order[i - 1]))),
                what_you_see=ctx.t("labs.shap.anim.see",
                                   "The running prediction, starting from the average and "
                                   "adding one feature's contribution at a time."),
                what_changed=(ctx.t("labs.shap.anim.changed_start",
                                    "Nothing is known yet, so the best guess is the "
                                    "average prediction, {b}.",
                                    b=fmt(result["baseline"], 3)) if i == 0 else
                              ctx.t("labs.shap.anim.changed",
                                    "Feature {j} contributed {v}.",
                                    j=int(order[i - 1]),
                                    v=fmt(float(phi[order[i - 1]]), 4))),
                why=ctx.t("labs.shap.anim.why",
                          "Each contribution is the average change in the prediction when "
                          "that feature joins, taken over every possible set of features "
                          "already present."),
                interpretation=ctx.t("labs.shap.anim.interpret",
                                     "The running total is {r}; {d} of the gap to the "
                                     "final prediction remains.",
                                     r=fmt(running, 4), d=fmt(remaining, 4)),
                conclusion=ctx.t("labs.shap.anim.conclude",
                                 "When every feature has been added, the total is exactly "
                                 "the prediction - never approximately."),
                warning=ctx.t("labs.shap.anim.warn",
                              "The order shown here is chosen for readability. The values "
                              "themselves are already averaged over all orders."),
                math="phi_j = sum over S of w(S) [f(S union j) - f(S)];  "
                     "baseline + sum phi_j = f(x)",
                outputs={"features_added": i, "running_total": round(running, 5),
                         "remaining": round(remaining, 5)},
                highlighted=("waterfall",),
            ))
        fig = ctx.figure(
            "labs.shap.figure.animation",
            xaxis_title=ctx.t("labs.shap.axis.step", "Contribution"),
            yaxis_title=ctx.t("labs.shap.axis.value", "Model output"),
            height=400,
        )
        fig.add_trace(go.Waterfall(
            x=[ctx.t("labs.shap.trace.baseline", "baseline")],
            y=[result["baseline"]], measure=["absolute"],
            increasing={"marker": {"color": ctx.color("positive")}},
            decreasing={"marker": {"color": ctx.color("negative")}},
            totals={"marker": {"color": ctx.color("primary")}},
            connector={"line": {"color": ctx.color("muted")}},
        ))
        P.add_hline(fig, result["prediction"],
                    ctx.t("labs.shap.trace.target",
                          "the prediction to be explained"),
                    "truth", theme=ctx.theme, dash="dash")
        build_frames(fig, frames, duration=760, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.shap.slider", "features added"))
        return animation(
            "additive_build", fig, steps,
            purpose=ctx.t("labs.shap.anim.purpose",
                          "Show attribution as a payout being divided, one player at a "
                          "time."),
            summary=ctx.t(
                "labs.shap.anim.summary",
                "Shapley values are the unique attribution that adds up exactly, treats "
                "identical features identically, and gives nothing to a feature that never "
                "changes the output. They explain the model's prediction - not the world "
                "the model was fitted to."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = SHAPLab(SPEC)
