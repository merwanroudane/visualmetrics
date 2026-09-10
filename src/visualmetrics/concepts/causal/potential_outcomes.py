"""Potential outcomes: both worlds, one observation, and the gap in between."""

from __future__ import annotations

from typing import Any

from ...backends import linear as LM
from ...data.generators.causal import potential_outcomes
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
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC"]

ESTIMATORS = ("naive", "regression", "matching", "ipw", "doubly_robust")


SPEC = make_spec(
    "causal.potential_outcomes",
    Domain.CAUSAL,
    "foundations",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        slider("ate", 2.0, -5.0, 5.0, 0.05, group="effects"),
        slider("heterogeneity", 0.0, 0.0, 3.0, 0.05, group="effects"),
        slider("confounding", 1.5, 0.0, 5.0, 0.05, group="selection"),
        slider("selection", 1.0, 0.0, 4.0, 0.05, group="selection"),
        toggle("randomized", False, group="selection"),
        int_slider("n", 400, 20, 20000, 10, group="dgp"),
        slider("noise", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        int_slider("reps", 400, 50, 4000, 50, group="simulation", expensive=True),
        toggle("show_both_worlds", True, group="views"),
        toggle("show_overlap", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", confounding=1.5, selection=1.0),
        scenario("randomized", "null", randomized=True),
        scenario("no_confounding", "null", confounding=0.0),
        scenario("strong_selection", "strong", selection=3.0, confounding=2.5),
        scenario("negative_selection", "negative", confounding=-2.0),
        scenario("heterogeneous_effects", "misspecification", heterogeneity=2.0),
        scenario("no_overlap", "violation", selection=4.0, n=400),
        scenario("small_sample", "small_sample", n=40),
        scenario("large_sample", "large_sample", n=8000),
        scenario("zero_effect", "null", ate=0.0, confounding=2.0),
    ),
    related=("causal.dag", "causal.did", "econometrics.omitted_variable_bias"),
    next_concepts=("causal.dag", "causal.did"),
    tags=("potential outcomes", "ate", "att", "selection bias", "overlap",
          "propensity score", "ipw"),
    aliases=("rubin causal model", "resultats potentiels", "النواتج المحتملة",
             "counterfactual", "average treatment effect"),
    backends=("numpy", "dowhy"),
    references=(
        ref("Rubin, D. B. (1974). Estimating causal effects of treatments in randomized "
            "and nonrandomized studies. Journal of Educational Psychology 66(5).",
            kind="paper", doi="10.1037/h0037350"),
        ref("Imbens, G. W. and Rubin, D. B. (2015). Causal Inference for Statistics, "
            "Social, and Biomedical Sciences.", kind="book"),
    ),
    curriculum_tags=("harvard.api114",),
)


class PotentialOutcomesLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = self._generate(p, int(p["n"]), state.seed)
        est = self._estimate(data)

        res.data = data
        res.dgp = data.dgp

        if p["show_both_worlds"]:
            res.add_panel(ctx.panel(
                "both_worlds", self._worlds_figure(ctx, data),
                "labs.po.figure.worlds", evidence=EvidenceType.SIMULATION,
            ))
        res.add_panel(ctx.panel(
            "decomposition", self._decomposition_figure(ctx, data, est),
            "labs.po.figure.decomposition", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_overlap"]:
            res.add_panel(ctx.panel(
                "overlap", self._overlap_figure(ctx, data),
                "labs.po.figure.overlap", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        draws = self._sampling(p, int(p["n"]), state.seed, int(p["reps"]))
        res.add_panel(ctx.panel(
            "estimators", self._estimator_figure(ctx, draws, data.truth["ate"]),
            "labs.po.figure.estimators", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))

        res.metric("true_ate", ctx.t("labs.po.metric.ate",
                                     "True average treatment effect"),
                   data.truth["ate"])
        res.metric("true_att", ctx.t("labs.po.metric.att",
                                     "True effect on the treated"), data.truth["att"])
        res.metric("naive", ctx.t("labs.po.metric.naive",
                                  "Naive difference in means"), est["naive"],
                   reference=data.truth["ate"])
        res.metric("selection_bias", ctx.t("labs.po.metric.bias",
                                           "Selection bias in the naive comparison"),
                   est["naive"] - data.truth["att"],
                   note=ctx.t("labs.po.metric.bias_note",
                              "E[Y(0) | treated] minus E[Y(0) | untreated]"))
        for key, label in (("regression", "regression adjustment"),
                           ("matching", "nearest-neighbour matching"),
                           ("ipw", "inverse probability weighting"),
                           ("doubly_robust", "doubly robust")):
            res.metric(key, ctx.t("labs.po.metric.estimator", "{m}", m=label),
                       est[key], reference=data.truth["ate"])
        res.metric("overlap_min", ctx.t("labs.po.metric.overlap",
                                        "Smallest propensity score among the treated"),
                   float(np.min(data["propensity"][data["d"] > 0])))
        res.metric("overlap_max", ctx.t("labs.po.metric.overlap_max",
                                        "Largest propensity score among the untreated"),
                   float(np.max(data["propensity"][data["d"] == 0])))
        res.metric("treated_share", ctx.t("labs.po.metric.share",
                                          "Share treated"), float(np.mean(data["d"])))
        for key in ("naive", "regression", "ipw", "doubly_robust"):
            res.metric(f"sim_bias_{key}",
                       ctx.t("labs.po.metric.sim_bias", "{m}: simulated bias", m=key),
                       float(np.mean(draws[key]) - data.truth["ate"]), reference=0.0)

        randomized = bool(p["randomized"]) or float(p["confounding"]) == 0
        overlap_ok = (float(np.min(data["propensity"])) > 0.05
                      and float(np.max(data["propensity"])) < 0.95)
        res.assume("exchangeability", ctx.t("assumptions.exchangeability"), randomized,
                   detail=ctx.t("labs.po.assume.exchangeability",
                                "Treatment here is assigned by {rule}.",
                                rule=("a coin flip" if p["randomized"]
                                      else "a rule that depends on X")),
                   consequence="" if randomized else ctx.t(
                       "labs.po.assume.selection",
                       "The treated and untreated groups differ in X, so their untreated "
                       "outcomes would have differed anyway. That difference is selection "
                       "bias, and it is not removed by a larger sample."))
        res.assume("positivity", ctx.t("assumptions.positivity"), overlap_ok,
                   detail=ctx.t("labs.po.assume.positivity",
                                "Every unit must have a non-trivial chance of either "
                                "treatment status, otherwise part of the population has no "
                                "comparison group at all."),
                   consequence="" if overlap_ok else ctx.t(
                       "labs.po.assume.no_overlap",
                       "Weighting estimators divide by the propensity score, so units with "
                       "extreme scores receive enormous weights and the estimate becomes "
                       "unstable."))
        res.assume("sutva", ctx.t("assumptions.sutva"), True,
                   detail=ctx.t("labs.po.assume.sutva",
                                "One unit's treatment must not affect another's outcome, "
                                "and there must be only one version of the treatment. Both "
                                "hold by construction in this simulation and often fail in "
                                "the field."))

        res.animations.append(self._animation(ctx, p, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.causal.potential_outcomes.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.causal.potential_outcomes.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.causal.potential_outcomes.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.causal.potential_outcomes.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.causal.potential_outcomes.warning"), kind="warning")

        if not overlap_ok:
            res.warnings.append(ctx.t(
                "labs.po.warn.overlap",
                "Propensity scores range from {lo} to {hi}. Where they approach 0 or 1 "
                "there is effectively no comparison group, and weighting estimators become "
                "dominated by a handful of observations.",
                lo=fmt(float(np.min(data["propensity"])), 3),
                hi=fmt(float(np.max(data["propensity"])), 3),
            ))
        if float(p["heterogeneity"]) > 0.1:
            res.warnings.append(ctx.t(
                "labs.po.warn.heterogeneity",
                "Effects vary across units, so the ATE ({a}) and the effect on the treated "
                "({t}) are different quantities. Different estimators target different ones, "
                "and a disagreement between them is not necessarily an error.",
                a=fmt(data.truth["ate"], 3), t=fmt(data.truth["att"], 3),
            ))
        return res

    @staticmethod
    def _generate(p, n, seed):
        return potential_outcomes(
            n=n, ate=float(p["ate"]), confounding=float(p["confounding"]),
            selection=float(p["selection"]), heterogeneity=float(p["heterogeneity"]),
            noise=float(p["noise"]), randomized=bool(p["randomized"]), seed=seed,
        )

    @staticmethod
    def _estimate(data):
        y, d, x = data["y"], data["d"], data["x"]
        n = y.size
        treated, control = d > 0, d == 0
        naive = float(y[treated].mean() - y[control].mean()) if treated.any() and control.any() else np.nan

        reg = LM.ols(y, np.column_stack([np.ones(n), d, x]), names=("const", "d", "x"))

        # nearest-neighbour matching on x, ATT
        matched = []
        xc, yc = x[control], y[control]
        for xi, yi in zip(x[treated], y[treated], strict=False):
            if xc.size:
                j = int(np.argmin(np.abs(xc - xi)))
                matched.append(yi - yc[j])
        matching = float(np.mean(matched)) if matched else np.nan

        ps = np.clip(data["propensity"], 0.01, 0.99)
        w = np.where(treated, 1.0 / ps, 1.0 / (1.0 - ps))
        ipw = float(np.sum(w * d * y) / np.sum(w * d)
                    - np.sum(w * (1 - d) * y) / np.sum(w * (1 - d)))

        m1 = LM.ols(y[treated], np.column_stack([np.ones(int(treated.sum())), x[treated]]))
        m0 = LM.ols(y[control], np.column_stack([np.ones(int(control.sum())), x[control]]))
        mu1 = m1.coefficients[0] + m1.coefficients[1] * x
        mu0 = m0.coefficients[0] + m0.coefficients[1] * x
        dr = float(np.mean(mu1 - mu0
                           + d * (y - mu1) / ps
                           - (1 - d) * (y - mu0) / (1 - ps)))
        return {"naive": naive, "regression": reg.coef("d"), "matching": matching,
                "ipw": ipw, "doubly_robust": dr, "regression_fit": reg}

    def _sampling(self, p, n, seed, reps):
        from ...simulation.monte_carlo import monte_carlo

        reps = min(reps, 2000)

        def experiment(gen):
            child = int(gen.integers(0, 2**31 - 1))
            d = self._generate(p, n, child)
            e = self._estimate(d)
            return {k: e[k] for k in ESTIMATORS}

        return monte_carlo(experiment, reps, seed).draws

    def _worlds_figure(self, ctx, data):
        go = P.require_plotly()
        n = min(data.n, 60)
        idx = np.argsort(data["x"])[:: max(data.n // n, 1)][:n]
        fig = ctx.figure(
            "labs.po.figure.worlds",
            xaxis_title=ctx.t("labs.po.axis.unit", "Unit (sorted by X)"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=430,
        )
        order = np.arange(idx.size)
        fig.add_trace(go.Scatter(
            x=np.repeat(order, 3),
            y=np.concatenate([[data["y0"][i], data["y1"][i], None] for i in idx]),
            mode="lines", line={"color": ctx.color("muted"), "width": 1.0},
            name=ctx.t("labs.po.trace.gap", "individual treatment effect"),
        ))
        P.add_points(fig, order, data["y0"][idx],
                     ctx.t("labs.po.trace.y0", "Y(0): outcome without treatment"),
                     "control", theme=ctx.theme, size=7)
        P.add_points(fig, order, data["y1"][idx],
                     ctx.t("labs.po.trace.y1", "Y(1): outcome with treatment"),
                     "treatment", theme=ctx.theme, size=7)
        observed = np.where(data["d"][idx] > 0, data["y1"][idx], data["y0"][idx])
        P.add_points(fig, order, observed,
                     ctx.t("labs.po.trace.observed", "the one you actually observe"),
                     "highlight", theme=ctx.theme, size=11, symbol="circle-open")
        P.add_legend_note(fig, ctx.t(
            "labs.po.legend_worlds",
            "Every unit has two outcomes and you see exactly one. The circled point is the "
            "observation; its partner is the counterfactual. This picture exists only "
            "because the data were simulated.",
        ), theme=ctx.theme)
        return fig

    def _decomposition_figure(self, ctx, data, est):
        go = P.require_plotly()
        att = data.truth["att"]
        bias = est["naive"] - att
        fig = ctx.figure(
            "labs.po.figure.decomposition",
            xaxis_title=ctx.t("labs.po.axis.component", "Component"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=370,
        )
        labels = [ctx.t("labs.po.trace.att", "true effect on the treated"),
                  ctx.t("labs.po.trace.bias", "selection bias"),
                  ctx.t("labs.po.trace.naive_total", "naive difference in means")]
        fig.add_trace(go.Bar(x=labels[:2], y=[att, bias],
                             marker={"color": [ctx.color("positive"),
                                               ctx.color("negative")]},
                             text=[fmt(att, 3), fmt(bias, 3)], textposition="outside",
                             name=ctx.t("labs.po.trace.parts", "parts")))
        fig.add_trace(go.Bar(x=[labels[2]], y=[est["naive"]],
                             marker={"color": ctx.color("primary")},
                             text=[fmt(est["naive"], 3)], textposition="outside",
                             name=ctx.t("labs.po.trace.total", "what you observe")))
        P.add_legend_note(fig, ctx.t(
            "labs.po.legend_decomposition",
            "The naive comparison is the effect PLUS the difference the two groups would "
            "have shown anyway. Every method in this lab is an attempt to remove the second "
            "bar - and none of them can verify that it succeeded.",
        ), theme=ctx.theme)
        return fig

    def _overlap_figure(self, ctx, data):
        fig = ctx.figure(
            "labs.po.figure.overlap",
            xaxis_title=ctx.t("labs.po.axis.propensity",
                              "Propensity score P(treated | X)"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=350,
        )
        ps = data["propensity"]
        P.add_histogram(fig, ps[data["d"] > 0], ctx.t("labs.common.trace.treated"),
                        "treatment", theme=ctx.theme, nbins=40, opacity=0.55)
        P.add_histogram(fig, ps[data["d"] == 0], ctx.t("labs.common.trace.control"),
                        "control", theme=ctx.theme, nbins=40, opacity=0.55)
        for c in (0.05, 0.95):
            P.add_vline(fig, c, "", "warning", theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.po.legend_overlap",
            "Adjustment can only work where the two histograms overlap. In regions covered "
            "by one colour only, the comparison group does not exist and any estimate there "
            "is extrapolation by the model.",
        ), theme=ctx.theme)
        return fig

    def _estimator_figure(self, ctx, draws, ate):
        fig = ctx.figure(
            "labs.po.figure.estimators",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=390,
        )
        roles = {"naive": "negative", "regression": "primary", "matching": "info",
                 "ipw": "warning", "doubly_robust": "positive"}
        for key, role in roles.items():
            vals = draws[key][np.isfinite(draws[key])]
            if vals.size:
                lo, hi = np.percentile(vals, [0.5, 99.5])
                P.add_histogram(fig, vals[(vals >= lo) & (vals <= hi)],
                                key.replace("_", " "), role, theme=ctx.theme,
                                nbins=45, opacity=0.42)
        P.add_vline(fig, ate, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.po.legend_estimators",
            "All five estimators use identical data. Their spread and their displacement "
            "are two different failures - only one of them shrinks as the sample grows.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, seed):
        go = P.require_plotly()
        selections = np.linspace(0.0, 3.5, 15)
        n = int(p["n"])
        frames, steps = [], []
        for i, s in enumerate(selections):
            q = dict(p)
            q["selection"] = float(s)
            q["randomized"] = False
            d = self._generate(q, n, seed)
            e = self._estimate(d)
            frames.append(go.Frame(name=f"{s:.2f}", data=[
                go.Bar(x=["naive", "regression", "IPW", "doubly robust"],
                       y=[e["naive"], e["regression"], e["ipw"], e["doubly_robust"]]),
            ]))
            ps = d["propensity"]
            steps.append(AnimationStep(
                id=f"sel_{i}", frame=i,
                title=ctx.t("labs.po.anim.title", "selection strength = {s}", s=fmt(s, 2)),
                what_you_see=ctx.t("labs.po.anim.see",
                                   "Four estimators of the same treatment effect."),
                what_changed=ctx.t("labs.po.anim.changed",
                                   "How strongly treatment depends on X moved to {s}.",
                                   s=fmt(s, 2)),
                why=ctx.t("labs.po.anim.why",
                          "Stronger selection makes the treated and untreated groups more "
                          "different in X, so the naive comparison confuses their "
                          "pre-existing gap with the effect."),
                interpretation=ctx.t("labs.po.anim.interpret",
                                     "Naive {a}, adjusted {b}, against a true effect of {t}. "
                                     "Propensity scores now span {lo} to {hi}.",
                                     a=fmt(e["naive"], 3), b=fmt(e["regression"], 3),
                                     t=fmt(p["ate"], 3),
                                     lo=fmt(float(np.min(ps)), 3),
                                     hi=fmt(float(np.max(ps)), 3)),
                conclusion=ctx.t("labs.po.anim.conclude",
                                 "Adjustment works while the groups still overlap in X. At "
                                 "the extreme it runs out of comparable units."),
                warning=ctx.t("labs.po.anim.warn",
                              "All of this rests on X being the ONLY thing that drives "
                              "selection. No test in the data can confirm that."),
                math="naive = ATT + (E[Y(0)|D=1] - E[Y(0)|D=0])",
                outputs={"selection": round(float(s), 3),
                         "naive": round(float(e["naive"]), 4),
                         "regression": round(float(e["regression"]), 4),
                         "min_propensity": round(float(np.min(ps)), 4)},
                violated_assumptions=(("exchangeability",) if s > 0.05 else ()) +
                                     (("positivity",) if float(np.min(ps)) < 0.05 else ()),
                highlighted=("estimator_bars",),
            ))
        fig = ctx.figure(
            "labs.po.figure.animation",
            xaxis_title=ctx.t("labs.po.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=380,
        )
        fig.add_trace(go.Bar(x=["naive", "regression", "IPW", "doubly robust"],
                             y=[0, 0, 0, 0],
                             marker={"color": [ctx.color("negative"), ctx.color("primary"),
                                               ctx.color("warning"),
                                               ctx.color("positive")]},
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, float(p["ate"]), ctx.t("labs.common.trace.truth"), "truth",
                    theme=ctx.theme, dash="dash")
        fig.update_yaxes(range=[float(p["ate"]) - 3, float(p["ate"]) + 6])
        build_frames(fig, frames, duration=440, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.po.slider", "selection strength"))
        return animation(
            "selection_grows", fig, steps,
            purpose=ctx.t("labs.po.anim.purpose",
                          "Watch selection bias appear and adjustment try to remove it."),
            summary=ctx.t(
                "labs.po.anim.summary",
                "Causal inference is a missing-data problem: half of every unit's outcomes "
                "are never observed. Randomization makes the missing half ignorable; "
                "adjustment tries to reconstruct it from X, and it works exactly as far as "
                "X really does explain who was treated."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = PotentialOutcomesLab(SPEC)
