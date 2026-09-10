"""Difference-in-differences, event studies and the staggered-adoption trap."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.causal import did_panel

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "causal.did",
    Domain.CAUSAL,
    "designs",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.VISUAL_DERIVATION,
    controls=(
        slider("effect", 2.0, -5.0, 5.0, 0.05, group="effects"),
        slider("pretrend", 0.0, -0.6, 0.6, 0.01, group="violations"),
        slider("anticipation", 0.0, 0.0, 1.0, 0.05, group="violations"),
        slider("heterogeneity", 0.0, 0.0, 4.0, 0.05, group="effects"),
        slider("dynamic_growth", 0.0, -0.3, 0.5, 0.01, group="effects"),
        toggle("staggered", False, group="design"),
        int_slider("n_units", 60, 4, 500, 2, group="design"),
        int_slider("n_periods", 10, 3, 60, 1, group="design"),
        int_slider("treat_period", 5, 1, 30, 1, group="design"),
        slider("treated_share", 0.5, 0.05, 0.95, 0.05, group="design"),
        slider("noise", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        toggle("show_event_study", True, group="views"),
        toggle("show_counterfactual", True, group="views"),
        int_slider("reps", 300, 50, 3000, 50, group="simulation", expensive=True),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", effect=2.0, pretrend=0.0),
        scenario("no_effect", "null", effect=0.0),
        scenario("parallel_trends_violated", "violation", pretrend=0.35),
        scenario("severe_pretrend", "violation", pretrend=0.6, effect=0.0),
        scenario("anticipation", "violation", anticipation=0.8),
        scenario("dynamic_effects", "compare_methods", dynamic_growth=0.25),
        scenario("staggered_homogeneous", "compare_methods", staggered=True,
                 heterogeneity=0.0),
        scenario("staggered_heterogeneous", "counterexample", staggered=True,
                 heterogeneity=3.5, n_periods=14),
        scenario("small_sample", "small_sample", n_units=8, n_periods=6),
        scenario("large_sample", "large_sample", n_units=300, n_periods=20),
        scenario("high_noise", "high_noise", noise=3.0),
        scenario("few_treated", "boundary", treated_share=0.1),
    ),
    prerequisites=("causal.potential_outcomes",),
    related=("causal.rdd", "panel.fixed_vs_random"),
    next_concepts=("causal.rdd",),
    tags=("difference in differences", "parallel trends", "event study", "twfe",
          "staggered adoption"),
    aliases=("did", "differences en differences", "الفروق في الفروق", "event study",
             "two way fixed effects"),
    backends=("numpy", "pyfixest"),
    references=(
        ref("Goodman-Bacon, A. (2021). Difference-in-differences with variation in "
            "treatment timing. Journal of Econometrics 225(2).", kind="paper",
            doi="10.1016/j.jeconom.2021.03.014"),
        ref("Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences with "
            "multiple time periods. Journal of Econometrics 225(2).", kind="paper",
            doi="10.1016/j.jeconom.2020.12.001"),
    ),
    curriculum_tags=("harvard.api114", "ksu.econ542"),
)


class DiDLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = self._generate(p, state.seed)
        twfe = self._twfe(data)
        two_by_two = self._two_by_two(data, int(p["treat_period"]))
        event = self._event_study(data)

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "trajectories", self._trajectory_figure(ctx, data, p, two_by_two),
            "labs.did.figure.trajectories", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_event_study"]:
            res.add_panel(ctx.panel(
                "event_study", self._event_figure(ctx, event, p),
                "labs.did.figure.event", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        res.add_panel(ctx.panel(
            "estimators", self._estimator_figure(ctx, twfe, two_by_two, data, p),
            "labs.did.figure.estimators", tab="compare",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        sim = self._sampling(p, state.seed, int(p["reps"]))
        res.add_panel(ctx.panel(
            "sampling", self._sampling_figure(ctx, sim, data.truth["att"]),
            "labs.did.figure.sampling", tab="simulation",
            evidence=EvidenceType.SIMULATION,
        ))

        res.metric("true_att", ctx.t("labs.did.metric.att",
                                     "True effect on the treated"), data.truth["att"])
        res.metric("two_by_two", ctx.t("labs.did.metric.two_by_two",
                                       "Classic 2x2 difference-in-differences"),
                   two_by_two["estimate"], reference=data.truth["att"])
        res.metric("twfe", ctx.t("labs.did.metric.twfe",
                                 "Two-way fixed effects estimate"), twfe["estimate"],
                   reference=data.truth["att"])
        res.metric("twfe_se", ctx.t("labs.did.metric.twfe_se",
                                    "Its clustered standard error"), twfe["se"],
                   note=ctx.t("labs.did.metric.cluster_note",
                              "clustered by unit; with few clusters this is optimistic"))
        res.metric("pretrend_slope", ctx.t("labs.did.metric.pretrend",
                                           "Estimated pre-treatment differential trend"),
                   event["pretrend_slope"], reference=float(p["pretrend"]))
        res.metric("pretrend_p", ctx.t("labs.did.metric.pretrend_p",
                                       "Joint p-value on the pre-treatment coefficients"),
                   event["pretrend_p"],
                   note=ctx.t("labs.did.metric.pretrend_note",
                              "a large value is reassuring, not a proof"))
        res.metric("bias", ctx.t("labs.did.metric.bias", "TWFE bias"),
                   twfe["estimate"] - data.truth["att"], reference=0.0)
        res.metric("n_treated", ctx.t("labs.did.metric.n_treated", "Treated units"),
                   int(data.meta["n_treated"]))
        res.metric("sim_bias", ctx.t("labs.did.metric.sim_bias",
                                     "TWFE bias across simulated studies"),
                   float(np.mean(sim["twfe"]) - data.truth["att"]), reference=0.0)

        notes = set(data.notes)
        res.assume("parallel_trends", ctx.t("assumptions.parallel_trends"),
                   "assumption_parallel_trends" not in notes,
                   detail=ctx.t("labs.did.assume.parallel",
                                "The identifying assumption is about counterfactual "
                                "POST-treatment trends, which are never observed. A flat "
                                "pre-period is suggestive evidence, not a test."),
                   consequence="" if "assumption_parallel_trends" not in notes else ctx.t(
                       "labs.did.assume.parallel_consequence",
                       "A differential trend of {t} per period is being attributed to the "
                       "treatment.", t=fmt(p["pretrend"], 3)))
        res.assume("no_anticipation", ctx.t("assumptions.no_anticipation"),
                   "assumption_no_anticipation" not in notes,
                   detail=ctx.t("labs.did.assume.anticipation",
                                "If units respond before adoption, the pre-period is "
                                "already contaminated and the baseline is wrong."))
        res.assume("homogeneous_effects", ctx.t("assumptions.homogeneous_effects"),
                   "assumption_homogeneous_effects" not in notes,
                   detail=ctx.t("labs.did.assume.homogeneous",
                                "With staggered adoption, two-way fixed effects uses "
                                "already-treated units as controls for later adopters."),
                   consequence="" if "assumption_homogeneous_effects" not in notes
                   else ctx.t("labs.did.assume.negative_weights",
                              "Some comparisons then receive negative weights, and the "
                              "estimate can fall outside the range of every individual "
                              "treatment effect."))
        res.assume("sutva", ctx.t("assumptions.sutva"), True,
                   detail=ctx.t("labs.did.assume.sutva",
                                "Untreated units must not be affected by others' treatment. "
                                "Spillovers contaminate the control group directly."))

        res.animations.append(self._animation(ctx, p, state.seed))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.causal.did.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.causal.did.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.causal.did.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.causal.did.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.causal.did.warning"),
                    kind="warning")

        if abs(twfe["estimate"] - data.truth["att"]) > 0.4 * max(
                abs(data.truth["att"]), 0.5):
            res.warnings.append(ctx.t(
                "labs.did.warn.bias",
                "The estimate is {e} against a true effect of {t}. Check the event-study "
                "panel: a sloped pre-period points at parallel trends, and staggered "
                "adoption with heterogeneous effects points at negative weighting.",
                e=fmt(twfe["estimate"], 3), t=fmt(data.truth["att"], 3),
            ))
        if bool(p["staggered"]) and float(p["heterogeneity"]) > 0.5:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.did.counterexample",
                "Adoption is staggered and effects differ across units. Two-way fixed "
                "effects reports {e} while the true average effect on the treated is {t}, "
                "and the simple 2x2 comparison gives {b}. The estimator is a weighted "
                "average of many 2x2 comparisons - including ones that use already-treated "
                "units as controls - and with staggered timing some of those weights are "
                "negative, so the result need not lie between the individual effects at all.",
                e=fmt(twfe["estimate"], 3), t=fmt(data.truth["att"], 3),
                b=fmt(two_by_two["estimate"], 3),
            ), kind="warning")
        return res

    @staticmethod
    def _generate(p, seed):
        return did_panel(
            n_units=int(p["n_units"]), n_periods=int(p["n_periods"]),
            treat_period=int(p["treat_period"]), effect=float(p["effect"]),
            treated_share=float(p["treated_share"]), pretrend=float(p["pretrend"]),
            anticipation=float(p["anticipation"]),
            heterogeneity=float(p["heterogeneity"]), staggered=bool(p["staggered"]),
            dynamic_growth=float(p["dynamic_growth"]), noise=float(p["noise"]),
            seed=seed,
        )

    @staticmethod
    def _twfe(data):
        y, d = data["y"], data["d"]
        unit, period = data["unit"], data["period"]
        units = np.unique(unit)
        periods = np.unique(period)
        cols = [d]
        for u in units[1:]:
            cols.append((unit == u).astype(float))
        for t in periods[1:]:
            cols.append((period == t).astype(float))
        X = np.column_stack([np.ones(y.size)] + cols)
        fit = LM.ols(y, X, cov_type="cluster", groups=unit,
                     names=tuple(["const", "d"] +
                                 [f"u{i}" for i in range(len(units) - 1)] +
                                 [f"t{i}" for i in range(len(periods) - 1)]))
        return {"estimate": fit.coef("d"), "se": fit.se("d"), "fit": fit}

    @staticmethod
    def _two_by_two(data, treat_period):
        y, treated, period = data["y"], data["treated"], data["period"]
        pre = period < treat_period
        post = ~pre
        groups = {}
        for name, mask in (("treated_pre", (treated > 0) & pre),
                           ("treated_post", (treated > 0) & post),
                           ("control_pre", (treated == 0) & pre),
                           ("control_post", (treated == 0) & post)):
            groups[name] = float(np.mean(y[mask])) if mask.any() else np.nan
        est = ((groups["treated_post"] - groups["treated_pre"])
               - (groups["control_post"] - groups["control_pre"]))
        return {"estimate": est, "means": groups}

    @staticmethod
    def _event_study(data):
        y = data["y"]
        rel = data["rel_time"]
        unit, period = data["unit"], data["period"]
        finite = np.isfinite(rel)
        offsets = sorted({int(v) for v in rel[finite]})
        offsets = [o for o in offsets if -6 <= o <= 8 and o != -1]
        cols = []
        names = []
        for o in offsets:
            cols.append(((rel == o) & finite).astype(float))
            names.append(f"rel_{o}")
        units = np.unique(unit)
        periods = np.unique(period)
        for u in units[1:]:
            cols.append((unit == u).astype(float))
            names.append(f"u{int(u)}")
        for t in periods[1:]:
            cols.append((period == t).astype(float))
            names.append(f"t{int(t)}")
        X = np.column_stack([np.ones(y.size)] + cols)
        fit = LM.ols(y, X, cov_type="cluster", groups=unit,
                     names=tuple(["const"] + names))
        coefs = {o: fit.coefficients[1 + i] for i, o in enumerate(offsets)}
        ses = {o: fit.standard_errors[1 + i] for i, o in enumerate(offsets)}
        pre_offsets = [o for o in offsets if o < 0]
        if pre_offsets:
            R = np.zeros((len(pre_offsets), X.shape[1]))
            for i, o in enumerate(pre_offsets):
                R[i, 1 + offsets.index(o)] = 1.0
            test = LM.f_test(fit, R)
            xs = np.asarray(pre_offsets, dtype=float)
            ys = np.asarray([coefs[o] for o in pre_offsets])
            slope = (float(np.polyfit(xs, ys, 1)[0]) if xs.size > 1 else float("nan"))
            pval = test.p_value
        else:
            slope, pval = float("nan"), float("nan")
        return {"coefs": coefs, "ses": ses, "offsets": offsets,
                "pretrend_slope": slope, "pretrend_p": pval}

    def _sampling(self, p, seed, reps):
        reps = min(reps, 1500)
        out = []
        for r in range(reps):
            d = self._generate(p, int(seed) * 274177 + r)
            out.append(self._twfe(d)["estimate"])
        return {"twfe": np.asarray(out)}

    def _trajectory_figure(self, ctx, data, p, two_by_two):
        periods = np.unique(data["period"])
        treated_mean = np.array([data["y"][(data["treated"] > 0) &
                                           (data["period"] == t)].mean()
                                 for t in periods])
        control_mean = np.array([data["y"][(data["treated"] == 0) &
                                           (data["period"] == t)].mean()
                                 for t in periods])
        cut = int(p["treat_period"])
        fig = ctx.figure(
            "labs.did.figure.trajectories",
            xaxis_title=ctx.t("labs.common.axis.period"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=440,
        )
        P.add_curve(fig, periods, treated_mean, ctx.t("labs.common.trace.treated"),
                    "treatment", theme=ctx.theme, mode="lines+markers", width=3.0)
        P.add_curve(fig, periods, control_mean, ctx.t("labs.common.trace.control"),
                    "control", theme=ctx.theme, mode="lines+markers", width=3.0)
        if p["show_counterfactual"] and cut < periods.size:
            shift = treated_mean[cut - 1] - control_mean[cut - 1] if cut >= 1 else 0.0
            counter = control_mean + shift
            P.add_curve(fig, periods[cut - 1:], counter[cut - 1:],
                        ctx.t("labs.common.trace.counterfactual"),
                        "baseline", theme=ctx.theme, dash="dash", width=2.6)
            go = P.require_plotly()
            gap_x = list(periods[cut:]) + list(periods[cut:])[::-1]
            gap_y = list(treated_mean[cut:]) + list(counter[cut:])[::-1]
            fig.add_trace(go.Scatter(x=gap_x, y=gap_y, fill="toself",
                                     fillcolor=P.rgba("power", 0.2, ctx.theme),
                                     line={"width": 0},
                                     name=ctx.t("labs.did.trace.gap",
                                                "estimated effect")))
        P.add_vline(fig, cut - 0.5,
                    ctx.t("labs.did.trace.adoption", "treatment begins"),
                    "warning", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.did.legend_trajectories",
            "The dashed line is not data. It is the control group's path shifted to match "
            "the treated group's last pre-period level - the counterfactual the design "
            "assumes into existence.",
        ), theme=ctx.theme)
        return fig

    def _event_figure(self, ctx, event, p):
        offsets = event["offsets"]
        coefs = [event["coefs"][o] for o in offsets]
        ses = [1.96 * event["ses"][o] for o in offsets]
        fig = ctx.figure(
            "labs.did.figure.event",
            xaxis_title=ctx.t("labs.did.axis.relative", "Periods relative to adoption"),
            yaxis_title=ctx.t("labs.did.axis.coefficient",
                              "Coefficient (period -1 normalized to zero)"),
            height=390,
        )
        go = P.require_plotly()
        pre = [i for i, o in enumerate(offsets) if o < 0]
        post = [i for i, o in enumerate(offsets) if o >= 0]
        for idx, role, label in (
            (pre, "muted", ctx.t("labs.did.trace.pre", "pre-treatment")),
            (post, "primary", ctx.t("labs.did.trace.post", "post-treatment")),
        ):
            if not idx:
                continue
            fig.add_trace(go.Scatter(
                x=[offsets[i] for i in idx], y=[coefs[i] for i in idx],
                error_y={"type": "data", "array": [ses[i] for i in idx]},
                mode="markers", marker={"color": ctx.color(role), "size": 9},
                name=label,
            ))
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dash")
        P.add_vline(fig, -0.5, ctx.t("labs.did.trace.adoption", "treatment begins"),
                    "warning", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.did.legend_event",
            "Flat grey points before adoption support the design; a slope there undermines "
            "it. Joint p-value on the pre-period coefficients: {p}. Passing this check is "
            "necessary, never sufficient.",
            p=fmt(event["pretrend_p"], 4),
        ), theme=ctx.theme)
        return fig

    def _estimator_figure(self, ctx, twfe, two_by_two, data, p):
        go = P.require_plotly()
        labels = [ctx.t("labs.did.metric.two_by_two",
                        "Classic 2x2 difference-in-differences"),
                  ctx.t("labs.did.metric.twfe", "Two-way fixed effects estimate")]
        values = [two_by_two["estimate"], twfe["estimate"]]
        fig = ctx.figure(
            "labs.did.figure.estimators",
            xaxis_title=ctx.t("labs.did.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=360,
        )
        fig.add_trace(go.Bar(x=labels, y=values,
                             marker={"color": [ctx.color("secondary"),
                                               ctx.color("primary")]},
                             error_y={"type": "data",
                                      "array": [0.0, 1.96 * twfe["se"]]},
                             text=[fmt(v, 3) for v in values], textposition="outside",
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, data.truth["att"], ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.did.legend_estimators",
            "With one adoption date and homogeneous effects the two agree. Where they "
            "separate, the two-way fixed-effects weighting is doing something the simple "
            "comparison is not.",
        ), theme=ctx.theme)
        return fig

    def _sampling_figure(self, ctx, sim, att):
        fig = ctx.figure(
            "labs.did.figure.sampling",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=350,
        )
        P.add_histogram(fig, sim["twfe"],
                        ctx.t("labs.did.trace.twfe_dist",
                              "TWFE estimate across simulated studies"),
                        "primary", theme=ctx.theme, nbins=45, opacity=0.6)
        P.add_vline(fig, att, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.did.legend_sampling",
            "If the histogram is centred away from the line, the design is biased - and no "
            "sample size fixes it. If it is merely wide, more data will help.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, seed):
        go = P.require_plotly()
        pretrends = np.linspace(-0.4, 0.4, 17)
        frames, steps = [], []
        for i, pt in enumerate(pretrends):
            q = dict(p)
            q["pretrend"] = float(pt)
            d = self._generate(q, seed)
            periods = np.unique(d["period"])
            tm = np.array([d["y"][(d["treated"] > 0) & (d["period"] == t)].mean()
                           for t in periods])
            cm = np.array([d["y"][(d["treated"] == 0) & (d["period"] == t)].mean()
                           for t in periods])
            est = self._twfe(d)["estimate"]
            frames.append(go.Frame(name=f"{pt:.2f}", data=[
                go.Scatter(x=periods, y=tm), go.Scatter(x=periods, y=cm),
            ]))
            steps.append(AnimationStep(
                id=f"pt_{i}", frame=i,
                title=ctx.t("labs.did.anim.title",
                            "differential pre-trend = {t} per period", t=fmt(pt, 2)),
                what_you_see=ctx.t("labs.did.anim.see",
                                   "Group means over time for the treated and control "
                                   "groups."),
                what_changed=ctx.t("labs.did.anim.changed",
                                   "The treated group's own trend moved to {t} per period.",
                                   t=fmt(pt, 2)),
                why=ctx.t("labs.did.anim.why",
                          "Difference-in-differences subtracts the control group's change. "
                          "If the treated group was already moving differently, that "
                          "difference is attributed to the treatment."),
                interpretation=ctx.t("labs.did.anim.interpret",
                                     "The estimate is {e} against a true effect of {t}.",
                                     e=fmt(est, 3), t=fmt(d.truth["att"], 3)),
                conclusion=ctx.t("labs.did.anim.conclude",
                                 "The pre-period slope and the bias move together - which "
                                 "is exactly why the event study is worth plotting."),
                warning=ctx.t("labs.did.anim.warn",
                              "A flat pre-period does not guarantee parallel counterfactual "
                              "trends AFTER treatment. That part is never observable."),
                math="DiD = (Ybar_T,post - Ybar_T,pre) - (Ybar_C,post - Ybar_C,pre)",
                outputs={"pretrend": round(float(pt), 3),
                         "estimate": round(float(est), 4),
                         "bias": round(float(est - d.truth["att"]), 4)},
                violated_assumptions=("parallel_trends",) if abs(pt) > 0.01 else (),
                highlighted=("group_means",),
            ))
        d0 = self._generate(p, seed)
        periods = np.unique(d0["period"])
        fig = ctx.figure(
            "labs.did.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.period"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=periods, y=np.zeros_like(periods, dtype=float),
                                 mode="lines+markers",
                                 line={"color": ctx.color("treatment"), "width": 3.0},
                                 name=ctx.t("labs.common.trace.treated")))
        fig.add_trace(go.Scatter(x=periods, y=np.zeros_like(periods, dtype=float),
                                 mode="lines+markers",
                                 line={"color": ctx.color("control"), "width": 3.0},
                                 name=ctx.t("labs.common.trace.control")))
        P.add_vline(fig, int(p["treat_period"]) - 0.5,
                    ctx.t("labs.did.trace.adoption", "treatment begins"),
                    "warning", theme=ctx.theme)
        fig.update_yaxes(range=[0, 16])
        build_frames(fig, frames, duration=440, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.did.slider", "pre-trend"))
        return animation(
            "parallel_trends", fig, steps,
            purpose=ctx.t("labs.did.anim.purpose",
                          "Show that the design's credibility lives entirely in a "
                          "counterfactual you cannot see."),
            summary=ctx.t(
                "labs.did.anim.summary",
                "Difference-in-differences needs the control group's change to be the "
                "treated group's counterfactual change. Break that and the estimate moves "
                "smoothly and silently. The event study is the closest thing to a check - "
                "and it only inspects the periods before treatment."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = DiDLab(SPEC)
