"""Regression discontinuity: the jump at the cutoff, and what can fake one."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.causal import rdd_dataset

__all__ = ["LAB", "SPEC", "local_linear_rd"]


def local_linear_rd(x, y, cutoff, bandwidth, degree: int = 1, d=None):
    """Local polynomial RD estimate with a triangular kernel."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centred = x - cutoff
    inside = np.abs(centred) <= bandwidth
    if inside.sum() < 2 * (degree + 2):
        return {"estimate": float("nan"), "se": float("nan"), "n_used": int(inside.sum()),
                "fuzzy": None}
    w = np.clip(1.0 - np.abs(centred[inside]) / bandwidth, 0.0, None)
    above = (centred[inside] >= 0).astype(float)
    cols = [np.ones(int(inside.sum())), above]
    for k in range(1, degree + 1):
        cols.append(centred[inside] ** k)
        cols.append(above * centred[inside] ** k)
    X = np.column_stack(cols)
    fit = LM.wls(y[inside], X, np.clip(w, 1e-8, None),
                 names=tuple(["const", "above"] +
                             [f"p{k}{s}" for k in range(1, degree + 1) for s in ("", "_x")]))
    out = {"estimate": fit.coef("above"), "se": fit.se("above"),
           "n_used": int(inside.sum()), "fuzzy": None}
    if d is not None:
        first = LM.wls(np.asarray(d, dtype=float)[inside], X, np.clip(w, 1e-8, None),
                       names=fit.names)
        jump_d = first.coef("above")
        if abs(jump_d) > 1e-6:
            out["fuzzy"] = float(out["estimate"] / jump_d)
            out["first_stage_jump"] = float(jump_d)
    return out


SPEC = make_spec(
    "causal.rdd",
    Domain.CAUSAL,
    "designs",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.VISUAL_DERIVATION,
    controls=(
        slider("effect", 2.0, -5.0, 5.0, 0.05, group="effects"),
        slider("cutoff", 0.0, -2.0, 2.0, 0.05, group="design"),
        slider("bandwidth", 1.0, 0.1, 3.0, 0.05, group="estimation"),
        int_slider("degree", 1, 0, 3, 1, group="estimation"),
        slider("slope_left", 1.0, -3.0, 3.0, 0.05, group="dgp"),
        slider("slope_right", 1.0, -3.0, 3.0, 0.05, group="dgp"),
        slider("curvature", 0.0, -1.5, 1.5, 0.05, group="dgp"),
        slider("manipulation", 0.0, 0.0, 1.0, 0.05, group="violations"),
        slider("fuzzy", 1.0, 0.5, 1.0, 0.01, group="design"),
        int_slider("n", 800, 50, 20000, 50, group="dgp"),
        slider("noise", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        toggle("show_density", True, group="views"),
        toggle("show_bandwidth_curve", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", effect=2.0, curvature=0.0),
        scenario("no_effect", "null", effect=0.0),
        scenario("different_slopes", "compare_methods", slope_left=1.5,
                 slope_right=-0.5),
        scenario("curvature_fakes_jump", "counterexample", effect=0.0, curvature=1.2,
                 degree=0, bandwidth=3.0),
        scenario("curvature_handled", "robustness", effect=0.0, curvature=1.2,
                 degree=1, bandwidth=0.6),
        scenario("narrow_bandwidth", "small_sample", bandwidth=0.2),
        scenario("wide_bandwidth", "misspecification", bandwidth=3.0, curvature=0.8),
        scenario("manipulation", "violation", manipulation=0.8),
        scenario("fuzzy_design", "compare_methods", fuzzy=0.75),
        scenario("high_noise", "high_noise", noise=3.0),
        scenario("large_sample", "large_sample", n=8000),
    ),
    prerequisites=("causal.potential_outcomes",),
    related=("causal.did", "econometrics.endogeneity_iv"),
    tags=("regression discontinuity", "cutoff", "bandwidth", "mccrary",
          "local polynomial", "fuzzy rd"),
    aliases=("rdd", "regression sur discontinuite", "الانقطاع الانحداري",
             "local randomization", "running variable"),
    backends=("numpy",),
    references=(
        ref("Imbens, G. W. and Lemieux, T. (2008). Regression discontinuity designs. "
            "Journal of Econometrics 142(2).", kind="paper",
            doi="10.1016/j.jeconom.2007.05.001"),
        ref("Gelman, A. and Imbens, G. (2019). Why high-order polynomials should not be "
            "used in regression discontinuity designs. JBES 37(3).", kind="paper",
            doi="10.1080/07350015.2017.1366909"),
    ),
    curriculum_tags=("harvard.api114",),
)


class RDDLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = self._generate(p, int(p["n"]), state.seed)
        cutoff = float(p["cutoff"])
        est = local_linear_rd(data["x"], data["y"], cutoff, float(p["bandwidth"]),
                              int(p["degree"]),
                              d=data["d"] if float(p["fuzzy"]) < 1.0 else None)
        globalfit = self._global_polynomial(data, cutoff, 4)
        density = self._density_test(data["x"], cutoff)

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "discontinuity", self._rd_figure(ctx, data, p, est, globalfit),
            "labs.rdd.figure.discontinuity", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_density"]:
            res.add_panel(ctx.panel(
                "density", self._density_figure(ctx, data, cutoff, density),
                "labs.rdd.figure.density", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_bandwidth_curve"]:
            res.add_panel(ctx.panel(
                "bandwidth", self._bandwidth_figure(ctx, data, p),
                "labs.rdd.figure.bandwidth", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("true_effect", ctx.t("labs.rdd.metric.truth",
                                        "True effect at the cutoff"), float(p["effect"]))
        res.metric("estimate", ctx.t("labs.rdd.metric.estimate",
                                     "Local polynomial estimate"), est["estimate"],
                   reference=float(p["effect"]))
        res.metric("se", ctx.t("labs.rdd.metric.se", "Its standard error"), est["se"])
        res.metric("n_used", ctx.t("labs.rdd.metric.n_used",
                                   "Observations inside the bandwidth"), est["n_used"],
                   note=ctx.t("labs.rdd.metric.n_used_note",
                              "out of {n} in total", n=int(p["n"])))
        if est.get("fuzzy") is not None:
            res.metric("fuzzy_estimate", ctx.t("labs.rdd.metric.fuzzy",
                                               "Fuzzy RD estimate (jump ratio)"),
                       est["fuzzy"], reference=float(p["effect"]))
            res.metric("first_stage_jump", ctx.t("labs.rdd.metric.first_jump",
                                                 "Jump in the treatment probability"),
                       est.get("first_stage_jump", float("nan")))
        res.metric("global_polynomial", ctx.t("labs.rdd.metric.global",
                                              "Global 4th-order polynomial estimate"),
                   globalfit["estimate"], reference=float(p["effect"]),
                   note=ctx.t("labs.rdd.metric.global_note",
                              "high-order global fits are notoriously unreliable at the "
                              "boundary"))
        res.metric("density_jump_p", ctx.t("labs.rdd.metric.density",
                                           "Density-continuity test p-value"),
                   density["p_value"],
                   note=ctx.t("labs.rdd.metric.density_note",
                              "small values suggest units sorted across the cutoff"))
        res.metric("bias", ctx.t("labs.rdd.metric.bias", "Bias at this bandwidth"),
                   est["estimate"] - float(p["effect"]), reference=0.0)

        manipulated = float(p["manipulation"]) > 0.01
        res.assume("no_manipulation", ctx.t("assumptions.no_manipulation"),
                   not manipulated and density["p_value"] > 0.05,
                   detail=ctx.t("labs.rdd.assume.manipulation",
                                "If units can choose which side of the cutoff to land on, "
                                "those just above and just below are no longer comparable."),
                   consequence="" if not manipulated else ctx.t(
                       "labs.rdd.assume.manipulation_consequence",
                       "The design collapses: the jump now mixes the treatment effect with "
                       "whatever made units sort."))
        res.assume("continuity", ctx.t("labs.rdd.assume.continuity_label",
                                       "Potential outcomes are continuous at the cutoff"),
                   True,
                   detail=ctx.t("labs.rdd.assume.continuity",
                                "Anything else that changes discontinuously at the same "
                                "threshold - another programme, another rule - is "
                                "indistinguishable from the treatment."))
        res.assume("correct_functional_form",
                   ctx.t("assumptions.correct_functional_form"),
                   float(p["curvature"]) == 0.0 or int(p["degree"]) >= 1,
                   detail=ctx.t("labs.rdd.assume.form",
                                "Curvature in the running variable can masquerade as a jump "
                                "if the local fit is too rigid or the bandwidth too wide."))

        res.animations.append(self._animation(ctx, data, p))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.causal.rdd.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.causal.rdd.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.causal.rdd.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.causal.rdd.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.causal.rdd.warning"),
                    kind="warning")
        res.explain("diagnostics", ctx.t("labs.rdd.checklist.title",
                                         "The standard credibility checks"), ctx.t(
            "labs.rdd.checklist",
            "1. Density of the running variable continuous at the cutoff? p = {d}.\n"
            "2. Estimate stable across bandwidths? See the bandwidth panel.\n"
            "3. Estimate stable across polynomial degrees? Compare {loc} with the global "
            "fit's {glob}.\n"
            "4. No jump at placebo cutoffs away from the real one.\n"
            "5. Covariates that should be unaffected show no jump.\n"
            "None of these proves the design; each of them can refute it.",
            d=fmt(density["p_value"], 4), loc=fmt(est["estimate"], 3),
            glob=fmt(globalfit["estimate"], 3),
        ))

        if manipulated or density["p_value"] < 0.05:
            res.warnings.append(ctx.t(
                "labs.rdd.warn.manipulation",
                "The density of the running variable jumps at the cutoff (p = {p}). That is "
                "evidence of sorting, and it invalidates the comparison rather than merely "
                "weakening it.", p=fmt(density["p_value"], 4),
            ))
        if float(p["curvature"]) != 0 and int(p["degree"]) == 0 \
                and abs(est["estimate"] - float(p["effect"])) > 0.5:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.rdd.counterexample",
                "The true effect here is {t}, but a flat local fit over a wide bandwidth "
                "reports {e}. Curvature in the running variable is being read as a jump. "
                "Narrow the bandwidth or allow a local slope and the artefact disappears.",
                t=fmt(p["effect"], 2), e=fmt(est["estimate"], 3),
            ), kind="warning")
        return res

    @staticmethod
    def _generate(p, n, seed):
        return rdd_dataset(
            n=n, cutoff=float(p["cutoff"]), effect=float(p["effect"]),
            slope_left=float(p["slope_left"]), slope_right=float(p["slope_right"]),
            curvature=float(p["curvature"]), noise=float(p["noise"]),
            manipulation=float(p["manipulation"]), fuzzy=float(p["fuzzy"]), seed=seed,
        )

    @staticmethod
    def _global_polynomial(data, cutoff, degree):
        x = data["x"] - cutoff
        above = (x >= 0).astype(float)
        cols = [np.ones(x.size), above]
        for k in range(1, degree + 1):
            cols.append(x**k)
            cols.append(above * x**k)
        fit = LM.ols(data["y"], np.column_stack(cols),
                     names=tuple(["const", "above"] +
                                 [f"p{k}{s}" for k in range(1, degree + 1)
                                  for s in ("", "_x")]))
        return {"estimate": fit.coef("above"), "fit": fit, "degree": degree}

    @staticmethod
    def _density_test(x, cutoff, bandwidth: float = 0.5):
        """A simple continuity check: compare counts just below and just above."""
        below = int(np.sum((x >= cutoff - bandwidth) & (x < cutoff)))
        above = int(np.sum((x >= cutoff) & (x < cutoff + bandwidth)))
        total = below + above
        if total < 10:
            return {"below": below, "above": above, "p_value": float("nan")}
        p = float(stats.binomtest(above, total, 0.5).pvalue)
        return {"below": below, "above": above, "p_value": p}

    def _rd_figure(self, ctx, data, p, est, globalfit):
        cutoff = float(p["cutoff"])
        bw = float(p["bandwidth"])
        x, y = data["x"], data["y"]
        fig = ctx.figure(
            "labs.rdd.figure.discontinuity",
            xaxis_title=ctx.t("labs.rdd.axis.running", "Running variable"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=450,
        )
        inside = np.abs(x - cutoff) <= bw
        P.add_points(fig, x[~inside], y[~inside],
                     ctx.t("labs.rdd.trace.outside", "outside the bandwidth"),
                     "muted", theme=ctx.theme, size=4, opacity=0.3)
        P.add_points(fig, x[inside], y[inside],
                     ctx.t("labs.rdd.trace.inside", "inside the bandwidth"),
                     "primary", theme=ctx.theme, size=6, opacity=0.75)
        bins = self._binned(x, y, cutoff, 24)
        P.add_points(fig, bins[0], bins[1],
                     ctx.t("labs.rdd.trace.binned", "binned means"),
                     "truth", theme=ctx.theme, size=10)
        for side, role in ((-1, "control"), (1, "treatment")):
            mask = inside & (np.sign(x - cutoff + 1e-12) == side)
            if mask.sum() > 3:
                f = LM.ols(y[mask],
                           np.column_stack([np.ones(int(mask.sum())), x[mask] - cutoff]))
                grid = (np.linspace(cutoff - bw, cutoff, 40) if side < 0
                        else np.linspace(cutoff, cutoff + bw, 40))
                P.add_curve(fig, grid,
                            f.coefficients[0] + f.coefficients[1] * (grid - cutoff),
                            ctx.t("labs.rdd.trace.local",
                                  "local fit {s} the cutoff",
                                  s=("below" if side < 0 else "above")),
                            role, theme=ctx.theme, width=3.2)
        P.add_vline(fig, cutoff, ctx.t("labs.rdd.trace.cutoff", "cutoff"),
                    "warning", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.rdd.legend_rd",
            "The estimate is the vertical distance between the two local fits AT the "
            "cutoff - {e}, against a true effect of {t}. Everything far from the line "
            "informs the shape, not the jump.",
            e=fmt(est["estimate"], 3), t=fmt(p["effect"], 3),
        ), theme=ctx.theme)
        return fig

    @staticmethod
    def _binned(x, y, cutoff, bins):
        centres, means = [], []
        for side in (-1, 1):
            mask = np.sign(x - cutoff + 1e-12) == side
            if mask.sum() < bins:
                continue
            edges = np.quantile(x[mask], np.linspace(0, 1, bins // 2 + 1))
            edges = np.unique(edges)
            idx = np.clip(np.digitize(x[mask], edges[1:-1]), 0, len(edges) - 2)
            for b in range(len(edges) - 1):
                sel = idx == b
                if sel.sum() > 2:
                    centres.append(float(np.mean(x[mask][sel])))
                    means.append(float(np.mean(y[mask][sel])))
        return np.asarray(centres), np.asarray(means)

    def _density_figure(self, ctx, data, cutoff, density):
        fig = ctx.figure(
            "labs.rdd.figure.density",
            xaxis_title=ctx.t("labs.rdd.axis.running", "Running variable"),
            yaxis_title=ctx.t("labs.common.axis.frequency"),
            height=340,
        )
        P.add_histogram(fig, data["x"],
                        ctx.t("labs.rdd.trace.density",
                              "density of the running variable"),
                        "primary", theme=ctx.theme, nbins=50, density=False)
        P.add_vline(fig, cutoff, ctx.t("labs.rdd.trace.cutoff", "cutoff"),
                    "warning", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.rdd.legend_density",
            "{below} observations just below the cutoff against {above} just above "
            "(p = {p}). A pile-up on one side is the clearest sign that units chose their "
            "position.",
            below=density["below"], above=density["above"],
            p=fmt(density["p_value"], 4),
        ), theme=ctx.theme)
        return fig

    def _bandwidth_figure(self, ctx, data, p):
        cutoff = float(p["cutoff"])
        bws = np.linspace(0.15, 3.0, 24)
        ests, los, his = [], [], []
        for bw in bws:
            e = local_linear_rd(data["x"], data["y"], cutoff, float(bw), int(p["degree"]))
            ests.append(e["estimate"])
            los.append(e["estimate"] - 1.96 * e["se"])
            his.append(e["estimate"] + 1.96 * e["se"])
        fig = ctx.figure(
            "labs.rdd.figure.bandwidth",
            xaxis_title=ctx.t("labs.rdd.axis.bandwidth", "Bandwidth"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=370,
        )
        P.shade_between(fig, bws, np.asarray(los), np.asarray(his),
                        ctx.t("labs.rdd.trace.ci", "95% interval"),
                        "info", theme=ctx.theme, alpha=0.18)
        P.add_curve(fig, bws, ests,
                    ctx.t("labs.rdd.trace.estimate", "estimate at each bandwidth"),
                    "primary", theme=ctx.theme)
        P.add_hline(fig, float(p["effect"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_vline(fig, float(p["bandwidth"]), ctx.t("labs.common.trace.current"),
                    "highlight", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.rdd.legend_bandwidth",
            "This is the bias-variance trade-off made visible: narrow bandwidths are noisy "
            "but honest, wide ones are precise but pick up curvature. A result that only "
            "exists at one bandwidth is not a result.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, data, p):
        go = P.require_plotly()
        cutoff = float(p["cutoff"])
        bws = np.linspace(2.8, 0.2, 16)
        x, y = data["x"], data["y"]
        frames, steps = [], []
        for i, bw in enumerate(bws):
            e = local_linear_rd(x, y, cutoff, float(bw), int(p["degree"]))
            inside = np.abs(x - cutoff) <= bw
            frames.append(go.Frame(name=f"{bw:.2f}", data=[
                go.Scatter(x=x[inside], y=y[inside]),
            ]))
            steps.append(AnimationStep(
                id=f"bw_{i}", frame=i,
                title=ctx.t("labs.rdd.anim.title", "bandwidth = {b}", b=fmt(bw, 2)),
                what_you_see=ctx.t("labs.rdd.anim.see",
                                   "Only the observations the estimator is currently "
                                   "allowed to use."),
                what_changed=ctx.t("labs.rdd.anim.changed",
                                   "The bandwidth narrowed to {b}, leaving {k} "
                                   "observations.", b=fmt(bw, 2), k=int(inside.sum())),
                why=ctx.t("labs.rdd.anim.why",
                          "Regression discontinuity is a local method. Units far from the "
                          "cutoff are informative about the shape of the relationship but "
                          "not about the jump."),
                interpretation=ctx.t("labs.rdd.anim.interpret",
                                     "Estimate {e} with standard error {s}, against a true "
                                     "effect of {t}.",
                                     e=fmt(e["estimate"], 3), s=fmt(e["se"], 3),
                                     t=fmt(p["effect"], 3)),
                conclusion=ctx.t("labs.rdd.anim.conclude",
                                 "Narrowing reduces bias from curvature and increases "
                                 "variance from having fewer points. There is no bandwidth "
                                 "that is best at both."),
                warning=ctx.t("labs.rdd.anim.warn",
                              "Choosing the bandwidth after seeing which one gives the "
                              "answer you wanted is not estimation."),
                math="tau = lim(x->c+) E[Y|X=x] - lim(x->c-) E[Y|X=x]",
                outputs={"bandwidth": round(float(bw), 3),
                         "estimate": round(float(e["estimate"]), 4),
                         "se": round(float(e["se"]), 4),
                         "n_used": int(inside.sum())},
                highlighted=("bandwidth_window",),
            ))
        fig = ctx.figure(
            "labs.rdd.figure.animation",
            xaxis_title=ctx.t("labs.rdd.axis.running", "Running variable"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=390,
        )
        P.add_points(fig, x, y, ctx.t("labs.rdd.trace.all", "all observations"),
                     "muted", theme=ctx.theme, size=4, opacity=0.25)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 6},
                                 name=ctx.t("labs.rdd.trace.inside",
                                            "inside the bandwidth")))
        P.add_vline(fig, cutoff, ctx.t("labs.rdd.trace.cutoff", "cutoff"),
                    "warning", theme=ctx.theme)
        build_frames(fig, frames, duration=440, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.rdd.slider", "bandwidth"))
        return animation(
            "shrinking_window", fig, steps,
            purpose=ctx.t("labs.rdd.anim.purpose",
                          "Show regression discontinuity as a deliberate act of throwing "
                          "data away."),
            summary=ctx.t(
                "labs.rdd.anim.summary",
                "The design's credibility comes from comparing units that are nearly "
                "identical, which means using only those near the cutoff. That is why the "
                "estimate is local, why the bandwidth matters so much, and why "
                "extrapolating it to units far from the threshold is unwarranted."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = RDDLab(SPEC)
