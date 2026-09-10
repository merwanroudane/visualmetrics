"""Frisch-Waugh-Lovell: a multiple-regression coefficient is a simple regression
on what is left over."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...backends import linear as LM
from ...simulation.random import rng

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "regression.fwl",
    Domain.REGRESSION,
    "partialling_out",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.GEOMETRIC_PROOF,
    controls=(
        int_slider("n", 200, 10, 5000, 1, group="dgp"),
        slider("beta_x", 1.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("beta_z", 2.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("corr_xz", 0.6, -0.98, 0.98, 0.01, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        int_slider("n_controls", 1, 1, 6, 1, group="dgp"),
        toggle("show_added_variable", True, group="views"),
        toggle("show_residual_scatter", True, group="views"),
        toggle("bad_control", False, group="violations"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", corr_xz=0.6, beta_x=1.0, beta_z=2.0),
        scenario("orthogonal_controls", "null", corr_xz=0.0),
        scenario("strongly_confounded", "strong", corr_xz=0.9, beta_z=3.0),
        scenario("near_collinear", "boundary", corr_xz=0.98),
        scenario("negative_correlation", "negative", corr_xz=-0.7),
        scenario("many_controls", "sensitivity", n_controls=5, n=400),
        scenario("small_sample", "small_sample", n=20),
        scenario("bad_control_collider", "counterexample", bad_control=True, n=400),
        scenario("high_noise", "high_noise", noise=5.0),
    ),
    prerequisites=("regression.ols_geometry",),
    related=("regression.multicollinearity", "econometrics.omitted_variable_bias"),
    tags=("frisch-waugh-lovell", "partial regression", "added variable plot",
          "partial correlation", "bad control"),
    aliases=("fwl", "frisch waugh", "الانحدار الجزئي", "partialling out"),
    backends=("numpy",),
    proof_ids=("regression.fwl.theorem",),
    references=(
        ref("Frisch, R. and Waugh, F. V. (1933). Partial time regressions as compared "
            "with individual trends. Econometrica 1(4).", kind="paper",
            doi="10.2307/1907330"),
        ref("Lovell, M. C. (1963). Seasonal adjustment of economic time series. JASA 58.",
            kind="paper", doi="10.1080/01621459.1963.10480682"),
    ),
    curriculum_tags=("dz.econometrics1",),
)


class FWLLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        gen = rng(state.seed, "fwl")
        x, Z, y, names = self._generate(gen, p, n)

        full_X = np.column_stack([np.ones(n), x, Z])
        full = LM.ols(y, full_X, names=("const", "x", *names))
        y_res, x_res = LM.partial_out(y, x, np.column_stack([np.ones(n), Z]))
        short = LM.ols(y_res, x_res.reshape(-1, 1), has_constant=False, names=("x_residual",))
        naive = LM.ols(y, np.column_stack([np.ones(n), x]), names=("const", "x"))

        gap = abs(full.coef("x") - short.coef("x_residual"))

        res.dgp = ctx.t(
            "labs.fwl.dgp",
            "y = {bx} x + {bz} z + noise, with corr(x, z) = {r} and n = {n}. The lab "
            "recovers the x coefficient twice: once from the full regression and once "
            "from residuals.",
            bx=fmt(p["beta_x"], 2), bz=fmt(p["beta_z"], 2), r=fmt(p["corr_xz"], 2), n=n,
        )

        res.add_panel(ctx.panel(
            "added_variable", self._added_variable_figure(ctx, x_res, y_res, short, full),
            "labs.fwl.figure.added_variable", evidence=EvidenceType.GEOMETRIC_PROOF,
        ))
        if p["show_residual_scatter"]:
            res.add_panel(ctx.panel(
                "residualization", self._residualization_figure(ctx, x, Z, y, n),
                "labs.fwl.figure.residualization", tab="math",
                evidence=EvidenceType.VISUAL_DERIVATION,
            ))
        res.add_panel(ctx.panel(
            "comparison", self._comparison_figure(ctx, naive, full, short, p),
            "labs.fwl.figure.comparison", tab="compare",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))

        res.metric("multiple_coefficient", ctx.t("labs.fwl.metric.full",
                                                 "x coefficient in the full regression"),
                   full.coef("x"), reference=float(p["beta_x"]))
        res.metric("fwl_coefficient", ctx.t("labs.fwl.metric.fwl",
                                            "x coefficient from residual-on-residual"),
                   short.coef("x_residual"), reference=full.coef("x"))
        res.metric("difference", ctx.t("labs.fwl.metric.gap",
                                       "Difference between the two (should be zero)"),
                   gap, reference=0.0)
        res.metric("residual_match", ctx.t("labs.fwl.metric.resid_match",
                                           "max |residual difference| between the two fits"),
                   float(np.max(np.abs(full.residuals - short.residuals))), reference=0.0)
        res.metric("naive_coefficient", ctx.t("labs.fwl.metric.naive",
                                              "x coefficient ignoring the controls"),
                   naive.coef("x"), reference=float(p["beta_x"]),
                   note=ctx.t("labs.fwl.metric.naive_note",
                              "this is the one that moves when controls are added"))
        res.metric("partial_correlation", ctx.t("labs.fwl.metric.partial_corr",
                                                "Partial correlation of x and y given the "
                                                "controls"),
                   float(np.corrcoef(x_res, y_res)[0, 1]))
        res.metric("raw_correlation", ctx.t("labs.fwl.metric.raw_corr",
                                            "Raw correlation of x and y"),
                   float(np.corrcoef(x, y)[0, 1]))
        res.metric("residual_variation", ctx.t("labs.fwl.metric.remaining",
                                               "Share of x variation left after "
                                               "partialling out"),
                   float(np.var(x_res, ddof=1) / max(np.var(x, ddof=1), 1e-12)),
                   note=ctx.t("labs.fwl.metric.remaining_note",
                              "this is all the information the coefficient rests on"))

        res.assume("full_rank", ctx.t("labs.fwl.assume.rank_label",
                                      "The controls do not span x"),
                   float(np.var(x_res, ddof=1)) > 1e-8,
                   detail=ctx.t("labs.fwl.assume.rank",
                                "If nothing is left of x after partialling out, no "
                                "coefficient can be identified from these data."))
        res.assume("good_controls", ctx.t("labs.fwl.assume.controls_label",
                                          "The controls are legitimate (not colliders or "
                                          "mediators)"), not p["bad_control"],
                   detail=ctx.t("labs.fwl.assume.controls",
                                "Partialling out is a statistical operation with no causal "
                                "content. Whether a control belongs in the model is a "
                                "question about the causal structure, not the data."))

        res.animations.append(self._animation(ctx, x, Z, y, n, p))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.regression.fwl.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.regression.fwl.intuition"))
        res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.regression.fwl.math"),
                    kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.fwl.proof",
            "Write X = [X1, X2] and let M2 = I - X2(X2'X2)^-1 X2' annihilate X2.\n"
            "1. The normal equations for y = X1 b1 + X2 b2 + e give X1'e = 0 and X2'e = 0.\n"
            "2. Pre-multiplying y = X1 b1 + X2 b2 + e by M2 kills the second term, since "
            "M2 X2 = 0, and leaves M2 y = M2 X1 b1 + M2 e.\n"
            "3. Because X2'e = 0, the residual is already orthogonal to X2, so M2 e = e.\n"
            "4. Therefore M2 y = (M2 X1) b1 + e with X1' e = 0, which are exactly the "
            "normal equations of regressing M2 y on M2 X1.\n"
            "5. Both the coefficient b1 and the residual vector e are therefore identical "
            "in the two regressions. The read-out above confirms this to machine precision.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.regression.fwl.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.regression.fwl.warning"), kind="warning")

        if p["bad_control"]:
            res.warnings.append(ctx.t(
                "labs.fwl.warn.bad_control",
                "The control in this scenario is a collider caused by both x and y. FWL "
                "still works perfectly - the two coefficients still match exactly - but "
                "both of them are now biased for the causal effect. The theorem is about "
                "algebra; it cannot tell you which variables belong in the model.",
            ))
        if float(np.var(x_res, ddof=1) / max(np.var(x, ddof=1), 1e-12)) < 0.05:
            res.warnings.append(ctx.t(
                "labs.fwl.warn.little_variation",
                "Less than 5% of the variation in x survives the controls. The coefficient "
                "is being identified by a very thin slice of the data, which is exactly "
                "what a large variance inflation factor reports.",
            ))
        return res

    @staticmethod
    def _generate(gen, p, n):
        rho = float(p["corr_xz"])
        k = int(p["n_controls"])
        x = gen.standard_normal(n)
        cols, names = [], []
        for j in range(k):
            base = gen.standard_normal(n)
            weight = rho if j == 0 else rho * 0.4
            cols.append(weight * x + np.sqrt(max(1 - weight**2, 1e-9)) * base)
            names.append(f"z{j + 1}")
        Z = np.column_stack(cols)
        y = (float(p["beta_x"]) * x + float(p["beta_z"]) * Z[:, 0]
             + float(p["noise"]) * gen.standard_normal(n))
        if p["bad_control"]:
            collider = 0.9 * x + 0.9 * y + 0.4 * gen.standard_normal(n)
            Z = np.column_stack([collider, *[Z[:, j] for j in range(1, Z.shape[1])]])
            names = ["collider", *names[1:]]
        return x, Z, y, tuple(names)

    def _added_variable_figure(self, ctx, x_res, y_res, short, full):
        fig = ctx.figure(
            "labs.fwl.figure.added_variable",
            xaxis_title=ctx.t("labs.fwl.axis.x_res",
                              "x with the controls partialled out"),
            yaxis_title=ctx.t("labs.fwl.axis.y_res",
                              "y with the controls partialled out"),
            height=430,
        )
        P.add_points(fig, x_res, y_res, ctx.t("labs.fwl.trace.residual_pairs",
                                              "residual pairs"),
                     "primary", theme=ctx.theme, size=6)
        order = np.argsort(x_res)
        P.add_curve(fig, x_res[order], short.fitted_values[order],
                    ctx.t("labs.fwl.trace.slope",
                          "slope = {b} (identical to the multiple-regression coefficient)",
                          b=fmt(short.coef("x_residual"), 4)),
                    "fitted", theme=ctx.theme, width=3.0)
        P.add_legend_note(fig, ctx.t(
            "labs.fwl.legend_added",
            "This is the added-variable plot. It shows exactly the variation that produces "
            "the multiple-regression coefficient - and if it looks like a vertical blob, "
            "that coefficient rests on almost nothing.",
        ), theme=ctx.theme)
        return fig

    def _residualization_figure(self, ctx, x, Z, y, n):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        Zc = np.column_stack([np.ones(n), Z])
        y_res, x_res = LM.partial_out(y, x, Zc)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.fwl.trace.step1", "step 1: raw x against raw y"),
            ctx.t("labs.fwl.trace.step2", "step 2: residual x against residual y"),
        ))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                 marker={"color": ctx.color("muted"), "size": 5,
                                         "opacity": 0.6}, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_res, y=y_res, mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 5,
                                         "opacity": 0.7}, showlegend=False), row=1, col=2)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=340, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.fwl.legend_residualization",
            "The left panel mixes the effect of x with everything the controls explain. The "
            "right panel has had that common part removed from BOTH variables, so its slope "
            "is the ceteris-paribus effect.",
        ), theme=ctx.theme)
        return fig

    def _comparison_figure(self, ctx, naive, full, short, p):
        labels = [
            ctx.t("labs.fwl.trace.naive", "x alone"),
            ctx.t("labs.fwl.trace.full", "full multiple regression"),
            ctx.t("labs.fwl.trace.fwl", "residual on residual"),
        ]
        values = [naive.coef("x"), full.coef("x"), short.coef("x_residual")]
        errors = [naive.se("x"), full.se("x"), short.se("x_residual")]
        fig = ctx.figure(
            "labs.fwl.figure.comparison",
            xaxis_title=ctx.t("labs.fwl.axis.method", "Method"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=350,
        )
        go = P.require_plotly()
        fig.add_trace(go.Bar(x=labels, y=values,
                             marker={"color": [ctx.color("muted"), ctx.color("primary"),
                                               ctx.color("secondary")]},
                             error_y={"type": "data", "array": [1.96 * e for e in errors]},
                             name=ctx.t("labs.common.axis.coefficient"),
                             text=[fmt(v, 4) for v in values], textposition="outside"))
        P.add_hline(fig, float(p["beta_x"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.fwl.legend_comparison",
            "The last two bars are equal to machine precision - that is the theorem. The "
            "first bar differs because it never removed the controls at all.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, x, Z, y, n, p):
        go = P.require_plotly()
        Zc = np.column_stack([np.ones(n), Z])
        y_res_full, x_res_full = LM.partial_out(y, x, Zc)
        ts = np.linspace(0.0, 1.0, 16)
        full = LM.ols(y, np.column_stack([np.ones(n), x, Z]),
                      names=("const", "x", *[f"z{j}" for j in range(1, Z.shape[1] + 1)]))
        frames, steps = [], []
        for i, t in enumerate(ts):
            xt = (1 - t) * x + t * x_res_full
            yt = (1 - t) * y + t * y_res_full
            slope = float(np.cov(xt, yt, ddof=1)[0, 1] / max(np.var(xt, ddof=1), 1e-12))
            frames.append(go.Frame(name=f"{t:.2f}", data=[go.Scatter(x=xt, y=yt)]))
            steps.append(AnimationStep(
                id=f"t_{i}", frame=i,
                title=ctx.t("labs.fwl.anim.title",
                            "{pct}% partialled out", pct=int(round(100 * t))),
                what_you_see=ctx.t("labs.fwl.anim.see",
                                   "The scatter of x against y, morphing from raw values "
                                   "towards control-adjusted residuals."),
                what_changed=ctx.t("labs.fwl.anim.changed",
                                   "{pct}% of what the controls explain has been removed "
                                   "from both variables.", pct=int(round(100 * t))),
                why=ctx.t("labs.fwl.anim.why",
                          "Partialling out subtracts from x and y whatever the controls can "
                          "predict, leaving only variation the controls cannot account for."),
                interpretation=ctx.t("labs.fwl.anim.interpret",
                                     "The simple slope through this cloud is now {s}; the "
                                     "multiple-regression coefficient is {m}.",
                                     s=fmt(slope, 4), m=fmt(full.coef("x"), 4)),
                conclusion=ctx.t("labs.fwl.anim.conclude",
                                 "At 100% the simple slope IS the multiple-regression "
                                 "coefficient - not approximately, exactly."),
                warning=ctx.t("labs.fwl.anim.warn",
                              "The cloud usually shrinks horizontally as controls are "
                              "removed. That shrinkage is the price of controlling, and it "
                              "is what inflates the standard error."),
                math="b1 from y on [X1, X2]  =  b1 from (M2 y) on (M2 X1)",
                outputs={"progress": round(float(t), 3), "simple_slope": round(slope, 5),
                         "multiple_coefficient": round(full.coef("x"), 5)},
                active_assumptions=("full_rank",),
                violated_assumptions=("good_controls",) if p["bad_control"] else (),
                highlighted=("scatter", "slope"),
            ))
        fig = ctx.figure(
            "labs.fwl.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=400,
        )
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 5,
                                         "opacity": 0.65},
                                 name=ctx.t("labs.common.trace.data")))
        build_frames(fig, frames, duration=340, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.fwl.slider", "partialled out"))
        return animation(
            "partialling_out", fig, steps,
            purpose=ctx.t("labs.fwl.anim.purpose",
                          "Make 'holding the other variables constant' a visible operation."),
            summary=ctx.t(
                "labs.fwl.anim.summary",
                "A multiple-regression coefficient is a simple slope through a cloud you "
                "cannot see in the raw data. Watching that cloud emerge - and shrink - is "
                "the clearest statement of both what controlling buys and what it costs."),
            evidence=EvidenceType.GEOMETRIC_PROOF,
        )


LAB = FWLLab(SPEC)
