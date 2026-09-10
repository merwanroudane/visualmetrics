"""Restricted regression, dummy variables and structural change."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...backends import linear as LM
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "RESTRICTIONS"]

RESTRICTIONS = ("beta1_equals_value", "beta1_plus_beta2", "both_zero",
                "constant_returns", "no_structural_break")


SPEC = make_spec(
    "regression.restricted",
    Domain.REGRESSION,
    "restricted_regression",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "experiment", "compare", "derive", "code", "quiz",
           "references"),
    evidence=EvidenceType.SYMBOLIC_DERIVATION,
    controls=(
        select("restriction", "beta1_plus_beta2", RESTRICTIONS, group="restriction"),
        slider("restriction_value", 1.0, -5.0, 5.0, 0.05, group="restriction"),
        slider("beta1", 0.6, -5.0, 5.0, 0.05, group="dgp"),
        slider("beta2", 0.4, -5.0, 5.0, 0.05, group="dgp"),
        int_slider("n", 200, 10, 5000, 1, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        slider("corr_x", 0.4, -0.95, 0.95, 0.01, group="dgp"),
        slider("break_shift", 0.0, -5.0, 5.0, 0.05, group="structural"),
        slider("break_slope", 0.0, -3.0, 3.0, 0.05, group="structural"),
        slider("break_fraction", 0.5, 0.1, 0.9, 0.01, group="structural"),
        slider("alpha", 0.05, 0.001, 0.20, 0.001, group="inference"),
        toggle("show_surface", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", restriction="beta1_plus_beta2",
                 restriction_value=1.0, beta1=0.6, beta2=0.4),
        scenario("restriction_true", "null", restriction="beta1_plus_beta2",
                 restriction_value=1.0, beta1=0.6, beta2=0.4, n=400),
        scenario("restriction_false", "violation", restriction="beta1_plus_beta2",
                 restriction_value=1.0, beta1=1.2, beta2=0.9, n=400),
        scenario("constant_returns", "compare_methods", restriction="constant_returns",
                 beta1=0.7, beta2=0.3),
        scenario("both_zero", "compare_methods", restriction="both_zero",
                 beta1=0.05, beta2=0.05, n=100),
        scenario("single_coefficient", "compare_methods",
                 restriction="beta1_equals_value", restriction_value=0.6),
        scenario("structural_break", "violation", restriction="no_structural_break",
                 break_shift=3.0, break_slope=1.0, n=300),
        scenario("no_break", "null", restriction="no_structural_break",
                 break_shift=0.0, break_slope=0.0, n=300),
        scenario("small_sample", "small_sample", n=15),
        scenario("bias_variance_tradeoff", "sensitivity", restriction="beta1_equals_value",
                 restriction_value=0.0, beta1=0.25, n=60),
    ),
    prerequisites=("regression.simple_linear",),
    related=("regression.multicollinearity", "econometrics.heteroskedasticity"),
    tags=("restrictions", "f test", "wald", "dummy variables", "chow test",
          "structural change"),
    aliases=("restricted least squares", "regression contrainte", "الانحدار المقيد",
             "chow test", "linear restriction"),
    backends=("numpy",),
    proof_ids=("regression.restricted.f_statistic",),
    references=(
        ref("Chow, G. C. (1960). Tests of equality between sets of coefficients in two "
            "linear regressions. Econometrica 28(3).", kind="paper", doi="10.2307/1910133"),
        ref("Greene, W. H. (2018). Econometric Analysis.", kind="book"),
    ),
    curriculum_tags=("dz.econometrics1",),
)


class RestrictedLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        gen = rng(state.seed, "restricted")
        y, X, names, break_info = self._generate(gen, p, n)
        R, r, description = self._restriction(p, names)

        unrestricted = LM.ols(y, X, names=names)
        restricted, test = LM.restricted_ols(y, X, R, r, names=names)
        wald = LM.wald_test(unrestricted, R, r)
        alpha = float(p["alpha"])

        res.dgp = ctx.t(
            "labs.restr.dgp",
            "y = {b1} x1 + {b2} x2 + noise with corr(x1, x2) = {c}, n = {n}. "
            "Restriction imposed: {desc}.",
            b1=fmt(p["beta1"], 2), b2=fmt(p["beta2"], 2), c=fmt(p["corr_x"], 2), n=n,
            desc=description,
        )

        if p["show_surface"] and X.shape[1] >= 3:
            res.add_panel(ctx.panel(
                "surface", self._surface_figure(ctx, y, X, unrestricted, restricted, R, r),
                "labs.restr.figure.surface", evidence=EvidenceType.GEOMETRIC_PROOF,
            ))
        res.add_panel(ctx.panel(
            "coefficients", self._coefficient_figure(ctx, unrestricted, restricted, p),
            "labs.restr.figure.coefficients", tab="compare",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if str(p["restriction"]) == "no_structural_break":
            res.add_panel(ctx.panel(
                "break", self._break_figure(ctx, y, X, unrestricted, restricted,
                                            break_info),
                "labs.restr.figure.break", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("f_statistic", ctx.t("labs.restr.metric.f", "F statistic"),
                   test.statistic)
        res.metric("f_p", ctx.t("labs.restr.metric.f_p", "F test p-value"), test.p_value)
        res.metric("wald_chi2", ctx.t("labs.restr.metric.wald",
                                      "Wald chi-square statistic"), wald.statistic)
        res.metric("wald_p", ctx.t("labs.restr.metric.wald_p", "Wald p-value"),
                   wald.p_value,
                   note=ctx.t("labs.restr.metric.wald_note",
                              "the F and Wald tests agree asymptotically; they differ in "
                              "finite samples"))
        res.metric("decision", ctx.t("labs.restr.metric.decision",
                                     "Decision at alpha = {a}", a=fmt(alpha, 3)),
                   ctx.t("labs.restr.reject", "reject the restriction")
                   if test.p_value < alpha
                   else ctx.t("labs.restr.fail", "do not reject the restriction"))
        res.metric("ssr_unrestricted", ctx.t("labs.restr.metric.ssr_u",
                                             "SSR without the restriction"),
                   unrestricted.ssr)
        res.metric("ssr_restricted", ctx.t("labs.restr.metric.ssr_r",
                                           "SSR with the restriction"), restricted.ssr)
        res.metric("fit_loss", ctx.t("labs.restr.metric.loss",
                                     "Loss of fit caused by the restriction"),
                   restricted.ssr - unrestricted.ssr,
                   note=ctx.t("labs.restr.metric.loss_note",
                              "never negative - a constraint can only hurt the fit"))
        res.metric("r2_unrestricted", ctx.t("labs.restr.metric.r2_u",
                                            "R-squared without the restriction"),
                   unrestricted.r_squared)
        res.metric("r2_restricted", ctx.t("labs.restr.metric.r2_r",
                                          "R-squared with the restriction"),
                   restricted.r_squared)
        for i, name in enumerate(names):
            if name == "const":
                continue
            res.metric(f"se_ratio_{name}",
                       ctx.t("labs.restr.metric.se_ratio",
                             "Standard error ratio for {n} (restricted / unrestricted)",
                             n=name),
                       float(restricted.standard_errors[i] /
                             max(unrestricted.standard_errors[i], 1e-12)),
                       note=ctx.t("labs.restr.metric.se_ratio_note",
                                  "below 1 means the restriction bought precision"))

        res.assume("restriction_true", ctx.t("labs.restr.assume.true_label",
                                             "The imposed restriction actually holds"),
                   test.p_value >= alpha,
                   detail=ctx.t("labs.restr.assume.true",
                                "A restriction that is true reduces variance for free. A "
                                "restriction that is false introduces bias that no sample "
                                "size will remove."),
                   consequence="" if test.p_value >= alpha else ctx.t(
                       "labs.restr.assume.false",
                       "The data reject this restriction, so the restricted estimates are "
                       "biased even though they look more precise."))
        res.assume("normal_errors", ctx.t("assumptions.normal_errors"), True,
                   detail=ctx.t("labs.restr.assume.normal",
                                "The exact F distribution needs normal errors; otherwise "
                                "the test is asymptotic."))

        res.animations.append(self._animation(ctx, y, X, names, p))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.regression.restricted.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.regression.restricted.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.regression.restricted.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.regression.restricted.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.regression.restricted.warning"), kind="warning")
        if str(p["restriction"]) == "no_structural_break":
            res.explain("diagnostics", ctx.t("labs.restr.chow.title",
                                             "This is the Chow test"), ctx.t(
                "labs.restr.chow.body",
                "Testing 'no structural break' means restricting every break dummy to zero. "
                "The F statistic printed above is exactly the Chow statistic, which is why "
                "the Chow test needs no separate machinery - it is one instance of the "
                "general linear-restriction test.",
            ))
        return res

    @staticmethod
    def _generate(gen, p, n):
        rho = float(p["corr_x"])
        x1 = gen.standard_normal(n)
        x2 = rho * x1 + np.sqrt(max(1 - rho**2, 1e-9)) * gen.standard_normal(n)
        y = (1.0 + float(p["beta1"]) * x1 + float(p["beta2"]) * x2
             + float(p["noise"]) * gen.standard_normal(n))
        break_info = None
        if str(p["restriction"]) == "no_structural_break":
            cut = int(float(p["break_fraction"]) * n)
            d = (np.arange(n) >= cut).astype(float)
            y = y + float(p["break_shift"]) * d + float(p["break_slope"]) * d * x1
            X = np.column_stack([np.ones(n), x1, x2, d, d * x1])
            names = ("const", "x1", "x2", "break", "break_x1")
            break_info = {"cut": cut, "d": d, "x1": x1}
            return y, X, names, break_info
        X = np.column_stack([np.ones(n), x1, x2])
        return y, X, ("const", "x1", "x2"), break_info

    @staticmethod
    def _restriction(p, names):
        k = len(names)
        kind = str(p["restriction"])
        value = float(p["restriction_value"])
        if kind == "beta1_equals_value":
            R = np.zeros((1, k))
            R[0, 1] = 1.0
            return R, np.array([value]), f"beta1 = {value:g}"
        if kind == "beta1_plus_beta2":
            R = np.zeros((1, k))
            R[0, 1] = R[0, 2] = 1.0
            return R, np.array([value]), f"beta1 + beta2 = {value:g}"
        if kind == "both_zero":
            R = np.zeros((2, k))
            R[0, 1] = 1.0
            R[1, 2] = 1.0
            return R, np.zeros(2), "beta1 = beta2 = 0"
        if kind == "constant_returns":
            R = np.zeros((1, k))
            R[0, 1] = R[0, 2] = 1.0
            return R, np.array([1.0]), "beta1 + beta2 = 1 (constant returns to scale)"
        R = np.zeros((2, k))
        R[0, 3] = 1.0
        R[1, 4] = 1.0
        return R, np.zeros(2), "no structural break (both break terms are zero)"

    def _surface_figure(self, ctx, y, X, unrestricted, restricted, R, r):
        go = P.require_plotly()
        b = unrestricted.coefficients
        se = unrestricted.standard_errors
        g1 = np.linspace(b[1] - 4 * se[1], b[1] + 4 * se[1], 60)
        g2 = np.linspace(b[2] - 4 * se[2], b[2] + 4 * se[2], 60)
        B1, B2 = np.meshgrid(g1, g2)
        base = X[:, 3:] @ b[3:] if X.shape[1] > 3 else 0.0
        resid = (y[:, None, None] - b[0]
                 - B1[None, :, :] * X[:, 1][:, None, None]
                 - B2[None, :, :] * X[:, 2][:, None, None]
                 - (base[:, None, None] if np.ndim(base) else 0.0))
        sse = np.sum(resid**2, axis=0)
        fig = ctx.figure(
            "labs.restr.figure.surface",
            xaxis_title="beta1", yaxis_title="beta2", height=440,
        )
        fig.add_trace(go.Contour(x=g1, y=g2, z=sse, colorscale=ctx.theme.colorscale,
                                 contours={"showlabels": True},
                                 name=ctx.t("labs.restr.trace.sse",
                                            "sum of squared residuals")))
        if R.shape[0] == 1 and abs(R[0, 2]) > 1e-12:
            line_b2 = (r[0] - R[0, 1] * g1) / R[0, 2]
            P.add_curve(fig, g1, line_b2,
                        ctx.t("labs.restr.trace.line", "the restriction"),
                        "warning", theme=ctx.theme, dash="dash", width=3.0)
        elif R.shape[0] == 1 and abs(R[0, 1]) > 1e-12:
            P.add_vline(fig, r[0] / R[0, 1],
                        ctx.t("labs.restr.trace.line", "the restriction"),
                        "warning", theme=ctx.theme)
        P.add_points(fig, [b[1]], [b[2]],
                     ctx.t("labs.restr.trace.unrestricted", "unrestricted optimum"),
                     "estimate", theme=ctx.theme, size=12)
        P.add_points(fig, [restricted.coefficients[1]], [restricted.coefficients[2]],
                     ctx.t("labs.restr.trace.restricted", "constrained optimum"),
                     "warning", theme=ctx.theme, size=12, symbol="square")
        P.add_legend_note(fig, ctx.t(
            "labs.restr.legend_surface",
            "The unrestricted estimate sits at the bottom of the bowl; the restricted one is "
            "the lowest point ON the dashed constraint. The vertical gap between them is "
            "the loss of fit that the F test evaluates.",
        ), theme=ctx.theme)
        return fig

    def _coefficient_figure(self, ctx, unrestricted, restricted, p):
        go = P.require_plotly()
        names = [n for n in unrestricted.names if n != "const"]
        idx = [unrestricted.names.index(n) for n in names]
        fig = ctx.figure(
            "labs.restr.figure.coefficients",
            xaxis_title=ctx.t("labs.restr.axis.coefficient", "Coefficient"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=360,
        )
        for label, fit, role in (
            (ctx.t("labs.restr.trace.unrestricted", "unrestricted"), unrestricted,
             "primary"),
            (ctx.t("labs.restr.trace.restricted", "restricted"), restricted, "warning"),
        ):
            fig.add_trace(go.Bar(
                x=names, y=[fit.coefficients[i] for i in idx], name=label,
                marker={"color": ctx.color(role)},
                error_y={"type": "data",
                         "array": [1.96 * fit.standard_errors[i] for i in idx]},
            ))
        P.add_legend_note(fig, ctx.t(
            "labs.restr.legend_coefficients",
            "Restricted estimates usually have shorter error bars. That extra precision is "
            "real only if the restriction is true - otherwise it is confidence in a wrong "
            "number.",
        ), theme=ctx.theme)
        return fig

    def _break_figure(self, ctx, y, X, unrestricted, restricted, break_info):
        cut = break_info["cut"]
        t = np.arange(y.size)
        fig = ctx.figure(
            "labs.restr.figure.break",
            xaxis_title=ctx.t("labs.common.axis.observations"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=360,
        )
        P.add_points(fig, t, y, ctx.t("labs.common.trace.data"), "primary",
                     theme=ctx.theme, size=5, opacity=0.6)
        P.add_curve(fig, t, unrestricted.fitted_values,
                    ctx.t("labs.restr.trace.with_break", "fit allowing a break"),
                    "fitted", theme=ctx.theme, mode="markers", showlegend=True)
        P.add_curve(fig, t, restricted.fitted_values,
                    ctx.t("labs.restr.trace.without_break", "fit forbidding a break"),
                    "warning", theme=ctx.theme, mode="markers", showlegend=True)
        P.add_vline(fig, cut, ctx.t("labs.restr.trace.cut", "assumed break date"),
                    "negative", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.restr.legend_break",
            "The break date is assumed known here. If it is estimated from the data, the "
            "usual F critical values are too small and the test rejects far too often.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, y, X, names, p):
        go = P.require_plotly()
        values = np.linspace(float(p["restriction_value"]) - 2.0,
                             float(p["restriction_value"]) + 2.0, 20)
        unrestricted = LM.ols(y, X, names=names)
        frames, steps = [], []
        for i, v in enumerate(values):
            q = dict(p)
            q["restriction_value"] = float(v)
            R, r, desc = self._restriction(q, names)
            restr, test = LM.restricted_ols(y, X, R, r, names=names)
            frames.append(go.Frame(name=f"{v:.2f}", data=[
                go.Bar(x=[n for n in names if n != "const"],
                       y=[restr.coefficients[names.index(n)]
                          for n in names if n != "const"]),
            ]))
            steps.append(AnimationStep(
                id=f"r_{i}", frame=i,
                title=ctx.t("labs.restr.anim.title",
                            "restriction value = {v}", v=fmt(v, 2)),
                what_you_see=ctx.t("labs.restr.anim.see",
                                   "Coefficients re-estimated under a restriction whose "
                                   "right-hand side is being swept."),
                what_changed=ctx.t("labs.restr.anim.changed",
                                   "The imposed value moved to {v}.", v=fmt(v, 2)),
                why=ctx.t("labs.restr.anim.why",
                          "The constrained optimum slides along the restriction surface as "
                          "that surface itself moves through the coefficient space."),
                interpretation=ctx.t("labs.restr.anim.interpret",
                                     "Loss of fit is {l}; the F test gives p = {p}.",
                                     l=fmt(restr.ssr - unrestricted.ssr, 3),
                                     p=fmt(test.p_value, 4)),
                conclusion=ctx.t("labs.restr.anim.conclude",
                                 "The set of restriction values the test does NOT reject is "
                                 "exactly a confidence interval for the restricted quantity."),
                warning=ctx.t("labs.restr.anim.warn",
                              "Sweeping the restriction after seeing the data and keeping "
                              "the value that is not rejected is not a test."),
                math="F = [(SSR_R - SSR_U)/q] / [SSR_U/(n - k)]",
                outputs={"restriction_value": round(float(v), 4),
                         "fit_loss": round(restr.ssr - unrestricted.ssr, 4),
                         "p_value": round(float(test.p_value), 5)},
                active_assumptions=("normal_errors",),
                highlighted=("coefficients",),
            ))
        fig = ctx.figure(
            "labs.restr.figure.animation",
            xaxis_title=ctx.t("labs.restr.axis.coefficient", "Coefficient"),
            yaxis_title=ctx.t("labs.common.axis.estimate"),
            height=380,
        )
        fig.add_trace(go.Bar(x=[n for n in names if n != "const"],
                             y=[0.0 for n in names if n != "const"],
                             marker={"color": ctx.color("warning")},
                             name=ctx.t("labs.restr.trace.restricted", "restricted")))
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.restr.slider", "restriction value"))
        return animation(
            "restriction_sweep", fig, steps,
            purpose=ctx.t("labs.restr.anim.purpose",
                          "Show a restriction as a place you force the estimate to stand."),
            summary=ctx.t(
                "labs.restr.anim.summary",
                "Every restriction costs fit and buys precision. The F test asks whether the "
                "cost is larger than sampling noise alone would produce - and the set of "
                "restrictions it fails to reject is precisely a confidence region."),
            evidence=EvidenceType.SYMBOLIC_DERIVATION,
        )


LAB = RestrictedLab(SPEC)
