"""Multicollinearity: unstable coefficients, stable predictions."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.regression import collinear_design
from ...simulation.monte_carlo import monte_carlo

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "regression.multicollinearity",
    Domain.REGRESSION,
    "diagnostics",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        slider("correlation", 0.9, -0.999, 0.999, 0.001, group="design"),
        int_slider("n", 150, 10, 5000, 1, group="design"),
        slider("beta1", 1.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("beta2", 1.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        int_slider("reps", 800, 100, 10000, 100, group="simulation", expensive=True),
        toggle("show_confidence_region", True, group="views"),
        toggle("show_prediction_stability", True, group="views"),
        toggle("standardize", False, group="transform"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", correlation=0.9, n=150),
        scenario("orthogonal", "null", correlation=0.0),
        scenario("mild", "weak", correlation=0.5),
        scenario("severe", "strong", correlation=0.98),
        scenario("near_perfect", "boundary", correlation=0.999),
        scenario("negative_collinearity", "negative", correlation=-0.95),
        scenario("small_sample", "small_sample", n=20, correlation=0.9),
        scenario("large_sample_rescues", "large_sample", n=4000, correlation=0.95),
        scenario("high_noise", "high_noise", noise=5.0, correlation=0.9),
        scenario("cancelling_effects", "counterexample", beta1=1.0, beta2=-1.0,
                 correlation=0.97),
    ),
    prerequisites=("regression.simple_linear",),
    related=("regression.fwl", "ml.regularization"),
    tags=("vif", "collinearity", "variance inflation", "condition number",
          "coefficient instability"),
    aliases=("multicollinearity", "multicolinearite", "الازدواج الخطي",
             "variance inflation factor"),
    backends=("numpy",),
    references=(
        ref("Belsley, D. A., Kuh, E. and Welsch, R. E. (1980). Regression Diagnostics.",
            kind="book"),
        ref("Wooldridge, J. M. (2019). Introductory Econometrics.", kind="book"),
    ),
    curriculum_tags=("dz.econometrics1", "cairo.eviews"),
)


class MulticollinearityLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        rho = float(p["correlation"])
        n = int(p["n"])
        data = collinear_design(n=n, correlation=rho, beta1=float(p["beta1"]),
                               beta2=float(p["beta2"]), noise=float(p["noise"]),
                               seed=state.seed)
        X = data.matrix("x1", "x2")
        if p["standardize"]:
            X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)
        fit = LM.ols(data["y"], X, names=("const", "x1", "x2"))
        fit.metadata["X"] = X
        vifs = LM.vif(X)
        cond = float(np.linalg.cond(X[:, 1:] - X[:, 1:].mean(axis=0)))

        draws = self._sampling(p, n, state.seed, int(p["reps"]))

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "coefficient_cloud",
            self._cloud_figure(ctx, draws, p, fit),
            "labs.mc.figure.cloud", evidence=EvidenceType.SIMULATION,
        ))
        res.add_panel(ctx.panel(
            "regressor_geometry", self._geometry_figure(ctx, X, rho),
            "labs.mc.figure.geometry", tab="math",
            evidence=EvidenceType.GEOMETRIC_PROOF,
        ))
        res.add_panel(ctx.panel(
            "vif_curve", self._vif_figure(ctx, rho),
            "labs.mc.figure.vif", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_prediction_stability"]:
            res.add_panel(ctx.panel(
                "prediction", self._prediction_figure(ctx, draws, p, state.seed),
                "labs.mc.figure.prediction", tab="compare",
                evidence=EvidenceType.SIMULATION,
            ))

        res.metric("correlation", ctx.t("labs.mc.metric.corr",
                                        "Correlation between the regressors"), rho)
        res.metric("vif1", ctx.t("labs.mc.metric.vif", "Variance inflation factor for x1"),
                   float(vifs[0]))
        res.metric("vif2", ctx.t("labs.mc.metric.vif2",
                                 "Variance inflation factor for x2"), float(vifs[1]))
        res.metric("condition_number", ctx.t("labs.mc.metric.condition",
                                             "Condition number of the centred design"),
                   cond)
        res.metric("beta1", ctx.t("labs.mc.metric.b1", "Estimated coefficient on x1"),
                   fit.coef("x1"), reference=float(p["beta1"]))
        res.metric("beta1_se", ctx.t("labs.mc.metric.b1_se", "Its standard error"),
                   fit.se("x1"))
        res.metric("beta1_p", ctx.t("labs.mc.metric.b1_p", "Its p-value"),
                   float(fit.pvalues[1]))
        res.metric("sum_p", ctx.t("labs.mc.metric.sum_p",
                                  "p-value for beta1 + beta2 = {v}",
                                  v=fmt(float(p["beta1"]) + float(p["beta2"]), 2)),
                   float(LM.f_test(fit, np.array([[0.0, 1.0, 1.0]]),
                                   np.array([float(p["beta1"]) + float(p["beta2"])])).p_value),
                   note=ctx.t("labs.mc.metric.sum_note",
                              "the SUM is estimated precisely even when each part is not"))
        res.metric("joint_f_p", ctx.t("labs.mc.metric.joint",
                                      "Joint F test that both coefficients are zero"),
                   float(LM.f_test(fit, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])).p_value))
        res.metric("simulated_sd_b1", ctx.t("labs.mc.metric.sim_sd",
                                            "Simulated sd of the x1 coefficient"),
                   float(np.std(draws["b1"], ddof=1)), reference=fit.se("x1"))
        res.metric("simulated_sd_sum", ctx.t("labs.mc.metric.sim_sum",
                                             "Simulated sd of the SUM of the coefficients"),
                   float(np.std(draws["b1"] + draws["b2"], ddof=1)))
        res.metric("simulated_sd_prediction",
                   ctx.t("labs.mc.metric.sim_pred",
                         "Simulated sd of a prediction at the sample centre"),
                   float(np.std(draws["prediction"], ddof=1)),
                   note=ctx.t("labs.mc.metric.sim_pred_note",
                              "barely affected by collinearity"))

        severe = float(np.max(vifs)) > 10.0
        res.assume("no_perfect_collinearity", ctx.t("assumptions.no_perfect_collinearity"),
                   abs(rho) < 0.9999,
                   detail=ctx.t("labs.mc.assume.perfect",
                                "At an exact correlation of one the design matrix loses rank "
                                "and the coefficients are not defined at all - which is a "
                                "different problem from a large but finite VIF."))
        res.assume("exogeneity", ctx.t("assumptions.exogeneity"), True,
                   detail=ctx.t("labs.mc.assume.unbiased",
                                "Collinearity does not touch this assumption. The estimates "
                                "remain unbiased and consistent throughout."))

        res.animations.append(self._animation(ctx, p, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.regression.multicollinearity.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.regression.multicollinearity.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.regression.multicollinearity.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.regression.multicollinearity.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.regression.multicollinearity.warning"), kind="warning")
        res.explain("diagnostics", ctx.t("labs.mc.verdict.title", "What to conclude"),
                    ctx.t(
            "labs.mc.verdict",
            "Individual coefficients have a simulated standard deviation of {sd1}, but their "
            "SUM has only {sds} and a prediction at the sample centre only {sdp}. The data "
            "identify the combination well and the parts badly - which is exactly what "
            "collinearity means, and why dropping a variable to 'fix' it usually makes "
            "things worse.",
            sd1=fmt(float(np.std(draws['b1'], ddof=1)), 4),
            sds=fmt(float(np.std(draws['b1'] + draws['b2'], ddof=1)), 4),
            sdp=fmt(float(np.std(draws['prediction'], ddof=1)), 4),
        ))
        if severe:
            res.warnings.append(ctx.t(
                "labs.mc.warn.severe",
                "The largest variance inflation factor is {v}: standard errors are about {s} "
                "times larger than they would be with orthogonal regressors. Nothing is "
                "biased - the data simply cannot separate these two variables.",
                v=fmt(float(np.max(vifs)), 1), s=fmt(float(np.sqrt(np.max(vifs))), 1),
            ))
        return res

    def _sampling(self, p, n, seed, reps):
        reps = min(reps, 4000)
        rho = float(p["correlation"])

        def experiment(gen):
            child = int(gen.integers(0, 2**31 - 1))
            d = collinear_design(n=n, correlation=rho, beta1=float(p["beta1"]),
                                 beta2=float(p["beta2"]), noise=float(p["noise"]),
                                 seed=child)
            f = LM.ols(d["y"], d.matrix("x1", "x2"), names=("const", "x1", "x2"))
            return {"b1": f.coef("x1"), "b2": f.coef("x2"),
                    "prediction": float(f.coefficients @ np.array([1.0, 0.0, 0.0])),
                    "r2": f.r_squared}

        return monte_carlo(experiment, reps, seed).draws

    def _cloud_figure(self, ctx, draws, p, fit):
        fig = ctx.figure(
            "labs.mc.figure.cloud",
            xaxis_title=ctx.t("labs.mc.axis.b1", "coefficient on x1"),
            yaxis_title=ctx.t("labs.mc.axis.b2", "coefficient on x2"),
            height=440,
        )
        P.add_points(fig, draws["b1"], draws["b2"],
                     ctx.t("labs.mc.trace.repeated", "estimates across repeated samples"),
                     "primary", theme=ctx.theme, size=4, opacity=0.4)
        P.add_points(fig, [fit.coef("x1")], [fit.coef("x2")],
                     ctx.t("labs.mc.trace.this", "this sample"), "estimate",
                     theme=ctx.theme, size=12)
        P.add_points(fig, [float(p["beta1"])], [float(p["beta2"])],
                     ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme, size=13,
                     symbol="star")
        total = float(p["beta1"]) + float(p["beta2"])
        lo = float(np.percentile(draws["b1"], 0.5))
        hi = float(np.percentile(draws["b1"], 99.5))
        P.add_curve(fig, [lo, hi], [total - lo, total - hi],
                    ctx.t("labs.mc.trace.ridge", "line where beta1 + beta2 is correct"),
                    "warning", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.mc.legend_cloud",
            "The cloud stretches along the dashed line: samples trade coefficient between "
            "the two variables while keeping their sum right. The estimates are unbiased - "
            "the cloud is centred on the star - just very elongated.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _geometry_figure(self, ctx, X, rho):
        x1 = X[:, 1] - X[:, 1].mean()
        x2 = X[:, 2] - X[:, 2].mean()
        x1 = x1 / max(np.linalg.norm(x1), 1e-12)
        x2 = x2 / max(np.linalg.norm(x2), 1e-12)
        angle = float(np.degrees(np.arccos(np.clip(x1 @ x2, -1, 1))))
        fig = ctx.figure(
            "labs.mc.figure.geometry",
            xaxis_title=ctx.t("labs.mc.axis.dir1", "direction 1"),
            yaxis_title=ctx.t("labs.mc.axis.dir2", "direction 2"),
            height=380,
        )
        theta = np.radians(angle)
        P.add_arrow(fig, 0, 0, 1.0, 0.0, "x1", "primary", theme=ctx.theme)
        P.add_arrow(fig, 0, 0, float(np.cos(theta)), float(np.sin(theta)), "x2",
                    "secondary", theme=ctx.theme)
        arc = np.linspace(0, theta, 60)
        P.add_curve(fig, 0.3 * np.cos(arc), 0.3 * np.sin(arc),
                    ctx.t("labs.mc.trace.angle", "angle = {a} degrees", a=fmt(angle, 1)),
                    "warning", theme=ctx.theme, dash="dot")
        fig.update_xaxes(range=[-0.3, 1.25], scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[-0.3, 1.25])
        P.add_legend_note(fig, ctx.t(
            "labs.mc.legend_geometry",
            "The angle between the regressor directions is the whole story. At 90 degrees "
            "each variable carries its own information; as the arrows close, the data lose "
            "the ability to tell them apart and the variance inflation factor is exactly "
            "1/sin^2 of this angle.",
        ), theme=ctx.theme)
        return fig

    def _vif_figure(self, ctx, rho):
        grid = np.linspace(0.0, 0.995, 200)
        vif = 1.0 / (1.0 - grid**2)
        fig = ctx.figure(
            "labs.mc.figure.vif",
            xaxis_title=ctx.t("labs.mc.axis.corr", "|correlation| between the regressors"),
            yaxis_title=ctx.t("labs.mc.axis.vif", "Variance inflation factor"),
            height=340,
        )
        P.add_curve(fig, grid, vif, "VIF = 1 / (1 - r^2)", "primary", theme=ctx.theme)
        P.add_hline(fig, 10.0, ctx.t("labs.mc.trace.rule", "the usual VIF > 10 convention"),
                    "warning", theme=ctx.theme, dash="dash")
        P.add_points(fig, [abs(rho)], [1.0 / max(1 - rho**2, 1e-9)],
                     ctx.t("labs.common.trace.current"), "highlight", theme=ctx.theme,
                     size=12)
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.mc.legend_vif",
            "The standard error is multiplied by the square root of this factor. The "
            "'VIF > 10' rule is a convention, not a theorem - what matters is whether the "
            "resulting interval is still useful for your question.",
        ), theme=ctx.theme)
        return fig

    def _prediction_figure(self, ctx, draws, p, seed):
        rhos = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.99])
        sd_b1, sd_sum, sd_pred = [], [], []
        for r in rhos:
            q = dict(p)
            q["correlation"] = float(r)
            d = self._sampling(q, int(p["n"]), seed, 500)
            sd_b1.append(float(np.std(d["b1"], ddof=1)))
            sd_sum.append(float(np.std(d["b1"] + d["b2"], ddof=1)))
            sd_pred.append(float(np.std(d["prediction"], ddof=1)))
        fig = ctx.figure(
            "labs.mc.figure.prediction",
            xaxis_title=ctx.t("labs.mc.axis.corr", "|correlation| between the regressors"),
            yaxis_title=ctx.t("labs.mc.axis.sd", "Standard deviation across samples"),
            height=360,
        )
        P.add_curve(fig, rhos, sd_b1,
                    ctx.t("labs.mc.trace.sd_b1", "individual coefficient"),
                    "negative", theme=ctx.theme)
        P.add_curve(fig, rhos, sd_sum,
                    ctx.t("labs.mc.trace.sd_sum", "sum of the coefficients"),
                    "warning", theme=ctx.theme)
        P.add_curve(fig, rhos, sd_pred,
                    ctx.t("labs.mc.trace.sd_pred", "prediction at the sample centre"),
                    "positive", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.mc.legend_prediction",
            "Collinearity destroys the precision of individual coefficients while leaving "
            "predictions almost untouched. If you only need forecasts, it is not your "
            "problem; if you need to attribute effects, it is the whole problem.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, seed):
        go = P.require_plotly()
        rhos = np.concatenate([np.linspace(0.0, 0.9, 12), np.linspace(0.92, 0.999, 8)])
        frames, steps = [], []
        for i, r in enumerate(rhos):
            q = dict(p)
            q["correlation"] = float(r)
            d = self._sampling(q, int(p["n"]), seed, 400)
            frames.append(go.Frame(name=f"{r:.3f}",
                                   data=[go.Scatter(x=d["b1"], y=d["b2"])]))
            vif = 1.0 / max(1 - r**2, 1e-9)
            steps.append(AnimationStep(
                id=f"rho_{i}", frame=i,
                title=ctx.t("labs.mc.anim.title", "correlation = {r}", r=fmt(r, 3)),
                what_you_see=ctx.t("labs.mc.anim.see",
                                   "The pair of estimated coefficients from each of many "
                                   "repeated samples."),
                what_changed=ctx.t("labs.mc.anim.changed",
                                   "The correlation between the two regressors moved to {r}.",
                                   r=fmt(r, 3)),
                why=ctx.t("labs.mc.anim.why",
                          "As the regressor directions align, many coefficient pairs fit the "
                          "data almost equally well, so sampling noise slides the estimate "
                          "along that ridge."),
                interpretation=ctx.t("labs.mc.anim.interpret",
                                     "The cloud's sd on the x1 axis is {s}; the variance "
                                     "inflation factor is {v}.",
                                     s=fmt(float(np.std(d["b1"], ddof=1)), 4),
                                     v=fmt(vif, 1)),
                conclusion=ctx.t("labs.mc.anim.conclude",
                                 "The cloud stretches but stays centred on the truth: the "
                                 "estimator is still unbiased, just imprecise."),
                warning=ctx.t("labs.mc.anim.warn",
                              "Nothing here is a bias. Reporting 'collinearity biased my "
                              "results' confuses precision with validity."),
                math="Var(b_j) = sigma^2 / (SST_j (1 - R_j^2)),  VIF_j = 1/(1 - R_j^2)",
                outputs={"correlation": round(float(r), 4), "vif": round(vif, 2),
                         "sd_b1": round(float(np.std(d["b1"], ddof=1)), 5)},
                active_assumptions=("exogeneity",),
                highlighted=("coefficient_cloud",),
            ))
        fig = ctx.figure(
            "labs.mc.figure.animation",
            xaxis_title=ctx.t("labs.mc.axis.b1", "coefficient on x1"),
            yaxis_title=ctx.t("labs.mc.axis.b2", "coefficient on x2"),
            height=400,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 4,
                                         "opacity": 0.45},
                                 name=ctx.t("labs.mc.trace.repeated",
                                            "estimates across repeated samples")))
        P.add_points(fig, [float(p["beta1"])], [float(p["beta2"])],
                     ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme, size=13,
                     symbol="star")
        span = 6.0
        fig.update_xaxes(range=[float(p["beta1"]) - span, float(p["beta1"]) + span])
        fig.update_yaxes(range=[float(p["beta2"]) - span, float(p["beta2"]) + span])
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.mc.slider", "correlation"))
        return animation(
            "collinearity_sweep", fig, steps,
            purpose=ctx.t("labs.mc.anim.purpose",
                          "Separate 'imprecise' from 'wrong'."),
            summary=ctx.t(
                "labs.mc.anim.summary",
                "The cloud never leaves the truth; it only stretches. Collinearity is a "
                "shortage of information about how to split an effect between two variables, "
                "not a defect in the estimator. More data or a better-designed sample "
                "helps; deleting a variable does not."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = MulticollinearityLab(SPEC)
