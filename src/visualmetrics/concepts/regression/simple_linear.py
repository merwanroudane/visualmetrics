"""Simple regression scenario explorer - the widest scenario space in the library."""

from __future__ import annotations

from typing import Any

from scipy import stats

from ...backends import linear as LM
from ...data.generators.regression import (
    ERROR_DISTRIBUTIONS,
    FUNCTIONAL_FORMS,
    HETERO_TYPES,
    X_DISTRIBUTIONS,
    simple_regression,
)
from ...simulation.monte_carlo import monte_carlo
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

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "regression.simple_linear",
    Domain.REGRESSION,
    "simple_regression",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "counterexample", "code", "quiz", "data", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        slider("beta1", 1.5, -5.0, 5.0, 0.05, group="dgp"),
        slider("beta0", 2.0, -20.0, 20.0, 0.1, group="dgp"),
        int_slider("n", 120, 5, 5000, 1, group="dgp"),
        slider("noise", 1.0, 0.0, 20.0, 0.05, group="dgp"),
        select("functional_form", "linear", FUNCTIONAL_FORMS, group="form"),
        select("x_distribution", "uniform", X_DISTRIBUTIONS, group="dgp"),
        slider("x_low", 0.0, -20.0, 20.0, 0.5, group="dgp", advanced=True),
        slider("x_high", 10.0, -10.0, 60.0, 0.5, group="dgp", advanced=True),
        select("error_distribution", "normal", ERROR_DISTRIBUTIONS, group="errors"),
        select("heteroskedasticity", "none", HETERO_TYPES, group="errors"),
        slider("hetero_strength", 0.0, 0.0, 1.0, 0.01, group="errors"),
        slider("autocorrelation", 0.0, -0.95, 0.95, 0.01, group="errors", advanced=True),
        int_slider("n_outliers", 0, 0, 20, 1, group="contamination"),
        slider("outlier_magnitude", 6.0, 1.0, 30.0, 0.5, group="contamination"),
        toggle("leverage_point", False, group="contamination"),
        slider("leverage_x", 20.0, -30.0, 60.0, 0.5, group="contamination", advanced=True),
        slider("leverage_y_shift", -8.0, -40.0, 40.0, 0.5, group="contamination",
               advanced=True),
        slider("measurement_error", 0.0, 0.0, 5.0, 0.05, group="violations"),
        slider("omitted_strength", 0.0, 0.0, 5.0, 0.05, group="violations"),
        slider("omitted_x_correlation", 0.0, -0.95, 0.95, 0.01, group="violations"),
        slider("endogeneity", 0.0, -0.9, 0.9, 0.01, group="violations"),
        slider("confidence_level", 0.95, 0.5, 0.999, 0.005, group="inference"),
        select("cov_type", "nonrobust", ("nonrobust", "HC1", "HC3", "HAC"),
               group="inference"),
        toggle("standardize", False, group="transform"),
        toggle("center", False, group="transform"),
        toggle("show_bands", True, group="views"),
        toggle("show_residuals", True, group="views"),
        toggle("show_sampling", True, group="views"),
        toggle("show_sse", False, group="views"),
        int_slider("reps", 1200, 100, 20000, 100, group="simulation", expensive=True),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", beta1=1.5, beta0=2.0, n=120, noise=1.0),
        scenario("clean_positive", "positive", beta1=2.0, noise=0.5, n=150),
        scenario("clean_negative", "negative", beta1=-1.8, noise=0.5, n=150),
        scenario("no_relationship", "null", beta1=0.0, noise=2.0, n=150),
        scenario("near_zero_slope", "weak", beta1=0.08, noise=2.0, n=60),
        scenario("high_noise", "high_noise", noise=6.0, n=120),
        scenario("low_noise", "low_noise", noise=0.2, n=120),
        scenario("small_sample", "small_sample", n=10),
        scenario("large_sample", "large_sample", n=2000),
        scenario("narrow_x_range", "boundary", x_low=4.0, x_high=6.0, n=120),
        scenario("quadratic_truth_linear_fit", "misspecification",
                 functional_form="quadratic", n=200, noise=1.0),
        scenario("log_log_elasticity", "compare_methods", functional_form="log_log",
                 beta1=0.7, noise=0.15, x_low=1.0, x_high=10.0),
        scenario("level_log", "compare_methods", functional_form="level_log",
                 x_low=1.0, x_high=20.0),
        scenario("threshold_effect", "misspecification", functional_form="threshold"),
        scenario("inverted_u", "misspecification", functional_form="inverted_u", n=200),
        scenario("heteroskedastic_fan", "violation", heteroskedasticity="increasing",
                 hetero_strength=0.9, n=200),
        scenario("non_normal_errors", "violation", error_distribution="student_t", n=100),
        scenario("skewed_errors", "violation", error_distribution="skewed", n=150),
        scenario("serially_correlated", "violation", autocorrelation=0.8, n=150),
        scenario("outlier_driven_slope", "counterexample", n=40, n_outliers=2,
                 outlier_magnitude=15.0, beta1=0.0),
        scenario("leverage_driven_fit", "counterexample", n=40, leverage_point=True,
                 leverage_x=25.0, leverage_y_shift=-20.0),
        scenario("omitted_variable_bias", "violation", omitted_strength=2.0,
                 omitted_x_correlation=0.7, n=300),
        scenario("measurement_error", "violation", measurement_error=2.0, n=300),
        scenario("endogeneity", "violation", endogeneity=0.6, n=300),
        scenario("robust_standard_errors", "robustness", heteroskedasticity="increasing",
                 hetero_strength=0.9, cov_type="HC3", n=200),
        scenario("standardized", "compare_methods", standardize=True),
    ),
    related=("regression.ols_geometry", "econometrics.heteroskedasticity",
             "econometrics.omitted_variable_bias"),
    next_concepts=("regression.ols_geometry", "regression.multicollinearity"),
    tags=("ols", "slope", "residuals", "functional form", "outliers", "leverage",
          "r-squared", "confidence band"),
    aliases=("simple regression", "regression simple", "الانحدار البسيط",
             "least squares", "linear regression"),
    backends=("numpy", "statsmodels"),
    misconceptions=("high_r2_means_correct", "extrapolation_is_estimation"),
    references=(
        ref("Wooldridge, J. M. (2019). Introductory Econometrics: A Modern Approach.",
            kind="book"),
        ref("Anscombe, F. J. (1973). Graphs in statistical analysis. The American "
            "Statistician 27(1).", kind="paper", doi="10.1080/00031305.1973.10478966"),
    ),
    curriculum_tags=("dz.econometrics1", "ksu.econ416"),
)


class SimpleRegressionLab(LabBase):
    DGP_KEYS = (
        "beta0", "beta1", "noise", "functional_form", "x_distribution", "x_low", "x_high",
        "error_distribution", "heteroskedasticity", "hetero_strength", "autocorrelation",
        "n_outliers", "outlier_magnitude", "leverage_point", "leverage_x",
        "leverage_y_shift", "measurement_error", "omitted_strength",
        "omitted_x_correlation", "endogeneity", "standardize", "center",
    )

    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        data = self._generate(p, n, state.seed)
        y = data["y"]
        X = data.matrix("x")
        fit = LM.ols(y, X, names=("const", "x"), cov_type=str(p["cov_type"]))
        fit.metadata["X"] = X
        level = float(p["confidence_level"])

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "scatter", self._scatter_figure(ctx, data, fit, p, level),
            "labs.reg.figure.scatter", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_residuals"]:
            res.add_panel(ctx.panel(
                "residuals", self._residual_figure(ctx, data, fit),
                "labs.reg.figure.residuals", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_sse"]:
            res.add_panel(ctx.panel(
                "sse", self._sse_figure(ctx, data, fit),
                "labs.reg.figure.sse", tab="math",
                evidence=EvidenceType.VISUAL_DERIVATION,
            ))
        if p["show_sampling"]:
            draws = self._sampling(p, n, state.seed, int(p["reps"]))
            res.add_panel(ctx.panel(
                "sampling", self._sampling_figure(ctx, draws, float(p["beta1"]), fit),
                "labs.reg.figure.sampling", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            res.metric("simulated_bias", ctx.t("labs.reg.metric.sim_bias",
                                               "Simulated bias of the slope"),
                       float(np.mean(draws["slope"]) - self._effective_truth(p)),
                       reference=0.0,
                       note=ctx.t("labs.reg.metric.sim_bias_note",
                                  "against the slope the DGP actually implies"))
            res.metric("simulated_se", ctx.t("labs.reg.metric.sim_se",
                                             "Simulated standard deviation of the slope"),
                       float(np.std(draws["slope"], ddof=1)),
                       reference=fit.se("x"),
                       note=ctx.t("labs.reg.metric.sim_se_note",
                                  "compared with the reported standard error"))

        ci = fit.conf_int(level)
        res.metric("slope", ctx.t("labs.reg.metric.slope", "Estimated slope"),
                   fit.coef("x"), reference=self._effective_truth(p))
        res.metric("slope_se", ctx.t("labs.reg.metric.slope_se",
                                     "Standard error ({c})", c=str(p["cov_type"])),
                   fit.se("x"))
        res.metric("slope_ci", ctx.t("labs.reg.metric.slope_ci",
                                     "Confidence interval for the slope"),
                   f"[{fmt(ci[1, 0], 4)}, {fmt(ci[1, 1], 4)}]")
        res.metric("slope_t", ctx.t("labs.reg.metric.t", "t statistic"),
                   float(fit.tvalues[1]))
        res.metric("slope_p", ctx.term("glossary.p_value"), float(fit.pvalues[1]))
        res.metric("intercept", ctx.t("labs.reg.metric.intercept", "Estimated intercept"),
                   fit.coef("const"), reference=float(p["beta0"]))
        res.metric("r_squared", ctx.term("glossary.r_squared"), fit.r_squared)
        res.metric("sigma_hat", ctx.t("labs.reg.metric.sigma",
                                      "Residual standard deviation"),
                   float(np.sqrt(fit.sigma2)), reference=float(p["noise"]))
        res.metric("ssr", ctx.t("labs.reg.metric.ssr", "Sum of squared residuals"),
                   fit.ssr)
        if float(p["omitted_strength"]) and float(p["omitted_x_correlation"]):
            res.metric("theoretical_bias", ctx.t("labs.reg.metric.theory_bias",
                                                 "Bias predicted by the OVB formula"),
                       data.truth["omitted_bias"])

        self._diagnose(ctx, res, data, fit, p)
        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.regression.simple_linear.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.regression.simple_linear.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.regression.simple_linear.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.regression.simple_linear.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.regression.simple_linear.warning"), kind="warning")
        return res

    # -- data ---------------------------------------------------------------
    def _generate(self, p, n, seed):
        kwargs = {k: p[k] for k in self.DGP_KEYS}
        return simple_regression(n=n, seed=seed, **kwargs)

    @staticmethod
    def _effective_truth(p):
        """The slope OLS is actually targeting under the current DGP."""
        if p["functional_form"] != "linear":
            return float("nan")
        beta1 = float(p["beta1"])
        if float(p["omitted_strength"]) and float(p["omitted_x_correlation"]):
            return float("nan")
        return beta1

    def _sampling(self, p, n, seed, reps):
        reps = min(reps, 4000)

        def experiment(gen):
            child = int(gen.integers(0, 2**31 - 1))
            d = self._generate(p, n, child)
            f = LM.ols(d["y"], d.matrix("x"), names=("const", "x"))
            return {"slope": f.coef("x"), "intercept": f.coef("const"),
                    "r2": f.r_squared}

        return monte_carlo(experiment, reps, seed).draws

    # -- figures ------------------------------------------------------------
    def _scatter_figure(self, ctx, data, fit, p, level):
        x, y = data["x"], data["y"]
        order = np.argsort(x)
        xs = x[order]
        fig = ctx.figure(
            "labs.reg.figure.scatter",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=450,
        )
        if p["show_bands"] and x.size > 3:
            grid = np.linspace(float(x.min()), float(x.max()), 200)
            Xg = np.column_stack([np.ones_like(grid), grid])
            se_mean = np.sqrt(np.einsum("ij,jk,ik->i", Xg, fit.covariance, Xg))
            crit = stats.t.ppf(0.5 + level / 2, max(fit.df_resid, 1))
            centre = Xg @ fit.coefficients
            P.shade_between(fig, grid, centre - crit * np.sqrt(se_mean**2 + fit.sigma2),
                            centre + crit * np.sqrt(se_mean**2 + fit.sigma2),
                            ctx.t("labs.common.trace.prediction_band"), "muted",
                            theme=ctx.theme, alpha=0.14)
            P.shade_between(fig, grid, centre - crit * se_mean, centre + crit * se_mean,
                            ctx.t("labs.common.trace.confidence_band"), "info",
                            theme=ctx.theme, alpha=0.22)
        if p["show_residuals"] and x.size <= 400:
            go = P.require_plotly()
            seg_x, seg_y = [], []
            for xi, yi, fi in zip(x, y, fit.fitted_values, strict=False):
                seg_x += [xi, xi, None]
                seg_y += [yi, fi, None]
            fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines",
                                     line={"color": ctx.color("residual"), "width": 1.0,
                                           "dash": "dot"},
                                     name=ctx.t("labs.common.trace.residuals"),
                                     hoverinfo="skip"))
        P.add_points(fig, x, y, ctx.t("labs.common.trace.data"), "primary",
                     theme=ctx.theme, size=7)
        P.add_curve(fig, xs, fit.fitted_values[order],
                    ctx.t("labs.reg.trace.fitted", "fitted line: y = {a} + {b} x",
                          a=fmt(fit.coef("const"), 3), b=fmt(fit.coef("x"), 3)),
                    "fitted", theme=ctx.theme, width=3.0)
        if p["functional_form"] != "linear":
            from ...data.generators.regression import apply_functional_form

            truth = apply_functional_form(xs, float(p["beta0"]), float(p["beta1"]),
                                          str(p["functional_form"]))
            P.add_curve(fig, xs, truth,
                        ctx.t("labs.reg.trace.truth_curve",
                              "true relationship ({f})", f=str(p["functional_form"])),
                        "truth", theme=ctx.theme, dash="dash")
        elif np.isfinite(self._effective_truth(p)):
            P.add_curve(fig, xs, float(p["beta0"]) + float(p["beta1"]) * xs,
                        ctx.t("labs.reg.trace.truth_line", "true line"),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_scatter",
            "The narrow band covers the average outcome at each x; the wide band covers a "
            "single new observation. Both widen away from the centre of the data - which is "
            "the geometry of extrapolation.",
        ), theme=ctx.theme)
        return fig

    def _residual_figure(self, ctx, data, fit):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.reg.trace.resid_vs_fitted", "residuals against fitted values"),
            ctx.t("labs.reg.trace.resid_qq", "residual normal quantile plot"),
        ))
        fig.add_trace(go.Scatter(x=fit.fitted_values, y=fit.residuals, mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 6,
                                         "opacity": 0.7}, showlegend=False), row=1, col=1)
        fig.add_hline(y=0, line={"color": ctx.color("baseline"), "dash": "dash"},
                      row=1, col=1)
        r = np.sort(fit.residuals)
        probs = (np.arange(1, r.size + 1) - 0.5) / r.size
        theo = stats.norm.ppf(probs) * float(np.std(fit.residuals, ddof=1))
        fig.add_trace(go.Scatter(x=theo, y=r, mode="markers",
                                 marker={"color": ctx.color("secondary"), "size": 5,
                                         "opacity": 0.7}, showlegend=False), row=1, col=2)
        lim = float(max(abs(theo[0]), abs(theo[-1])))
        fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                                 line={"color": ctx.color("truth"), "dash": "dash"},
                                 showlegend=False), row=1, col=2)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=350, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_residuals",
            "A residual plot with any visible pattern - a curve, a fan, a drift - is telling "
            "you the model is wrong in a specific and fixable way.",
        ), theme=ctx.theme)
        return fig

    def _sse_figure(self, ctx, data, fit):
        y, X = data["y"], data.matrix("x")
        b0_hat, b1_hat = fit.coefficients
        b0 = np.linspace(b0_hat - 4 * fit.standard_errors[0],
                         b0_hat + 4 * fit.standard_errors[0], 70)
        b1 = np.linspace(b1_hat - 4 * fit.standard_errors[1],
                         b1_hat + 4 * fit.standard_errors[1], 70)
        B0, B1 = np.meshgrid(b0, b1)
        resid = y[:, None, None] - (B0[None, :, :] + B1[None, :, :] * X[:, 1][:, None, None])
        sse = np.sum(resid**2, axis=0)
        go = P.require_plotly()
        fig = ctx.figure(
            "labs.reg.figure.sse",
            xaxis_title=ctx.t("labs.reg.axis.b0", "intercept"),
            yaxis_title=ctx.t("labs.reg.axis.b1", "slope"),
            height=420,
        )
        fig.add_trace(go.Contour(x=b0, y=b1, z=sse, colorscale=ctx.theme.colorscale,
                                 contours={"showlabels": True},
                                 name=ctx.t("labs.reg.trace.sse", "sum of squared residuals")))
        P.add_points(fig, [b0_hat], [b1_hat],
                     ctx.t("labs.reg.trace.optimum", "least-squares optimum"),
                     "estimate", theme=ctx.theme, size=13)
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_sse",
            "Least squares is the bottom of this bowl. The bowl's curvature in each "
            "direction is exactly what the standard errors report.",
        ), theme=ctx.theme)
        return fig

    def _sampling_figure(self, ctx, draws, beta1, fit):
        fig = ctx.figure(
            "labs.reg.figure.sampling",
            xaxis_title=ctx.t("labs.reg.axis.slope_estimate", "Slope estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=360,
        )
        P.add_histogram(fig, draws["slope"],
                        ctx.t("labs.reg.trace.slope_dist",
                              "slope across repeated samples"),
                        "primary", theme=ctx.theme, nbins=55, opacity=0.6)
        P.add_vline(fig, beta1, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_vline(fig, fit.coef("x"),
                    ctx.t("labs.reg.trace.this_sample", "estimate from this sample"),
                    "estimate", theme=ctx.theme, dash="solid")
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_sampling",
            "A real study draws ONE value from this histogram. Whether the histogram is "
            "centred on the truth is the question of bias; how wide it is, the question of "
            "precision.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    # -- diagnostics ---------------------------------------------------------
    def _diagnose(self, ctx, res, data, fit, p):
        X = data.matrix("x")
        notes = set(data.notes)
        res.assume("linearity", ctx.t("assumptions.linearity"),
                   p["functional_form"] == "linear",
                   detail=ctx.t("labs.reg.assume.linearity",
                                "The DGP here uses the '{f}' form.",
                                f=str(p["functional_form"])),
                   consequence="" if p["functional_form"] == "linear" else ctx.t(
                       "labs.reg.assume.linearity_consequence",
                       "Fitting a straight line to a curved truth leaves a systematic "
                       "pattern in the residuals and makes the slope an average of very "
                       "different local slopes."))
        res.assume("exogeneity", ctx.t("assumptions.exogeneity"),
                   "assumption_exogeneity" not in notes,
                   detail=ctx.t("labs.reg.assume.exogeneity",
                                "Omitted variables, measurement error and simultaneity all "
                                "break this one, and all three are available as controls."))
        res.assume("homoskedasticity", ctx.t("assumptions.homoskedasticity"),
                   "assumption_homoskedasticity" not in notes,
                   consequence=ctx.t("labs.reg.assume.hetero_consequence",
                                     "Coefficients stay unbiased; the classical standard "
                                     "error is what breaks."))
        res.assume("no_autocorrelation", ctx.t("assumptions.no_autocorrelation"),
                   "assumption_no_autocorrelation" not in notes)
        res.assume("normal_errors", ctx.t("assumptions.normal_errors"),
                   "assumption_normal_errors" not in notes,
                   detail=ctx.t("labs.reg.assume.normality",
                                "Needed for exact finite-sample t and F inference; with a "
                                "large n the central limit theorem usually rescues it."))

        if fit.nobs > 6:
            bp = LM.breusch_pagan(fit, X)
            jb = LM.jarque_bera(fit.residuals)
            dw = LM.durbin_watson(fit.residuals)
            reset = LM.reset_test(fit, X)
            res.metric("breusch_pagan_p", ctx.t("labs.reg.metric.bp",
                                                "Breusch-Pagan p-value"), bp.p_value)
            res.metric("jarque_bera_p", ctx.t("labs.reg.metric.jb",
                                              "Jarque-Bera p-value"), jb.p_value)
            res.metric("durbin_watson", ctx.t("labs.reg.metric.dw",
                                              "Durbin-Watson statistic"), dw,
                       reference=2.0)
            res.metric("reset_p", ctx.t("labs.reg.metric.reset",
                                        "Ramsey RESET p-value"), reset.p_value,
                       note=ctx.t("labs.reg.metric.reset_note",
                                  "small values point at a wrong functional form"))

        h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
        cooks = (fit.residuals**2 / (2 * fit.sigma2)) * h / np.clip(1 - h, 1e-9, None) ** 2
        res.metric("max_leverage", ctx.t("labs.reg.metric.leverage", "Maximum leverage"),
                   float(np.max(h)),
                   note=ctx.t("labs.reg.metric.leverage_note",
                              "the average is k/n = {v}", v=fmt(2 / fit.nobs, 4)))
        res.metric("max_cooks", ctx.t("labs.reg.metric.cooks",
                                      "Largest Cook's distance"), float(np.max(cooks)))
        if float(np.max(cooks)) > 1.0:
            res.warnings.append(ctx.t(
                "labs.reg.warn.influential",
                "One observation has a Cook's distance above 1: deleting it alone would move "
                "the fitted line materially. Look at the point before trusting the slope.",
            ))
        if p["functional_form"] != "linear" and fit.r_squared > 0.8:
            res.warnings.append(ctx.t(
                "labs.reg.warn.r2_trap",
                "R-squared is {r} even though the fitted form is wrong. Goodness of fit "
                "measures how close the line is to the points, not whether the model is right.",
                r=fmt(fit.r_squared, 3),
            ))

    # -- animation -----------------------------------------------------------
    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        base = self._generate(p, n, seed)
        x = base["x"]
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        slopes = np.linspace(float(p["beta1"]) - 3.0, float(p["beta1"]) + 3.0, 24)
        y = base["y"]
        X = base.matrix("x")
        fit = LM.ols(y, X, names=("const", "x"))
        b0 = fit.coef("const")
        frames, steps = [], []
        for i, b in enumerate(slopes):
            pred = b0 + b * X[:, 1]
            sse = float(np.sum((y - pred) ** 2))
            frames.append(go.Frame(name=f"{b:.2f}", data=[
                go.Scatter(x=xs, y=b0 + b * xs),
            ]))
            steps.append(AnimationStep(
                id=f"slope_{i}", frame=i,
                title=ctx.t("labs.reg.anim.title", "candidate slope = {b}", b=fmt(b, 2)),
                what_you_see=ctx.t("labs.reg.anim.see",
                                   "The same {n} points with a candidate line drawn through "
                                   "them.", n=n),
                what_changed=ctx.t("labs.reg.anim.changed",
                                   "The candidate slope moved to {b}.", b=fmt(b, 2)),
                why=ctx.t("labs.reg.anim.why",
                          "Least squares scores every candidate line by the total squared "
                          "vertical distance to the points, and keeps the lowest score."),
                interpretation=ctx.t("labs.reg.anim.interpret",
                                     "Sum of squared residuals is {s}; the minimum available "
                                     "is {m} at slope {o}.",
                                     s=fmt(sse, 2), m=fmt(fit.ssr, 2),
                                     o=fmt(fit.coef("x"), 3)),
                conclusion=ctx.t("labs.reg.anim.conclude",
                                 "The estimated slope is not a guess about the world; it is "
                                 "the unique minimiser of this score for these data."),
                warning=ctx.t("labs.reg.anim.warn",
                              "Squaring is a choice. It is what makes one far-away point "
                              "able to dominate every other."),
                math="SSR(b) = sum (y_i - b0 - b*x_i)^2;  minimised at b = Cov(x,y)/Var(x)",
                outputs={"slope": round(float(b), 4), "ssr": round(sse, 3)},
                active_assumptions=("linearity",),
                highlighted=("candidate_line",),
            ))
        fig = ctx.figure(
            "labs.reg.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=400,
        )
        fig.add_trace(go.Scatter(x=xs, y=b0 + slopes[0] * xs, mode="lines",
                                 line={"color": ctx.color("fitted"), "width": 3.0},
                                 name=ctx.t("labs.reg.trace.candidate", "candidate line")))
        P.add_points(fig, x, y, ctx.t("labs.common.trace.data"), "primary",
                     theme=ctx.theme, size=6)
        build_frames(fig, frames, duration=340, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.reg.slider", "slope"))
        return animation(
            "least_squares_search", fig, steps,
            purpose=ctx.t("labs.reg.anim.purpose",
                          "Show what 'least squares' is actually minimising."),
            summary=ctx.t(
                "labs.reg.anim.summary",
                "The fitted line is whichever candidate makes the total squared vertical "
                "distance smallest. Everything else - standard errors, R-squared, the "
                "sensitivity to outliers - follows from that one rule."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = SimpleRegressionLab(SPEC)
