"""Heteroskedasticity: diagnosis, consequence and remedy."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.regression import HETERO_TYPES, simple_regression
from ...simulation.monte_carlo import monte_carlo

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "econometrics.heteroskedasticity",
    Domain.ECONOMETRICS,
    "diagnostics_and_remedies",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("heteroskedasticity", "increasing", HETERO_TYPES, group="violation"),
        slider("hetero_strength", 0.8, 0.0, 1.0, 0.01, group="violation"),
        int_slider("n", 200, 10, 5000, 1, group="dgp"),
        slider("beta1", 1.5, -5.0, 5.0, 0.05, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        slider("alpha", 0.05, 0.001, 0.20, 0.001, group="inference"),
        int_slider("reps", 1000, 100, 10000, 100, group="simulation", expensive=True),
        toggle("show_wls", True, group="remedies"),
        toggle("show_size_study", True, group="simulation"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", heteroskedasticity="increasing",
                 hetero_strength=0.8, n=200),
        scenario("homoskedastic", "null", heteroskedasticity="none", hetero_strength=0.0),
        scenario("mild", "weak", hetero_strength=0.25),
        scenario("severe", "strong", hetero_strength=1.0),
        scenario("decreasing", "negative", heteroskedasticity="decreasing",
                 hetero_strength=0.8),
        scenario("u_shaped", "misspecification", heteroskedasticity="u_shaped",
                 hetero_strength=0.9),
        scenario("grouped", "compare_methods", heteroskedasticity="grouped",
                 hetero_strength=0.9),
        scenario("small_sample", "small_sample", n=25, hetero_strength=0.9),
        scenario("large_sample", "large_sample", n=3000, hetero_strength=0.9),
        scenario("robust_repairs", "robustness", hetero_strength=1.0, n=400),
    ),
    prerequisites=("regression.simple_linear",),
    related=("econometrics.autocorrelation", "regression.simple_linear"),
    next_concepts=("econometrics.autocorrelation",),
    tags=("heteroskedasticity", "white", "breusch-pagan", "robust standard errors",
          "wls", "gls"),
    aliases=("heteroskedasticity", "heteroscedasticite", "عدم تجانس التباين",
             "robust standard errors", "white test"),
    backends=("numpy", "statsmodels"),
    references=(
        ref("White, H. (1980). A heteroskedasticity-consistent covariance matrix "
            "estimator. Econometrica 48(4).", kind="paper", doi="10.2307/1912934"),
        ref("Breusch, T. S. and Pagan, A. R. (1979). A simple test for heteroscedasticity. "
            "Econometrica 47(5).", kind="paper", doi="10.2307/1911963"),
    ),
    curriculum_tags=("dz.econometrics1", "cairo.eviews", "ksu.econ416"),
)


class HeteroskedasticityLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        data = self._generate(p, n, state.seed)
        X = data.matrix("x")
        y = data["y"]
        fits = {
            "classical": LM.ols(y, X, names=("const", "x"), cov_type="nonrobust"),
            "hc1": LM.ols(y, X, names=("const", "x"), cov_type="HC1"),
            "hc3": LM.ols(y, X, names=("const", "x"), cov_type="HC3"),
        }
        for f in fits.values():
            f.metadata["X"] = X
        if p["show_wls"]:
            fits["wls"] = self._wls(y, X, fits["classical"])

        bp = LM.breusch_pagan(fits["classical"], X)
        white = LM.white_test(fits["classical"], X)

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "scatter", self._scatter_figure(ctx, data, fits["classical"]),
            "labs.het.figure.scatter", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        res.add_panel(ctx.panel(
            "residual_spread", self._spread_figure(ctx, data, fits["classical"]),
            "labs.het.figure.spread", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "standard_errors", self._se_figure(ctx, fits, p, state.seed, n),
            "labs.het.figure.se", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))
        if p["show_size_study"]:
            study = self._size_study(p, n, state.seed, int(p["reps"]))
            res.add_panel(ctx.panel(
                "size", self._size_figure(ctx, study, float(p["alpha"])),
                "labs.het.figure.size", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            for key, label in (("classical", "classical"), ("hc1", "HC1"),
                               ("hc3", "HC3"), ("wls", "WLS")):
                if key in study:
                    res.metric(f"size_{key}",
                               ctx.t("labs.het.metric.size",
                                     "Actual rejection rate of a true null using {m}",
                                     m=label),
                               study[key], reference=float(p["alpha"]))

        res.metric("bp_statistic", ctx.t("labs.het.metric.bp", "Breusch-Pagan statistic"),
                   bp.statistic)
        res.metric("bp_p", ctx.t("labs.het.metric.bp_p", "Breusch-Pagan p-value"),
                   bp.p_value)
        res.metric("white_p", ctx.t("labs.het.metric.white_p", "White test p-value"),
                   white.p_value)
        res.metric("slope", ctx.t("labs.het.metric.slope", "OLS slope"),
                   fits["classical"].coef("x"), reference=float(p["beta1"]),
                   note=ctx.t("labs.het.metric.slope_note",
                              "identical whichever standard error you report"))
        for key, label in (("classical", "classical"), ("hc1", "HC1"), ("hc3", "HC3")):
            res.metric(f"se_{key}",
                       ctx.t("labs.het.metric.se", "Standard error ({m})", m=label),
                       fits[key].se("x"))
        if "wls" in fits:
            res.metric("se_wls", ctx.t("labs.het.metric.se", "Standard error ({m})",
                                       m="WLS"), fits["wls"].se("x"))
            res.metric("wls_slope", ctx.t("labs.het.metric.wls_slope", "WLS slope"),
                       fits["wls"].coef("x"), reference=float(p["beta1"]))
        res.metric("se_ratio", ctx.t("labs.het.metric.ratio",
                                     "HC1 divided by the classical standard error"),
                   float(fits["hc1"].se("x") / max(fits["classical"].se("x"), 1e-12)),
                   note=ctx.t("labs.het.metric.ratio_note",
                              "far from 1 means the classical formula is misleading"))

        violated = str(p["heteroskedasticity"]) != "none" and float(p["hetero_strength"]) > 0
        res.assume("homoskedasticity", ctx.t("assumptions.homoskedasticity"), not violated,
                   detail=ctx.t("labs.het.assume.homoskedasticity",
                                "The error variance is a function of x in this DGP."),
                   consequence=ctx.t("labs.het.assume.consequence",
                                     "Coefficients remain unbiased and consistent. The "
                                     "classical standard error, its t statistics, its "
                                     "confidence intervals and its F tests are all wrong."))
        res.assume("exogeneity", ctx.t("assumptions.exogeneity"), True,
                   detail=ctx.t("labs.het.assume.exogeneity",
                                "Untouched by heteroskedasticity - which is precisely why "
                                "the estimator stays unbiased."))
        res.assume("known_variance_form",
                   ctx.t("labs.het.assume.wls_label",
                         "The variance function is known well enough for weighting"),
                   False,
                   detail=ctx.t("labs.het.assume.wls",
                                "Weighted least squares needs the variance structure. "
                                "Guessing it wrong can be worse than doing nothing, which "
                                "is why robust standard errors are the default in practice."))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.econometrics.heteroskedasticity.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.econometrics.heteroskedasticity.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.econometrics.heteroskedasticity.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.econometrics.heteroskedasticity.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.econometrics.heteroskedasticity.warning"), kind="warning")
        res.explain("diagnostics", ctx.t("labs.het.workflow.title",
                                         "Diagnosis to remedy"), ctx.t(
            "labs.het.workflow",
            "1. Suspect: the residual plot fans out.\n"
            "2. Test: Breusch-Pagan p = {bp}, White p = {w}.\n"
            "3. Consequence: the robust standard error is {r} times the classical one.\n"
            "4. Remedy: report robust standard errors, or weight if you genuinely know the "
            "variance function.\n"
            "5. What the remedy does NOT fix: nothing about bias, endogeneity, functional "
            "form or omitted variables. Robust standard errors repair inference only.",
            bp=fmt(bp.p_value, 4), w=fmt(white.p_value, 4),
            r=fmt(float(fits["hc1"].se("x") / max(fits["classical"].se("x"), 1e-12)), 2),
        ))
        return res

    @staticmethod
    def _generate(p, n, seed):
        return simple_regression(
            n=n, beta1=float(p["beta1"]), noise=float(p["noise"]),
            heteroskedasticity=str(p["heteroskedasticity"]),
            hetero_strength=float(p["hetero_strength"]), seed=seed,
        )

    @staticmethod
    def _wls(y, X, ols_fit):
        """Feasible WLS with the variance modelled as exp of a linear index."""
        log_u2 = np.log(np.clip(ols_fit.residuals**2, 1e-12, None))
        aux = LM.ols(log_u2, X)
        weights = 1.0 / np.clip(np.exp(aux.fitted_values), 1e-9, None)
        fit = LM.wls(y, X, weights, names=("const", "x"))
        fit.metadata["X"] = X
        return fit

    def _scatter_figure(self, ctx, data, fit):
        x, y = data["x"], data["y"]
        order = np.argsort(x)
        fig = ctx.figure(
            "labs.het.figure.scatter",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=420,
        )
        P.add_points(fig, x, y, ctx.t("labs.common.trace.data"), "primary",
                     theme=ctx.theme, size=6)
        P.add_curve(fig, x[order], fit.fitted_values[order],
                    ctx.t("labs.common.trace.fitted_line"), "fitted", theme=ctx.theme,
                    width=3.0)
        band = 1.96 * np.sqrt(fit.sigma2) * data["hetero_weight"][order]
        P.shade_between(fig, x[order], fit.fitted_values[order] - band,
                        fit.fitted_values[order] + band,
                        ctx.t("labs.het.trace.true_spread",
                              "true +/- 2 sd of the error (varies with x)"),
                        "warning", theme=ctx.theme, alpha=0.16)
        P.add_legend_note(fig, ctx.t(
            "labs.het.legend_scatter",
            "The line is still in the right place - every point is centred on it. What "
            "changes with x is how far the points scatter, and that is what the classical "
            "standard error assumed away.",
        ), theme=ctx.theme)
        return fig

    def _spread_figure(self, ctx, data, fit):
        x = data["x"]
        u2 = fit.residuals**2
        order = np.argsort(x)
        window = max(int(0.1 * x.size), 5)
        smooth = np.convolve(u2[order], np.ones(window) / window, mode="same")
        fig = ctx.figure(
            "labs.het.figure.spread",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.het.axis.squared_residual", "Squared residual"),
            height=350,
        )
        P.add_points(fig, x, u2, ctx.t("labs.het.trace.u2", "squared residuals"),
                     "primary", theme=ctx.theme, size=5, opacity=0.5)
        P.add_curve(fig, x[order], smooth,
                    ctx.t("labs.het.trace.smooth", "local average"),
                    "negative", theme=ctx.theme, width=3.0)
        P.add_hline(fig, float(np.mean(u2)),
                    ctx.t("labs.het.trace.constant",
                          "what homoskedasticity would predict"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.het.legend_spread",
            "The Breusch-Pagan test is exactly a regression of these squared residuals on "
            "the regressors: it asks whether the red curve has a slope.",
        ), theme=ctx.theme)
        return fig

    def _se_figure(self, ctx, fits, p, seed, n):
        draws = self._sampling(p, n, seed, 600)
        true_sd = float(np.std(draws["slope"], ddof=1))
        labels, values = [], []
        for key, label in (("classical", "classical"), ("hc1", "HC1"), ("hc3", "HC3"),
                           ("wls", "WLS")):
            if key in fits:
                labels.append(label)
                values.append(fits[key].se("x"))
        fig = ctx.figure(
            "labs.het.figure.se",
            xaxis_title=ctx.t("labs.het.axis.method", "Covariance estimator"),
            yaxis_title=ctx.t("labs.het.axis.se", "Reported standard error"),
            height=350,
        )
        P.add_bar(fig, labels, values, ctx.t("labs.het.trace.reported", "reported"),
                  "primary", theme=ctx.theme, text=[fmt(v, 4) for v in values])
        P.add_hline(fig, true_sd,
                    ctx.t("labs.het.trace.true_sd",
                          "actual sampling sd of the slope = {v}", v=fmt(true_sd, 4)),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.het.legend_se",
            "A standard error is only honest if it matches the dashed line. Under "
            "heteroskedasticity the classical bar misses it while the robust bars do not.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _sampling(self, p, n, seed, reps):
        def experiment(gen):
            child = int(gen.integers(0, 2**31 - 1))
            d = self._generate(p, n, child)
            f = LM.ols(d["y"], d.matrix("x"), names=("const", "x"))
            return {"slope": f.coef("x")}

        return monte_carlo(experiment, reps, seed).draws

    def _size_study(self, p, n, seed, reps):
        """Rejection rate of a TRUE null - the size of each procedure."""
        reps = min(reps, 3000)
        q = dict(p)
        q["beta1"] = 0.0
        counts = {"classical": 0, "hc1": 0, "hc3": 0}
        wls_count = 0
        alpha = float(p["alpha"])
        for r in range(reps):
            d = self._generate(q, n, int(seed) * 7919 + r)
            X = d.matrix("x")
            y = d["y"]
            for key, cov in (("classical", "nonrobust"), ("hc1", "HC1"), ("hc3", "HC3")):
                f = LM.ols(y, X, names=("const", "x"), cov_type=cov)
                if float(f.pvalues[1]) < alpha:
                    counts[key] += 1
            if p["show_wls"]:
                base = LM.ols(y, X, names=("const", "x"))
                try:
                    w = self._wls(y, X, base)
                    if float(w.pvalues[1]) < alpha:
                        wls_count += 1
                except Exception:  # noqa: BLE001
                    pass
        out = {k: v / reps for k, v in counts.items()}
        if p["show_wls"]:
            out["wls"] = wls_count / reps
        return out

    def _size_figure(self, ctx, study, alpha):
        labels = list(study)
        values = [study[k] for k in labels]
        fig = ctx.figure(
            "labs.het.figure.size",
            xaxis_title=ctx.t("labs.het.axis.method", "Covariance estimator"),
            yaxis_title=ctx.t("labs.het.axis.rejection",
                              "Rejection rate when the null is TRUE"),
            height=350,
        )
        P.add_bar(fig, labels, values, ctx.t("labs.het.trace.size", "actual size"),
                  "negative", theme=ctx.theme, text=[pct(v) for v in values])
        P.add_hline(fig, alpha, ctx.t("labs.het.trace.nominal",
                                      "nominal alpha = {a}", a=fmt(alpha, 3)),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.het.legend_size",
            "This is the honest test of an inference procedure: with no effect at all, how "
            "often does it claim one? A bar above the dashed line means the method rejects "
            "true nulls more often than advertised.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        strengths = np.linspace(0.0, 1.0, 16)
        frames, steps = [], []
        for i, s in enumerate(strengths):
            q = dict(p)
            q["hetero_strength"] = float(s)
            d = self._generate(q, n, seed)
            X = d.matrix("x")
            classical = LM.ols(d["y"], X, names=("const", "x"))
            classical.metadata["X"] = X
            robust = LM.ols(d["y"], X, names=("const", "x"), cov_type="HC1")
            bp = LM.breusch_pagan(classical, X)
            frames.append(go.Frame(name=f"{s:.2f}", data=[
                go.Scatter(x=d["x"], y=classical.residuals),
            ]))
            steps.append(AnimationStep(
                id=f"s_{i}", frame=i,
                title=ctx.t("labs.het.anim.title",
                            "heteroskedasticity strength = {s}", s=fmt(s, 2)),
                what_you_see=ctx.t("labs.het.anim.see",
                                   "Residuals plotted against x as the error variance is "
                                   "made progressively more dependent on x."),
                what_changed=ctx.t("labs.het.anim.changed",
                                   "The strength of the variance-x relationship moved "
                                   "to {s}.", s=fmt(s, 2)),
                why=ctx.t("labs.het.anim.why",
                          "Observations in the high-variance region carry less information "
                          "about the line, but least squares weights them equally - so the "
                          "classical variance formula overstates the precision it achieves."),
                interpretation=ctx.t("labs.het.anim.interpret",
                                     "Breusch-Pagan p = {bp}; the robust standard error is "
                                     "{r} times the classical one.",
                                     bp=fmt(bp.p_value, 4),
                                     r=fmt(robust.se("x") / max(classical.se("x"), 1e-12), 2)),
                conclusion=ctx.t("labs.het.anim.conclude",
                                 "The fan opens, the test reacts, and the two standard "
                                 "errors separate. The slope itself barely moves."),
                warning=ctx.t("labs.het.anim.warn",
                              "Watch the slope estimate throughout: it is not drifting. "
                              "This is a variance problem, not a bias problem."),
                math="Var(b) = (X'X)^-1 X' Omega X (X'X)^-1",
                outputs={"strength": round(float(s), 3),
                         "bp_p": round(float(bp.p_value), 5),
                         "slope": round(classical.coef("x"), 5),
                         "se_ratio": round(float(robust.se("x") /
                                                 max(classical.se("x"), 1e-12)), 4)},
                active_assumptions=("exogeneity",),
                violated_assumptions=() if s == 0 else ("homoskedasticity",),
                highlighted=("residual_fan",),
            ))
        fig = ctx.figure(
            "labs.het.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.residual"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 6,
                                         "opacity": 0.6},
                                 name=ctx.t("labs.common.trace.residuals")))
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme)
        build_frames(fig, frames, duration=440, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.het.slider", "strength"))
        return animation(
            "fan_opens", fig, steps,
            purpose=ctx.t("labs.het.anim.purpose",
                          "Introduce the violation gradually so its symptom, its test and "
                          "its consequence can be watched separately."),
            summary=ctx.t(
                "labs.het.anim.summary",
                "As the fan opens, the diagnostic reacts and the two standard errors "
                "diverge - but the coefficient stays put. Heteroskedasticity is a problem "
                "of how confident you are allowed to be, not of where the line is."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = HeteroskedasticityLab(SPEC)
