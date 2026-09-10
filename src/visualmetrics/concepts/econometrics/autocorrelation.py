"""Serial correlation: diagnosis, consequence and remedy."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.timeseries import acf
from ...simulation.random import rng

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "econometrics.autocorrelation",
    Domain.ECONOMETRICS,
    "diagnostics_and_remedies",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        slider("rho", 0.7, -0.95, 0.95, 0.01, group="violation"),
        slider("x_persistence", 0.8, 0.0, 0.99, 0.01, group="dgp"),
        int_slider("n", 150, 20, 3000, 1, group="dgp"),
        slider("beta1", 1.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        int_slider("lags", 2, 1, 12, 1, group="diagnostics"),
        int_slider("hac_lags", 0, 0, 40, 1, group="remedies",
                   help_key="labs.auto.controls.hac_lags.help"),
        toggle("lagged_dependent", False, group="violation"),
        toggle("show_fgls", True, group="remedies"),
        toggle("show_size_study", True, group="simulation"),
        slider("alpha", 0.05, 0.001, 0.20, 0.001, group="inference"),
        int_slider("reps", 600, 100, 5000, 100, group="simulation", expensive=True),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", rho=0.7, x_persistence=0.8, n=150),
        scenario("no_autocorrelation", "null", rho=0.0),
        scenario("mild", "weak", rho=0.3),
        scenario("severe", "strong", rho=0.92),
        scenario("negative", "negative", rho=-0.7),
        scenario("white_noise_regressor", "compare_methods", x_persistence=0.0, rho=0.8),
        scenario("persistent_regressor", "violation", x_persistence=0.95, rho=0.9),
        scenario("lagged_dependent_variable", "counterexample", lagged_dependent=True,
                 rho=0.6, n=300),
        scenario("small_sample", "small_sample", n=30, rho=0.8),
        scenario("large_sample", "large_sample", n=2000, rho=0.8),
    ),
    prerequisites=("regression.simple_linear",),
    related=("econometrics.heteroskedasticity", "timeseries.stationarity"),
    tags=("serial correlation", "durbin-watson", "breusch-godfrey", "hac",
          "newey-west", "fgls", "cochrane-orcutt"),
    aliases=("autocorrelation", "الارتباط الذاتي", "correlation serielle",
             "newey-west", "durbin watson"),
    backends=("numpy", "statsmodels"),
    references=(
        ref("Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, "
            "heteroskedasticity and autocorrelation consistent covariance matrix. "
            "Econometrica 55(3).", kind="paper", doi="10.2307/1913610"),
        ref("Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear "
            "models. Australian Economic Papers 17.", kind="paper"),
    ),
    curriculum_tags=("dz.econometrics1", "cairo.eviews"),
)


class AutocorrelationLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        rho = float(p["rho"])
        y, X, names = self._generate(p, n, state.seed)

        classical = LM.ols(y, X, names=names, cov_type="nonrobust")
        classical.metadata["X"] = X
        maxlags = int(p["hac_lags"]) or None
        hac = LM.ols(y, X, names=names, cov_type="HAC", maxlags=maxlags)
        fits = {"classical": classical, "hac": hac}
        if p["show_fgls"] and not p["lagged_dependent"]:
            fits["fgls"] = self._fgls(y, X, classical, names)

        dw = LM.durbin_watson(classical.residuals)
        bg = LM.breusch_godfrey(classical, X, lags=int(p["lags"]))

        res.dgp = ctx.t(
            "labs.auto.dgp",
            "y_t = {b} x_t + u_t with AR(1) errors of coefficient rho = {r}; the regressor "
            "itself has persistence {px}. n = {n}{ld}.",
            b=fmt(p["beta1"], 2), r=fmt(rho, 2), px=fmt(p["x_persistence"], 2), n=n,
            ld=(", and a lagged dependent variable is included"
                if p["lagged_dependent"] else ""),
        )

        res.add_panel(ctx.panel(
            "series", self._series_figure(ctx, y, X, classical),
            "labs.auto.figure.series", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        res.add_panel(ctx.panel(
            "acf", self._acf_figure(ctx, classical.residuals, n),
            "labs.auto.figure.acf", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "standard_errors", self._se_figure(ctx, fits, p, n, state.seed, names),
            "labs.auto.figure.se", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))
        if p["show_size_study"]:
            study = self._size_study(p, n, state.seed, int(p["reps"]), names)
            res.add_panel(ctx.panel(
                "size", self._size_figure(ctx, study, float(p["alpha"])),
                "labs.auto.figure.size", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            for key in study:
                res.metric(f"size_{key}",
                           ctx.t("labs.auto.metric.size",
                                 "Rejection rate of a true null using {m}", m=key),
                           study[key], reference=float(p["alpha"]))

        res.metric("durbin_watson", ctx.t("labs.auto.metric.dw",
                                          "Durbin-Watson statistic"), dw, reference=2.0,
                   note=ctx.t("labs.auto.metric.dw_note",
                              "approximately 2(1 - rho_hat); invalid with a lagged "
                              "dependent variable"))
        res.metric("implied_rho", ctx.t("labs.auto.metric.rho",
                                        "Residual AR(1) coefficient"),
                   float(acf(classical.residuals, 1)[1]), reference=rho)
        res.metric("bg_statistic", ctx.t("labs.auto.metric.bg",
                                         "Breusch-Godfrey statistic ({l} lags)",
                                         l=int(p["lags"])), bg.statistic)
        res.metric("bg_p", ctx.t("labs.auto.metric.bg_p",
                                 "Breusch-Godfrey p-value"), bg.p_value)
        res.metric("slope", ctx.t("labs.auto.metric.slope", "OLS slope"),
                   classical.coef(names[1]), reference=float(p["beta1"]))
        res.metric("se_classical", ctx.t("labs.auto.metric.se_classical",
                                         "Classical standard error"),
                   classical.se(names[1]))
        res.metric("se_hac", ctx.t("labs.auto.metric.se_hac",
                                   "Newey-West HAC standard error"), hac.se(names[1]))
        if "fgls" in fits:
            res.metric("se_fgls", ctx.t("labs.auto.metric.se_fgls",
                                        "Feasible GLS standard error"),
                       fits["fgls"].se(names[1]))
            res.metric("slope_fgls", ctx.t("labs.auto.metric.slope_fgls",
                                           "Feasible GLS slope"),
                       fits["fgls"].coef(names[1]), reference=float(p["beta1"]))
        res.metric("se_ratio", ctx.t("labs.auto.metric.ratio",
                                     "HAC divided by classical standard error"),
                   float(hac.se(names[1]) / max(classical.se(names[1]), 1e-12)))

        biased = bool(p["lagged_dependent"]) and abs(rho) > 0.01
        res.assume("no_autocorrelation", ctx.t("assumptions.no_autocorrelation"),
                   abs(rho) < 0.01,
                   detail=ctx.t("labs.auto.assume.autocorrelation",
                                "The errors follow an AR(1) process here."),
                   consequence=ctx.t("labs.auto.assume.consequence",
                                     "With a strictly exogenous regressor the coefficient "
                                     "stays unbiased and only the standard error is wrong."))
        res.assume("exogeneity", ctx.t("assumptions.exogeneity"), not biased,
                   detail=ctx.t("labs.auto.assume.lagged",
                                "A lagged dependent variable is correlated with past errors "
                                "by construction. Combined with serial correlation, that "
                                "makes least squares inconsistent, not merely imprecise."),
                   consequence="" if not biased else ctx.t(
                       "labs.auto.assume.lagged_consequence",
                       "Robust standard errors cannot repair this: the point estimate "
                       "itself converges to the wrong number."))

        res.animations.append(self._animation(ctx, p, n, state.seed, names))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.econometrics.autocorrelation.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.econometrics.autocorrelation.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.econometrics.autocorrelation.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.econometrics.autocorrelation.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.econometrics.autocorrelation.warning"), kind="warning")
        res.explain("diagnostics", ctx.t("labs.auto.workflow.title",
                                         "Diagnosis to remedy"), ctx.t(
            "labs.auto.workflow",
            "1. Suspect: residuals wander in runs rather than alternating.\n"
            "2. Test: Durbin-Watson = {dw} (against 2), Breusch-Godfrey p = {bg}.\n"
            "3. Consequence: the HAC standard error is {r} times the classical one.\n"
            "4. Remedy: HAC standard errors, or feasible GLS if you trust the AR structure.\n"
            "5. What the remedy does NOT fix: if the model contains a lagged dependent "
            "variable, serial correlation makes the estimator inconsistent, and no "
            "covariance correction touches that.",
            dw=fmt(dw, 3), bg=fmt(bg.p_value, 4),
            r=fmt(float(hac.se(names[1]) / max(classical.se(names[1]), 1e-12)), 2),
        ))
        if biased:
            res.warnings.append(ctx.t(
                "labs.auto.warn.lagged",
                "This scenario combines a lagged dependent variable with serially "
                "correlated errors. The slope is now inconsistent - watch it drift away "
                "from the true value as n grows, which no standard-error fix can repair.",
            ))
        return res

    @staticmethod
    def _generate(p, n, seed):
        gen = rng(seed, "autocorr")
        rho = float(p["rho"])
        px = float(p["x_persistence"])
        x = np.empty(n)
        x[0] = gen.standard_normal()
        for t in range(1, n):
            x[t] = px * x[t - 1] + np.sqrt(max(1 - px**2, 1e-9)) * gen.standard_normal()
        e = gen.standard_normal(n) * float(p["noise"])
        u = np.empty(n)
        u[0] = e[0] / np.sqrt(max(1 - rho**2, 1e-6))
        for t in range(1, n):
            u[t] = rho * u[t - 1] + e[t]
        if p["lagged_dependent"]:
            y = np.empty(n)
            y[0] = u[0]
            for t in range(1, n):
                y[t] = 0.5 * y[t - 1] + float(p["beta1"]) * x[t] + u[t]
            X = np.column_stack([np.ones(n - 1), x[1:], y[:-1]])
            return y[1:], X, ("const", "x", "y_lag")
        y = float(p["beta1"]) * x + u
        return y, np.column_stack([np.ones(n), x]), ("const", "x")

    @staticmethod
    def _fgls(y, X, ols_fit, names):
        """Cochrane-Orcutt style feasible GLS with an estimated AR(1) coefficient."""
        u = ols_fit.residuals
        rho_hat = float(np.clip(acf(u, 1)[1], -0.95, 0.95))
        yt = y[1:] - rho_hat * y[:-1]
        Xt = X[1:] - rho_hat * X[:-1]
        fit = LM.ols(yt, Xt, names=names)
        fit.metadata["rho_hat"] = rho_hat
        return fit

    def _series_figure(self, ctx, y, X, fit):
        t = np.arange(y.size)
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(
                                ctx.t("labs.auto.trace.fit", "observed and fitted series"),
                                ctx.t("labs.auto.trace.resid_time",
                                      "residuals over time"),
                            ))
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines",
                                 line={"color": ctx.color("primary")},
                                 name=ctx.t("labs.common.trace.data")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=fit.fitted_values, mode="lines",
                                 line={"color": ctx.color("fitted"), "dash": "dash"},
                                 name=ctx.t("labs.common.trace.fitted_line")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=fit.residuals, mode="lines",
                                 line={"color": ctx.color("residual")},
                                 name=ctx.t("labs.common.trace.residuals")), row=2, col=1)
        fig.add_hline(y=0, line={"color": ctx.color("baseline"), "dash": "dot"},
                      row=2, col=1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=430, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.auto.legend_series",
            "Serially correlated residuals wander in long runs above and below zero. "
            "Independent residuals would cross zero far more often.",
        ), theme=ctx.theme)
        return fig

    def _acf_figure(self, ctx, resid, n):
        nlags = min(24, max(n // 4, 4))
        a = acf(resid, nlags)
        band = 1.96 / np.sqrt(n)
        fig = ctx.figure(
            "labs.auto.figure.acf",
            xaxis_title=ctx.t("labs.common.axis.lag"),
            yaxis_title=ctx.t("labs.common.axis.autocorrelation"),
            height=350,
        )
        P.add_bar(fig, np.arange(nlags + 1), a,
                  ctx.t("labs.auto.trace.acf", "residual autocorrelation"),
                  "primary", theme=ctx.theme)
        P.shade_between(fig, np.arange(nlags + 1), np.full(nlags + 1, -band),
                        np.full(nlags + 1, band),
                        ctx.t("labs.auto.trace.band",
                              "approximate 95% band under no autocorrelation"),
                        "muted", theme=ctx.theme, alpha=0.2)
        P.add_legend_note(fig, ctx.t(
            "labs.auto.legend_acf",
            "Bars outside the band at low lags are the signature of serial correlation. A "
            "geometric decay pattern points specifically at an AR process.",
        ), theme=ctx.theme)
        return fig

    def _se_figure(self, ctx, fits, p, n, seed, names):
        draws = self._sampling(p, n, seed, 400, names)
        true_sd = float(np.std(draws, ddof=1))
        labels = list(fits)
        values = [fits[k].se(names[1]) for k in labels]
        fig = ctx.figure(
            "labs.auto.figure.se",
            xaxis_title=ctx.t("labs.auto.axis.method", "Covariance estimator"),
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
            "labs.auto.legend_se",
            "With a persistent regressor AND persistent errors, the classical bar can be "
            "far below the dashed line - the regression behaves as if it had far fewer "
            "independent observations than it appears to have.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _sampling(self, p, n, seed, reps, names):
        out = np.empty(reps)
        for r in range(reps):
            y, X, nm = self._generate(p, n, int(seed) * 104729 + r)
            f = LM.ols(y, X, names=nm)
            out[r] = f.coef(nm[1])
        return out

    def _size_study(self, p, n, seed, reps, names):
        reps = min(reps, 2000)
        q = dict(p)
        q["beta1"] = 0.0
        alpha = float(p["alpha"])
        counts = {"classical": 0, "hac": 0}
        if p["show_fgls"] and not p["lagged_dependent"]:
            counts["fgls"] = 0
        for r in range(reps):
            y, X, nm = self._generate(q, n, int(seed) * 15485863 + r)
            cl = LM.ols(y, X, names=nm)
            if float(cl.pvalues[1]) < alpha:
                counts["classical"] += 1
            hc = LM.ols(y, X, names=nm, cov_type="HAC",
                        maxlags=int(p["hac_lags"]) or None)
            if float(hc.pvalues[1]) < alpha:
                counts["hac"] += 1
            if "fgls" in counts:
                try:
                    g = self._fgls(y, X, cl, nm)
                    if float(g.pvalues[1]) < alpha:
                        counts["fgls"] += 1
                except Exception:  # noqa: BLE001
                    pass
        return {k: v / reps for k, v in counts.items()}

    def _size_figure(self, ctx, study, alpha):
        labels = list(study)
        values = [study[k] for k in labels]
        fig = ctx.figure(
            "labs.auto.figure.size",
            xaxis_title=ctx.t("labs.auto.axis.method", "Covariance estimator"),
            yaxis_title=ctx.t("labs.het.axis.rejection",
                              "Rejection rate when the null is TRUE"),
            height=340,
        )
        P.add_bar(fig, labels, values, ctx.t("labs.het.trace.size", "actual size"),
                  "negative", theme=ctx.theme, text=[pct(v) for v in values])
        P.add_hline(fig, alpha, ctx.t("labs.het.trace.nominal",
                                      "nominal alpha = {a}", a=fmt(alpha, 3)),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.auto.legend_size",
            "This is the practical damage: with no relationship at all, how often does each "
            "method announce one? Serial correlation is what turns a 5% procedure into a "
            "20% one.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed, names):
        go = P.require_plotly()
        rhos = np.linspace(0.0, 0.95, 16)
        frames, steps = [], []
        for i, r in enumerate(rhos):
            q = dict(p)
            q["rho"] = float(r)
            y, X, nm = self._generate(q, n, seed)
            cl = LM.ols(y, X, names=nm)
            cl.metadata["X"] = X
            hc = LM.ols(y, X, names=nm, cov_type="HAC")
            dw = LM.durbin_watson(cl.residuals)
            frames.append(go.Frame(name=f"{r:.2f}", data=[
                go.Scatter(x=np.arange(cl.residuals.size), y=cl.residuals),
            ]))
            steps.append(AnimationStep(
                id=f"rho_{i}", frame=i,
                title=ctx.t("labs.auto.anim.title", "error persistence rho = {r}",
                            r=fmt(r, 2)),
                what_you_see=ctx.t("labs.auto.anim.see",
                                   "Regression residuals plotted in time order."),
                what_changed=ctx.t("labs.auto.anim.changed",
                                   "The AR(1) coefficient of the errors moved to {r}.",
                                   r=fmt(r, 2)),
                why=ctx.t("labs.auto.anim.why",
                          "A persistent error carries part of yesterday's shock into today, "
                          "so consecutive observations repeat information instead of adding "
                          "new information."),
                interpretation=ctx.t("labs.auto.anim.interpret",
                                     "Durbin-Watson is {dw} and the HAC standard error is "
                                     "{k} times the classical one.",
                                     dw=fmt(dw, 3),
                                     k=fmt(hc.se(nm[1]) / max(cl.se(nm[1]), 1e-12), 2)),
                conclusion=ctx.t("labs.auto.anim.conclude",
                                 "The residual runs get longer, the diagnostic falls away "
                                 "from 2, and the classical standard error becomes an "
                                 "understatement."),
                warning=ctx.t("labs.auto.anim.warn",
                              "The effective sample size is falling even though n on the "
                              "screen never changes."),
                math="Var(b) with AR(1) errors is inflated by roughly (1+rho)/(1-rho) for a "
                     "persistent regressor",
                outputs={"rho": round(float(r), 3), "durbin_watson": round(dw, 4),
                         "se_ratio": round(float(hc.se(nm[1]) /
                                                 max(cl.se(nm[1]), 1e-12)), 4)},
                violated_assumptions=() if r == 0 else ("no_autocorrelation",),
                highlighted=("residual_series",),
            ))
        fig = ctx.figure(
            "labs.auto.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.residual"),
            height=380,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line={"color": ctx.color("residual"), "width": 1.8},
                                 name=ctx.t("labs.common.trace.residuals")))
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme)
        build_frames(fig, frames, duration=440, reduced_motion=ctx.reduced_motion,
                     slider_label="rho")
        return animation(
            "persistence_grows", fig, steps,
            purpose=ctx.t("labs.auto.anim.purpose",
                          "Connect the visual symptom to the diagnostic and the damage."),
            summary=ctx.t(
                "labs.auto.anim.summary",
                "Serial correlation does not move the line; it inflates how much you think "
                "you know about it. Long runs in the residual plot, a Durbin-Watson far "
                "from 2, and a HAC standard error much larger than the classical one are "
                "three views of the same shortage of independent information."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = AutocorrelationLab(SPEC)
