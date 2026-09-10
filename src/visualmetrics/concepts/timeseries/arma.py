"""ARMA modelling and the Box-Jenkins identification loop."""

from __future__ import annotations

from typing import Any

from scipy import optimize, stats

from ...backends import linear as LM
from ...data.generators.timeseries import acf, arma_process, pacf
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

__all__ = ["LAB", "SPEC", "theoretical_acf", "fit_arma_css"]


def theoretical_acf(ar, ma, nlags: int = 24) -> np.ndarray:
    """Theoretical ACF of an ARMA process via its MA(infinity) representation."""
    psi = np.zeros(400)
    psi[0] = 1.0
    for j in range(1, psi.size):
        val = ma[j - 1] if j - 1 < len(ma) else 0.0
        for i, phi in enumerate(ar, start=1):
            if j - i >= 0:
                val += phi * psi[j - i]
        psi[j] = val
    gamma = np.array([float(np.sum(psi[: psi.size - k] * psi[k:])) for k in range(nlags + 1)])
    return gamma / gamma[0] if gamma[0] > 0 else gamma


def fit_arma_css(y, p_order: int, q_order: int, burn: int | None = None) -> dict[str, Any]:
    """Conditional sum-of-squares estimation of an ARMA(p, q) model.

    ``burn`` fixes how many initial observations are discarded. Model-selection
    criteria are only comparable when every candidate is scored on the SAME
    observations, so the search passes a common value rather than letting each
    model choose its own.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    mean = float(np.mean(y))
    z = y - mean
    burn = max(p_order, q_order) if burn is None else max(burn, max(p_order, q_order))

    def residuals(theta):
        ar = theta[:p_order]
        ma = theta[p_order:]
        e = np.zeros(n)
        for t in range(n):
            pred = 0.0
            for i, phi in enumerate(ar, start=1):
                if t - i >= 0:
                    pred += phi * z[t - i]
            for j, th in enumerate(ma, start=1):
                if t - j >= 0:
                    pred += th * e[t - j]
            e[t] = z[t] - pred
        return e

    if p_order == 0 and q_order == 0:
        theta = np.zeros(0)
        e = z.copy()
    elif q_order == 0:
        # pure AR: conditional least squares has an exact closed form
        cols = [z[burn - i: n - i] for i in range(1, p_order + 1)]
        fit = LM.ols(z[burn:], np.column_stack(cols), has_constant=False)
        theta = fit.coefficients
        e = residuals(theta)
    else:
        if p_order:
            cols = [z[burn - i: n - i] for i in range(1, p_order + 1)]
            ar_start = LM.ols(z[burn:], np.column_stack(cols),
                              has_constant=False).coefficients
        else:
            ar_start = np.zeros(0)
        start = np.concatenate([ar_start, np.full(q_order, 0.1)])

        def objective(theta):
            return float(np.sum(residuals(theta)[burn:] ** 2))

        best = optimize.minimize(objective, start, method="Nelder-Mead",
                                 options={"maxiter": 4000, "xatol": 1e-7,
                                          "fatol": 1e-9})
        best = optimize.minimize(objective, best.x, method="Nelder-Mead",
                                 options={"maxiter": 4000, "xatol": 1e-8,
                                          "fatol": 1e-10})
        theta = best.x
        e = residuals(theta)

    eff = n - burn
    ssr = float(np.sum(e[burn:] ** 2))
    k = p_order + q_order + 1
    sigma2 = ssr / max(eff, 1)
    loglik = -0.5 * eff * (np.log(2 * np.pi * sigma2) + 1.0)
    return {
        "params": theta, "residuals": e, "sigma2": sigma2, "loglik": float(loglik),
        "aic": float(-2 * loglik + 2 * k), "bic": float(-2 * loglik + k * np.log(eff)),
        "p": p_order, "q": q_order, "mean": mean, "burn": burn,
    }


SPEC = make_spec(
    "timeseries.arma",
    Domain.TIMESERIES,
    "arima_family",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        slider("ar1", 0.6, -0.98, 0.98, 0.01, group="process"),
        slider("ar2", 0.0, -0.98, 0.98, 0.01, group="process"),
        slider("ma1", 0.0, -0.98, 0.98, 0.01, group="process"),
        slider("ma2", 0.0, -0.98, 0.98, 0.01, group="process"),
        int_slider("n", 300, 30, 5000, 10, group="dgp"),
        slider("sigma", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        int_slider("nlags", 24, 5, 60, 1, group="views"),
        int_slider("fit_p", 1, 0, 4, 1, group="estimation"),
        int_slider("fit_q", 0, 0, 4, 1, group="estimation"),
        toggle("show_model_search", True, group="estimation"),
        toggle("show_forecast", True, group="views"),
        int_slider("horizon", 20, 1, 100, 1, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("ar1", "canonical", ar1=0.6, ma1=0.0, fit_p=1, fit_q=0),
        scenario("white_noise", "null", ar1=0.0, ma1=0.0, fit_p=0, fit_q=0),
        scenario("ma1", "compare_methods", ar1=0.0, ma1=0.7, fit_p=0, fit_q=1),
        scenario("arma11", "compare_methods", ar1=0.5, ma1=0.4, fit_p=1, fit_q=1),
        scenario("ar2_cyclical", "compare_methods", ar1=1.2, ar2=-0.7, fit_p=2, fit_q=0),
        scenario("persistent", "strong", ar1=0.95, fit_p=1),
        scenario("near_non_invertible", "boundary", ar1=0.0, ma1=0.98, fit_p=0, fit_q=1),
        scenario("negative_ar", "negative", ar1=-0.7, fit_p=1),
        scenario("overfitted", "misspecification", ar1=0.6, fit_p=4, fit_q=4),
        scenario("underfitted", "misspecification", ar1=0.5, ma1=0.6, fit_p=1, fit_q=0),
        scenario("small_sample", "small_sample", n=50),
        scenario("large_sample", "large_sample", n=3000),
    ),
    prerequisites=("timeseries.stationarity",),
    related=("timeseries.var", "econometrics.autocorrelation"),
    next_concepts=("timeseries.cointegration", "timeseries.var"),
    tags=("ar", "ma", "arma", "acf", "pacf", "box-jenkins", "aic", "invertibility"),
    aliases=("arima", "box jenkins", "بوكس جنكينز", "autocorrelation function",
             "partial autocorrelation"),
    backends=("numpy", "scipy", "statsmodels"),
    references=(
        ref("Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (2008). Time Series "
            "Analysis: Forecasting and Control.", kind="book"),
        ref("Hamilton, J. D. (1994). Time Series Analysis.", kind="book"),
    ),
    curriculum_tags=("dz.econometrics2", "cairo.eviews"),
)


class ARMALab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        ar = tuple(v for v in (float(p["ar1"]), float(p["ar2"])) if abs(v) > 1e-9)
        ma = tuple(v for v in (float(p["ma1"]), float(p["ma2"])) if abs(v) > 1e-9)
        n = int(p["n"])
        data = arma_process(n=n, ar=ar, ma=ma, sigma=float(p["sigma"]), seed=state.seed)
        y = data["y"]
        nlags = int(p["nlags"])

        fit = fit_arma_css(y, int(p["fit_p"]), int(p["fit_q"]))
        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "series", self._series_figure(ctx, data, fit, p),
            "labs.arma.figure.series", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        res.add_panel(ctx.panel(
            "acf_pacf", self._acf_pacf_figure(ctx, y, ar, ma, nlags, n),
            "labs.arma.figure.acf", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "residual_diagnostics", self._residual_figure(ctx, fit, n, nlags),
            "labs.arma.figure.residuals", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_model_search"]:
            table = self._model_search(y)
            res.add_panel(ctx.panel(
                "model_search", self._search_figure(ctx, table),
                "labs.arma.figure.search", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
            res.tables["model_search"] = table
            best = min(table, key=lambda row: row["aic"])
            res.metric("best_aic_model", ctx.t("labs.arma.metric.best",
                                               "Model selected by AIC"),
                       f"ARMA({best['p']}, {best['q']})",
                       note=ctx.t("labs.arma.metric.best_note",
                                  "the true process is ARMA({p}, {q})",
                                  p=len(ar), q=len(ma)))

        res.metric("stationary", ctx.t("labs.arma.metric.stationary",
                                       "Process is stationary"),
                   ctx.t("labs.arma.yes", "yes") if data.meta["stationary"]
                   else ctx.t("labs.arma.no", "no"))
        res.metric("invertible", ctx.t("labs.arma.metric.invertible",
                                       "MA part is invertible"),
                   ctx.t("labs.arma.yes", "yes") if self._invertible(ma)
                   else ctx.t("labs.arma.no", "no"),
                   note=ctx.t("labs.arma.metric.invertible_note",
                              "invertibility is what lets an MA model be written as an "
                              "infinite AR, and it is required for identification"))
        for i, v in enumerate(fit["params"][: fit["p"]], start=1):
            res.metric(f"ar{i}_hat", ctx.t("labs.arma.metric.ar",
                                           "Estimated AR({i})", i=i), float(v),
                       reference=ar[i - 1] if i <= len(ar) else 0.0)
        for j, v in enumerate(fit["params"][fit["p"]:], start=1):
            res.metric(f"ma{j}_hat", ctx.t("labs.arma.metric.ma",
                                           "Estimated MA({j})", j=j), float(v),
                       reference=ma[j - 1] if j <= len(ma) else 0.0)
        res.metric("sigma2_hat", ctx.t("labs.arma.metric.sigma2",
                                       "Residual variance"), fit["sigma2"],
                   reference=float(p["sigma"]) ** 2)
        res.metric("aic", ctx.t("labs.arma.metric.aic", "AIC"), fit["aic"])
        res.metric("bic", ctx.t("labs.arma.metric.bic", "BIC"), fit["bic"])
        lb = self._ljung_box(fit["residuals"], min(10, nlags),
                            fit["p"] + fit["q"])
        res.metric("ljung_box_p", ctx.t("labs.arma.metric.lb",
                                        "Ljung-Box p-value on the residuals"),
                   lb, note=ctx.t("labs.arma.metric.lb_note",
                                  "large values mean no autocorrelation is left - which is "
                                  "what an adequate model should achieve"))

        res.assume("stationarity", ctx.t("assumptions.stationarity"),
                   bool(data.meta["stationary"]),
                   detail=ctx.t("labs.arma.assume.stationarity",
                                "The AR polynomial's roots must lie outside the unit "
                                "circle for the process to be stationary."))
        res.assume("invertibility", ctx.t("labs.arma.assume.invertible_label",
                                          "The MA polynomial is invertible"),
                   self._invertible(ma),
                   detail=ctx.t("labs.arma.assume.invertible",
                                "Without invertibility, two different MA parameters give "
                                "exactly the same autocovariances, so the model is not "
                                "identified from the data."))
        res.assume("adequacy", ctx.t("labs.arma.assume.adequacy_label",
                                     "The fitted model leaves white-noise residuals"),
                   lb > 0.05,
                   detail=ctx.t("labs.arma.assume.adequacy",
                                "The Ljung-Box test on the residuals is the diagnostic "
                                "step of the Box-Jenkins loop."))

        res.animations.append(self._animation(ctx, p, n, state.seed, nlags))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.timeseries.arma.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.timeseries.arma.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.timeseries.arma.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.timeseries.arma.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.timeseries.arma.warning"), kind="warning")
        res.explain("diagnostics", ctx.t("labs.arma.loop.title",
                                         "The Box-Jenkins loop"), ctx.t(
            "labs.arma.loop",
            "1. Identify: read the ACF and PACF shapes to propose orders.\n"
            "2. Estimate: fit the proposed ARMA({p}, {q}).\n"
            "3. Diagnose: Ljung-Box on the residuals gives p = {lb}.\n"
            "4. Iterate if the residuals are not white noise; stop when they are and no "
            "simpler model does as well.\n"
            "Information criteria help at step 4, but only among models fitted to the same "
            "data - never across different degrees of differencing.",
            p=fit["p"], q=fit["q"], lb=fmt(lb, 4),
        ))

        if not self._invertible(ma):
            res.warnings.append(ctx.t(
                "labs.arma.warn.invertibility",
                "The MA coefficient is at or beyond the invertibility boundary. Two "
                "different parameter values produce identical autocovariances, so no "
                "estimator can distinguish them from the data alone.",
            ))
        if fit["p"] + fit["q"] > len(ar) + len(ma) + 1:
            res.warnings.append(ctx.t(
                "labs.arma.warn.overfit",
                "The fitted model has more parameters than the process that generated the "
                "data. Extra lags will fit sampling noise, which shows up as worse "
                "forecasts even though the in-sample fit improves.",
            ))
        return res

    @staticmethod
    def _invertible(ma):
        if not ma:
            return True
        roots = np.roots(np.r_[1.0, np.asarray(ma, dtype=float)][::-1])
        return bool(np.all(np.abs(roots) > 1.0)) if roots.size else True

    @staticmethod
    def _ljung_box(resid, lags, dof):
        r = np.asarray(resid, dtype=float)
        n = r.size
        a = acf(r, lags)
        stat = n * (n + 2) * float(np.sum([a[k] ** 2 / (n - k) for k in range(1, lags + 1)]))
        df = max(lags - dof, 1)
        return float(stats.chi2.sf(stat, df))

    def _model_search(self, y):
        rows = []
        for pp in range(3):
            for qq in range(3):
                try:
                    f = fit_arma_css(y, pp, qq, burn=2)
                    rows.append({"p": pp, "q": qq, "aic": f["aic"], "bic": f["bic"],
                                 "sigma2": f["sigma2"]})
                except Exception:
                    continue
        return rows

    def _series_figure(self, ctx, data, fit, p):
        y = data["y"]
        t = data["t"]
        fitted = y - fit["residuals"]
        fig = ctx.figure(
            "labs.arma.figure.series",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=400,
        )
        P.add_curve(fig, t, y, ctx.t("labs.common.trace.data"), "primary",
                    theme=ctx.theme)
        P.add_curve(fig, t, fitted,
                    ctx.t("labs.arma.trace.fitted", "one-step-ahead fitted values"),
                    "fitted", theme=ctx.theme, dash="dash")
        if p["show_forecast"]:
            fc, lo, hi = self._forecast(y, fit, int(p["horizon"]))
            future = np.arange(t[-1] + 1, t[-1] + 1 + fc.size, dtype=float)
            P.add_curve(fig, future, fc,
                        ctx.t("labs.arma.trace.forecast", "forecast"),
                        "secondary", theme=ctx.theme, width=3.0)
            P.shade_between(fig, future, lo, hi,
                            ctx.t("labs.arma.trace.forecast_band",
                                  "95% forecast interval"),
                            "secondary", theme=ctx.theme, alpha=0.18)
        P.add_legend_note(fig, ctx.t(
            "labs.arma.legend_series",
            "Forecasts from a stationary ARMA model decay towards the mean, and the interval "
            "widens to the unconditional standard deviation. A model that forecasts a trend "
            "forever is not stationary.",
        ), theme=ctx.theme)
        return fig

    @staticmethod
    def _forecast(y, fit, horizon):
        mean = fit["mean"]
        z = y - mean
        e = fit["residuals"]
        pp, qq = fit["p"], fit["q"]
        ar = fit["params"][:pp]
        ma = fit["params"][pp:]
        hist = list(z[-max(pp, 1):]) if pp else []
        errs = list(e[-max(qq, 1):]) if qq else []
        out = []
        for h in range(horizon):
            val = 0.0
            for i, phi in enumerate(ar, start=1):
                val += phi * (out[-i] if i <= len(out) else hist[-i + len(out)])
            for j, th in enumerate(ma, start=1):
                if h < qq and j <= len(errs):
                    val += th * errs[-j]
            out.append(val)
        fc = np.asarray(out) + mean
        psi = np.zeros(horizon)
        psi[0] = 1.0
        for j in range(1, horizon):
            v = ma[j - 1] if j - 1 < len(ma) else 0.0
            for i, phi in enumerate(ar, start=1):
                if j - i >= 0:
                    v += phi * psi[j - i]
            psi[j] = v
        se = np.sqrt(fit["sigma2"] * np.cumsum(psi**2))
        return fc, fc - 1.96 * se, fc + 1.96 * se

    def _acf_pacf_figure(self, ctx, y, ar, ma, nlags, n):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        sample_acf = acf(y, nlags)
        sample_pacf = pacf(y, nlags)
        theo = theoretical_acf(ar, ma, nlags)
        band = 1.96 / np.sqrt(n)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.arma.trace.acf", "autocorrelation (ACF)"),
            ctx.t("labs.arma.trace.pacf", "partial autocorrelation (PACF)"),
        ))
        fig.add_trace(go.Bar(x=np.arange(nlags + 1), y=sample_acf,
                             marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.common.trace.empirical")), row=1, col=1)
        fig.add_trace(go.Scatter(x=np.arange(nlags + 1), y=theo, mode="lines+markers",
                                 line={"color": ctx.color("truth"), "dash": "dash"},
                                 name=ctx.t("labs.common.trace.theoretical")),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=np.arange(nlags + 1), y=sample_pacf,
                             marker={"color": ctx.color("secondary")},
                             name=ctx.t("labs.arma.trace.pacf_short", "PACF")),
                      row=1, col=2)
        for col in (1, 2):
            for sign in (1, -1):
                fig.add_hline(y=sign * band, line={"color": ctx.color("muted"),
                                                   "dash": "dot"}, row=1, col=col)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=380, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.arma.legend_acf",
            "AR processes: ACF decays, PACF cuts off after lag p. MA processes: ACF cuts "
            "off after lag q, PACF decays. ARMA: both decay - which is why mixed models are "
            "hard to identify by eye.",
        ), theme=ctx.theme)
        return fig

    def _residual_figure(self, ctx, fit, n, nlags):
        r = fit["residuals"]
        a = acf(r, min(nlags, 24))
        band = 1.96 / np.sqrt(n)
        fig = ctx.figure(
            "labs.arma.figure.residuals",
            xaxis_title=ctx.t("labs.common.axis.lag"),
            yaxis_title=ctx.t("labs.common.axis.autocorrelation"),
            height=340,
        )
        P.add_bar(fig, np.arange(a.size), a,
                  ctx.t("labs.arma.trace.resid_acf", "residual ACF"),
                  "primary", theme=ctx.theme)
        P.shade_between(fig, np.arange(a.size), np.full(a.size, -band),
                        np.full(a.size, band),
                        ctx.t("labs.arma.trace.band", "white-noise band"),
                        "muted", theme=ctx.theme, alpha=0.2)
        P.add_legend_note(fig, ctx.t(
            "labs.arma.legend_residuals",
            "This is the diagnostic step: if bars still poke outside the band, structure "
            "remains and the model is not yet adequate.",
        ), theme=ctx.theme)
        return fig

    def _search_figure(self, ctx, table):
        go = P.require_plotly()
        ps = sorted({row["p"] for row in table})
        qs = sorted({row["q"] for row in table})
        z = np.full((len(qs), len(ps)), np.nan)
        for row in table:
            z[qs.index(row["q"]), ps.index(row["p"])] = row["aic"]
        fig = ctx.figure(
            "labs.arma.figure.search",
            xaxis_title="AR order p", yaxis_title="MA order q", height=350,
        )
        fig.add_trace(go.Heatmap(z=z, x=ps, y=qs, colorscale=ctx.theme.colorscale,
                                 reversescale=True,
                                 text=[[fmt(v, 1) for v in row] for row in z],
                                 texttemplate="%{text}",
                                 name="AIC"))
        P.add_legend_note(fig, ctx.t(
            "labs.arma.legend_search",
            "Lower AIC is better. Information criteria are comparable only across models "
            "fitted to the same observations - never compare a levels model with a "
            "differenced one this way.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed, nlags):
        go = P.require_plotly()
        values = np.linspace(-0.95, 0.95, 20)
        frames, steps = [], []
        for i, v in enumerate(values):
            ar = (float(v),) if abs(v) > 1e-9 else ()
            ma = tuple(m for m in (float(p["ma1"]),) if abs(m) > 1e-9)
            d = arma_process(n=n, ar=ar, ma=ma, sigma=float(p["sigma"]), seed=seed)
            sample_acf = acf(d["y"], nlags)
            frames.append(go.Frame(name=f"{v:.2f}", data=[
                go.Scatter(x=d["t"], y=d["y"]),
                go.Bar(x=np.arange(nlags + 1), y=sample_acf),
            ]))
            steps.append(AnimationStep(
                id=f"ar_{i}", frame=i,
                title=ctx.t("labs.arma.anim.title", "AR(1) coefficient = {v}",
                            v=fmt(v, 2)),
                what_you_see=ctx.t("labs.arma.anim.see",
                                   "The realized series above and its sample "
                                   "autocorrelation function below."),
                what_changed=ctx.t("labs.arma.anim.changed",
                                   "The AR(1) coefficient moved to {v}.", v=fmt(v, 2)),
                why=ctx.t("labs.arma.anim.why",
                          "The theoretical ACF of an AR(1) is rho^k. Positive coefficients "
                          "give smooth decay; negative ones give an alternating pattern."),
                interpretation=ctx.t("labs.arma.anim.interpret",
                                     "Lag-1 autocorrelation is {a}; the series looks {look}.",
                                     a=fmt(float(sample_acf[1]), 3),
                                     look=("smooth and persistent" if v > 0.3 else
                                           ("jagged and alternating" if v < -0.3
                                            else "close to white noise"))),
                conclusion=ctx.t("labs.arma.anim.conclude",
                                 "The shape of the ACF is the fingerprint you use to "
                                 "identify the model order."),
                warning=ctx.t("labs.arma.anim.warn",
                              "The sample ACF is an estimate. Individual bars wobble around "
                              "the theoretical curve even when the model is exactly right."),
                math="ACF of AR(1): rho_k = phi^k",
                outputs={"ar1": round(float(v), 3),
                         "sample_acf1": round(float(sample_acf[1]), 4)},
                highlighted=("series", "acf"),
            ))
        make_subplots = P.SUBPLOT()
        fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45],
                            vertical_spacing=0.12,
                            subplot_titles=(
                                ctx.t("labs.arma.trace.series", "series"),
                                ctx.t("labs.arma.trace.acf", "autocorrelation (ACF)"),
                            ))
        d0 = arma_process(n=n, ar=(float(values[0]),), sigma=float(p["sigma"]), seed=seed)
        fig.add_trace(go.Scatter(x=d0["t"], y=d0["y"], mode="lines",
                                 line={"color": ctx.color("primary")},
                                 name=ctx.t("labs.common.trace.data")), row=1, col=1)
        fig.add_trace(go.Bar(x=np.arange(nlags + 1), y=acf(d0["y"], nlags),
                             marker={"color": ctx.color("secondary")},
                             name=ctx.t("labs.arma.trace.acf_short", "ACF")), row=2, col=1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=470, **layout)
        fig.update_yaxes(range=[-1.05, 1.05], row=2, col=1)
        build_frames(fig, frames, duration=460, reduced_motion=ctx.reduced_motion,
                     slider_label="AR(1)")
        return animation(
            "acf_fingerprint", fig, steps,
            purpose=ctx.t("labs.arma.anim.purpose",
                          "Link a parameter value to the ACF shape it produces."),
            summary=ctx.t(
                "labs.arma.anim.summary",
                "Box-Jenkins identification is pattern recognition on the ACF and PACF. "
                "Learning the fingerprints - geometric decay for AR, sharp cut-off for MA - "
                "is what makes the first step of the loop possible at all."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = ARMALab(SPEC)
