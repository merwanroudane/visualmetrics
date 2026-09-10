"""ARCH, GARCH and volatility clustering."""

from __future__ import annotations

from typing import Any

from scipy import optimize, stats

from ...backends import linear as LM
from ...data.generators.timeseries import acf, garch_process
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
    pct,
    ref,
    scenario,
    seed_control,
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC", "fit_garch11", "arch_lm_test"]


def fit_garch11(r, leverage: bool = False) -> dict[str, Any]:
    """Quasi-maximum-likelihood GARCH(1,1), optionally with a GJR leverage term."""
    r = np.asarray(r, dtype=float)
    mu0 = float(np.mean(r))
    var0 = float(np.var(r, ddof=1))

    def negll(theta):
        mu, omega, alpha, beta = theta[:4]
        gamma = theta[4] if leverage else 0.0
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta + 0.5 * gamma >= 0.999:
            return 1e12
        e = r - mu
        h = np.empty(e.size)
        h[0] = var0
        for t in range(1, e.size):
            neg = 1.0 if e[t - 1] < 0 else 0.0
            h[t] = omega + (alpha + gamma * neg) * e[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0:
                return 1e12
        return 0.5 * float(np.sum(np.log(2 * np.pi * h) + e**2 / h))

    start = [mu0, var0 * 0.1, 0.1, 0.8] + ([0.0] if leverage else [])
    res = optimize.minimize(negll, start, method="Nelder-Mead",
                            options={"maxiter": 6000, "fatol": 1e-8})
    theta = res.x
    mu, omega, alpha, beta = theta[:4]
    gamma = theta[4] if leverage else 0.0
    e = r - mu
    h = np.empty(e.size)
    h[0] = var0
    for t in range(1, e.size):
        neg = 1.0 if e[t - 1] < 0 else 0.0
        h[t] = max(omega + (alpha + gamma * neg) * e[t - 1] ** 2 + beta * h[t - 1], 1e-12)
    persistence = alpha + beta + 0.5 * gamma
    return {
        "mu": float(mu), "omega": float(omega), "alpha": float(alpha),
        "beta": float(beta), "gamma": float(gamma), "sigma2": h,
        "persistence": float(persistence),
        "unconditional": float(omega / max(1 - persistence, 1e-9)),
        "std_resid": e / np.sqrt(h), "loglik": -float(res.fun),
        "converged": bool(res.success),
    }


def arch_lm_test(resid, lags: int = 5) -> dict[str, float]:
    """Engle's LM test for ARCH effects."""
    u2 = np.asarray(resid, dtype=float) ** 2
    n = u2.size
    X = np.column_stack([np.ones(n - lags)] +
                        [u2[lags - i: n - i] for i in range(1, lags + 1)])
    fit = LM.ols(u2[lags:], X)
    stat = (n - lags) * fit.r_squared
    return {"statistic": float(stat), "p_value": float(stats.chi2.sf(stat, lags)),
            "lags": lags}


SPEC = make_spec(
    "timeseries.garch",
    Domain.TIMESERIES,
    "volatility",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        slider("omega", 0.05, 0.001, 1.0, 0.001, group="process"),
        slider("alpha", 0.10, 0.0, 0.6, 0.005, group="process"),
        slider("beta", 0.85, 0.0, 0.99, 0.005, group="process"),
        slider("leverage", 0.0, 0.0, 0.4, 0.005, group="process"),
        slider("mu", 0.0, -1.0, 1.0, 0.01, group="process"),
        int_slider("n", 1000, 100, 20000, 50, group="dgp"),
        int_slider("arch_lags", 5, 1, 20, 1, group="tests"),
        slider("var_level", 0.01, 0.001, 0.10, 0.001, group="risk"),
        toggle("fit_leverage", False, group="estimation"),
        toggle("show_var", True, group="views"),
        toggle("show_tails", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", omega=0.05, alpha=0.10, beta=0.85),
        scenario("no_arch", "null", alpha=0.0, beta=0.0, omega=1.0),
        scenario("pure_arch", "compare_methods", alpha=0.5, beta=0.0, omega=0.5),
        scenario("highly_persistent", "boundary", alpha=0.08, beta=0.915),
        scenario("integrated_garch", "violation", alpha=0.10, beta=0.90),
        scenario("low_persistence", "weak", alpha=0.05, beta=0.4, omega=0.5),
        scenario("leverage_effect", "compare_methods", leverage=0.15, alpha=0.03,
                 beta=0.85, fit_leverage=True),
        scenario("small_sample", "small_sample", n=200),
        scenario("large_sample", "large_sample", n=8000),
        scenario("calm_market", "low_noise", omega=0.01, alpha=0.05, beta=0.8),
    ),
    prerequisites=("timeseries.arma",),
    related=("timeseries.stationarity", "econometrics.heteroskedasticity"),
    tags=("arch", "garch", "volatility clustering", "leverage", "value at risk",
          "heavy tails", "persistence"),
    aliases=("garch", "volatilite", "التقلب", "volatility clustering",
             "conditional heteroskedasticity"),
    backends=("numpy", "scipy", "arch"),
    references=(
        ref("Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. "
            "Econometrica 50(4).", kind="paper", doi="10.2307/1912773"),
        ref("Bollerslev, T. (1986). Generalized autoregressive conditional "
            "heteroskedasticity. Journal of Econometrics 31(3).", kind="paper",
            doi="10.1016/0304-4076(86)90063-1"),
    ),
    curriculum_tags=("dz.econometrics2", "aub.econ306"),
)


class GARCHLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        data = garch_process(n=n, omega=float(p["omega"]), alpha=float(p["alpha"]),
                             beta=float(p["beta"]), mu=float(p["mu"]),
                             leverage=float(p["leverage"]), seed=state.seed)
        r = data["r"]
        fit = fit_garch11(r, leverage=bool(p["fit_leverage"]))
        arch_lm = arch_lm_test(r - float(np.mean(r)), lags=int(p["arch_lags"]))

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "returns", self._returns_figure(ctx, data, fit),
            "labs.garch.figure.returns", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        res.add_panel(ctx.panel(
            "acf", self._acf_figure(ctx, r, n),
            "labs.garch.figure.acf", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_tails"]:
            res.add_panel(ctx.panel(
                "tails", self._tails_figure(ctx, r, fit),
                "labs.garch.figure.tails", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_var"]:
            res.add_panel(ctx.panel(
                "var", self._var_figure(ctx, data, fit, float(p["var_level"])),
                "labs.garch.figure.var", tab="simulation",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("alpha_hat", ctx.t("labs.garch.metric.alpha",
                                      "Estimated ARCH coefficient"), fit["alpha"],
                   reference=float(p["alpha"]))
        res.metric("beta_hat", ctx.t("labs.garch.metric.beta",
                                     "Estimated GARCH coefficient"), fit["beta"],
                   reference=float(p["beta"]))
        res.metric("omega_hat", ctx.t("labs.garch.metric.omega",
                                      "Estimated constant"), fit["omega"],
                   reference=float(p["omega"]))
        if p["fit_leverage"]:
            res.metric("gamma_hat", ctx.t("labs.garch.metric.gamma",
                                          "Estimated leverage coefficient"),
                       fit["gamma"], reference=float(p["leverage"]))
        res.metric("persistence", ctx.t("labs.garch.metric.persistence",
                                        "Persistence (alpha + beta)"), fit["persistence"],
                   reference=float(p["alpha"]) + float(p["beta"])
                   + 0.5 * float(p["leverage"]),
                   note=ctx.t("labs.garch.metric.persistence_note",
                              "at or above 1 the unconditional variance does not exist"))
        res.metric("unconditional_vol", ctx.t("labs.garch.metric.uncond",
                                              "Implied long-run volatility"),
                   float(np.sqrt(max(fit["unconditional"], 0.0))))
        res.metric("volatility_half_life", ctx.t("labs.garch.metric.half_life",
                                                 "Half-life of a volatility shock"),
                   self._half_life(fit["persistence"]))
        res.metric("arch_lm_p", ctx.t("labs.garch.metric.arch_lm",
                                      "Engle ARCH LM test p-value on the raw returns"),
                   arch_lm["p_value"],
                   note=ctx.t("labs.garch.metric.arch_lm_note",
                              "small values confirm that volatility is predictable"))
        res.metric("resid_arch_p", ctx.t("labs.garch.metric.resid_arch",
                                         "ARCH LM p-value on the standardized residuals"),
                   arch_lm_test(fit["std_resid"], lags=int(p["arch_lags"]))["p_value"],
                   note=ctx.t("labs.garch.metric.resid_arch_note",
                              "large values mean the model absorbed the clustering"))
        res.metric("return_acf1", ctx.t("labs.garch.metric.acf_r",
                                        "Lag-1 autocorrelation of returns"),
                   float(acf(r, 1)[1]), reference=0.0)
        res.metric("squared_acf1", ctx.t("labs.garch.metric.acf_r2",
                                         "Lag-1 autocorrelation of SQUARED returns"),
                   float(acf(r**2, 1)[1]),
                   note=ctx.t("labs.garch.metric.acf_r2_note",
                              "this is volatility clustering in one number"))
        res.metric("excess_kurtosis", ctx.t("labs.garch.metric.kurtosis",
                                            "Excess kurtosis of returns"),
                   float(stats.kurtosis(r, fisher=True)), reference=0.0,
                   note=ctx.t("labs.garch.metric.kurtosis_note",
                              "positive even though every innovation was normal"))

        true_persistence = float(p["alpha"] + p["beta"] + 0.5 * p.get("gamma", 0.0))
        stationary = fit["persistence"] < 1.0 and true_persistence < 1.0
        res.assume("stationarity", ctx.t("labs.garch.assume.stationary_label",
                                         "The variance process is stationary"),
                   stationary,
                   detail=ctx.t("labs.garch.assume.stationary",
                                "alpha + beta must be below 1 for an unconditional "
                                "variance to exist."),
                   consequence="" if stationary else ctx.t(
                       "labs.garch.assume.integrated",
                       "At persistence 1 the model is integrated GARCH: shocks to "
                       "volatility never die out and long-horizon variance forecasts "
                       "diverge."))
        res.assume("mean_unpredictable",
                   ctx.t("labs.garch.assume.mean_label",
                         "Returns are unpredictable in level"),
                   abs(float(acf(r, 1)[1])) < 3 / np.sqrt(n),
                   detail=ctx.t("labs.garch.assume.mean",
                                "GARCH models the variance, not the mean. Predictable "
                                "volatility does not imply predictable returns."))
        res.assume("normal_errors", ctx.t("labs.garch.assume.innov_label",
                                          "The standardized innovations are normal"),
                   float(stats.kurtosis(fit["std_resid"], fisher=True)) < 1.0,
                   detail=ctx.t("labs.garch.assume.innov",
                                "If the standardized residuals still have heavy tails, a "
                                "Student-t innovation would fit better and the normal-based "
                                "Value at Risk would understate the tail."))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.timeseries.garch.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.timeseries.garch.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.timeseries.garch.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.timeseries.garch.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.timeseries.garch.warning"), kind="warning")

        if true_persistence >= 1.0:
            res.warnings.append(ctx.t(
                "labs.garch.warn.integrated",
                "The simulated process has alpha + beta = {t}, so it is integrated "
                "GARCH: no unconditional variance exists and volatility shocks never "
                "decay. The estimate came out at {e} because maximum likelihood is "
                "biased downward here - reading that estimate as evidence of "
                "stationarity is exactly the mistake this scenario is built to expose.",
                t=fmt(true_persistence, 3), e=fmt(fit["persistence"], 4),
            ))
        elif fit["persistence"] > 0.99:
            res.warnings.append(ctx.t(
                "labs.garch.warn.persistence",
                "Estimated persistence is {v}. This close to one, the unconditional "
                "variance is effectively undefined and long-horizon volatility forecasts "
                "stop being meaningful - a common and easily missed result in practice.",
                v=fmt(fit["persistence"], 4),
            ))
        if not fit["converged"]:
            res.warnings.append(ctx.t(
                "labs.garch.warn.convergence",
                "The likelihood optimizer did not report convergence. Treat the coefficients "
                "below as unreliable rather than as estimates.",
            ))
        return res

    @staticmethod
    def _half_life(persistence):
        if not (0 < persistence < 1):
            return float("inf")
        return float(np.log(0.5) / np.log(persistence))

    def _returns_figure(self, ctx, data, fit):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        t = data["t"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                            subplot_titles=(
                                ctx.t("labs.garch.trace.returns", "returns"),
                                ctx.t("labs.garch.trace.vol",
                                      "conditional volatility"),
                            ))
        fig.add_trace(go.Scatter(x=t, y=data["r"], mode="lines",
                                 line={"color": ctx.color("primary"), "width": 1.0},
                                 name=ctx.t("labs.garch.trace.returns", "returns")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=data["sigma"], mode="lines",
                                 line={"color": ctx.color("truth"), "width": 2.0},
                                 name=ctx.t("labs.garch.trace.true_vol",
                                            "true volatility")), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=np.sqrt(fit["sigma2"]), mode="lines",
                                 line={"color": ctx.color("secondary"), "width": 1.8,
                                       "dash": "dash"},
                                 name=ctx.t("labs.garch.trace.fitted_vol",
                                            "GARCH-fitted volatility")), row=2, col=1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=450, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.garch.legend_returns",
            "Calm and turbulent stretches alternate in the top panel, and the bottom panel "
            "is the model's read of how turbulent each day was. The true volatility is "
            "visible only because this is a simulation.",
        ), theme=ctx.theme)
        return fig

    def _acf_figure(self, ctx, r, n):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        nlags = 24
        band = 1.96 / np.sqrt(n)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.garch.trace.acf_r", "ACF of returns"),
            ctx.t("labs.garch.trace.acf_r2", "ACF of squared returns"),
        ))
        for col, series in ((1, acf(r, nlags)), (2, acf(r**2, nlags))):
            fig.add_trace(go.Bar(x=np.arange(nlags + 1), y=series,
                                 marker={"color": ctx.color("primary" if col == 1
                                                            else "warning")},
                                 showlegend=False), row=1, col=col)
            for sign in (1, -1):
                fig.add_hline(y=sign * band, line={"color": ctx.color("muted"),
                                                   "dash": "dot"}, row=1, col=col)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=350, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.garch.legend_acf",
            "This pair of pictures is the whole point: returns are nearly unpredictable "
            "(left) while their magnitudes are strongly predictable (right).",
        ), theme=ctx.theme)
        return fig

    def _tails_figure(self, ctx, r, fit):
        z = (r - fit["mu"]) / float(np.std(r, ddof=1))
        std_resid = fit["std_resid"]
        grid = np.linspace(-6, 6, 400)
        fig = ctx.figure(
            "labs.garch.figure.tails",
            xaxis_title=ctx.t("labs.garch.axis.standardized",
                              "Standardized value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=370,
        )
        P.add_histogram(fig, z[np.abs(z) < 6],
                        ctx.t("labs.garch.trace.raw",
                              "returns standardized by the CONSTANT sd"),
                        "warning", theme=ctx.theme, nbins=70, opacity=0.5)
        P.add_histogram(fig, std_resid[np.abs(std_resid) < 6],
                        ctx.t("labs.garch.trace.std",
                              "returns standardized by the CONDITIONAL sd"),
                        "primary", theme=ctx.theme, nbins=70, opacity=0.5)
        P.add_curve(fig, grid, stats.norm.pdf(grid),
                    ctx.t("labs.common.trace.normal_ref"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.garch.legend_tails",
            "The heavy tails of financial returns are largely an artefact of mixing calm "
            "and turbulent periods. Divide by the RIGHT day's volatility and the histogram "
            "collapses back towards the normal curve.",
        ), theme=ctx.theme)
        return fig

    def _var_figure(self, ctx, data, fit, level):
        z = float(stats.norm.ppf(level))
        cond_var = fit["mu"] + z * np.sqrt(fit["sigma2"])
        uncond_var = fit["mu"] + z * float(np.std(data["r"], ddof=1))
        r = data["r"]
        breaches_cond = int(np.sum(r < cond_var))
        breaches_uncond = int(np.sum(r < uncond_var))
        fig = ctx.figure(
            "labs.garch.figure.var",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.return"),
            height=380,
        )
        P.add_curve(fig, data["t"], r, ctx.t("labs.garch.trace.returns", "returns"),
                    "muted", theme=ctx.theme, width=1.0)
        P.add_curve(fig, data["t"], cond_var,
                    ctx.t("labs.garch.trace.var_cond",
                          "{l} Value at Risk from the GARCH model", l=pct(level, 0)),
                    "primary", theme=ctx.theme, width=2.0)
        P.add_hline(fig, uncond_var,
                    ctx.t("labs.garch.trace.var_uncond",
                          "{l} Value at Risk from a constant variance", l=pct(level, 0)),
                    "negative", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.garch.legend_var",
            "A conditional risk limit tightens in turbulent periods and relaxes in calm "
            "ones. Breaches: {a} with GARCH, {b} with a fixed variance, against an expected "
            "{e} at this level.",
            a=breaches_cond, b=breaches_uncond, e=int(round(level * r.size)),
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        betas = np.linspace(0.0, 0.96, 16)
        n_anim = min(n, 800)
        frames, steps = [], []
        for i, b in enumerate(betas):
            alpha = min(float(p["alpha"]), max(0.98 - b, 0.005))
            d = garch_process(n=n_anim, omega=float(p["omega"]), alpha=alpha,
                              beta=float(b), seed=seed)
            persistence = alpha + b
            frames.append(go.Frame(name=f"{b:.2f}", data=[
                go.Scatter(x=d["t"], y=d["r"]),
                go.Scatter(x=d["t"], y=d["sigma"]),
            ]))
            steps.append(AnimationStep(
                id=f"beta_{i}", frame=i,
                title=ctx.t("labs.garch.anim.title", "GARCH coefficient = {b}",
                            b=fmt(b, 2)),
                what_you_see=ctx.t("labs.garch.anim.see",
                                   "Simulated returns above their own conditional "
                                   "volatility path."),
                what_changed=ctx.t("labs.garch.anim.changed",
                                   "beta moved to {b}, so persistence is now {pp}.",
                                   b=fmt(b, 2), pp=fmt(persistence, 3)),
                why=ctx.t("labs.garch.anim.why",
                          "beta is how much of yesterday's variance carries into today. "
                          "Larger values make volatility regimes last longer."),
                interpretation=ctx.t("labs.garch.anim.interpret",
                                     "Squared-return autocorrelation at lag 1 is {a}; the "
                                     "volatility half-life is {h} periods.",
                                     a=fmt(float(acf(d["r"] ** 2, 1)[1]), 3),
                                     h=("infinite" if not np.isfinite(
                                         self._half_life(persistence))
                                        else fmt(self._half_life(persistence), 1))),
                conclusion=ctx.t("labs.garch.anim.conclude",
                                 "Clustering is a property of the variance equation. The "
                                 "mean of the returns never changed in any frame."),
                warning=ctx.t("labs.garch.anim.warn",
                              "As persistence approaches one, the unconditional variance "
                              "diverges and the process approaches integrated GARCH."),
                math="sigma2_t = omega + alpha e2_(t-1) + beta sigma2_(t-1)",
                outputs={"beta": round(float(b), 3),
                         "persistence": round(float(persistence), 4)},
                violated_assumptions=("stationarity",) if persistence >= 0.999 else (),
                highlighted=("returns", "volatility"),
            ))
        make_subplots = P.SUBPLOT()
        d0 = garch_process(n=n_anim, omega=float(p["omega"]), alpha=float(p["alpha"]),
                           beta=0.0, seed=seed)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=(
                                ctx.t("labs.garch.trace.returns", "returns"),
                                ctx.t("labs.garch.trace.vol", "conditional volatility"),
                            ))
        fig.add_trace(go.Scatter(x=d0["t"], y=d0["r"], mode="lines",
                                 line={"color": ctx.color("primary"), "width": 1.0},
                                 name=ctx.t("labs.garch.trace.returns", "returns")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d0["t"], y=d0["sigma"], mode="lines",
                                 line={"color": ctx.color("truth"), "width": 2.0},
                                 name=ctx.t("labs.garch.trace.true_vol",
                                            "true volatility")), row=2, col=1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=430, **layout)
        build_frames(fig, frames, duration=460, reduced_motion=ctx.reduced_motion,
                     slider_label="beta")
        return animation(
            "clustering_emerges", fig, steps,
            purpose=ctx.t("labs.garch.anim.purpose",
                          "Turn one coefficient into visible volatility clustering."),
            summary=ctx.t(
                "labs.garch.anim.summary",
                "Volatility clustering is not an exotic phenomenon: one persistence "
                "parameter produces it, and with it the heavy tails and the predictable "
                "risk that make conditional models worth fitting. Returns remain "
                "unpredictable in level throughout."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = GARCHLab(SPEC)
