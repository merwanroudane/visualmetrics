"""VAR, impulse responses, variance decomposition and Granger causality."""

from __future__ import annotations

from typing import Any

from scipy import stats

from ...backends import linear as LM
from ...data.generators.timeseries import var_process
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

__all__ = ["LAB", "SPEC", "impulse_responses", "variance_decomposition"]


def impulse_responses(A: np.ndarray, chol: np.ndarray, horizon: int) -> np.ndarray:
    """Orthogonalized impulse responses of a VAR(1): psi_h = A^h * P."""
    out = np.empty((horizon + 1, A.shape[0], A.shape[0]))
    power = np.eye(A.shape[0])
    for h in range(horizon + 1):
        out[h] = power @ chol
        power = power @ A
    return out


def variance_decomposition(psi: np.ndarray) -> np.ndarray:
    """Share of each variable's forecast-error variance due to each shock."""
    cum = np.cumsum(psi**2, axis=0)
    total = cum.sum(axis=2, keepdims=True)
    return cum / np.clip(total, 1e-12, None)


SPEC = make_spec(
    "timeseries.var",
    Domain.TIMESERIES,
    "var_and_structural",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        slider("a11", 0.5, -1.2, 1.2, 0.01, group="dynamics"),
        slider("a12", 0.3, -1.2, 1.2, 0.01, group="dynamics"),
        slider("a21", 0.1, -1.2, 1.2, 0.01, group="dynamics"),
        slider("a22", 0.4, -1.2, 1.2, 0.01, group="dynamics"),
        slider("corr", 0.0, -0.95, 0.95, 0.01, group="shocks"),
        slider("sigma1", 1.0, 0.05, 5.0, 0.05, group="shocks"),
        slider("sigma2", 1.0, 0.05, 5.0, 0.05, group="shocks"),
        int_slider("n", 300, 40, 5000, 10, group="dgp"),
        int_slider("horizon", 20, 1, 60, 1, group="views"),
        select("ordering", "y1_first", ("y1_first", "y2_first"), group="identification"),
        int_slider("lags", 1, 1, 4, 1, group="estimation"),
        toggle("show_fevd", True, group="views"),
        toggle("show_granger", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", a11=0.5, a12=0.3, a21=0.1, a22=0.4),
        scenario("no_feedback", "null", a12=0.0, a21=0.0),
        scenario("one_way_causality", "positive", a12=0.6, a21=0.0),
        scenario("mutual_feedback", "strong", a12=0.5, a21=0.5),
        scenario("persistent", "strong", a11=0.9, a22=0.9, a12=0.2, a21=0.1),
        scenario("unstable", "boundary", a11=0.95, a12=0.6, a21=0.6, a22=0.95),
        scenario("correlated_shocks", "violation", corr=0.85),
        scenario("ordering_matters", "counterexample", corr=0.9, ordering="y2_first"),
        scenario("small_sample", "small_sample", n=60),
        scenario("large_sample", "large_sample", n=3000),
    ),
    prerequisites=("timeseries.arma",),
    related=("timeseries.cointegration",),
    tags=("var", "impulse response", "fevd", "granger causality", "cholesky",
          "identification"),
    aliases=("vector autoregression", "irf", "الانحدار الذاتي المتجه",
             "impulse response", "variance decomposition"),
    backends=("numpy", "statsmodels"),
    references=(
        ref("Sims, C. A. (1980). Macroeconomics and reality. Econometrica 48(1).",
            kind="paper", doi="10.2307/1912017"),
        ref("Lutkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.",
            kind="book"),
    ),
    curriculum_tags=("dz.econometrics2", "aub.econ306"),
)


class VARLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        horizon = int(p["horizon"])
        data = var_process(n=n, a11=float(p["a11"]), a12=float(p["a12"]),
                           a21=float(p["a21"]), a22=float(p["a22"]),
                           sigma1=float(p["sigma1"]), sigma2=float(p["sigma2"]),
                           corr=float(p["corr"]), seed=state.seed)
        Y = np.column_stack([data["y1"], data["y2"]])
        est = self._estimate(Y, int(p["lags"]))
        order = [0, 1] if str(p["ordering"]) == "y1_first" else [1, 0]
        chol = self._cholesky(est["sigma"], order)
        psi = impulse_responses(est["A"], chol, horizon)
        fevd = variance_decomposition(psi)
        granger = self._granger(Y, int(p["lags"]))

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "series", self._series_figure(ctx, data),
            "labs.var.figure.series", tab="data",
            evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        res.add_panel(ctx.panel(
            "irf", self._irf_figure(ctx, psi, horizon, p),
            "labs.var.figure.irf", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_fevd"]:
            res.add_panel(ctx.panel(
                "fevd", self._fevd_figure(ctx, fevd, horizon, p),
                "labs.var.figure.fevd", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_granger"]:
            res.add_panel(ctx.panel(
                "ordering", self._ordering_figure(ctx, est, horizon),
                "labs.var.figure.ordering", tab="diagnostics",
                evidence=EvidenceType.COUNTEREXAMPLE,
            ))

        eig = np.abs(np.linalg.eigvals(est["A"]))
        res.metric("max_eigenvalue", ctx.t("labs.var.metric.eig",
                                           "Largest absolute eigenvalue of A"),
                   float(eig.max()), reference=1.0,
                   note=ctx.t("labs.var.metric.eig_note",
                              "the system is stable only below 1"))
        for i in range(2):
            for j in range(2):
                res.metric(f"a{i + 1}{j + 1}",
                           ctx.t("labs.var.metric.coef",
                                 "Estimated A[{i}, {j}]", i=i + 1, j=j + 1),
                           float(est["A"][i, j]),
                           reference=float(p[f"a{i + 1}{j + 1}"]))
        res.metric("shock_correlation", ctx.t("labs.var.metric.corr",
                                              "Correlation of the reduced-form shocks"),
                   float(est["sigma"][0, 1] /
                         np.sqrt(max(est["sigma"][0, 0] * est["sigma"][1, 1], 1e-12))),
                   reference=float(p["corr"]))
        res.metric("granger_1_to_2", ctx.t("labs.var.metric.granger12",
                                           "p-value: does y1 Granger-cause y2?"),
                   granger["p_1_to_2"])
        res.metric("granger_2_to_1", ctx.t("labs.var.metric.granger21",
                                           "p-value: does y2 Granger-cause y1?"),
                   granger["p_2_to_1"])
        res.metric("peak_response", ctx.t("labs.var.metric.peak",
                                          "Peak response of y1 to a shock in y2"),
                   float(np.max(np.abs(psi[:, 0, 1]))))
        res.metric("cumulative_response", ctx.t("labs.var.metric.cumulative",
                                                "Cumulative response of y1 to a shock in y2"),
                   float(np.sum(psi[:, 0, 1])))
        res.metric("fevd_final", ctx.t("labs.var.metric.fevd",
                                       "Share of y1's variance at horizon {h} due to the "
                                       "y2 shock", h=horizon),
                   float(fevd[-1, 0, 1]))

        stable = bool(eig.max() < 1.0)
        res.assume("stationarity", ctx.t("assumptions.stationarity"), stable,
                   detail=ctx.t("labs.var.assume.stability",
                                "Stability requires every eigenvalue of A to lie inside "
                                "the unit circle."),
                   consequence="" if stable else ctx.t(
                       "labs.var.assume.unstable",
                       "An unstable VAR has impulse responses that grow without bound, so "
                       "the moving-average representation and the variance decomposition "
                       "do not exist."))
        res.assume("recursive_identification",
                   ctx.t("labs.var.assume.cholesky_label",
                         "The Cholesky ordering is a valid identifying assumption"),
                   abs(float(p["corr"])) < 0.05,
                   detail=ctx.t("labs.var.assume.cholesky",
                                "With correlated reduced-form shocks, orthogonalized "
                                "responses depend on the ordering you chose. That ordering "
                                "is an assumption about which variable cannot react within "
                                "the period - it is not estimated."),
                   consequence="" if abs(float(p["corr"])) < 0.05 else ctx.t(
                       "labs.var.assume.cholesky_consequence",
                       "Swap the ordering control and the impact responses change. Any "
                       "conclusion that flips with the ordering is a conclusion about your "
                       "assumption, not about the data."))

        res.animations.append(self._animation(ctx, est, chol, horizon))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.timeseries.var.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.timeseries.var.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.timeseries.var.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.timeseries.var.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.timeseries.var.warning"), kind="warning")

        if not stable:
            res.warnings.append(ctx.t(
                "labs.var.warn.unstable",
                "The largest eigenvalue is {e}, so this VAR is explosive. The impulse "
                "responses shown grow without limit and the variance decomposition is "
                "not meaningful.", e=fmt(float(eig.max()), 3),
            ))
        if abs(float(p["corr"])) > 0.3:
            res.warnings.append(ctx.t(
                "labs.var.warn.ordering",
                "The reduced-form shocks are correlated at {c}. Switch the ordering control "
                "and compare: the differences you see are pure identifying assumption.",
                c=fmt(float(p["corr"]), 2),
            ))
        return res

    @staticmethod
    def _estimate(Y, lags):
        n, k = Y.shape
        X = np.column_stack([np.ones(n - lags)] +
                            [Y[lags - i: n - i] for i in range(1, lags + 1)])
        coefs, resid = [], []
        for j in range(k):
            fit = LM.ols(Y[lags:, j], X)
            coefs.append(fit.coefficients)
            resid.append(fit.residuals)
        coefs = np.asarray(coefs)
        U = np.column_stack(resid)
        A = coefs[:, 1: 1 + k]
        sigma = (U.T @ U) / max(U.shape[0] - X.shape[1], 1)
        return {"A": A, "coefs": coefs, "sigma": sigma, "resid": U, "lags": lags}

    @staticmethod
    def _cholesky(sigma, order):
        perm = np.eye(sigma.shape[0])[order]
        s = perm @ sigma @ perm.T
        try:
            L = np.linalg.cholesky(s)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(s)
            L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
        return perm.T @ L @ perm

    @staticmethod
    def _granger(Y, lags):
        n = Y.shape[0]
        out = {}
        for target, source, key in ((0, 1, "p_2_to_1"), (1, 0, "p_1_to_2")):
            y = Y[lags:, target]
            own = np.column_stack([np.ones(n - lags)] +
                                  [Y[lags - i: n - i, target] for i in range(1, lags + 1)])
            full = np.column_stack([own] +
                                   [Y[lags - i: n - i, source] for i in range(1, lags + 1)])
            r_res = LM.ols(y, own)
            r_full = LM.ols(y, full)
            num = (r_res.ssr - r_full.ssr) / lags
            den = r_full.ssr / r_full.df_resid
            f = float(num / den) if den > 0 else float("nan")
            out[key] = float(stats.f.sf(f, lags, r_full.df_resid))
            out[key + "_f"] = f
        return out

    def _series_figure(self, ctx, data):
        fig = ctx.figure(
            "labs.var.figure.series",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=330,
        )
        P.add_curve(fig, data["t"], data["y1"], "y1", "primary", theme=ctx.theme)
        P.add_curve(fig, data["t"], data["y2"], "y2", "secondary", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.var.legend_series",
            "A VAR lets each variable respond to the recent past of both. Everything else "
            "in this lab is a rearrangement of the estimated coefficient matrix.",
        ), theme=ctx.theme)
        return fig

    def _irf_figure(self, ctx, psi, horizon, p):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        h = np.arange(horizon + 1)
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "y1 <- shock to y1", "y1 <- shock to y2",
            "y2 <- shock to y1", "y2 <- shock to y2",
        ))
        for i in range(2):
            for j in range(2):
                fig.add_trace(go.Scatter(
                    x=h, y=psi[:, i, j], mode="lines+markers",
                    line={"color": ctx.color("primary" if i == j else "secondary"),
                          "width": 2.6}, showlegend=False,
                ), row=i + 1, col=j + 1)
                fig.add_hline(y=0, line={"color": ctx.color("baseline"), "dash": "dot"},
                              row=i + 1, col=j + 1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=460, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.var.legend_irf",
            "Each panel traces one variable's path after a one-standard-deviation shock. In "
            "a stable system every path returns to zero; how fast, and whether it "
            "overshoots, is the whole dynamic story. Ordering used: {o}.",
            o=str(p["ordering"]).replace("_", " "),
        ), theme=ctx.theme)
        return fig

    def _fevd_figure(self, ctx, fevd, horizon, p):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        h = np.arange(horizon + 1)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.var.trace.fevd1", "forecast-error variance of y1"),
            ctx.t("labs.var.trace.fevd2", "forecast-error variance of y2"),
        ))
        for i in range(2):
            for j in range(2):
                fig.add_trace(go.Scatter(
                    x=h, y=fevd[:, i, j], mode="lines", stackgroup=f"g{i}",
                    line={"width": 0.5},
                    fillcolor=P.rgba("primary" if j == 0 else "secondary", 0.6, ctx.theme),
                    name=ctx.t("labs.var.trace.shock", "shock to y{j}", j=j + 1),
                    showlegend=i == 0,
                ), row=1, col=i + 1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=360, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.var.legend_fevd",
            "At horizon 1 a variable's forecast error is mostly its own shock; as the "
            "horizon grows, the other variable's shocks propagate in. Like the impulse "
            "responses, these shares depend on the Cholesky ordering.",
        ), theme=ctx.theme)
        return fig

    def _ordering_figure(self, ctx, est, horizon):
        h = np.arange(horizon + 1)
        fig = ctx.figure(
            "labs.var.figure.ordering",
            xaxis_title=ctx.t("labs.common.axis.horizon"),
            yaxis_title=ctx.t("labs.var.axis.response",
                              "Response of y1 to a shock in y2"),
            height=360,
        )
        for order, role, label in (
            ([0, 1], "primary", ctx.t("labs.var.trace.order1", "ordering: y1 first")),
            ([1, 0], "warning", ctx.t("labs.var.trace.order2", "ordering: y2 first")),
        ):
            chol = self._cholesky(est["sigma"], order)
            psi = impulse_responses(est["A"], chol, horizon)
            P.add_curve(fig, h, psi[:, 0, 1], label, role, theme=ctx.theme,
                        mode="lines+markers")
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.var.legend_ordering",
            "Same data, same estimated VAR, two orderings. Where the curves separate, the "
            "result is being produced by the identifying assumption rather than by the "
            "data. With uncorrelated shocks the two curves coincide exactly.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, est, chol, horizon):
        go = P.require_plotly()
        psi = impulse_responses(est["A"], chol, horizon)
        frames, steps = [], []
        for h in range(horizon + 1):
            frames.append(go.Frame(name=str(h), data=[
                go.Scatter(x=np.arange(h + 1), y=psi[: h + 1, 0, 1]),
                go.Scatter(x=np.arange(h + 1), y=psi[: h + 1, 1, 1]),
            ]))
            steps.append(AnimationStep(
                id=f"h_{h}", frame=h,
                title=ctx.t("labs.var.anim.title", "{h} periods after the shock", h=h),
                what_you_see=ctx.t("labs.var.anim.see",
                                   "The path of both variables following a single "
                                   "one-standard-deviation shock to y2."),
                what_changed=ctx.t("labs.var.anim.changed",
                                   "The horizon advanced to period {h}.", h=h),
                why=ctx.t("labs.var.anim.why",
                          "The response at horizon h is A^h times the impact matrix: each "
                          "extra period passes the effect through the coefficient matrix "
                          "once more."),
                interpretation=ctx.t("labs.var.anim.interpret",
                                     "y1 has moved {a} and y2 {b} relative to its "
                                     "no-shock path.",
                                     a=fmt(float(psi[h, 0, 1]), 4),
                                     b=fmt(float(psi[h, 1, 1]), 4)),
                conclusion=ctx.t("labs.var.anim.conclude",
                                 "A stable system absorbs the shock and returns to its "
                                 "path; the shape of the return is the dynamics."),
                warning=ctx.t("labs.var.anim.warn",
                              "The shock is orthogonalized under an assumed ordering. It is "
                              "a structural interpretation layered onto a reduced form."),
                math="psi_h = A^h P where P P' = Sigma",
                outputs={"horizon": h, "y1_response": round(float(psi[h, 0, 1]), 5),
                         "y2_response": round(float(psi[h, 1, 1]), 5)},
                active_assumptions=("stationarity", "recursive_identification"),
                highlighted=("irf_paths",),
            ))
        fig = ctx.figure(
            "labs.var.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.horizon"),
            yaxis_title=ctx.t("labs.var.axis.response", "Response to a shock in y2"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=[0], y=[psi[0, 0, 1]], mode="lines+markers",
                                 line={"color": ctx.color("primary"), "width": 2.8},
                                 name="y1"))
        fig.add_trace(go.Scatter(x=[0], y=[psi[0, 1, 1]], mode="lines+markers",
                                 line={"color": ctx.color("secondary"), "width": 2.8},
                                 name="y2"))
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        span = float(np.max(np.abs(psi[:, :, 1]))) * 1.3 + 0.1
        fig.update_xaxes(range=[0, horizon])
        fig.update_yaxes(range=[-span, span])
        build_frames(fig, frames, duration=320, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.common.axis.horizon"))
        return animation(
            "impulse_propagation", fig, steps,
            purpose=ctx.t("labs.var.anim.purpose",
                          "Watch one shock travel through a dynamic system."),
            summary=ctx.t(
                "labs.var.anim.summary",
                "An impulse response is not a forecast and not a causal effect: it is what "
                "the estimated coefficient matrix implies, given an identifying assumption "
                "you supplied. Change the ordering and the first period changes with it."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = VARLab(SPEC)
