"""Stationarity, unit roots and spurious regression."""

from __future__ import annotations

from typing import Any

from scipy import stats

from ...backends import linear as LM
from ...data.generators.timeseries import acf, ar_process, structural_break_series, trend_stationary
from ...simulation.random import rng
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
    select,
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC", "adf_test"]

PROCESSES = ("ar1", "trend_stationary", "structural_break")


def adf_test(y, lags: int = 1, trend: str = "c") -> dict[str, float]:
    """Augmented Dickey-Fuller test with response-surface critical values.

    The statistic is the t ratio on the lagged level in
    ``delta y_t = a + rho_star y_(t-1) + sum(gamma_i delta y_(t-i)) + e_t``.
    Critical values follow MacKinnon (1996); they are NOT normal quantiles.
    """
    y = np.asarray(y, dtype=float).ravel()
    dy = np.diff(y)
    n = dy.size - lags
    if n < 10:
        return {"statistic": float("nan"), "critical_5": float("nan"),
                "reject": False, "lags": lags}
    Y = dy[lags:]
    cols = [y[lags:-1]]
    for i in range(1, lags + 1):
        cols.append(dy[lags - i: -i] if i < dy.size else np.zeros_like(Y))
    X = np.column_stack(cols)
    if trend in ("c", "ct"):
        X = np.column_stack([np.ones(Y.size), X])
    if trend == "ct":
        X = np.column_stack([X, np.arange(Y.size, dtype=float)])
    fit = LM.ols(Y, X, has_constant=trend in ("c", "ct"))
    idx = 1 if trend in ("c", "ct") else 0
    stat = float(fit.tvalues[idx])
    # MacKinnon (1996) asymptotic critical values
    table = {"n": (-1.94, -1.62, -1.28), "c": (-3.43, -2.86, -2.57),
             "ct": (-3.96, -3.41, -3.13)}
    c1, c5, c10 = table.get(trend, table["c"])
    return {"statistic": stat, "critical_1": c1, "critical_5": c5, "critical_10": c10,
            "reject": stat < c5, "lags": lags, "rho_star": float(fit.coefficients[idx])}


SPEC = make_spec(
    "timeseries.stationarity",
    Domain.TIMESERIES,
    "unit_roots",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("process", "ar1", PROCESSES, group="dgp"),
        slider("rho", 0.5, -1.2, 1.2, 0.005, group="dgp"),
        slider("drift", 0.0, -1.0, 1.0, 0.01, group="dgp"),
        slider("trend", 0.0, -0.2, 0.2, 0.005, group="dgp"),
        int_slider("n", 200, 20, 5000, 10, group="dgp"),
        slider("sigma", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        slider("break_shift", 3.0, -10.0, 10.0, 0.1, group="dgp",
               depends_on=("process", ("structural_break",))),
        int_slider("adf_lags", 1, 0, 12, 1, group="tests"),
        select("adf_trend", "c", ("n", "c", "ct"), group="tests"),
        int_slider("n_paths", 6, 1, 40, 1, group="views"),
        int_slider("reps", 400, 50, 4000, 50, group="simulation", expensive=True),
        toggle("show_spurious", True, group="views"),
        toggle("show_variance", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("white_noise", "null", rho=0.0),
        scenario("canonical", "canonical", rho=0.5),
        scenario("persistent", "strong", rho=0.9),
        scenario("near_unit_root", "boundary", rho=0.98),
        scenario("unit_root", "violation", rho=1.0),
        scenario("random_walk_with_drift", "violation", rho=1.0, drift=0.15),
        scenario("explosive", "boundary", rho=1.03, n=120),
        scenario("negative_persistence", "negative", rho=-0.7),
        scenario("trend_stationary_lookalike", "counterexample",
                 process="trend_stationary", rho=0.4, trend=0.05),
        scenario("structural_break_lookalike", "counterexample",
                 process="structural_break", break_shift=4.0),
        scenario("small_sample", "small_sample", n=40, rho=0.9),
        scenario("large_sample", "large_sample", n=3000, rho=0.95),
    ),
    related=("timeseries.arma", "timeseries.cointegration"),
    next_concepts=("timeseries.arma", "timeseries.cointegration"),
    tags=("stationarity", "unit root", "random walk", "adf", "spurious regression",
          "dickey-fuller"),
    aliases=("unit root", "racine unitaire", "جذر الوحدة", "stationarity",
             "augmented dickey fuller", "spurious regression"),
    backends=("numpy", "arch", "statsmodels"),
    references=(
        ref("Dickey, D. A. and Fuller, W. A. (1979). Distribution of the estimators for "
            "autoregressive time series with a unit root. JASA 74(366).", kind="paper",
            doi="10.1080/01621459.1979.10482531"),
        ref("Granger, C. W. J. and Newbold, P. (1974). Spurious regressions in "
            "econometrics. Journal of Econometrics 2(2).", kind="paper",
            doi="10.1016/0304-4076(74)90034-7"),
        ref("MacKinnon, J. G. (1996). Numerical distribution functions for unit root and "
            "cointegration tests. Journal of Applied Econometrics 11(6).", kind="paper",
            doi="10.1002/(SICI)1099-1255(199611)11:6<601::AID-JAE417>3.0.CO;2-T"),
    ),
    curriculum_tags=("dz.econometrics2", "cairo.eviews", "aub.econ306"),
)


class StationarityLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        data = self._generate(p, n, state.seed)
        y = data["y"]
        adf = adf_test(y, lags=int(p["adf_lags"]), trend=str(p["adf_trend"]))

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "paths", self._paths_figure(ctx, p, n, state.seed),
            "labs.ur.figure.paths", evidence=EvidenceType.SIMULATION,
        ))
        res.add_panel(ctx.panel(
            "acf", self._acf_figure(ctx, y, n),
            "labs.ur.figure.acf", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_variance"]:
            res.add_panel(ctx.panel(
                "variance", self._variance_figure(ctx, p, n, state.seed),
                "labs.ur.figure.variance", tab="compare",
                evidence=EvidenceType.SIMULATION,
            ))
        if p["show_spurious"]:
            spurious = self._spurious_study(p, n, state.seed, int(p["reps"]))
            res.add_panel(ctx.panel(
                "spurious", self._spurious_figure(ctx, spurious, float(p["rho"])),
                "labs.ur.figure.spurious", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            res.metric("spurious_rejection",
                       ctx.t("labs.ur.metric.spurious",
                             "Share of INDEPENDENT series pairs found 'significant' at 5%"),
                       spurious["rejection_rate"], reference=0.05,
                       note=ctx.t("labs.ur.metric.spurious_note",
                                  "should be 5% if the regression were valid"))
            res.metric("spurious_median_r2",
                       ctx.t("labs.ur.metric.spurious_r2",
                             "Median R-squared between independent series"),
                       spurious["median_r2"])

        res.metric("rho_estimate", ctx.t("labs.ur.metric.rho",
                                         "Estimated AR(1) coefficient"),
                   float(acf(y, 1)[1]), reference=float(p["rho"]))
        res.metric("adf_statistic", ctx.t("labs.ur.metric.adf", "ADF test statistic"),
                   adf["statistic"])
        res.metric("adf_critical_5", ctx.t("labs.ur.metric.adf_crit",
                                           "5% Dickey-Fuller critical value"),
                   adf["critical_5"],
                   note=ctx.t("labs.ur.metric.adf_note",
                              "not -1.96: the null distribution is non-standard"))
        res.metric("adf_decision", ctx.t("labs.ur.metric.adf_decision",
                                         "ADF conclusion at 5%"),
                   ctx.t("labs.ur.reject", "reject the unit root (looks stationary)")
                   if adf["reject"]
                   else ctx.t("labs.ur.fail", "cannot reject a unit root"))
        res.metric("variance_ratio", ctx.t("labs.ur.metric.var_ratio",
                                           "Variance of the last third over the first third"),
                   float(np.var(y[-n // 3:], ddof=1) /
                         max(np.var(y[: n // 3], ddof=1), 1e-12)),
                   note=ctx.t("labs.ur.metric.var_note",
                              "far above 1 means the variance is growing with time"))
        res.metric("half_life", ctx.t("labs.ur.metric.half_life",
                                      "Shock half-life in periods"),
                   self._half_life(float(p["rho"])))

        stationary = bool(data.meta.get("stationary", abs(float(p["rho"])) < 1.0))
        res.assume("stationarity", ctx.t("assumptions.stationarity"), stationary,
                   detail=ctx.t("labs.ur.assume.stationarity",
                                "The process is {kind}.",
                                kind=str(data.meta.get("kind", "unknown"))),
                   consequence="" if stationary else ctx.t(
                       "labs.ur.assume.consequence",
                       "Shocks never fade, the variance grows without bound, and t "
                       "statistics from a levels regression follow non-standard "
                       "distributions - so ordinary critical values reject far too often."))
        res.assume("correct_deterministic_terms",
                   ctx.t("labs.ur.assume.terms_label",
                         "The test includes the right deterministic terms"), True,
                   detail=ctx.t("labs.ur.assume.terms",
                                "Omitting a trend that is genuinely present destroys the "
                                "power of the ADF test; including one that is not present "
                                "wastes it. This choice is not innocuous."))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.timeseries.stationarity.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.timeseries.stationarity.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.timeseries.stationarity.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.timeseries.stationarity.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.timeseries.stationarity.warning"), kind="warning")

        if str(p["process"]) in ("trend_stationary", "structural_break"):
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.ur.counterexample",
                "This series is stationary around a deterministic component, yet the ADF "
                "statistic is {s} against a 5% critical value of {c}. Trends and level "
                "shifts both make a stationary series look like a unit-root process to a "
                "test that does not know about them.",
                s=fmt(adf["statistic"], 3), c=fmt(adf["critical_5"], 2),
            ), kind="warning")
        return res

    @staticmethod
    def _generate(p, n, seed):
        kind = str(p["process"])
        if kind == "trend_stationary":
            return trend_stationary(n=n, trend=float(p["trend"]) or 0.05,
                                    rho=float(p["rho"]), sigma=float(p["sigma"]),
                                    seed=seed)
        if kind == "structural_break":
            return structural_break_series(n=n, level_shift=float(p["break_shift"]),
                                           rho=float(p["rho"]),
                                           sigma=float(p["sigma"]), seed=seed)
        return ar_process(n=n, rho=float(p["rho"]), drift=float(p["drift"]),
                          trend=float(p["trend"]), sigma=float(p["sigma"]), seed=seed)

    @staticmethod
    def _half_life(rho):
        if abs(rho) >= 1.0 or rho <= 0:
            return float("inf") if abs(rho) >= 1.0 else 0.0
        return float(np.log(0.5) / np.log(rho))

    def _paths_figure(self, ctx, p, n, seed):
        fig = ctx.figure(
            "labs.ur.figure.paths",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=430,
        )
        for i in range(int(p["n_paths"])):
            d = self._generate(p, n, int(seed) * 1000003 + i)
            P.add_curve(fig, d["t"], d["y"],
                        ctx.t("labs.ur.trace.path", "path {i}", i=i + 1),
                        "primary" if i == 0 else "muted", theme=ctx.theme,
                        width=2.4 if i == 0 else 1.1,
                        opacity=1.0 if i == 0 else 0.55, showlegend=i < 2)
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.ur.legend_paths",
            "Stationary paths keep returning to a common level; unit-root paths fan apart "
            "and never come back. Watching several realizations at once is the clearest "
            "single diagnostic there is.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _acf_figure(self, ctx, y, n):
        nlags = min(30, max(n // 4, 5))
        a = acf(y, nlags)
        dy = np.diff(y)
        ad = acf(dy, nlags)
        band = 1.96 / np.sqrt(n)
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.ur.trace.acf_level", "ACF of the level"),
            ctx.t("labs.ur.trace.acf_diff", "ACF of the first difference"),
        ))
        for col, series in ((1, a), (2, ad)):
            fig.add_trace(go.Bar(x=np.arange(series.size), y=series,
                                 marker={"color": ctx.color("primary")},
                                 showlegend=False), row=1, col=col)
            for sign in (1, -1):
                fig.add_hline(y=sign * band, line={"color": ctx.color("muted"),
                                                   "dash": "dot"}, row=1, col=col)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=340, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.ur.legend_acf",
            "A unit-root series has an ACF that decays almost linearly and stays high for "
            "many lags. If differencing collapses it to near zero, the series was I(1).",
        ), theme=ctx.theme)
        return fig

    def _variance_figure(self, ctx, p, n, seed):
        paths = np.array([self._generate(p, n, int(seed) * 99991 + i)["y"]
                          for i in range(60)])
        var_t = np.var(paths, axis=0, ddof=1)
        t = np.arange(n, dtype=float)
        fig = ctx.figure(
            "labs.ur.figure.variance",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.ur.axis.cross_var",
                              "Variance across 60 independent paths"),
            height=350,
        )
        P.add_curve(fig, t, var_t,
                    ctx.t("labs.ur.trace.var", "cross-sectional variance at each date"),
                    "primary", theme=ctx.theme)
        rho = float(p["rho"])
        if abs(rho) < 1:
            level = float(p["sigma"]) ** 2 / (1 - rho**2)
            P.add_hline(fig, level,
                        ctx.t("labs.ur.trace.stationary_var",
                              "stationary variance sigma^2/(1 - rho^2)"),
                        "truth", theme=ctx.theme, dash="dash")
        else:
            P.add_curve(fig, t, float(p["sigma"]) ** 2 * np.maximum(t, 1),
                        ctx.t("labs.ur.trace.linear_var",
                              "t * sigma^2 (the unit-root prediction)"),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.ur.legend_variance",
            "This is the sharpest definition of the difference: a stationary process has a "
            "variance that settles at a constant; a unit-root process has one that grows "
            "linearly forever.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _spurious_study(self, p, n, seed, reps):
        reps = min(reps, 2000)
        rho = float(p["rho"])
        tstats, r2s = [], []
        for r in range(reps):
            g = rng(int(seed) * 2971215073 + r, "spurious")
            a = self._ar_path(g, n, rho, float(p["sigma"]))
            b = self._ar_path(g, n, rho, float(p["sigma"]))
            fit = LM.ols(a, np.column_stack([np.ones(n), b]), names=("const", "x"))
            tstats.append(float(fit.tvalues[1]))
            r2s.append(fit.r_squared)
        tstats = np.asarray(tstats)
        return {"tstats": tstats, "r2": np.asarray(r2s),
                "rejection_rate": float(np.mean(np.abs(tstats) > 1.96)),
                "median_r2": float(np.median(r2s)), "reps": reps}

    @staticmethod
    def _ar_path(gen, n, rho, sigma):
        e = sigma * gen.standard_normal(n)
        y = np.empty(n)
        y[0] = e[0] / np.sqrt(max(1 - min(rho**2, 0.999), 1e-3))
        for t in range(1, n):
            y[t] = rho * y[t - 1] + e[t]
        return y

    def _spurious_figure(self, ctx, study, rho):
        fig = ctx.figure(
            "labs.ur.figure.spurious",
            xaxis_title=ctx.t("labs.ur.axis.tstat",
                              "t statistic from regressing one series on the other"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=370,
        )
        vals = study["tstats"]
        clipped = vals[np.abs(vals) < np.percentile(np.abs(vals), 99)]
        P.add_histogram(fig, clipped,
                        ctx.t("labs.ur.trace.tdist",
                              "t statistics between INDEPENDENT series"),
                        "negative", theme=ctx.theme, nbins=60, opacity=0.6)
        grid = np.linspace(float(clipped.min()), float(clipped.max()), 300)
        P.add_curve(fig, grid, stats.norm.pdf(grid),
                    ctx.t("labs.ur.trace.normal",
                          "what the standard theory promises"),
                    "truth", theme=ctx.theme, dash="dash")
        for c in (-1.96, 1.96):
            P.add_vline(fig, c, "", "type_i", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.ur.legend_spurious",
            "Both series are pure noise with no relationship whatsoever. At rho = {r} the "
            "regression calls them significantly related {p} of the time, with a median "
            "R-squared of {r2}. This is spurious regression.",
            r=fmt(rho, 2), p=pct(study["rejection_rate"]),
            r2=fmt(study["median_r2"], 3),
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        rhos = np.concatenate([np.linspace(0.0, 0.9, 10), [0.95, 0.98, 0.99, 1.0, 1.01]])
        frames, steps = [], []
        for i, r in enumerate(rhos):
            q = dict(p)
            q["rho"] = float(r)
            q["process"] = "ar1"
            paths = [self._generate(q, n, int(seed) * 104729 + j)["y"] for j in range(5)]
            a = adf_test(paths[0], lags=int(p["adf_lags"]), trend=str(p["adf_trend"]))
            frames.append(go.Frame(name=f"{r:.2f}",
                                   data=[go.Scatter(x=np.arange(n), y=path)
                                         for path in paths]))
            steps.append(AnimationStep(
                id=f"rho_{i}", frame=i,
                title=ctx.t("labs.ur.anim.title", "rho = {r}", r=fmt(r, 3)),
                what_you_see=ctx.t("labs.ur.anim.see",
                                   "Five independent realizations of the same AR(1) process."),
                what_changed=ctx.t("labs.ur.anim.changed",
                                   "The autoregressive coefficient moved to {r}.",
                                   r=fmt(r, 3)),
                why=ctx.t("labs.ur.anim.why",
                          "rho is the fraction of yesterday that survives into today. Below "
                          "one, shocks decay geometrically; at exactly one, they never "
                          "decay at all."),
                interpretation=ctx.t("labs.ur.anim.interpret",
                                     "Shock half-life is {h} periods; the ADF statistic on "
                                     "the first path is {s} against a critical value of {c}.",
                                     h=("infinite" if not np.isfinite(self._half_life(r))
                                        else fmt(self._half_life(r), 1)),
                                     s=fmt(a["statistic"], 2), c=fmt(a["critical_5"], 2)),
                conclusion=ctx.t("labs.ur.anim.conclude",
                                 "The change at rho = 1 is not gradual in its consequences: "
                                 "the variance stops converging and standard inference stops "
                                 "being valid."),
                warning=ctx.t("labs.ur.anim.warn",
                              "At 0.95 and at 1.00 the pictures look almost identical over "
                              "200 observations. Tests, not eyes, are needed - and even they "
                              "have very little power to tell these apart."),
                math="y_t = rho y_(t-1) + e_t;  Var(y_t) = sigma^2/(1 - rho^2) if |rho| < 1",
                outputs={"rho": round(float(r), 3),
                         "adf_statistic": round(float(a["statistic"]), 3),
                         "half_life": (None if not np.isfinite(self._half_life(r))
                                       else round(self._half_life(r), 2))},
                violated_assumptions=() if abs(r) < 1 else ("stationarity",),
                highlighted=("paths",),
            ))
        q = dict(p)
        q["rho"] = 0.0
        q["process"] = "ar1"
        fig = ctx.figure(
            "labs.ur.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=390,
        )
        for j in range(5):
            d = self._generate(q, n, int(seed) * 104729 + j)
            P.add_curve(fig, d["t"], d["y"],
                        ctx.t("labs.ur.trace.path", "path {i}", i=j + 1),
                        "primary" if j == 0 else "muted", theme=ctx.theme,
                        showlegend=j < 2)
        fig.update_yaxes(range=[-30, 30])
        build_frames(fig, frames, duration=520, reduced_motion=ctx.reduced_motion,
                     slider_label="rho")
        return animation(
            "rho_to_one", fig, steps,
            purpose=ctx.t("labs.ur.anim.purpose",
                          "Approach the unit root and watch what changes qualitatively."),
            summary=ctx.t(
                "labs.ur.anim.summary",
                "Between rho = 0.95 and rho = 1.00 the pictures are nearly identical but "
                "the statistical world is completely different: mean reversion versus "
                "permanent shocks, bounded versus exploding variance, standard versus "
                "non-standard inference. This is why unit-root testing is hard and why its "
                "results should be reported with the deterministic terms you chose."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = StationarityLab(SPEC)
