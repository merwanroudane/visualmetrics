"""Cointegration and the error-correction mechanism."""

from __future__ import annotations

from typing import Any

from ...backends import linear as LM
from ...data.generators.timeseries import cointegrated_pair
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
from .stationarity import adf_test

__all__ = ["LAB", "SPEC"]

#: Engle-Granger critical values for the residual ADF test with one regressor
#: (MacKinnon 1991): the estimated cointegrating vector makes the usual
#: Dickey-Fuller values far too permissive.
EG_CRITICAL = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


SPEC = make_spec(
    "timeseries.cointegration",
    Domain.TIMESERIES,
    "cointegration_and_ecm",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        toggle("cointegrated", True, group="dgp"),
        slider("beta", 1.5, -4.0, 4.0, 0.05, group="dgp"),
        slider("adjustment", -0.25, -1.5, 0.0, 0.01, group="dgp"),
        int_slider("n", 250, 30, 5000, 10, group="dgp"),
        slider("sigma_x", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        slider("sigma_e", 0.8, 0.05, 5.0, 0.05, group="dgp"),
        int_slider("adf_lags", 1, 0, 8, 1, group="tests"),
        int_slider("reps", 300, 50, 3000, 50, group="simulation", expensive=True),
        toggle("show_ecm", True, group="views"),
        toggle("show_spread", True, group="views"),
        toggle("show_size_study", True, group="simulation"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", cointegrated=True, adjustment=-0.25),
        scenario("no_cointegration", "counterexample", cointegrated=False),
        scenario("fast_adjustment", "strong", adjustment=-0.8),
        scenario("slow_adjustment", "weak", adjustment=-0.05),
        scenario("no_adjustment", "boundary", adjustment=-0.005),
        scenario("negative_beta", "negative", beta=-1.2),
        scenario("noisy_equilibrium", "high_noise", sigma_e=3.0),
        scenario("tight_equilibrium", "low_noise", sigma_e=0.2),
        scenario("small_sample", "small_sample", n=60),
        scenario("large_sample", "large_sample", n=2000),
    ),
    prerequisites=("timeseries.stationarity",),
    related=("timeseries.var", "timeseries.stationarity"),
    tags=("cointegration", "engle-granger", "error correction", "long run",
          "spurious regression", "ardl"),
    aliases=("ecm", "cointegration", "التكامل المشترك", "error correction model",
             "engle granger"),
    backends=("numpy", "arch", "statsmodels"),
    references=(
        ref("Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error "
            "correction. Econometrica 55(2).", kind="paper", doi="10.2307/1913236"),
        ref("MacKinnon, J. G. (1991). Critical values for cointegration tests.",
            kind="paper"),
    ),
    curriculum_tags=("dz.econometrics2", "aub.econ306", "ksu.econ542"),
)


class CointegrationLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        data = self._generate(p, n, state.seed)
        y, x = data["y"], data["x"]

        levels = LM.ols(y, np.column_stack([np.ones(n), x]), names=("const", "x"))
        resid = levels.residuals
        eg = adf_test(resid, lags=int(p["adf_lags"]), trend="n")
        eg["critical_5"] = EG_CRITICAL["5%"]
        eg["reject"] = eg["statistic"] < EG_CRITICAL["5%"]
        ecm = self._ecm(y, x, resid)

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "series", self._series_figure(ctx, data),
            "labs.coint.figure.series", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        if p["show_spread"]:
            res.add_panel(ctx.panel(
                "spread", self._spread_figure(ctx, data, resid, levels),
                "labs.coint.figure.spread", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_ecm"]:
            res.add_panel(ctx.panel(
                "ecm", self._ecm_figure(ctx, ecm, resid),
                "labs.coint.figure.ecm", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_size_study"]:
            study = self._size_study(p, n, state.seed, int(p["reps"]))
            res.add_panel(ctx.panel(
                "size", self._size_figure(ctx, study),
                "labs.coint.figure.size", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            res.metric("naive_rejection",
                       ctx.t("labs.coint.metric.naive",
                             "Levels regression 'significant' at 5% when the series are "
                             "INDEPENDENT"),
                       study["naive"], reference=0.05)
            res.metric("eg_size", ctx.t("labs.coint.metric.eg_size",
                                        "Engle-Granger rejection rate when there is NO "
                                        "cointegration"),
                       study["eg_no_coint"], reference=0.05)
            res.metric("eg_power", ctx.t("labs.coint.metric.eg_power",
                                         "Engle-Granger rejection rate when there IS "
                                         "cointegration"),
                       study["eg_coint"])

        res.metric("levels_beta", ctx.t("labs.coint.metric.beta",
                                        "Long-run coefficient from the levels regression"),
                   levels.coef("x"),
                   reference=float(p["beta"]) if p["cointegrated"] else None)
        res.metric("levels_r2", ctx.t("labs.coint.metric.r2",
                                      "R-squared of the levels regression"),
                   levels.r_squared,
                   note=ctx.t("labs.coint.metric.r2_note",
                              "high R-squared between integrated series proves nothing"))
        res.metric("levels_dw", ctx.t("labs.coint.metric.dw",
                                      "Durbin-Watson of the levels regression"),
                   LM.durbin_watson(resid),
                   note=ctx.t("labs.coint.metric.dw_note",
                              "a value far below the R-squared is the classic "
                              "spurious-regression signature"))
        res.metric("eg_statistic", ctx.t("labs.coint.metric.eg",
                                         "Engle-Granger residual ADF statistic"),
                   eg["statistic"])
        res.metric("eg_critical", ctx.t("labs.coint.metric.eg_crit",
                                        "Engle-Granger 5% critical value"),
                   EG_CRITICAL["5%"],
                   note=ctx.t("labs.coint.metric.eg_crit_note",
                              "not the ordinary ADF value: beta was estimated, which makes "
                              "the residuals look more stationary than they are"))
        res.metric("eg_decision", ctx.t("labs.coint.metric.eg_decision",
                                        "Engle-Granger conclusion at 5%"),
                   ctx.t("labs.coint.reject", "cointegrated") if eg["reject"]
                   else ctx.t("labs.coint.fail", "no cointegration found"))
        res.metric("adjustment_speed", ctx.t("labs.coint.metric.alpha",
                                             "Estimated error-correction coefficient"),
                   ecm["alpha"], reference=float(p["adjustment"]))
        res.metric("adjustment_t", ctx.t("labs.coint.metric.alpha_t",
                                         "Its t statistic"), ecm["alpha_t"])
        res.metric("half_life", ctx.t("labs.coint.metric.half_life",
                                      "Half-life of a disequilibrium, in periods"),
                   self._half_life(ecm["alpha"]))
        res.metric("short_run", ctx.t("labs.coint.metric.short_run",
                                      "Short-run coefficient on the change in x"),
                   ecm["short_run"])

        res.assume("cointegration", ctx.t("labs.coint.assume.coint_label",
                                          "The series are cointegrated"),
                   bool(p["cointegrated"]),
                   detail=ctx.t("labs.coint.assume.coint",
                                "Both series are I(1); the question is whether some linear "
                                "combination of them is I(0)."),
                   consequence="" if p["cointegrated"] else ctx.t(
                       "labs.coint.assume.spurious",
                       "Without cointegration, a levels regression between two integrated "
                       "series is spurious: high R-squared, large t statistics and no "
                       "relationship whatsoever."))
        res.assume("stable_adjustment",
                   ctx.t("labs.coint.assume.stable_label",
                         "The adjustment coefficient is negative and above -2"),
                   -2.0 < ecm["alpha"] < 0.0,
                   detail=ctx.t("labs.coint.assume.stable",
                                "A positive coefficient means the system is pushed apart "
                                "rather than pulled together, which is not an equilibrium."))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.timeseries.cointegration.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.timeseries.cointegration.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.timeseries.cointegration.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.timeseries.cointegration.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.timeseries.cointegration.warning"), kind="warning")

        if not p["cointegrated"]:
            res.warnings.append(ctx.t(
                "labs.coint.warn.spurious",
                "These two series are completely independent random walks, yet the levels "
                "regression reports R-squared {r} with a t statistic of {t}. The "
                "Durbin-Watson of {dw} is the giveaway.",
                r=fmt(levels.r_squared, 3), t=fmt(float(levels.tvalues[1]), 2),
                dw=fmt(LM.durbin_watson(resid), 3),
            ))
        return res

    @staticmethod
    def _generate(p, n, seed):
        return cointegrated_pair(n=n, beta=float(p["beta"]),
                                 adjustment=float(p["adjustment"]),
                                 sigma_x=float(p["sigma_x"]),
                                 sigma_e=float(p["sigma_e"]),
                                 seed=seed, cointegrated=bool(p["cointegrated"]))

    @staticmethod
    def _ecm(y, x, resid):
        dy = np.diff(y)
        dx = np.diff(x)
        ect = resid[:-1]
        X = np.column_stack([np.ones(dy.size), ect, dx])
        fit = LM.ols(dy, X, names=("const", "ect", "dx"))
        return {"fit": fit, "alpha": fit.coef("ect"), "alpha_t": float(fit.tvalues[1]),
                "short_run": fit.coef("dx"), "r2": fit.r_squared}

    @staticmethod
    def _half_life(alpha):
        rate = 1.0 + alpha
        if not (0 < rate < 1):
            return float("inf")
        return float(np.log(0.5) / np.log(rate))

    def _series_figure(self, ctx, data):
        fig = ctx.figure(
            "labs.coint.figure.series",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.common.axis.value"),
            height=400,
        )
        P.add_curve(fig, data["t"], data["y"], "y", "primary", theme=ctx.theme)
        P.add_curve(fig, data["t"], data["x"], "x", "secondary", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.coint.legend_series",
            "Both series wander without bound - each on its own is I(1). Cointegration is "
            "not visible here; it is a statement about their DIFFERENCE.",
        ), theme=ctx.theme)
        return fig

    def _spread_figure(self, ctx, data, resid, levels):
        fig = ctx.figure(
            "labs.coint.figure.spread",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.coint.axis.spread",
                              "Deviation from the estimated long-run relation"),
            height=360,
        )
        P.add_curve(fig, data["t"], resid,
                    ctx.t("labs.coint.trace.resid",
                          "y - {a} - {b} x", a=fmt(levels.coef("const"), 2),
                          b=fmt(levels.coef("x"), 3)),
                    "primary", theme=ctx.theme)
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.coint.legend_spread",
            "THIS is where cointegration lives. If this line keeps crossing zero and never "
            "drifts away, a stationary long-run relation exists. If it wanders like the "
            "levels did, there is none.",
        ), theme=ctx.theme)
        return fig

    def _ecm_figure(self, ctx, ecm, resid):
        fit = ecm["fit"]
        ect = resid[:-1]
        dy = fit.fitted_values + fit.residuals
        order = np.argsort(ect)
        fig = ctx.figure(
            "labs.coint.figure.ecm",
            xaxis_title=ctx.t("labs.coint.axis.ect",
                              "Last period's disequilibrium"),
            yaxis_title=ctx.t("labs.coint.axis.dy", "Change in y this period"),
            height=380,
        )
        P.add_points(fig, ect, dy, ctx.t("labs.common.trace.data"), "primary",
                     theme=ctx.theme, size=5, opacity=0.55)
        P.add_curve(fig, ect[order],
                    (fit.coef("const") + fit.coef("ect") * ect)[order],
                    ctx.t("labs.coint.trace.correction",
                          "error correction: slope = {a}", a=fmt(ecm["alpha"], 3)),
                    "fitted", theme=ctx.theme, width=3.0)
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        P.add_vline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.coint.legend_ecm",
            "A negative slope is the whole mechanism: when y sat above its long-run "
            "relation last period, it falls this period. The slope says what fraction of "
            "the gap closes each period.",
        ), theme=ctx.theme)
        return fig

    def _size_study(self, p, n, seed, reps):
        reps = min(reps, 1500)
        naive = eg_no = eg_yes = 0
        for r in range(reps):
            q = dict(p)
            q["cointegrated"] = False
            d0 = self._generate(q, n, int(seed) * 433494437 + r)
            f0 = LM.ols(d0["y"], np.column_stack([np.ones(n), d0["x"]]),
                        names=("const", "x"))
            if abs(float(f0.tvalues[1])) > 1.96:
                naive += 1
            a0 = adf_test(f0.residuals, lags=int(p["adf_lags"]), trend="n")
            if a0["statistic"] < EG_CRITICAL["5%"]:
                eg_no += 1
            q["cointegrated"] = True
            d1 = self._generate(q, n, int(seed) * 433494437 + r)
            f1 = LM.ols(d1["y"], np.column_stack([np.ones(n), d1["x"]]),
                        names=("const", "x"))
            a1 = adf_test(f1.residuals, lags=int(p["adf_lags"]), trend="n")
            if a1["statistic"] < EG_CRITICAL["5%"]:
                eg_yes += 1
        return {"naive": naive / reps, "eg_no_coint": eg_no / reps,
                "eg_coint": eg_yes / reps, "reps": reps}

    def _size_figure(self, ctx, study):
        labels = [
            ctx.t("labs.coint.trace.naive",
                  "levels t test, series independent"),
            ctx.t("labs.coint.trace.eg_size",
                  "Engle-Granger, no cointegration"),
            ctx.t("labs.coint.trace.eg_power",
                  "Engle-Granger, cointegration present"),
        ]
        values = [study["naive"], study["eg_no_coint"], study["eg_coint"]]
        fig = ctx.figure(
            "labs.coint.figure.size",
            xaxis_title=ctx.t("labs.coint.axis.procedure", "Procedure"),
            yaxis_title=ctx.t("labs.coint.axis.rate", "Rejection rate"),
            height=360,
        )
        P.add_bar(fig, labels, values,
                  ctx.t("labs.coint.trace.rate", "rejection rate"),
                  "primary", theme=ctx.theme, text=[pct(v) for v in values])
        P.add_hline(fig, 0.05, ctx.t("labs.het.trace.nominal", "nominal alpha = 0.05"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.coint.legend_size",
            "The first bar is the spurious-regression disaster. The second shows that "
            "Engle-Granger with the right critical values keeps its promise; the third is "
            "its power to detect a relation that really is there.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        alphas = np.concatenate([[-0.001], np.linspace(-0.05, -0.9, 14)])
        frames, steps = [], []
        for i, a in enumerate(alphas):
            q = dict(p)
            q["adjustment"] = float(a)
            q["cointegrated"] = True
            d = self._generate(q, n, seed)
            f = LM.ols(d["y"], np.column_stack([np.ones(n), d["x"]]), names=("const", "x"))
            eg = adf_test(f.residuals, lags=int(p["adf_lags"]), trend="n")
            frames.append(go.Frame(name=f"{a:.3f}",
                                   data=[go.Scatter(x=d["t"], y=f.residuals)]))
            steps.append(AnimationStep(
                id=f"alpha_{i}", frame=i,
                title=ctx.t("labs.coint.anim.title",
                            "adjustment speed = {a}", a=fmt(a, 3)),
                what_you_see=ctx.t("labs.coint.anim.see",
                                   "The deviation from the estimated long-run relation, "
                                   "over time."),
                what_changed=ctx.t("labs.coint.anim.changed",
                                   "The error-correction coefficient moved to {a}: about "
                                   "{k} of any gap now closes each period.",
                                   a=fmt(a, 3), k=pct(abs(a))),
                why=ctx.t("labs.coint.anim.why",
                          "A stronger pull back towards equilibrium makes the spread mean "
                          "revert faster, which is exactly what makes it stationary."),
                interpretation=ctx.t("labs.coint.anim.interpret",
                                     "The Engle-Granger statistic is {s} against a critical "
                                     "value of {c}; the disequilibrium half-life is {h} "
                                     "periods.",
                                     s=fmt(eg["statistic"], 2), c=fmt(EG_CRITICAL["5%"], 2),
                                     h=("infinite" if not np.isfinite(self._half_life(a))
                                        else fmt(self._half_life(a), 1))),
                conclusion=ctx.t("labs.coint.anim.conclude",
                                 "Cointegration is not about the levels looking alike; it "
                                 "is about whether a force pulls their combination back."),
                warning=ctx.t("labs.coint.anim.warn",
                              "Near zero adjustment the spread is indistinguishable from a "
                              "random walk over a finite sample - cointegration tests have "
                              "very little power there."),
                math="delta y_t = alpha (y_(t-1) - beta x_(t-1)) + short-run terms + e_t",
                outputs={"alpha": round(float(a), 4),
                         "eg_statistic": round(float(eg["statistic"]), 3)},
                violated_assumptions=("cointegration",) if abs(a) < 0.01 else (),
                highlighted=("spread",),
            ))
        q = dict(p)
        q["adjustment"] = float(alphas[0])
        d0 = self._generate(q, n, seed)
        f0 = LM.ols(d0["y"], np.column_stack([np.ones(n), d0["x"]]), names=("const", "x"))
        fig = ctx.figure(
            "labs.coint.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.time"),
            yaxis_title=ctx.t("labs.coint.axis.spread",
                              "Deviation from the estimated long-run relation"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=d0["t"], y=f0.residuals, mode="lines",
                                 line={"color": ctx.color("primary"), "width": 2.0},
                                 name=ctx.t("labs.coint.trace.spread_short", "spread")))
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dash")
        build_frames(fig, frames, duration=480, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.coint.slider", "adjustment speed"))
        return animation(
            "adjustment_speed", fig, steps,
            purpose=ctx.t("labs.coint.anim.purpose",
                          "Show cointegration as a restoring force, not a resemblance."),
            summary=ctx.t(
                "labs.coint.anim.summary",
                "Two series can wander anywhere as long as something keeps pulling their "
                "combination back. That pull is the error-correction coefficient, and its "
                "size is the whole difference between a genuine long-run relation and a "
                "spurious regression."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = CointegrationLab(SPEC)
