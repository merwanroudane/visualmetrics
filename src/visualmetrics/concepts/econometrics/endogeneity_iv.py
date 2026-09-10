"""Endogeneity and instrumental variables, including weak and invalid instruments."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.regression import iv_design
from ...simulation.monte_carlo import monte_carlo

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "econometrics.endogeneity_iv",
    Domain.ECONOMETRICS,
    "endogeneity_and_identification",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive",
           "simulate", "diagnose", "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        slider("beta", 1.0, -5.0, 5.0, 0.05, group="structure"),
        slider("endogeneity", 0.6, 0.0, 0.95, 0.01, group="structure"),
        slider("instrument_strength", 0.8, 0.0, 3.0, 0.01, group="instrument"),
        slider("invalid_instrument", 0.0, 0.0, 2.0, 0.01, group="instrument"),
        int_slider("n_instruments", 1, 1, 10, 1, group="instrument"),
        int_slider("n", 300, 20, 20000, 10, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        int_slider("reps", 800, 100, 8000, 100, group="simulation", expensive=True),
        toggle("show_first_stage", True, group="views"),
        toggle("show_sampling", True, group="views"),
        toggle("show_dag", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", endogeneity=0.6, instrument_strength=0.8),
        scenario("no_endogeneity", "null", endogeneity=0.0),
        scenario("strong_instrument", "strong", instrument_strength=2.0),
        scenario("weak_instrument", "weak", instrument_strength=0.06, n=300),
        scenario("very_weak_instrument", "counterexample", instrument_strength=0.02,
                 n=300),
        scenario("invalid_instrument", "violation", invalid_instrument=0.8,
                 instrument_strength=1.0),
        scenario("overidentified", "compare_methods", n_instruments=4,
                 instrument_strength=0.6),
        scenario("overidentified_invalid", "violation", n_instruments=4,
                 invalid_instrument=0.6, instrument_strength=0.8),
        scenario("severe_endogeneity", "strong", endogeneity=0.9),
        scenario("small_sample", "small_sample", n=40),
        scenario("large_sample", "large_sample", n=10000),
    ),
    prerequisites=("econometrics.omitted_variable_bias",),
    related=("causal.dag", "econometrics.simultaneous_equations"),
    next_concepts=("econometrics.simultaneous_equations",),
    tags=("endogeneity", "instrumental variables", "2sls", "weak instruments",
          "sargan", "exclusion restriction", "first stage"),
    aliases=("iv", "2sls", "variables instrumentales", "المتغيرات الوسيطة",
             "instrumental variables", "two stage least squares"),
    backends=("numpy", "linearmodels"),
    references=(
        ref("Staiger, D. and Stock, J. H. (1997). Instrumental variables regression with "
            "weak instruments. Econometrica 65(3).", kind="paper", doi="10.2307/2171753"),
        ref("Angrist, J. D. and Pischke, J.-S. (2009). Mostly Harmless Econometrics.",
            kind="book"),
    ),
    curriculum_tags=("dz.econometrics2", "ksu.econ541", "aub.econ305"),
)


class IVLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        data = self._generate(p, n, state.seed)
        n_inst = int(p["n_instruments"])
        y = data["y"]
        X = data.matrix("x")
        Z = np.column_stack([np.ones(n)] + [data[f"z{j + 1}"] for j in range(n_inst)])

        ols = LM.ols(y, X, names=("const", "x"))
        iv = LM.iv_2sls(y, X, Z, endog_index=1, names=("const", "x"))

        res.data = data
        res.dgp = data.dgp

        if p["show_dag"]:
            res.add_panel(ctx.panel(
                "dag", self._dag_figure(ctx, p),
                "labs.iv.figure.dag", evidence=EvidenceType.VISUAL_DERIVATION,
            ))
        res.add_panel(ctx.panel(
            "comparison", self._comparison_figure(ctx, ols, iv, p),
            "labs.iv.figure.comparison", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_first_stage"]:
            res.add_panel(ctx.panel(
                "first_stage", self._first_stage_figure(ctx, data, iv, n_inst),
                "labs.iv.figure.first_stage", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_sampling"]:
            draws = self._sampling(p, n, state.seed, int(p["reps"]))
            res.add_panel(ctx.panel(
                "sampling", self._sampling_figure(ctx, draws, float(p["beta"])),
                "labs.iv.figure.sampling", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            for key, label in (("ols", "OLS"), ("iv", "2SLS")):
                vals = draws[key]
                finite = vals[np.isfinite(vals)]
                res.metric(f"sim_bias_{key}",
                           ctx.t("labs.iv.metric.sim_bias", "{m}: median bias", m=label),
                           float(np.median(finite) - float(p["beta"])), reference=0.0)
                res.metric(f"sim_iqr_{key}",
                           ctx.t("labs.iv.metric.sim_iqr",
                                 "{m}: interquartile range", m=label),
                           float(stats.iqr(finite)))

        res.metric("ols_estimate", ctx.t("labs.iv.metric.ols", "OLS estimate"),
                   ols.coef("x"), reference=float(p["beta"]))
        res.metric("ols_theoretical_bias", ctx.t("labs.iv.metric.ols_bias",
                                                 "OLS bias implied by this DGP"),
                   data.truth["ols_bias"])
        res.metric("iv_estimate", ctx.t("labs.iv.metric.iv", "2SLS estimate"),
                   iv.second_stage.coef("x"), reference=float(p["beta"]))
        res.metric("iv_se", ctx.t("labs.iv.metric.iv_se", "2SLS standard error"),
                   iv.second_stage.se("x"))
        res.metric("ols_se", ctx.t("labs.iv.metric.ols_se", "OLS standard error"),
                   ols.se("x"))
        res.metric("se_cost", ctx.t("labs.iv.metric.cost",
                                    "Precision cost: 2SLS SE / OLS SE"),
                   float(iv.second_stage.se("x") / max(ols.se("x"), 1e-12)))
        res.metric("first_stage_f", ctx.t("labs.iv.metric.f",
                                          "First-stage F on the excluded instruments"),
                   iv.first_stage_f,
                   note=ctx.t("labs.iv.metric.f_note",
                              "below 10 is the classic weak-instrument warning"))
        res.metric("partial_r2", ctx.t("labs.iv.metric.partial",
                                       "Partial R-squared of the instruments"),
                   iv.partial_r2)
        if iv.endogeneity_test is not None:
            res.metric("hausman_p", ctx.t("labs.iv.metric.hausman",
                                          "Durbin-Wu-Hausman p-value"),
                       iv.endogeneity_test.p_value,
                       note=ctx.t("labs.iv.metric.hausman_note",
                                  "small values say OLS and 2SLS disagree more than "
                                  "sampling noise explains"))
        if iv.overid is not None:
            res.metric("sargan_p", ctx.t("labs.iv.metric.sargan",
                                         "Sargan over-identification p-value"),
                       iv.overid.p_value,
                       note=ctx.t("labs.iv.metric.sargan_note",
                                  "tests joint validity; it cannot say WHICH instrument "
                                  "is bad, and it is powerless if all are equally invalid"))
        else:
            res.metric("sargan_p", ctx.t("labs.iv.metric.sargan_na",
                                         "Over-identification test"),
                       ctx.t("labs.iv.metric.exactly_identified",
                             "not available - the model is exactly identified"))

        endogenous = float(p["endogeneity"]) > 0.01
        weak = iv.weak_instruments
        invalid = float(p["invalid_instrument"]) > 0.01
        res.assume("exogeneity", ctx.t("assumptions.exogeneity"), not endogenous,
                   detail=ctx.t("labs.iv.assume.exogeneity",
                                "The regressor is correlated with the structural error by "
                                "construction here."),
                   consequence="" if not endogenous else ctx.t(
                       "labs.iv.assume.ols_inconsistent",
                       "OLS converges to beta + Cov(x, u)/Var(x), not to beta."))
        res.assume("instrument_relevance", ctx.t("assumptions.instrument_relevance"),
                   not weak,
                   detail=ctx.t("labs.iv.assume.relevance",
                                "The first-stage F is {f}. Relevance is testable, and this "
                                "is the test.", f=fmt(iv.first_stage_f, 2)),
                   consequence="" if not weak else ctx.t(
                       "labs.iv.assume.weak_consequence",
                       "With weak instruments 2SLS is biased towards OLS in finite samples "
                       "and its confidence intervals under-cover badly."))
        res.assume("instrument_exogeneity", ctx.t("assumptions.instrument_exogeneity"),
                   not invalid,
                   detail=ctx.t("labs.iv.assume.exclusion",
                                "The exclusion restriction is an assumption, not a "
                                "testable claim - except partially, when the model is "
                                "over-identified."),
                   consequence="" if not invalid else ctx.t(
                       "labs.iv.assume.invalid_consequence",
                       "A direct instrument-to-outcome path makes 2SLS inconsistent too, "
                       "and often more so than OLS."))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.econometrics.endogeneity_iv.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.econometrics.endogeneity_iv.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.econometrics.endogeneity_iv.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.econometrics.endogeneity_iv.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.econometrics.endogeneity_iv.warning"), kind="warning")

        if weak and endogenous:
            res.warnings.append(ctx.t(
                "labs.iv.warn.weak",
                "The first-stage F is {f}. With an instrument this weak, 2SLS is not a cure: "
                "its finite-sample distribution is pulled towards OLS and its standard error "
                "understates the real uncertainty. A weak instrument is often worse than "
                "admitting the problem.", f=fmt(iv.first_stage_f, 2),
            ))
        if invalid:
            res.warnings.append(ctx.t(
                "labs.iv.warn.invalid",
                "This instrument has a direct effect on the outcome, so the exclusion "
                "restriction fails. 2SLS is now inconsistent - and note that with a single "
                "instrument no test can detect this.",
            ))
        return res

    @staticmethod
    def _generate(p, n, seed):
        return iv_design(
            n=n, beta=float(p["beta"]), endogeneity=float(p["endogeneity"]),
            instrument_strength=float(p["instrument_strength"]),
            invalid_instrument=float(p["invalid_instrument"]),
            n_instruments=int(p["n_instruments"]), noise=float(p["noise"]), seed=seed,
        )

    def _sampling(self, p, n, seed, reps):
        reps = min(reps, 3000)
        n_inst = int(p["n_instruments"])

        def experiment(gen):
            child = int(gen.integers(0, 2**31 - 1))
            d = self._generate(p, n, child)
            X = d.matrix("x")
            Z = np.column_stack([np.ones(n)] + [d[f"z{j + 1}"] for j in range(n_inst)])
            o = LM.ols(d["y"], X, names=("const", "x"))
            try:
                i = LM.iv_2sls(d["y"], X, Z, endog_index=1, names=("const", "x"))
                iv_val = i.second_stage.coef("x")
                f_val = i.first_stage_f
            except Exception:  # noqa: BLE001
                iv_val, f_val = np.nan, np.nan
            return {"ols": o.coef("x"), "iv": iv_val, "first_stage_f": f_val}

        return monte_carlo(experiment, reps, seed).draws

    def _dag_figure(self, ctx, p):
        go = P.require_plotly()
        fig = ctx.figure("labs.iv.figure.dag", height=340, showlegend=False)
        nodes = {"Z": (0.0, 0.0), "X": (1.0, 0.0), "Y": (2.0, 0.0), "U": (1.5, 1.0)}
        for name, (nx, ny) in nodes.items():
            observed = name != "U"
            P.add_points(fig, [nx], [ny], name,
                         "primary" if observed else "muted", theme=ctx.theme,
                         size=42, symbol="circle" if observed else "circle-open")
            P.add_annotation(fig, nx, ny, name, theme=ctx.theme)
        edges = [("Z", "X", "instrument_strength", "positive"),
                 ("X", "Y", "beta", "primary"),
                 ("U", "X", "endogeneity", "negative"),
                 ("U", "Y", "endogeneity", "negative")]
        if float(p["invalid_instrument"]) > 0.01:
            edges.append(("Z", "Y", "invalid_instrument", "negative"))
        for a, b, key, role in edges:
            (ax, ay), (bx, by) = nodes[a], nodes[b]
            strength = abs(float(p.get(key, 1.0)))
            if strength < 0.01:
                continue
            dx, dy = bx - ax, by - ay
            norm = np.hypot(dx, dy)
            pad = 0.13
            P.add_arrow(fig, ax + pad * dx / norm, ay + pad * dy / norm,
                        bx - pad * dx / norm, by - pad * dy / norm, f"{a}->{b}",
                        role, theme=ctx.theme,
                        width=1.2 + 2.5 * min(strength, 2.0), showlegend=False)
        fig.update_xaxes(visible=False, range=[-0.4, 2.4])
        fig.update_yaxes(visible=False, range=[-0.5, 1.4])
        P.add_legend_note(fig, ctx.t(
            "labs.iv.legend_dag",
            "U is unobserved. The instrument earns its keep only if the Z->X arrow is "
            "strong AND there is no Z->Y arrow at all. Line thickness shows the strength "
            "you have set.",
        ), theme=ctx.theme)
        return fig

    def _comparison_figure(self, ctx, ols, iv, p):
        go = P.require_plotly()
        labels = ["OLS", "2SLS"]
        values = [ols.coef("x"), iv.second_stage.coef("x")]
        errors = [1.96 * ols.se("x"), 1.96 * iv.second_stage.se("x")]
        fig = ctx.figure(
            "labs.iv.figure.comparison",
            xaxis_title=ctx.t("labs.iv.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=380,
        )
        fig.add_trace(go.Bar(x=labels, y=values,
                             marker={"color": [ctx.color("negative"),
                                               ctx.color("primary")]},
                             error_y={"type": "data", "array": errors},
                             text=[fmt(v, 4) for v in values], textposition="outside",
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, float(p["beta"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.iv.legend_comparison",
            "2SLS trades precision for consistency: the bar usually moves towards the "
            "dashed line while its error bar grows. Both changes are the point.",
        ), theme=ctx.theme)
        return fig

    def _first_stage_figure(self, ctx, data, iv, n_inst):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        z = data["z1"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.iv.trace.first", "first stage: z against x"),
            ctx.t("labs.iv.trace.reduced", "reduced form: z against y"),
        ))
        fig.add_trace(go.Scatter(x=z, y=data["x"], mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 5,
                                         "opacity": 0.6}, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=z, y=data["y"], mode="markers",
                                 marker={"color": ctx.color("secondary"), "size": 5,
                                         "opacity": 0.6}, showlegend=False), row=1, col=2)
        for col, target in ((1, data["x"]), (2, data["y"])):
            f = LM.ols(target, np.column_stack([np.ones(z.size), z]))
            order = np.argsort(z)
            fig.add_trace(go.Scatter(x=z[order], y=f.fitted_values[order], mode="lines",
                                     line={"color": ctx.color("fitted"), "width": 2.6},
                                     showlegend=False), row=1, col=col)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=350, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.iv.legend_first_stage",
            "The IV estimate is the ratio of these two slopes. A flat left panel means you "
            "are dividing by something close to zero - which is exactly what a weak "
            "instrument is. First-stage F = {f}.", f=fmt(iv.first_stage_f, 2),
        ), theme=ctx.theme)
        return fig

    def _sampling_figure(self, ctx, draws, beta):
        fig = ctx.figure(
            "labs.iv.figure.sampling",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=380,
        )
        ols_v = draws["ols"][np.isfinite(draws["ols"])]
        iv_v = draws["iv"][np.isfinite(draws["iv"])]
        lo = float(np.percentile(np.concatenate([ols_v, iv_v]), 1))
        hi = float(np.percentile(np.concatenate([ols_v, iv_v]), 99))
        pad = 0.3 * (hi - lo + 1e-9)
        clip = lambda v: v[(v > lo - pad) & (v < hi + pad)]  # noqa: E731
        P.add_histogram(fig, clip(ols_v), "OLS", "negative", theme=ctx.theme,
                        nbins=60, opacity=0.55)
        P.add_histogram(fig, clip(iv_v), "2SLS", "primary", theme=ctx.theme,
                        nbins=60, opacity=0.55)
        P.add_vline(fig, beta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.iv.legend_sampling",
            "OLS is tight and in the wrong place; 2SLS is wide and centred correctly - "
            "unless the instrument is weak, in which case it is wide, skewed AND pulled "
            "back towards OLS.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        strengths = np.concatenate([np.linspace(0.01, 0.3, 6), np.linspace(0.45, 2.5, 8)])
        frames, steps = [], []
        reps = 250
        for i, s in enumerate(strengths):
            q = dict(p)
            q["instrument_strength"] = float(s)
            d = self._sampling(q, n, seed, reps)
            iv_v = d["iv"][np.isfinite(d["iv"])]
            f_med = float(np.nanmedian(d["first_stage_f"]))
            lo, hi = float(p["beta"]) - 4, float(p["beta"]) + 4
            hist, edges = np.histogram(np.clip(iv_v, lo, hi), bins=np.linspace(lo, hi, 51),
                                       density=True)
            centres = 0.5 * (edges[1:] + edges[:-1])
            frames.append(go.Frame(name=f"{s:.2f}", data=[go.Bar(x=centres, y=hist)]))
            steps.append(AnimationStep(
                id=f"strength_{i}", frame=i,
                title=ctx.t("labs.iv.anim.title", "first-stage strength = {s}",
                            s=fmt(s, 2)),
                what_you_see=ctx.t("labs.iv.anim.see",
                                   "The sampling distribution of the 2SLS estimator across "
                                   "{r} simulated studies.", r=reps),
                what_changed=ctx.t("labs.iv.anim.changed",
                                   "The instrument's effect on the endogenous regressor "
                                   "moved to {s}, giving a median first-stage F of {f}.",
                                   s=fmt(s, 2), f=fmt(f_med, 1)),
                why=ctx.t("labs.iv.anim.why",
                          "2SLS divides the reduced-form slope by the first-stage slope. As "
                          "the denominator approaches zero, the ratio becomes unstable and "
                          "its distribution develops heavy tails."),
                interpretation=ctx.t("labs.iv.anim.interpret",
                                     "Median estimate {m} against a true value of {t}; "
                                     "interquartile range {q}.",
                                     m=fmt(float(np.median(iv_v)), 3),
                                     t=fmt(p["beta"], 3),
                                     q=fmt(float(stats.iqr(iv_v)), 3)),
                conclusion=ctx.t("labs.iv.anim.conclude",
                                 "A strong instrument makes 2SLS centred and reasonably "
                                 "tight; a weak one makes it neither."),
                warning=ctx.t("labs.iv.anim.warn",
                              "At the weakest settings the distribution is not merely wide, "
                              "it is dragged towards the OLS bias. Weak-instrument 2SLS can "
                              "be worse than doing nothing."),
                math="plim(2SLS) = beta + Cov(z, u) / Cov(z, x)",
                outputs={"instrument_strength": round(float(s), 3),
                         "median_first_stage_F": round(f_med, 2),
                         "median_estimate": round(float(np.median(iv_v)), 4)},
                violated_assumptions=("instrument_relevance",) if f_med < 10 else (),
                highlighted=("iv_distribution",),
            ))
        lo, hi = float(p["beta"]) - 4, float(p["beta"]) + 4
        centres = np.linspace(lo, hi, 50)
        fig = ctx.figure(
            "labs.iv.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=390,
        )
        fig.add_trace(go.Bar(x=centres, y=np.zeros_like(centres),
                             marker={"color": ctx.color("primary")}, opacity=0.7,
                             name="2SLS"))
        P.add_vline(fig, float(p["beta"]), ctx.t("labs.common.trace.truth"), "truth",
                    theme=ctx.theme)
        build_frames(fig, frames, duration=480, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.iv.slider", "instrument strength"))
        return animation(
            "weak_to_strong", fig, steps,
            purpose=ctx.t("labs.iv.anim.purpose",
                          "Show that 'having an instrument' and 'having a usable "
                          "instrument' are different things."),
            summary=ctx.t(
                "labs.iv.anim.summary",
                "Instrumental variables buy consistency with a division, and dividing by a "
                "small number is dangerous. Strong instruments give a well-behaved, roughly "
                "normal estimator; weak ones give a skewed, heavy-tailed distribution "
                "leaning back towards the very bias you were trying to remove."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = IVLab(SPEC)
