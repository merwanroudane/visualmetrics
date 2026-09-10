"""Fixed versus random effects, and the Hausman logic that chooses between them."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.causal import panel_dataset
from ...simulation.monte_carlo import monte_carlo

__all__ = ["LAB", "SPEC", "panel_estimators"]


def panel_estimators(unit, period, y, x, sigma_alpha=1.0, sigma_e=1.0):
    """Pooled OLS, between, within (fixed effects), first differences and random effects."""
    unit = np.asarray(unit)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    units = np.unique(unit)
    N, T = units.size, n // max(units.size, 1)

    pooled = LM.ols(y, np.column_stack([np.ones(n), x]), names=("const", "x"))

    ybar = np.array([y[unit == u].mean() for u in units])
    xbar = np.array([x[unit == u].mean() for u in units])
    between = LM.ols(ybar, np.column_stack([np.ones(N), xbar]), names=("const", "x"))

    ymean = np.array([y[unit == u].mean() for u in unit])
    xmean = np.array([x[unit == u].mean() for u in unit])
    yw, xw = y - ymean, x - xmean
    within = LM.ols(yw, xw.reshape(-1, 1), has_constant=False, names=("x",))
    # correct the degrees of freedom for the N estimated unit means
    dof_scale = np.sqrt((n - 1) / max(n - N - 1, 1))
    within.standard_errors = within.standard_errors * dof_scale
    within.covariance = within.covariance * dof_scale**2
    within.df_resid = max(n - N - 1, 1)

    fd_y, fd_x = [], []
    for u in units:
        mask = unit == u
        fd_y.append(np.diff(y[mask]))
        fd_x.append(np.diff(x[mask]))
    fd = LM.ols(np.concatenate(fd_y), np.concatenate(fd_x).reshape(-1, 1),
                has_constant=False, names=("x",))

    # feasible random effects: quasi-demeaning with estimated variance components
    sigma2_e = float(within.sigma2)
    sigma2_between = float(between.sigma2)
    sigma2_alpha = max(sigma2_between - sigma2_e / max(T, 1), 1e-9)
    theta = 1.0 - np.sqrt(sigma2_e / max(T * sigma2_alpha + sigma2_e, 1e-12))
    yr = y - theta * ymean
    xr = x - theta * xmean
    cr = 1.0 - theta
    random = LM.ols(yr, np.column_stack([np.full(n, cr), xr]), names=("const", "x"))

    diff = within.coef("x") - random.coef("x")
    var_diff = within.se("x") ** 2 - random.se("x") ** 2
    hausman = float(diff**2 / var_diff) if var_diff > 1e-12 else float("nan")
    hausman_p = float(stats.chi2.sf(hausman, 1)) if np.isfinite(hausman) else float("nan")

    return {
        "pooled": pooled, "between": between, "within": within, "fd": fd,
        "random": random, "theta": float(theta), "hausman": hausman,
        "hausman_p": hausman_p, "N": int(N), "T": int(T),
        "sigma2_alpha": float(sigma2_alpha), "sigma2_e": sigma2_e,
    }


SPEC = make_spec(
    "panel.fixed_vs_random",
    Domain.PANEL,
    "static_panel",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        slider("effect_x_correlation", 0.8, -0.95, 0.95, 0.01, group="dgp"),
        slider("beta", 1.0, -5.0, 5.0, 0.05, group="dgp"),
        int_slider("n_units", 40, 3, 500, 1, group="dgp"),
        int_slider("n_periods", 8, 2, 100, 1, group="dgp"),
        slider("sigma_alpha", 1.0, 0.0, 5.0, 0.05, group="dgp"),
        slider("sigma_e", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        slider("time_effects", 0.0, 0.0, 3.0, 0.05, group="dgp"),
        int_slider("reps", 400, 50, 4000, 50, group="simulation", expensive=True),
        toggle("show_unit_lines", True, group="views"),
        toggle("show_sampling", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", effect_x_correlation=0.8),
        scenario("random_effects_valid", "null", effect_x_correlation=0.0),
        scenario("mild_correlation", "weak", effect_x_correlation=0.3),
        scenario("severe_correlation", "strong", effect_x_correlation=0.95),
        scenario("negative_correlation", "negative", effect_x_correlation=-0.8),
        scenario("no_unit_heterogeneity", "boundary", sigma_alpha=0.0,
                 effect_x_correlation=0.0),
        scenario("short_panel", "small_sample", n_periods=2, n_units=60),
        scenario("long_panel", "large_sample", n_periods=40, n_units=20),
        scenario("few_units", "small_sample", n_units=6, n_periods=10),
        scenario("noisy", "high_noise", sigma_e=3.0),
        scenario("time_effects", "misspecification", time_effects=2.0),
    ),
    prerequisites=("regression.simple_linear",),
    related=("regression.fwl", "econometrics.omitted_variable_bias"),
    tags=("fixed effects", "random effects", "within", "between", "hausman",
          "first differences", "panel"),
    aliases=("fe re", "panel", "المعطيات الطولية", "donnees de panel",
             "within estimator", "hausman test"),
    backends=("numpy", "linearmodels"),
    references=(
        ref("Hausman, J. A. (1978). Specification tests in econometrics. "
            "Econometrica 46(6).", kind="paper", doi="10.2307/1913827"),
        ref("Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel "
            "Data.", kind="book"),
    ),
    curriculum_tags=("dz.econometrics2", "ksu.econ541", "aub.econ305"),
)


class PanelLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = self._generate(p, state.seed)
        est = panel_estimators(data["unit"], data["period"], data["y"], data["x"],
                               float(p["sigma_alpha"]), float(p["sigma_e"]))

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "units", self._units_figure(ctx, data, est, p),
            "labs.panel.figure.units", evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        res.add_panel(ctx.panel(
            "comparison", self._comparison_figure(ctx, est, p),
            "labs.panel.figure.comparison", tab="compare",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "within_between", self._within_between_figure(ctx, data, est),
            "labs.panel.figure.within_between", tab="math",
            evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_sampling"]:
            draws = self._sampling(p, state.seed, int(p["reps"]))
            res.add_panel(ctx.panel(
                "sampling", self._sampling_figure(ctx, draws, float(p["beta"])),
                "labs.panel.figure.sampling", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
            for key, label in (("pooled", "pooled OLS"), ("within", "fixed effects"),
                               ("random", "random effects"), ("fd", "first differences")):
                res.metric(f"sim_bias_{key}",
                           ctx.t("labs.panel.metric.sim_bias", "{m}: simulated bias",
                                 m=label),
                           float(np.mean(draws[key]) - float(p["beta"])), reference=0.0)

        beta = float(p["beta"])
        res.metric("pooled", ctx.t("labs.panel.metric.pooled", "Pooled OLS"),
                   est["pooled"].coef("x"), reference=beta)
        res.metric("between", ctx.t("labs.panel.metric.between", "Between estimator"),
                   est["between"].coef("x"), reference=beta)
        res.metric("within", ctx.t("labs.panel.metric.within",
                                   "Fixed effects (within)"),
                   est["within"].coef("x"), reference=beta)
        res.metric("within_se", ctx.t("labs.panel.metric.within_se",
                                      "Fixed-effects standard error"),
                   est["within"].se("x"))
        res.metric("first_differences", ctx.t("labs.panel.metric.fd",
                                              "First differences"),
                   est["fd"].coef("x"), reference=beta,
                   note=ctx.t("labs.panel.metric.fd_note",
                              "identical to fixed effects when T = 2"))
        res.metric("random", ctx.t("labs.panel.metric.random", "Random effects"),
                   est["random"].coef("x"), reference=beta)
        res.metric("random_se", ctx.t("labs.panel.metric.random_se",
                                      "Random-effects standard error"),
                   est["random"].se("x"))
        res.metric("theta", ctx.t("labs.panel.metric.theta",
                                  "Quasi-demeaning weight theta"), est["theta"],
                   note=ctx.t("labs.panel.metric.theta_note",
                              "0 gives pooled OLS, 1 gives fixed effects"))
        res.metric("hausman", ctx.t("labs.panel.metric.hausman",
                                    "Hausman statistic"), est["hausman"])
        res.metric("hausman_p", ctx.t("labs.panel.metric.hausman_p",
                                      "Hausman p-value"), est["hausman_p"],
                   note=ctx.t("labs.panel.metric.hausman_note",
                              "small values reject random effects in favour of fixed "
                              "effects"))
        res.metric("efficiency_cost", ctx.t("labs.panel.metric.cost",
                                            "Fixed-effects SE divided by random-effects SE"),
                   float(est["within"].se("x") / max(est["random"].se("x"), 1e-12)),
                   note=ctx.t("labs.panel.metric.cost_note",
                              "the price of robustness when random effects would have "
                              "been valid"))
        res.metric("rho", ctx.t("labs.panel.metric.rho",
                                "Share of variance from the unit effects"),
                   float(est["sigma2_alpha"] /
                         max(est["sigma2_alpha"] + est["sigma2_e"], 1e-12)))

        re_valid = abs(float(p["effect_x_correlation"])) < 0.05
        res.assume("random_effects", ctx.t("assumptions.random_effects"), re_valid,
                   detail=ctx.t("labs.panel.assume.re",
                                "Random effects requires the unit effect to be "
                                "uncorrelated with the regressors. Here the correlation is "
                                "set to {r}.", r=fmt(p["effect_x_correlation"], 2)),
                   consequence="" if re_valid else ctx.t(
                       "labs.panel.assume.re_violated",
                       "Pooled OLS and random effects are inconsistent; fixed effects "
                       "remains consistent because the within transformation removes the "
                       "offending term entirely."))
        res.assume("strict_exogeneity",
                   ctx.t("labs.panel.assume.strict_label",
                         "The regressor is strictly exogenous"), True,
                   detail=ctx.t("labs.panel.assume.strict",
                                "Fixed effects requires x_it to be uncorrelated with the "
                                "error in EVERY period, not just the current one. A lagged "
                                "dependent variable breaks this and creates Nickell bias."))
        res.assume("within_variation",
                   ctx.t("labs.panel.assume.variation_label",
                         "The regressor varies within units"),
                   float(np.var(data["x"] - np.array(
                       [data["x"][data["unit"] == u].mean() for u in data["unit"]]))) > 1e-8,
                   detail=ctx.t("labs.panel.assume.variation",
                                "Fixed effects cannot estimate the coefficient of anything "
                                "constant within a unit - the within transformation "
                                "annihilates it."))

        res.animations.append(self._animation(ctx, p, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.panel.fixed_vs_random.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.panel.fixed_vs_random.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.panel.fixed_vs_random.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.panel.fixed_vs_random.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.panel.fixed_vs_random.warning"), kind="warning")
        res.explain("diagnostics", ctx.t("labs.panel.decision.title",
                                         "Choosing between them"), ctx.t(
            "labs.panel.decision",
            "Hausman p = {p}. {verdict} But note what the test cannot do: it compares two "
            "estimators, so it is silent about time-varying unobservables, which bias both.",
            p=fmt(est["hausman_p"], 4),
            verdict=(ctx.t("labs.panel.decision_fe",
                           "The data reject the random-effects assumption, so fixed effects "
                           "is the safer choice here.")
                     if est["hausman_p"] < 0.05 else
                     ctx.t("labs.panel.decision_re",
                           "The data do not reject the random-effects assumption, so random "
                           "effects may be used for its extra precision.")),
        ))

        if not re_valid and est["hausman_p"] > 0.05:
            res.warnings.append(ctx.t(
                "labs.panel.warn.hausman_power",
                "The unit effect IS correlated with the regressor in this DGP, yet the "
                "Hausman test fails to reject. With few units or short panels the test has "
                "little power - failing to reject is not evidence that random effects is "
                "safe.",
            ))
        return res

    @staticmethod
    def _generate(p, seed):
        return panel_dataset(
            n_units=int(p["n_units"]), n_periods=int(p["n_periods"]),
            beta=float(p["beta"]),
            effect_x_correlation=float(p["effect_x_correlation"]),
            sigma_alpha=float(p["sigma_alpha"]), sigma_e=float(p["sigma_e"]),
            time_effects=float(p["time_effects"]), seed=seed,
        )

    def _sampling(self, p, seed, reps):
        reps = min(reps, 2000)

        def experiment(gen):
            child = int(gen.integers(0, 2**31 - 1))
            d = self._generate(p, child)
            e = panel_estimators(d["unit"], d["period"], d["y"], d["x"])
            return {"pooled": e["pooled"].coef("x"), "within": e["within"].coef("x"),
                    "random": e["random"].coef("x"), "fd": e["fd"].coef("x"),
                    "between": e["between"].coef("x")}

        return monte_carlo(experiment, reps, seed).draws

    def _units_figure(self, ctx, data, est, p):
        fig = ctx.figure(
            "labs.panel.figure.units",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=440,
        )
        units = np.unique(data["unit"])
        show = units[: min(units.size, 14)]
        for i, u in enumerate(show):
            mask = data["unit"] == u
            P.add_points(fig, data["x"][mask], data["y"][mask],
                         ctx.t("labs.panel.trace.unit", "unit {u}", u=int(u)),
                         "primary" if i % 2 else "secondary", theme=ctx.theme,
                         size=7, opacity=0.75, showlegend=False)
            if p["show_unit_lines"] and mask.sum() > 2:
                f = LM.ols(data["y"][mask],
                           np.column_stack([np.ones(int(mask.sum())), data["x"][mask]]))
                order = np.argsort(data["x"][mask])
                P.add_curve(fig, data["x"][mask][order], f.fitted_values[order],
                            "", "muted", theme=ctx.theme, width=1.2, opacity=0.6,
                            showlegend=False)
        xs = np.linspace(float(data["x"].min()), float(data["x"].max()), 50)
        P.add_curve(fig, xs, est["pooled"].coef("const") + est["pooled"].coef("x") * xs,
                    ctx.t("labs.panel.trace.pooled_line",
                          "pooled OLS: slope {b}", b=fmt(est["pooled"].coef("x"), 3)),
                    "negative", theme=ctx.theme, width=3.2)
        centre_y = float(np.mean(data["y"]))
        centre_x = float(np.mean(data["x"]))
        P.add_curve(fig, xs, centre_y + est["within"].coef("x") * (xs - centre_x),
                    ctx.t("labs.panel.trace.within_line",
                          "fixed effects: slope {b}", b=fmt(est["within"].coef("x"), 3)),
                    "positive", theme=ctx.theme, width=3.2, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.panel.legend_units",
            "Each cluster is one unit. The thin lines are the within-unit slopes that fixed "
            "effects averages; the thick red line cuts across clusters and picks up their "
            "level differences as if they were slope.",
        ), theme=ctx.theme)
        return fig

    def _comparison_figure(self, ctx, est, p):
        go = P.require_plotly()
        names = ["pooled", "between", "within", "fd", "random"]
        labels = [ctx.t("labs.panel.metric.pooled", "Pooled OLS"),
                  ctx.t("labs.panel.metric.between", "Between estimator"),
                  ctx.t("labs.panel.metric.within", "Fixed effects (within)"),
                  ctx.t("labs.panel.metric.fd", "First differences"),
                  ctx.t("labs.panel.metric.random", "Random effects")]
        values = [est[k].coef("x") for k in names]
        errors = [1.96 * est[k].se("x") for k in names]
        fig = ctx.figure(
            "labs.panel.figure.comparison",
            xaxis_title=ctx.t("labs.panel.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=390,
        )
        fig.add_trace(go.Bar(x=labels, y=values, error_y={"type": "data", "array": errors},
                             marker={"color": [ctx.color("negative"), ctx.color("muted"),
                                               ctx.color("positive"), ctx.color("info"),
                                               ctx.color("warning")]},
                             text=[fmt(v, 3) for v in values], textposition="outside",
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, float(p["beta"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.panel.legend_comparison",
            "Only estimators that eliminate the unit effect stay on the dashed line when "
            "that effect is correlated with x. Fixed effects and first differences do; "
            "pooled OLS, between and random effects do not.",
        ), theme=ctx.theme)
        return fig

    def _within_between_figure(self, ctx, data, est):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        unit = data["unit"]
        ymean = np.array([data["y"][unit == u].mean() for u in unit])
        xmean = np.array([data["x"][unit == u].mean() for u in unit])
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.panel.trace.between_var",
                  "between variation: unit means"),
            ctx.t("labs.panel.trace.within_var",
                  "within variation: deviations from unit means"),
        ))
        units = np.unique(unit)
        fig.add_trace(go.Scatter(
            x=[data["x"][unit == u].mean() for u in units],
            y=[data["y"][unit == u].mean() for u in units],
            mode="markers", marker={"color": ctx.color("muted"), "size": 9},
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data["x"] - xmean, y=data["y"] - ymean, mode="markers",
            marker={"color": ctx.color("positive"), "size": 5, "opacity": 0.6},
            showlegend=False), row=1, col=2)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=360, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.panel.legend_within_between",
            "Pooled OLS is a weighted average of these two pictures. Fixed effects throws "
            "the left one away entirely - which is what makes it robust and also what makes "
            "it less precise.",
        ), theme=ctx.theme)
        return fig

    def _sampling_figure(self, ctx, draws, beta):
        fig = ctx.figure(
            "labs.panel.figure.sampling",
            xaxis_title=ctx.t("labs.common.axis.estimate"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=370,
        )
        for key, role, label in (
            ("pooled", "negative", ctx.t("labs.panel.metric.pooled", "Pooled OLS")),
            ("random", "warning", ctx.t("labs.panel.metric.random", "Random effects")),
            ("within", "positive", ctx.t("labs.panel.metric.within",
                                         "Fixed effects (within)")),
        ):
            P.add_histogram(fig, draws[key], label, role, theme=ctx.theme,
                            nbins=50, opacity=0.5)
        P.add_vline(fig, beta, ctx.t("labs.common.trace.truth"), "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.panel.legend_sampling",
            "Bias and precision are visible at once: fixed effects is centred on the truth "
            "but wider; random effects is narrow and displaced. Only one of those two "
            "problems can be fixed with more data.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, seed):
        go = P.require_plotly()
        corrs = np.linspace(0.0, 0.95, 16)
        frames, steps = [], []
        for i, c in enumerate(corrs):
            q = dict(p)
            q["effect_x_correlation"] = float(c)
            d = self._generate(q, seed)
            e = panel_estimators(d["unit"], d["period"], d["y"], d["x"])
            frames.append(go.Frame(name=f"{c:.2f}", data=[
                go.Bar(x=["pooled", "random", "fixed effects"],
                       y=[e["pooled"].coef("x"), e["random"].coef("x"),
                          e["within"].coef("x")]),
            ]))
            steps.append(AnimationStep(
                id=f"corr_{i}", frame=i,
                title=ctx.t("labs.panel.anim.title",
                            "corr(unit effect, x) = {c}", c=fmt(c, 2)),
                what_you_see=ctx.t("labs.panel.anim.see",
                                   "The three main panel estimators computed on the same "
                                   "data."),
                what_changed=ctx.t("labs.panel.anim.changed",
                                   "The correlation between the unobserved unit effect and "
                                   "the regressor moved to {c}.", c=fmt(c, 2)),
                why=ctx.t("labs.panel.anim.why",
                          "Pooled OLS leaves the unit effect in the error; random effects "
                          "only partly removes it; the within transformation removes it "
                          "exactly, whatever its correlation with x."),
                interpretation=ctx.t("labs.panel.anim.interpret",
                                     "Pooled {a}, random {b}, fixed effects {c} against a "
                                     "true value of {t}. Hausman p = {h}.",
                                     a=fmt(e["pooled"].coef("x"), 3),
                                     b=fmt(e["random"].coef("x"), 3),
                                     c=fmt(e["within"].coef("x"), 3),
                                     t=fmt(p["beta"], 3), h=fmt(e["hausman_p"], 4)),
                conclusion=ctx.t("labs.panel.anim.conclude",
                                 "Only the estimator that eliminates the unit effect stays "
                                 "put as the correlation grows."),
                warning=ctx.t("labs.panel.anim.warn",
                              "Fixed effects removes only what is constant within a unit. "
                              "Anything unobserved that varies over time still biases it."),
                math="within: y_it - ybar_i = beta (x_it - xbar_i) + (e_it - ebar_i)",
                outputs={"correlation": round(float(c), 3),
                         "pooled": round(e["pooled"].coef("x"), 4),
                         "random": round(e["random"].coef("x"), 4),
                         "within": round(e["within"].coef("x"), 4),
                         "hausman_p": round(float(e["hausman_p"]), 5)},
                violated_assumptions=("random_effects",) if c > 0.05 else (),
                highlighted=("estimator_bars",),
            ))
        fig = ctx.figure(
            "labs.panel.figure.animation",
            xaxis_title=ctx.t("labs.panel.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=380,
        )
        fig.add_trace(go.Bar(x=["pooled", "random", "fixed effects"], y=[0, 0, 0],
                             marker={"color": [ctx.color("negative"), ctx.color("warning"),
                                               ctx.color("positive")]},
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, float(p["beta"]), ctx.t("labs.common.trace.truth"), "truth",
                    theme=ctx.theme, dash="dash")
        fig.update_yaxes(range=[min(0, float(p["beta"])) - 1.0,
                                float(p["beta"]) + 3.0])
        build_frames(fig, frames, duration=440, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.panel.slider", "correlation"))
        return animation(
            "correlation_sweep", fig, steps,
            purpose=ctx.t("labs.panel.anim.purpose",
                          "Show exactly which assumption separates the panel estimators."),
            summary=ctx.t(
                "labs.panel.anim.summary",
                "The choice between fixed and random effects is a single question: is the "
                "unobserved unit effect related to your regressor? If yes, only the "
                "estimators that eliminate it survive. Random effects is more efficient "
                "precisely because it does not throw that variation away - which is also "
                "why it breaks."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = PanelLab(SPEC)
