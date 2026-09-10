"""Simultaneous equations, identification, 2SLS/3SLS and SUR."""

from __future__ import annotations

from typing import Any

from ...backends import linear as LM
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
    ref,
    scenario,
    seed_control,
    select,
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC"]

SYSTEMS = ("supply_demand", "wage_price", "sur_only")


SPEC = make_spec(
    "econometrics.simultaneous_equations",
    Domain.ECONOMETRICS,
    "systems",
    module=__name__,
    levels=("advanced", "phd"),
    modes=("learn", "visualize", "experiment", "compare", "derive", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        select("system", "supply_demand", SYSTEMS, group="system"),
        slider("demand_slope", -1.0, -4.0, -0.05, 0.05, group="structure"),
        slider("supply_slope", 0.8, 0.05, 4.0, 0.05, group="structure"),
        slider("demand_shifter", 1.0, 0.0, 3.0, 0.05, group="identification"),
        slider("supply_shifter", 1.0, 0.0, 3.0, 0.05, group="identification"),
        slider("cross_equation_corr", 0.0, -0.95, 0.95, 0.01, group="structure"),
        int_slider("n", 400, 30, 20000, 10, group="dgp"),
        slider("shock_sd", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        int_slider("reps", 500, 100, 5000, 50, group="simulation", expensive=True),
        toggle("show_scatter", True, group="views"),
        toggle("show_sur", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", system="supply_demand", demand_shifter=1.0,
                 supply_shifter=1.0),
        scenario("under_identified", "boundary", demand_shifter=0.0, supply_shifter=0.0),
        scenario("demand_only_identified", "violation", demand_shifter=0.0,
                 supply_shifter=1.5),
        scenario("exactly_identified", "canonical", demand_shifter=1.5,
                 supply_shifter=1.5),
        scenario("weak_shifters", "weak", demand_shifter=0.08, supply_shifter=0.08),
        scenario("correlated_shocks", "compare_methods", cross_equation_corr=0.8),
        scenario("sur_gains", "compare_methods", system="sur_only",
                 cross_equation_corr=0.85),
        scenario("sur_no_gain", "null", system="sur_only", cross_equation_corr=0.0),
        scenario("wage_price", "compare_methods", system="wage_price"),
        scenario("small_sample", "small_sample", n=50),
        scenario("large_sample", "large_sample", n=8000),
    ),
    prerequisites=("econometrics.endogeneity_iv",),
    related=("econometrics.endogeneity_iv",),
    tags=("simultaneity", "2sls", "3sls", "sur", "identification", "order condition",
          "reduced form"),
    aliases=("simultaneous equations", "equations simultanees", "المعادلات الآنية",
             "seemingly unrelated regressions", "three stage least squares"),
    backends=("numpy", "linearmodels"),
    references=(
        ref("Zellner, A. (1962). An efficient method of estimating seemingly unrelated "
            "regressions. JASA 57(298).", kind="paper", doi="10.1080/01621459.1962.10480664"),
        ref("Greene, W. H. (2018). Econometric Analysis.", kind="book"),
    ),
    curriculum_tags=("dz.econometrics2", "ksu.econ541"),
)


class SystemsLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        system = str(p["system"])
        n = int(p["n"])
        data = self._generate(p, n, state.seed)

        if system == "sur_only":
            return self._sur_lab(ctx, res, data, p, state)

        order_ok = float(p["supply_shifter"]) > 1e-6
        estimates = self._estimate_system(data, p, order_ok)

        res.dgp = data["dgp"]
        res.add_panel(ctx.panel(
            "structure", self._structure_figure(ctx, data, p),
            "labs.sys.figure.structure", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_scatter"]:
            res.add_panel(ctx.panel(
                "scatter", self._scatter_figure(ctx, data, estimates, p),
                "labs.sys.figure.scatter", tab="visualize",
                evidence=EvidenceType.EMPIRICAL_EXAMPLE,
            ))
        res.add_panel(ctx.panel(
            "comparison", self._comparison_figure(ctx, estimates, p),
            "labs.sys.figure.comparison", tab="compare",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        sim = self._sampling(p, n, state.seed, int(p["reps"]), order_ok)
        res.add_panel(ctx.panel(
            "sampling", self._sampling_figure(ctx, sim, p),
            "labs.sys.figure.sampling", tab="simulation",
            evidence=EvidenceType.SIMULATION,
        ))

        res.metric("true_demand_slope", ctx.t("labs.sys.metric.true_demand",
                                              "True demand slope"),
                   float(p["demand_slope"]))
        res.metric("ols_demand", ctx.t("labs.sys.metric.ols",
                                       "OLS estimate of the demand slope"),
                   estimates["ols"], reference=float(p["demand_slope"]),
                   note=ctx.t("labs.sys.metric.ols_note",
                              "a mixture of both curves, not either one"))
        if order_ok:
            res.metric("tsls_demand", ctx.t("labs.sys.metric.tsls",
                                            "2SLS estimate of the demand slope"),
                       estimates["2sls"], reference=float(p["demand_slope"]))
            res.metric("first_stage_f", ctx.t("labs.iv.metric.f",
                                              "First-stage F on the excluded shifter"),
                       estimates["first_stage_f"])
        res.metric("reduced_form_price", ctx.t("labs.sys.metric.rf_price",
                                               "Reduced-form coefficient of price on the "
                                               "supply shifter"),
                   estimates["rf_price"])
        res.metric("reduced_form_quantity", ctx.t("labs.sys.metric.rf_quantity",
                                                  "Reduced-form coefficient of quantity on "
                                                  "the supply shifter"),
                   estimates["rf_quantity"])
        res.metric("indirect_least_squares", ctx.t("labs.sys.metric.ils",
                                                   "Indirect least squares (ratio of the "
                                                   "two reduced-form coefficients)"),
                   estimates["ils"], reference=float(p["demand_slope"]),
                   note=ctx.t("labs.sys.metric.ils_note",
                              "identical to 2SLS when the equation is exactly identified"))
        res.metric("order_condition", ctx.t("labs.sys.metric.order",
                                            "Order condition for the demand equation"),
                   ctx.t("labs.sys.satisfied", "satisfied") if order_ok
                   else ctx.t("labs.sys.failed", "FAILED - no excluded shifter"))
        res.metric("sim_median_ols", ctx.t("labs.sys.metric.sim_ols",
                                           "Median OLS estimate across studies"),
                   float(np.median(sim["ols"])), reference=float(p["demand_slope"]))
        if order_ok:
            res.metric("sim_median_2sls", ctx.t("labs.sys.metric.sim_2sls",
                                                "Median 2SLS estimate across studies"),
                       float(np.nanmedian(sim["2sls"])),
                       reference=float(p["demand_slope"]))

        res.assume("exogeneity", ctx.t("assumptions.exogeneity"), False,
                   detail=ctx.t("labs.sys.assume.simultaneity",
                                "Price and quantity are determined together, so price is "
                                "correlated with the demand shock by construction."),
                   consequence=ctx.t("labs.sys.assume.consequence",
                                     "Least squares on the demand equation estimates "
                                     "neither the demand nor the supply slope, but a "
                                     "variance-weighted mixture of the two."))
        res.assume("order_condition", ctx.t("labs.sys.assume.order_label",
                                            "Order condition: at least one excluded "
                                            "exogenous variable per endogenous regressor"),
                   order_ok,
                   detail=ctx.t("labs.sys.assume.order",
                                "The demand equation is identified only by a shifter that "
                                "moves supply but is excluded from demand."),
                   consequence="" if order_ok else ctx.t(
                       "labs.sys.assume.order_failed",
                       "Without an excluded shifter, no estimator - however sophisticated - "
                       "can recover the structural slopes from these data."))
        res.assume("exclusion_valid",
                   ctx.t("labs.sys.assume.exclusion_label",
                         "The excluded shifter really is excluded"), True,
                   detail=ctx.t("labs.sys.assume.exclusion",
                                "Identification here is an assumption about which variables "
                                "belong in which equation. It is not testable in an exactly "
                                "identified system."))

        res.animations.append(self._animation(ctx, p, n, state.seed))
        self._explain(ctx, res, order_ok, estimates, p)
        if not order_ok:
            res.warnings.append(ctx.t(
                "labs.sys.warn.unidentified",
                "With no excluded shifter, the scatter of price against quantity traces the "
                "intersection of two moving curves. Any line through that cloud is "
                "consistent with infinitely many pairs of structural slopes - the "
                "parameters are simply not identified.",
            ))
        return res

    # -- SUR branch --------------------------------------------------------
    def _sur_lab(self, ctx, res, data, p, state):
        y1, y2, X1, X2 = data["y1"], data["y2"], data["X1"], data["X2"]
        f1 = LM.ols(y1, X1, names=("const", "x1"))
        f2 = LM.ols(y2, X2, names=("const", "x2", "x3"))
        sur = self._sur(y1, y2, X1, X2)
        res.dgp = data["dgp"]

        res.add_panel(ctx.panel(
            "residual_correlation", self._sur_residual_figure(ctx, f1, f2),
            "labs.sys.figure.sur_resid", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "sur_gain", self._sur_gain_figure(ctx, p, state.seed),
            "labs.sys.figure.sur_gain", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))

        res.metric("residual_correlation", ctx.t("labs.sys.metric.resid_corr",
                                                 "Correlation of the two equations' "
                                                 "residuals"),
                   float(np.corrcoef(f1.residuals, f2.residuals)[0, 1]),
                   reference=float(p["cross_equation_corr"]))
        res.metric("ols_se_eq1", ctx.t("labs.sys.metric.ols_se",
                                       "Equation 1 slope SE, equation-by-equation OLS"),
                   f1.se("x1"))
        res.metric("sur_se_eq1", ctx.t("labs.sys.metric.sur_se",
                                       "Equation 1 slope SE, SUR"), sur["se1"])
        res.metric("efficiency_gain", ctx.t("labs.sys.metric.gain",
                                            "SUR standard error divided by the OLS one"),
                   float(sur["se1"] / max(f1.se("x1"), 1e-12)),
                   note=ctx.t("labs.sys.metric.gain_note",
                              "below 1 means SUR was worth it"))
        res.assume("cross_equation_correlation",
                   ctx.t("labs.sys.assume.sur_label",
                         "The equations' errors are correlated"),
                   abs(float(p["cross_equation_corr"])) > 0.05,
                   detail=ctx.t("labs.sys.assume.sur",
                                "SUR gains efficiency only from correlated errors combined "
                                "with different regressor sets. With identical regressors "
                                "it collapses exactly to equation-by-equation OLS."))
        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.econometrics.simultaneous_equations.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.econometrics.simultaneous_equations.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.econometrics.simultaneous_equations.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.econometrics.simultaneous_equations.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.econometrics.simultaneous_equations.warning"),
                    kind="warning")
        return res

    # -- generation --------------------------------------------------------
    def _generate(self, p, n, seed):
        gen = rng(seed, "systems")
        system = str(p["system"])
        corr = float(p["cross_equation_corr"])
        sd = float(p["shock_sd"])
        cov = np.array([[sd**2, corr * sd * sd], [corr * sd * sd, sd**2]])
        shocks = gen.multivariate_normal(np.zeros(2), cov, size=n)

        if system == "sur_only":
            x1 = gen.standard_normal(n)
            x2 = gen.standard_normal(n)
            x3 = gen.standard_normal(n)
            y1 = 1.0 + 1.5 * x1 + shocks[:, 0]
            y2 = 0.5 - 0.8 * x2 + 1.2 * x3 + shocks[:, 1]
            return {
                "y1": y1, "y2": y2,
                "X1": np.column_stack([np.ones(n), x1]),
                "X2": np.column_stack([np.ones(n), x2, x3]),
                "dgp": (f"Two unrelated-looking equations with different regressors and "
                        f"corr(e1, e2) = {corr:g}; n = {n}."),
            }

        bd, bs = float(p["demand_slope"]), float(p["supply_slope"])
        wd, ws = float(p["demand_shifter"]), float(p["supply_shifter"])
        income = gen.standard_normal(n)
        cost = gen.standard_normal(n)
        # demand:  q = a_d + bd * price + wd * income + e_d
        # supply:  q = a_s + bs * price + ws * cost   + e_s
        denom = bs - bd
        price = ((wd * income - ws * cost) + (shocks[:, 0] - shocks[:, 1])) / denom
        quantity = bs * price + ws * cost + shocks[:, 1]
        label = ("wage and price equations" if system == "wage_price"
                 else "supply and demand")
        return {
            "price": price, "quantity": quantity, "income": income, "cost": cost,
            "dgp": (f"Structural {label}: demand slope {bd:g} shifted by income "
                    f"({wd:g}), supply slope {bs:g} shifted by cost ({ws:g}); "
                    f"corr(shocks) = {corr:g}; n = {n}."),
        }

    def _estimate_system(self, data, p, order_ok):
        price, quantity = data["price"], data["quantity"]
        income, cost = data["income"], data["cost"]
        n = price.size
        X = np.column_stack([np.ones(n), price, income])
        ols = LM.ols(quantity, X, names=("const", "price", "income"))
        rf_p = LM.ols(price, np.column_stack([np.ones(n), income, cost]),
                      names=("const", "income", "cost"))
        rf_q = LM.ols(quantity, np.column_stack([np.ones(n), income, cost]),
                      names=("const", "income", "cost"))
        out = {
            "ols": ols.coef("price"),
            "ols_se": ols.se("price"),
            "rf_price": rf_p.coef("cost"),
            "rf_quantity": rf_q.coef("cost"),
            "ils": (rf_q.coef("cost") / rf_p.coef("cost")
                    if abs(rf_p.coef("cost")) > 1e-9 else float("nan")),
            "2sls": float("nan"),
            "2sls_se": float("nan"),
            "first_stage_f": float("nan"),
        }
        if order_ok:
            Z = np.column_stack([np.ones(n), income, cost])
            iv = LM.iv_2sls(quantity, X, Z, endog_index=1,
                            names=("const", "price", "income"))
            out["2sls"] = iv.second_stage.coef("price")
            out["2sls_se"] = iv.second_stage.se("price")
            out["first_stage_f"] = iv.first_stage_f
        return out

    @staticmethod
    def _sur(y1, y2, X1, X2):
        """Two-equation feasible SUR (Zellner) via a stacked GLS step."""
        f1 = LM.ols(y1, X1)
        f2 = LM.ols(y2, X2)
        u = np.column_stack([f1.residuals, f2.residuals])
        sigma = (u.T @ u) / u.shape[0]
        sigma_inv = np.linalg.pinv(sigma)
        k1, k2 = X1.shape[1], X2.shape[1]
        n = y1.size
        XtX = np.zeros((k1 + k2, k1 + k2))
        Xty = np.zeros(k1 + k2)
        blocks = [(X1, y1), (X2, y2)]
        for i, (Xi, _yi) in enumerate(blocks):
            for j, (Xj, _yj) in enumerate(blocks):
                r0 = 0 if i == 0 else k1
                c0 = 0 if j == 0 else k1
                XtX[r0:r0 + Xi.shape[1], c0:c0 + Xj.shape[1]] += sigma_inv[i, j] * (Xi.T @ Xj)
            r0 = 0 if i == 0 else k1
            Xty[r0:r0 + Xi.shape[1]] += sum(
                sigma_inv[i, j] * (Xi.T @ blocks[j][1]) for j in range(2))
        cov = np.linalg.pinv(XtX)
        beta = cov @ Xty
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        del n
        return {"beta": beta, "se": se, "se1": float(se[1]), "sigma": sigma}

    def _sampling(self, p, n, seed, reps, order_ok):
        reps = min(reps, 2000)
        ols_vals, iv_vals = [], []
        for r in range(reps):
            d = self._generate(p, n, int(seed) * 32452843 + r)
            e = self._estimate_system(d, p, order_ok)
            ols_vals.append(e["ols"])
            iv_vals.append(e["2sls"])
        return {"ols": np.asarray(ols_vals), "2sls": np.asarray(iv_vals)}

    # -- figures -----------------------------------------------------------
    def _structure_figure(self, ctx, data, p):
        bd, bs = float(p["demand_slope"]), float(p["supply_slope"])
        price = data["price"]
        grid = np.linspace(float(np.percentile(price, 2)),
                           float(np.percentile(price, 98)), 60)
        fig = ctx.figure(
            "labs.sys.figure.structure",
            xaxis_title=ctx.t("labs.sys.axis.price", "Price"),
            yaxis_title=ctx.t("labs.sys.axis.quantity", "Quantity"),
            height=420,
        )
        gen = rng(1, "curves")
        for i in range(6):
            shift_d = gen.normal(0, float(p["demand_shifter"]) + 0.4)
            shift_s = gen.normal(0, float(p["supply_shifter"]) + 0.4)
            P.add_curve(fig, grid, bd * grid + shift_d,
                        ctx.t("labs.sys.trace.demand", "demand curves"),
                        "negative", theme=ctx.theme, opacity=0.55, showlegend=i == 0)
            P.add_curve(fig, grid, bs * grid + shift_s,
                        ctx.t("labs.sys.trace.supply", "supply curves"),
                        "positive", theme=ctx.theme, opacity=0.55, showlegend=i == 0)
        P.add_points(fig, price, data["quantity"],
                     ctx.t("labs.sys.trace.equilibria", "observed equilibria"),
                     "primary", theme=ctx.theme, size=5, opacity=0.5)
        P.add_legend_note(fig, ctx.t(
            "labs.sys.legend_structure",
            "You never observe a curve - only where two moving curves crossed. Recovering a "
            "slope requires knowing something that shifts the OTHER curve and not this one.",
        ), theme=ctx.theme)
        return fig

    def _scatter_figure(self, ctx, data, estimates, p):
        price, quantity = data["price"], data["quantity"]
        order = np.argsort(price)
        fig = ctx.figure(
            "labs.sys.figure.scatter",
            xaxis_title=ctx.t("labs.sys.axis.price", "Price"),
            yaxis_title=ctx.t("labs.sys.axis.quantity", "Quantity"),
            height=400,
        )
        P.add_points(fig, price, quantity,
                     ctx.t("labs.sys.trace.equilibria", "observed equilibria"),
                     "primary", theme=ctx.theme, size=5, opacity=0.5)
        centre_q = float(np.mean(quantity))
        centre_p = float(np.mean(price))
        for slope, role, label in (
            (estimates["ols"], "negative", ctx.t("labs.sys.trace.ols_line", "OLS slope")),
            (estimates["2sls"], "primary", ctx.t("labs.sys.trace.tsls_line", "2SLS slope")),
            (float(p["demand_slope"]), "truth",
             ctx.t("labs.sys.trace.true_demand", "true demand slope")),
        ):
            if not np.isfinite(slope):
                continue
            P.add_curve(fig, price[order],
                        centre_q + slope * (price[order] - centre_p), label,
                        role, theme=ctx.theme, width=3.0,
                        dash="dash" if role == "truth" else "solid")
        P.add_legend_note(fig, ctx.t(
            "labs.sys.legend_scatter",
            "All three lines are drawn through the same cloud. Only the one built from an "
            "excluded shifter lands on the structural slope.",
        ), theme=ctx.theme)
        return fig

    def _comparison_figure(self, ctx, estimates, p):
        go = P.require_plotly()
        labels = [ctx.t("labs.sys.trace.ols_line", "OLS slope"),
                  ctx.t("labs.sys.metric.ils_short", "indirect least squares"),
                  ctx.t("labs.sys.trace.tsls_line", "2SLS slope")]
        values = [estimates["ols"], estimates["ils"], estimates["2sls"]]
        fig = ctx.figure(
            "labs.sys.figure.comparison",
            xaxis_title=ctx.t("labs.iv.axis.estimator", "Estimator"),
            yaxis_title=ctx.t("labs.sys.axis.slope", "Estimated demand slope"),
            height=360,
        )
        keep = [i for i, v in enumerate(values) if np.isfinite(v)]
        fig.add_trace(go.Bar(x=[labels[i] for i in keep], y=[values[i] for i in keep],
                             marker={"color": [ctx.color("negative"), ctx.color("secondary"),
                                               ctx.color("primary")][: len(keep)]},
                             text=[fmt(values[i], 4) for i in keep],
                             textposition="outside",
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, float(p["demand_slope"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.sys.legend_comparison",
            "Indirect least squares and 2SLS coincide exactly when the equation is exactly "
            "identified. With more instruments than needed, 2SLS pools them and ILS is no "
            "longer unique.",
        ), theme=ctx.theme)
        return fig

    def _sampling_figure(self, ctx, sim, p):
        fig = ctx.figure(
            "labs.sys.figure.sampling",
            xaxis_title=ctx.t("labs.sys.axis.slope", "Estimated demand slope"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=360,
        )
        P.add_histogram(fig, sim["ols"], "OLS", "negative", theme=ctx.theme,
                        nbins=50, opacity=0.55)
        finite = sim["2sls"][np.isfinite(sim["2sls"])]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            P.add_histogram(fig, finite[(finite > lo) & (finite < hi)], "2SLS",
                            "primary", theme=ctx.theme, nbins=50, opacity=0.55)
        P.add_vline(fig, float(p["demand_slope"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.sys.legend_sampling",
            "Simultaneity bias is systematic: the OLS histogram sits away from the truth in "
            "every study, not merely in this one.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _sur_residual_figure(self, ctx, f1, f2):
        fig = ctx.figure(
            "labs.sys.figure.sur_resid",
            xaxis_title=ctx.t("labs.sys.axis.resid1", "Residual, equation 1"),
            yaxis_title=ctx.t("labs.sys.axis.resid2", "Residual, equation 2"),
            height=400,
        )
        P.add_points(fig, f1.residuals, f2.residuals,
                     ctx.t("labs.sys.trace.resid_pairs", "residual pairs"),
                     "primary", theme=ctx.theme, size=5, opacity=0.55)
        P.add_legend_note(fig, ctx.t(
            "labs.sys.legend_sur_resid",
            "SUR exploits exactly this cloud. A tilt means one equation's residual carries "
            "information about the other's, and system estimation can use it. A round cloud "
            "means there is nothing to gain.",
        ), theme=ctx.theme)
        return fig

    def _sur_gain_figure(self, ctx, p, seed):
        corrs = np.linspace(0.0, 0.95, 10)
        ratios = []
        for c in corrs:
            q = dict(p)
            q["cross_equation_corr"] = float(c)
            gains = []
            for r in range(30):
                d = self._generate(q, int(p["n"]), int(seed) * 7 + r)
                f1 = LM.ols(d["y1"], d["X1"], names=("const", "x1"))
                sur = self._sur(d["y1"], d["y2"], d["X1"], d["X2"])
                gains.append(sur["se1"] / max(f1.se("x1"), 1e-12))
            ratios.append(float(np.mean(gains)))
        fig = ctx.figure(
            "labs.sys.figure.sur_gain",
            xaxis_title=ctx.t("labs.sys.axis.corr",
                              "Correlation between the equations' errors"),
            yaxis_title=ctx.t("labs.sys.axis.ratio", "SUR SE / OLS SE"),
            height=350,
        )
        P.add_curve(fig, corrs, ratios,
                    ctx.t("labs.sys.trace.gain", "efficiency ratio"),
                    "primary", theme=ctx.theme, mode="lines+markers")
        P.add_hline(fig, 1.0, ctx.t("labs.sys.trace.no_gain", "no gain"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.sys.legend_sur_gain",
            "SUR is worth using only where this curve sits below the dashed line. At zero "
            "correlation - or with identical regressors in both equations - it reduces "
            "exactly to running each equation separately.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        shifters = np.linspace(0.0, 2.5, 14)
        frames, steps = [], []
        for i, s in enumerate(shifters):
            q = dict(p)
            q["supply_shifter"] = float(s)
            d = self._generate(q, n, seed)
            e = self._estimate_system(d, q, s > 1e-6)
            frames.append(go.Frame(name=f"{s:.2f}", data=[
                go.Scatter(x=d["price"], y=d["quantity"]),
            ]))
            steps.append(AnimationStep(
                id=f"shift_{i}", frame=i,
                title=ctx.t("labs.sys.anim.title", "supply shifter = {s}", s=fmt(s, 2)),
                what_you_see=ctx.t("labs.sys.anim.see",
                                   "Observed price-quantity equilibria as the cost shifter "
                                   "is strengthened."),
                what_changed=ctx.t("labs.sys.anim.changed",
                                   "The variable that moves supply but not demand now has "
                                   "coefficient {s}.", s=fmt(s, 2)),
                why=ctx.t("labs.sys.anim.why",
                          "When only supply moves, the equilibria trace out the demand "
                          "curve. That tracing is what identification means."),
                interpretation=ctx.t("labs.sys.anim.interpret",
                                     "2SLS gives {iv} against a true demand slope of {t}; "
                                     "OLS gives {ols}.",
                                     iv=fmt(e["2sls"], 3), t=fmt(p["demand_slope"], 3),
                                     ols=fmt(e["ols"], 3)),
                conclusion=ctx.t("labs.sys.anim.conclude",
                                 "At zero the cloud is an uninformative blob; as the shifter "
                                 "strengthens, the cloud aligns with the demand curve."),
                warning=ctx.t("labs.sys.anim.warn",
                              "OLS never converges to either structural slope, at any "
                              "shifter strength."),
                math="order condition: excluded exogenous variables >= endogenous regressors",
                outputs={"supply_shifter": round(float(s), 3),
                         "tsls": None if not np.isfinite(e["2sls"]) else round(e["2sls"], 4),
                         "ols": round(e["ols"], 4)},
                violated_assumptions=("order_condition",) if s <= 1e-6 else (),
                highlighted=("equilibria",),
            ))
        d0 = self._generate(p, n, seed)
        fig = ctx.figure(
            "labs.sys.figure.animation",
            xaxis_title=ctx.t("labs.sys.axis.price", "Price"),
            yaxis_title=ctx.t("labs.sys.axis.quantity", "Quantity"),
            height=390,
        )
        fig.add_trace(go.Scatter(x=d0["price"], y=d0["quantity"], mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 5,
                                         "opacity": 0.5},
                                 name=ctx.t("labs.sys.trace.equilibria",
                                            "observed equilibria")))
        build_frames(fig, frames, duration=460, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.sys.slider", "supply shifter"))
        return animation(
            "identification_emerges", fig, steps,
            purpose=ctx.t("labs.sys.anim.purpose",
                          "Show identification appearing as one curve is made to move."),
            summary=ctx.t(
                "labs.sys.anim.summary",
                "Identification in a simultaneous system is not a property of the estimator "
                "but of the data-generating process: something must move one curve while "
                "leaving the other alone. Without that, no method recovers the structure."),
            evidence=EvidenceType.SIMULATION,
        )

    def _explain(self, ctx, res, order_ok, estimates, p):
        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.econometrics.simultaneous_equations.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.econometrics.simultaneous_equations.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.econometrics.simultaneous_equations.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.econometrics.simultaneous_equations.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.econometrics.simultaneous_equations.warning"),
                    kind="warning")


LAB = SystemsLab(SPEC)
