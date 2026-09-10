"""Omitted-variable bias: the coefficient moves, and the formula says by how much."""

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


SPEC = make_spec(
    "econometrics.omitted_variable_bias",
    Domain.ECONOMETRICS,
    "specification",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive",
           "simulate", "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SYMBOLIC_DERIVATION,
    controls=(
        slider("beta_x", 1.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("beta_z", 2.0, -5.0, 5.0, 0.05, group="dgp"),
        slider("corr_xz", 0.6, -0.95, 0.95, 0.01, group="dgp"),
        int_slider("n", 300, 20, 20000, 10, group="dgp"),
        slider("noise", 1.0, 0.05, 10.0, 0.05, group="dgp"),
        select("omitted_role", "confounder",
               ("confounder", "mediator", "collider", "irrelevant"), group="structure"),
        int_slider("reps", 1000, 100, 10000, 100, group="simulation", expensive=True),
        toggle("show_sample_path", True, group="views"),
        toggle("show_bias_surface", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", beta_x=1.0, beta_z=2.0, corr_xz=0.6),
        scenario("no_bias_uncorrelated", "null", corr_xz=0.0),
        scenario("no_bias_irrelevant", "null", beta_z=0.0, corr_xz=0.8),
        scenario("upward_bias", "positive", beta_z=2.0, corr_xz=0.7),
        scenario("downward_bias", "negative", beta_z=-2.0, corr_xz=0.7),
        scenario("sign_reversal", "counterexample", beta_x=0.4, beta_z=-3.0,
                 corr_xz=0.8),
        scenario("effect_hidden", "counterexample", beta_x=1.0, beta_z=-2.0,
                 corr_xz=0.5),
        scenario("large_sample_no_rescue", "large_sample", n=20000, corr_xz=0.7),
        scenario("mediator", "misspecification", omitted_role="mediator"),
        scenario("collider", "misspecification", omitted_role="collider"),
    ),
    prerequisites=("regression.simple_linear",),
    related=("causal.dag", "econometrics.endogeneity_iv", "regression.fwl"),
    next_concepts=("econometrics.endogeneity_iv", "causal.dag"),
    tags=("omitted variable bias", "confounding", "specification error",
          "bad control", "consistency"),
    aliases=("ovb", "biais de variable omise", "تحيز المتغير المحذوف",
             "confounding bias"),
    backends=("numpy",),
    proof_ids=("econometrics.ovb.formula",),
    references=(
        ref("Angrist, J. D. and Pischke, J.-S. (2009). Mostly Harmless Econometrics.",
            kind="book"),
        ref("Cinelli, C. and Hazlett, C. (2020). Making sense of sensitivity. JRSS-B 82.",
            kind="paper", doi="10.1111/rssb.12348"),
    ),
    curriculum_tags=("dz.econometrics1",),
)


class OVBLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        n = int(p["n"])
        gen = rng(state.seed, "ovb")
        x, z, y = self._generate(gen, p, n)

        short = LM.ols(y, np.column_stack([np.ones(n), x]), names=("const", "x"))
        long = LM.ols(y, np.column_stack([np.ones(n), x, z]), names=("const", "x", "z"))
        auxiliary = LM.ols(z, np.column_stack([np.ones(n), x]), names=("const", "x"))

        delta = auxiliary.coef("x")
        predicted_bias = long.coef("z") * delta
        observed_gap = short.coef("x") - long.coef("x")

        role = str(p["omitted_role"])
        res.dgp = ctx.t(
            "labs.ovb.dgp",
            "y = {bx} x + {bz} z + noise where z acts as a {role} and corr(x, z) = {c}; "
            "n = {n}. The short regression leaves z out.",
            bx=fmt(p["beta_x"], 2), bz=fmt(p["beta_z"], 2), role=role,
            c=fmt(p["corr_xz"], 2), n=n,
        )

        res.add_panel(ctx.panel(
            "decomposition", self._decomposition_figure(ctx, short, long, auxiliary, p),
            "labs.ovb.figure.decomposition", evidence=EvidenceType.SYMBOLIC_DERIVATION,
        ))
        res.add_panel(ctx.panel(
            "scatter", self._scatter_figure(ctx, x, z, y, short, long),
            "labs.ovb.figure.scatter", tab="visualize",
            evidence=EvidenceType.EMPIRICAL_EXAMPLE,
        ))
        if p["show_sample_path"]:
            res.add_panel(ctx.panel(
                "consistency", self._consistency_figure(ctx, p, state.seed),
                "labs.ovb.figure.consistency", tab="simulation",
                evidence=EvidenceType.SIMULATION,
            ))
        if p["show_bias_surface"]:
            res.add_panel(ctx.panel(
                "surface", self._surface_figure(ctx, p),
                "labs.ovb.figure.surface", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("short_coefficient", ctx.t("labs.ovb.metric.short",
                                              "x coefficient with z omitted"),
                   short.coef("x"), reference=float(p["beta_x"]))
        res.metric("long_coefficient", ctx.t("labs.ovb.metric.long",
                                             "x coefficient with z included"),
                   long.coef("x"), reference=float(p["beta_x"]))
        res.metric("observed_gap", ctx.t("labs.ovb.metric.gap",
                                         "Difference between the two"), observed_gap)
        res.metric("formula_bias", ctx.t("labs.ovb.metric.formula",
                                         "beta_z times delta (the OVB formula)"),
                   predicted_bias, reference=observed_gap,
                   note=ctx.t("labs.ovb.metric.formula_note",
                              "these agree to machine precision - it is an identity, "
                              "not an approximation"))
        res.metric("identity_error", ctx.t("labs.ovb.metric.identity",
                                           "|gap - formula|"),
                   abs(observed_gap - predicted_bias), reference=0.0)
        res.metric("beta_z", ctx.t("labs.ovb.metric.beta_z",
                                   "Effect of z on y, holding x fixed"), long.coef("z"),
                   reference=float(p["beta_z"]))
        res.metric("delta", ctx.t("labs.ovb.metric.delta",
                                  "Slope of z on x (the auxiliary regression)"), delta)
        res.metric("bias_direction", ctx.t("labs.ovb.metric.direction",
                                           "Predicted direction"),
                   ctx.t("labs.ovb.upward", "upward") if predicted_bias > 0
                   else (ctx.t("labs.ovb.downward", "downward") if predicted_bias < 0
                         else ctx.t("labs.ovb.none", "none")))

        confounded = role == "confounder" and abs(predicted_bias) > 1e-8
        res.assume("exogeneity", ctx.t("assumptions.exogeneity"), not confounded,
                   detail=ctx.t("labs.ovb.assume.exogeneity",
                                "With z omitted, z sits inside the error term. If z is also "
                                "correlated with x, the error is correlated with the "
                                "regressor and least squares is inconsistent."),
                   consequence="" if not confounded else ctx.t(
                       "labs.ovb.assume.consequence",
                       "This is a bias, not an imprecision. Increasing n shrinks the "
                       "standard error around the WRONG number."))
        if role in ("mediator", "collider"):
            res.assume("good_controls",
                       ctx.t("labs.ovb.assume.control_label",
                             "Including z is the right thing to do"), False,
                       detail=ctx.t("labs.ovb.assume.bad_control",
                                    "z is a {role} here. Controlling for it is a mistake: "
                                    "it removes part of the effect (mediator) or "
                                    "manufactures a spurious one (collider). 'Long' is not "
                                    "automatically 'better'.", role=role))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.econometrics.omitted_variable_bias.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.econometrics.omitted_variable_bias.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.econometrics.omitted_variable_bias.math"), kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.ovb.derivation",
            "Let the long model be y = b0 + b1 x + b2 z + u with E[u | x, z] = 0, and let "
            "the auxiliary regression be z = d0 + d1 x + v with E[v | x] = 0.\n"
            "Substituting: y = (b0 + b2 d0) + (b1 + b2 d1) x + (b2 v + u).\n"
            "The composite error b2 v + u is uncorrelated with x by construction, so the "
            "short regression consistently estimates the coefficient in front of x - which "
            "is b1 + b2 d1, not b1.\n"
            "Therefore plim(short) - b1 = b2 * d1: the effect of the omitted variable on "
            "the outcome, times its relationship with the included regressor.\n"
            "Both signs are knowable before estimating anything, which is why the direction "
            "of an omitted-variable bias can often be argued from theory alone.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.econometrics.omitted_variable_bias.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.econometrics.omitted_variable_bias.warning"),
                    kind="warning")

        if np.sign(short.coef("x")) != np.sign(float(p["beta_x"])) and \
                abs(float(p["beta_x"])) > 0.05:
            res.warnings.append(ctx.t(
                "labs.ovb.warn.sign_flip",
                "The short regression reports {s} while the true effect is {t}: omitting one "
                "variable has reversed the sign. No amount of data would reveal this - only "
                "knowing that z exists would.",
                s=fmt(short.coef("x"), 3), t=fmt(p["beta_x"], 3),
            ))
        return res

    @staticmethod
    def _generate(gen, p, n):
        rho = float(p["corr_xz"])
        role = str(p["omitted_role"])
        x = gen.standard_normal(n)
        noise = float(p["noise"])
        if role == "mediator":
            z = 1.2 * x + gen.standard_normal(n) * 0.5
            y = float(p["beta_x"]) * x + float(p["beta_z"]) * z + noise * gen.standard_normal(n)
            return x, z, y
        if role == "collider":
            y = float(p["beta_x"]) * x + noise * gen.standard_normal(n)
            z = 1.0 * x + 1.0 * y + 0.5 * gen.standard_normal(n)
            return x, z, y
        z = rho * x + np.sqrt(max(1 - rho**2, 1e-9)) * gen.standard_normal(n)
        beta_z = 0.0 if role == "irrelevant" else float(p["beta_z"])
        y = float(p["beta_x"]) * x + beta_z * z + noise * gen.standard_normal(n)
        return x, z, y

    def _decomposition_figure(self, ctx, short, long, auxiliary, p):
        go = P.require_plotly()
        b_long = long.coef("x")
        bias = long.coef("z") * auxiliary.coef("x")
        fig = ctx.figure(
            "labs.ovb.figure.decomposition",
            xaxis_title=ctx.t("labs.ovb.axis.part", "Component"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=400,
        )
        labels = [
            ctx.t("labs.ovb.trace.direct", "direct effect (long regression)"),
            ctx.t("labs.ovb.trace.bias", "bias = beta_z x delta"),
            ctx.t("labs.ovb.trace.short", "what the short regression reports"),
        ]
        fig.add_trace(go.Bar(x=labels[:2], y=[b_long, bias],
                             marker={"color": [ctx.color("positive"),
                                               ctx.color("negative")]},
                             name=ctx.t("labs.ovb.trace.components", "components"),
                             text=[fmt(b_long, 4), fmt(bias, 4)],
                             textposition="outside"))
        fig.add_trace(go.Bar(x=[labels[2]], y=[short.coef("x")],
                             marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.ovb.trace.total", "their sum"),
                             text=[fmt(short.coef("x"), 4)], textposition="outside"))
        P.add_hline(fig, float(p["beta_x"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.ovb.legend_decomposition",
            "The first two bars add up to the third exactly. That addition is the whole "
            "omitted-variable-bias formula, and both of its factors can often be signed "
            "from theory before any data are collected.",
        ), theme=ctx.theme)
        return fig

    def _scatter_figure(self, ctx, x, z, y, short, long):
        go = P.require_plotly()
        fig = ctx.figure(
            "labs.ovb.figure.scatter",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=420,
        )
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker={"color": z, "colorscale": ctx.theme.colorscale, "size": 6,
                    "opacity": 0.75, "showscale": True,
                    "colorbar": {"title": ctx.t("labs.ovb.trace.z", "omitted z")}},
            name=ctx.t("labs.common.trace.data"),
        ))
        order = np.argsort(x)
        P.add_curve(fig, x[order], short.fitted_values[order],
                    ctx.t("labs.ovb.trace.short_line", "short regression (z omitted)"),
                    "negative", theme=ctx.theme, width=3.0)
        for q, label in ((0.15, "low z"), (0.85, "high z")):
            level = float(np.quantile(z, q))
            xs = np.linspace(float(x.min()), float(x.max()), 50)
            ys = (long.coef("const") + long.coef("x") * xs + long.coef("z") * level)
            P.add_curve(fig, xs, ys,
                        ctx.t("labs.ovb.trace.long_line",
                              "long regression at {label}", label=label),
                        "positive", theme=ctx.theme, dash="dash", width=2.2,
                        showlegend=q > 0.5)
        P.add_legend_note(fig, ctx.t(
            "labs.ovb.legend_scatter",
            "The two dashed lines are parallel - that is the ceteris-paribus effect. The "
            "solid line cuts across groups of different z and picks up their level "
            "differences as if they were slope.",
        ), theme=ctx.theme)
        return fig

    def _consistency_figure(self, ctx, p, seed):
        sizes = np.unique(np.round(np.geomspace(20, 20000, 14)).astype(int))
        short_vals, long_vals = [], []
        for ns in sizes:
            gen = rng(seed, "ovb_consistency", int(ns))
            x, z, y = self._generate(gen, p, int(ns))
            s = LM.ols(y, np.column_stack([np.ones(int(ns)), x]), names=("const", "x"))
            l_ = LM.ols(y, np.column_stack([np.ones(int(ns)), x, z]),
                        names=("const", "x", "z"))
            short_vals.append(s.coef("x"))
            long_vals.append(l_.coef("x"))
        fig = ctx.figure(
            "labs.ovb.figure.consistency",
            xaxis_title=ctx.t("labs.common.axis.sample_size"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=350,
        )
        P.add_curve(fig, sizes, short_vals,
                    ctx.t("labs.ovb.trace.short_line", "short regression (z omitted)"),
                    "negative", theme=ctx.theme, mode="lines+markers")
        P.add_curve(fig, sizes, long_vals,
                    ctx.t("labs.ovb.trace.long_series", "long regression (z included)"),
                    "positive", theme=ctx.theme, mode="lines+markers")
        P.add_hline(fig, float(p["beta_x"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.ovb.legend_consistency",
            "This is the difference between imprecision and inconsistency: the green curve "
            "converges to the truth while the red one converges confidently to something "
            "else. More data cannot fix a missing variable.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _surface_figure(self, ctx, p):
        go = P.require_plotly()
        deltas = np.linspace(-1.5, 1.5, 60)
        betas = np.linspace(-3.0, 3.0, 60)
        D, B = np.meshgrid(deltas, betas)
        bias = B * D
        fig = ctx.figure(
            "labs.ovb.figure.surface",
            xaxis_title=ctx.t("labs.ovb.axis.delta", "delta: how z moves with x"),
            yaxis_title=ctx.t("labs.ovb.axis.beta_z", "beta_z: effect of z on y"),
            height=400,
        )
        fig.add_trace(go.Contour(x=deltas, y=betas, z=bias,
                                 colorscale=ctx.theme.diverging_colorscale, zmid=0,
                                 contours={"showlabels": True},
                                 name=ctx.t("labs.ovb.trace.bias_surface", "bias")))
        P.add_points(fig, [float(p["corr_xz"])], [float(p["beta_z"])],
                     ctx.t("labs.common.trace.current"), "highlight", theme=ctx.theme,
                     size=12)
        P.add_legend_note(fig, ctx.t(
            "labs.ovb.legend_surface",
            "Bias is a product of two signs. Both must be non-zero for a problem to exist, "
            "and knowing only their signs already tells you the direction of the error.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        rhos = np.linspace(-0.9, 0.9, 19)
        frames, steps = [], []
        for i, r in enumerate(rhos):
            q = dict(p)
            q["corr_xz"] = float(r)
            gen = rng(seed, "ovb_anim", i)
            x, z, y = self._generate(gen, q, n)
            s = LM.ols(y, np.column_stack([np.ones(n), x]), names=("const", "x"))
            l_ = LM.ols(y, np.column_stack([np.ones(n), x, z]),
                        names=("const", "x", "z"))
            aux = LM.ols(z, np.column_stack([np.ones(n), x]), names=("const", "x"))
            bias = l_.coef("z") * aux.coef("x")
            order = np.argsort(x)
            frames.append(go.Frame(name=f"{r:.2f}", data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x[order], y=s.fitted_values[order]),
            ]))
            steps.append(AnimationStep(
                id=f"rho_{i}", frame=i,
                title=ctx.t("labs.ovb.anim.title", "corr(x, z) = {r}", r=fmt(r, 2)),
                what_you_see=ctx.t("labs.ovb.anim.see",
                                   "The scatter of x against y and the line the short "
                                   "regression fits to it."),
                what_changed=ctx.t("labs.ovb.anim.changed",
                                   "The correlation between the included and the omitted "
                                   "variable moved to {r}.", r=fmt(r, 2)),
                why=ctx.t("labs.ovb.anim.why",
                          "The omitted variable is inside the error term. As it becomes "
                          "correlated with x, so does the error - and the slope absorbs "
                          "that relationship."),
                interpretation=ctx.t("labs.ovb.anim.interpret",
                                     "The short slope is {s} against a true effect of {t}; "
                                     "the formula predicts a bias of {b}.",
                                     s=fmt(s.coef("x"), 3), t=fmt(p["beta_x"], 3),
                                     b=fmt(bias, 3)),
                conclusion=ctx.t("labs.ovb.anim.conclude",
                                 "At zero correlation the bias vanishes even though z still "
                                 "matters for y. BOTH links are needed."),
                warning=ctx.t("labs.ovb.anim.warn",
                              "Nothing in the data reveals this. The scatter looks perfectly "
                              "ordinary at every frame."),
                math="plim(short) = beta_x + beta_z * delta",
                outputs={"corr_xz": round(float(r), 3),
                         "short_slope": round(s.coef("x"), 4),
                         "predicted_bias": round(bias, 4)},
                violated_assumptions=() if abs(bias) < 1e-6 else ("exogeneity",),
                highlighted=("short_line",),
            ))
        gen = rng(seed, "ovb_anim", 0)
        x, z, y = self._generate(gen, p, n)
        fig = ctx.figure(
            "labs.ovb.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=400,
        )
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 5,
                                         "opacity": 0.55},
                                 name=ctx.t("labs.common.trace.data")))
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line={"color": ctx.color("negative"), "width": 3.0},
                                 name=ctx.t("labs.ovb.trace.short_line",
                                            "short regression (z omitted)")))
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label="corr(x, z)")
        return animation(
            "confounding_sweep", fig, steps,
            purpose=ctx.t("labs.ovb.anim.purpose",
                          "Show that bias needs two links, not one."),
            summary=ctx.t(
                "labs.ovb.anim.summary",
                "An omitted variable is harmless unless it is connected to BOTH the "
                "regressor and the outcome. When both links exist, the slope silently "
                "reports their product added to the effect you wanted - and the data give "
                "no sign that anything is wrong."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = OVBLab(SPEC)
