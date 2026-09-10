"""Bivariate random variables: joint, marginal, conditional, covariance, independence."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "JOINTS"]

JOINTS = (
    "bivariate_normal",
    "discrete_table",
    "independent_uniform",
    "uniform_triangle",
    "circle_dependence",
    "quadratic_dependence",
    "mixture",
)


SPEC = make_spec(
    "probability.bivariate",
    Domain.PROBABILITY,
    "bivariate_random_variables",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "experiment", "compare", "simulate", "counterexample",
           "code", "quiz", "data", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("joint", "bivariate_normal", JOINTS, group="model"),
        slider("rho", 0.6, -0.99, 0.99, 0.01, group="model",
               depends_on=("joint", ("bivariate_normal", "mixture"))),
        slider("sigma_x", 1.0, 0.1, 5.0, 0.05, group="model"),
        slider("sigma_y", 1.0, 0.1, 5.0, 0.05, group="model"),
        slider("condition_x", 0.0, -4.0, 4.0, 0.05, group="conditional"),
        int_slider("n", 2000, 100, 40000, 100, group="sampling", expensive=True),
        int_slider("grid", 60, 20, 160, 10, group="views", advanced=True),
        toggle("show_marginals", True, group="views"),
        toggle("show_conditional", True, group="views"),
        toggle("show_surface", False, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", joint="bivariate_normal", rho=0.6),
        scenario("independent", "null", joint="bivariate_normal", rho=0.0),
        scenario("strong_positive", "positive", rho=0.92),
        scenario("strong_negative", "negative", rho=-0.92),
        scenario("weak", "weak", rho=0.15),
        scenario("near_singular", "boundary", rho=0.995),
        scenario("discrete_joint", "compare_methods", joint="discrete_table"),
        scenario("uncorrelated_but_dependent", "counterexample", joint="circle_dependence"),
        scenario("quadratic_dependence", "counterexample", joint="quadratic_dependence"),
        scenario("dependent_support", "violation", joint="uniform_triangle"),
        scenario("mixture_of_groups", "misspecification", joint="mixture", rho=0.0),
        scenario("unequal_scales", "sensitivity", sigma_x=1.0, sigma_y=4.0, rho=0.6),
    ),
    prerequisites=("probability.distributions",),
    related=("probability.distributions", "multivariate.pca"),
    tags=("joint", "marginal", "conditional", "covariance", "independence", "correlation"),
    aliases=("joint distribution", "bivariate", "المتغيرات الثنائية", "loi conjointe",
             "conditional distribution"),
    backends=("scipy",),
    references=(
        ref("Casella, G. and Berger, R. L. (2002). Statistical Inference, chapter 4.",
            kind="book"),
    ),
    curriculum_tags=("dz.stat3",),
)


class BivariateLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        joint = str(p["joint"])
        rho = float(p["rho"])
        sx, sy = float(p["sigma_x"]), float(p["sigma_y"])
        n = int(p["n"])
        cx = float(p["condition_x"])

        gen = rng(state.seed, "bivariate", joint)
        x, y, table = self._sample(gen, joint, rho, sx, sy, n)

        corr = float(np.corrcoef(x, y)[0, 1])
        cov = float(np.cov(x, y, ddof=1)[0, 1])
        independent = self._independence(joint, rho)

        res.dgp = ctx.t(
            "labs.biv.dgp",
            "{joint} joint distribution; n = {n} draws; parameters rho = {rho}, "
            "sd(X) = {sx}, sd(Y) = {sy}.",
            joint=joint, n=n, rho=fmt(rho, 2), sx=fmt(sx, 2), sy=fmt(sy, 2),
        )

        res.add_panel(ctx.panel(
            "joint", self._joint_figure(ctx, x, y, table, joint, p),
            "labs.biv.figure.joint", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_marginals"]:
            res.add_panel(ctx.panel(
                "marginals", self._marginal_figure(ctx, x, y, table),
                "labs.biv.figure.marginals", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_conditional"]:
            res.add_panel(ctx.panel(
                "conditional", self._conditional_figure(ctx, x, y, cx, joint, rho, sx, sy),
                "labs.biv.figure.conditional", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_surface"] and joint == "bivariate_normal":
            res.add_panel(ctx.panel(
                "surface", self._surface_figure(ctx, rho, sx, sy),
                "labs.biv.figure.surface", tab="visualize",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("covariance", ctx.t("labs.biv.metric.cov", "Sample covariance"), cov)
        res.metric("correlation", ctx.t("labs.biv.metric.corr", "Sample correlation"),
                   corr, reference=rho if joint in ("bivariate_normal",) else None)
        res.metric("spearman", ctx.t("labs.biv.metric.spearman",
                                     "Spearman rank correlation"),
                   float(stats.spearmanr(x, y).statistic),
                   note=ctx.t("labs.biv.metric.spearman_note",
                              "picks up monotone dependence that Pearson can miss"))
        mi = self._mutual_information(x, y)
        res.metric("mutual_information", ctx.t("labs.biv.metric.mi",
                                               "Estimated mutual information"), mi,
                   note=ctx.t("labs.biv.metric.mi_note",
                              "zero only under genuine independence"))
        res.metric("conditional_mean", ctx.t("labs.biv.metric.cond_mean",
                                             "E[Y | X near {x}]", x=fmt(cx, 2)),
                   self._conditional_mean(x, y, cx))
        res.metric("marginal_mean_y", ctx.t("labs.biv.metric.marg_mean",
                                            "E[Y] (unconditional)"), float(np.mean(y)))

        res.assume("independence", ctx.t("labs.biv.assume.independence_label",
                                         "X and Y are independent"), independent,
                   detail=ctx.t("labs.biv.assume.independence",
                                "Independence requires the joint density to factor into the "
                                "product of the marginals at EVERY point, not merely a "
                                "correlation of zero."))
        res.assume("linear_dependence",
                   ctx.t("labs.biv.assume.linear_label",
                         "Any dependence present is linear"),
                   joint in ("bivariate_normal", "independent_uniform"),
                   detail=ctx.t("labs.biv.assume.linear",
                                "Correlation only measures the linear part of a relationship."))

        res.animations.append(self._animation(ctx, joint, sx, sy, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.probability.bivariate.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.probability.bivariate.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.probability.bivariate.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.probability.bivariate.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.probability.bivariate.warning"), kind="warning")

        if abs(corr) < 0.1 and mi > 0.05:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.biv.counterexample",
                "Correlation here is {c} - essentially zero - yet mutual information is {m} "
                "and the scatter plot shows an obvious structure. Zero correlation does not "
                "mean independence; it means no LINEAR association.",
                c=fmt(corr, 3), m=fmt(mi, 3),
            ), kind="warning")
            res.warnings.append(ctx.t(
                "labs.biv.warn.uncorrelated",
                "This scenario is a genuine counterexample to 'uncorrelated implies "
                "independent'.",
            ))
        return res

    @staticmethod
    def _sample(gen, joint, rho, sx, sy, n):
        table = None
        if joint == "bivariate_normal":
            cov = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
            draws = gen.multivariate_normal([0, 0], cov, size=n)
            return draws[:, 0], draws[:, 1], None
        if joint == "discrete_table":
            table = np.array([[0.10, 0.05, 0.05],
                              [0.05, 0.30, 0.05],
                              [0.05, 0.05, 0.30]])
            flat = table.ravel() / table.sum()
            idx = gen.choice(flat.size, size=n, p=flat)
            xs, ys = np.divmod(idx, table.shape[1])
            return xs.astype(float), ys.astype(float), table / table.sum()
        if joint == "independent_uniform":
            return gen.uniform(-2, 2, n), gen.uniform(-2, 2, n), None
        if joint == "uniform_triangle":
            u = gen.uniform(0, 1, n)
            v = gen.uniform(0, 1, n)
            mask = v > u
            u[mask], v[mask] = v[mask], u[mask]
            return (u * 4 - 2) * sx, (v * 4 - 2) * sy, None
        if joint == "circle_dependence":
            theta = gen.uniform(0, 2 * np.pi, n)
            r = 2.0 + gen.normal(0, 0.12, n)
            return r * np.cos(theta) * sx, r * np.sin(theta) * sy, None
        if joint == "quadratic_dependence":
            xs = gen.uniform(-2, 2, n) * sx
            return xs, (xs**2 - 2.0) * sy / max(sx**2, 1e-9) + gen.normal(0, 0.3, n), None
        pick = gen.random(n) < 0.5
        cov = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
        a = gen.multivariate_normal([-2, -2], cov, size=n)
        b = gen.multivariate_normal([2, 2], cov, size=n)
        return (np.where(pick, a[:, 0], b[:, 0]), np.where(pick, a[:, 1], b[:, 1]), None)

    @staticmethod
    def _independence(joint, rho):
        if joint == "independent_uniform":
            return True
        if joint == "bivariate_normal":
            return abs(rho) < 1e-9
        return False

    @staticmethod
    def _mutual_information(x, y, bins: int = 24):
        joint, _, _ = np.histogram2d(x, y, bins=bins)
        joint = joint / joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint * np.log(joint / (px @ py))
        return float(np.nansum(term))

    @staticmethod
    def _conditional_mean(x, y, cx, width: float = 0.35):
        mask = np.abs(x - cx) < width * max(np.std(x), 1e-9)
        return float(np.mean(y[mask])) if mask.sum() > 3 else float("nan")

    def _joint_figure(self, ctx, x, y, table, joint, p):
        if table is not None:
            fig = ctx.figure(
                "labs.biv.figure.joint",
                xaxis_title="X", yaxis_title="Y", height=430,
            )
            P.add_heatmap(fig, table.T, x=list(range(table.shape[0])),
                          y=list(range(table.shape[1])), theme=ctx.theme,
                          colorscale=ctx.theme.colorscale,
                          text=[[fmt(v, 3) for v in row] for row in table.T],
                          texttemplate="%{text}")
            P.add_legend_note(fig, ctx.t(
                "labs.biv.legend_table",
                "The joint probability mass function as a table. Row sums are the marginal "
                "of X, column sums the marginal of Y, and independence would require every "
                "cell to equal the product of its row and column sums.",
            ), theme=ctx.theme)
            return fig

        fig = ctx.figure(
            "labs.biv.figure.joint",
            xaxis_title="X", yaxis_title="Y", height=440,
        )
        grid = int(p["grid"])
        H, xe, ye = np.histogram2d(x, y, bins=grid)
        P.add_heatmap(fig, (H / H.sum()).T, x=0.5 * (xe[1:] + xe[:-1]),
                      y=0.5 * (ye[1:] + ye[:-1]), theme=ctx.theme,
                      colorscale=ctx.theme.colorscale, showscale=True,
                      name=ctx.t("labs.biv.trace.joint", "joint density"))
        sub = min(x.size, 1200)
        P.add_points(fig, x[:sub], y[:sub], ctx.t("labs.biv.trace.draws", "draws"),
                     "highlight", theme=ctx.theme, size=3, opacity=0.35)
        P.add_legend_note(fig, ctx.t(
            "labs.biv.legend_joint",
            "Colour is joint density; the dots are actual draws. Everything else in this lab "
            "is derived from this one surface by summing it in different directions.",
        ), theme=ctx.theme)
        return fig

    def _marginal_figure(self, ctx, x, y, table):
        make_subplots = P.SUBPLOT()
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.biv.trace.marg_x", "marginal of X"),
            ctx.t("labs.biv.trace.marg_y", "marginal of Y"),
        ))
        go = P.require_plotly()
        for i, (vals, role) in enumerate(((x, "primary"), (y, "secondary"))):
            fig.add_trace(go.Histogram(x=vals, nbinsx=45, histnorm="probability density",
                                       marker={"color": ctx.color(role)},
                                       showlegend=False), row=1, col=i + 1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=330, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.biv.legend_marginals",
            "A marginal is the joint distribution with one dimension summed away. Two joint "
            "distributions with completely different dependence can share these two pictures.",
        ), theme=ctx.theme)
        return fig

    def _conditional_figure(self, ctx, x, y, cx, joint, rho, sx, sy):
        width = 0.35 * max(float(np.std(x)), 1e-9)
        mask = np.abs(x - cx) < width
        fig = ctx.figure(
            "labs.biv.figure.conditional",
            xaxis_title="Y",
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=360,
        )
        P.add_histogram(fig, y, ctx.t("labs.biv.trace.marg_y", "marginal of Y"),
                        "muted", theme=ctx.theme, nbins=45, opacity=0.35)
        if mask.sum() > 5:
            P.add_histogram(fig, y[mask],
                            ctx.t("labs.biv.trace.cond",
                                  "Y given X near {x} ({k} draws)", x=fmt(cx, 2),
                                  k=int(mask.sum())),
                            "primary", theme=ctx.theme, nbins=35, opacity=0.6)
        if joint == "bivariate_normal":
            mean = rho * (sy / sx) * cx
            sd = sy * np.sqrt(max(1 - rho**2, 0.0))
            grid = np.linspace(mean - 4 * sd, mean + 4 * sd, 300)
            P.add_curve(fig, grid, stats.norm.pdf(grid, mean, sd),
                        ctx.t("labs.biv.trace.cond_exact",
                              "exact conditional N({m}, {s}^2)", m=fmt(mean, 3),
                              s=fmt(sd, 3)),
                        "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.biv.legend_conditional",
            "Conditioning slices the joint distribution and renormalizes the slice. If the "
            "slice looks the same wherever you cut, X carries no information about Y.",
        ), theme=ctx.theme)
        return fig

    def _surface_figure(self, ctx, rho, sx, sy):
        go = P.require_plotly()
        xs = np.linspace(-3 * sx, 3 * sx, 60)
        ys = np.linspace(-3 * sy, 3 * sy, 60)
        X, Y = np.meshgrid(xs, ys)
        cov = np.array([[sx**2, rho * sx * sy], [rho * sx * sy, sy**2]])
        pos = np.dstack([X, Y])
        Z = stats.multivariate_normal([0, 0], cov, allow_singular=True).pdf(pos)
        fig = ctx.figure("labs.biv.figure.surface", height=460)
        fig.add_trace(go.Surface(x=xs, y=ys, z=Z, colorscale=ctx.theme.colorscale,
                                 showscale=False))
        fig.update_layout(scene={"xaxis_title": "X", "yaxis_title": "Y",
                                 "zaxis_title": ctx.t("labs.common.axis.density")})
        return fig

    def _animation(self, ctx, joint, sx, sy, n, seed):
        go = P.require_plotly()
        rhos = np.linspace(-0.95, 0.95, 20)
        sub = min(n, 1200)
        frames, steps = [], []
        for i, r in enumerate(rhos):
            gen = rng(seed, "biv_anim", i)
            x, y, _ = self._sample(gen, "bivariate_normal", float(r), sx, sy, sub)
            frames.append(go.Frame(name=f"{r:.2f}", data=[go.Scatter(x=x, y=y)]))
            steps.append(AnimationStep(
                id=f"rho_{i}", frame=i,
                title=ctx.t("labs.biv.anim.title", "correlation rho = {r}", r=fmt(r, 2)),
                what_you_see=ctx.t("labs.biv.anim.see",
                                   "Draws from a bivariate normal with the marginals held "
                                   "fixed and only the correlation changing."),
                what_changed=ctx.t("labs.biv.anim.changed", "rho moved to {r}.", r=fmt(r, 2)),
                why=ctx.t("labs.biv.anim.why",
                          "rho enters only the off-diagonal of the covariance matrix, so it "
                          "tilts and squeezes the cloud without touching either marginal."),
                interpretation=ctx.t("labs.biv.anim.interpret",
                                     "Knowing X now reduces the conditional variance of Y by "
                                     "a factor of 1 - rho^2 = {v}.", v=fmt(1 - r**2, 3)),
                conclusion=ctx.t("labs.biv.anim.conclude",
                                 "Correlation is exactly how much the cloud leans - nothing "
                                 "more and nothing less."),
                warning=ctx.t("labs.biv.anim.warn",
                              "Both marginal histograms are identical in every frame. "
                              "Marginals cannot reveal dependence."),
                math="Var(Y | X = x) = sigma_Y^2 (1 - rho^2)",
                outputs={"rho": round(float(r), 3),
                         "conditional_variance_factor": round(float(1 - r**2), 4)},
                highlighted=("scatter_cloud",),
            ))
        fig = ctx.figure(
            "labs.biv.figure.animation", xaxis_title="X", yaxis_title="Y", height=420,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("primary"), "size": 4,
                                         "opacity": 0.55},
                                 name=ctx.t("labs.biv.trace.draws", "draws")))
        fig.update_xaxes(range=[-4 * sx, 4 * sx])
        fig.update_yaxes(range=[-4 * sy, 4 * sy])
        build_frames(fig, frames, duration=380, reduced_motion=ctx.reduced_motion,
                     slider_label="rho")
        return animation(
            "correlation_sweep", fig, steps,
            purpose=ctx.t("labs.biv.anim.purpose",
                          "Separate what correlation controls from what it does not."),
            summary=ctx.t(
                "labs.biv.anim.summary",
                "Sweeping rho tilts the cloud while both marginals stay exactly the same. "
                "Dependence lives in the joint distribution, and no amount of looking at one "
                "variable at a time will reveal it."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = BivariateLab(SPEC)
