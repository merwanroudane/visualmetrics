"""OLS as orthogonal projection - a geometric proof, not an analogy."""

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
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "regression.ols_geometry",
    Domain.REGRESSION,
    "ols_geometry",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.GEOMETRIC_PROOF,
    controls=(
        int_slider("n_obs", 3, 3, 3, 1, group="geometry",
                   help_key="labs.geom.controls.n_obs.help"),
        int_slider("k", 1, 1, 2, 1, group="geometry"),
        slider("y1", 3.0, -6.0, 6.0, 0.1, group="vector"),
        slider("y2", 1.0, -6.0, 6.0, 0.1, group="vector"),
        slider("y3", 4.0, -6.0, 6.0, 0.1, group="vector"),
        slider("x_correlation", 0.0, -0.98, 0.98, 0.01, group="geometry",
               depends_on=("k", (2,))),
        toggle("include_intercept", True, group="geometry"),
        toggle("show_residual", True, group="views"),
        toggle("show_right_angle", True, group="views"),
        toggle("show_matrix_check", True, group="views"),
        int_slider("n_large", 200, 20, 5000, 10, group="algebra"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", k=1, y1=3.0, y2=1.0, y3=4.0),
        scenario("plane_projection", "compare_methods", k=2, x_correlation=0.0),
        scenario("y_in_column_space", "boundary", y1=1.0, y2=2.0, y3=3.0, k=1),
        scenario("y_orthogonal", "boundary", y1=1.0, y2=-1.0, y3=0.0, k=1),
        scenario("near_collinear_basis", "violation", k=2, x_correlation=0.97),
        scenario("no_intercept", "compare_methods", include_intercept=False),
        scenario("large_sample_algebra", "large_sample", n_large=2000),
    ),
    prerequisites=("regression.simple_linear", "math.projection"),
    related=("regression.fwl", "math.projection"),
    next_concepts=("regression.fwl",),
    tags=("projection", "orthogonality", "column space", "normal equations",
          "hat matrix", "idempotent"),
    aliases=("ols geometry", "projection", "geometrie des mco", "الإسقاط",
             "hat matrix", "normal equations"),
    backends=("numpy",),
    proof_ids=("regression.ols.residual_orthogonality",),
    references=(
        ref("Strang, G. (2016). Introduction to Linear Algebra.", kind="book"),
        ref("Davidson, R. and MacKinnon, J. G. (2004). Econometric Theory and Methods.",
            kind="book"),
    ),
)


class GeometryLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        k = int(p["k"])
        y = np.array([float(p["y1"]), float(p["y2"]), float(p["y3"])])
        X = self._design(p, k)
        P_mat, M_mat = LM.projection_matrices(X)
        y_hat = P_mat @ y
        e = M_mat @ y

        res.dgp = ctx.t(
            "labs.geom.dgp",
            "A 3-dimensional outcome vector projected onto the {k}-dimensional column space "
            "of X. Nothing is estimated statistically here - this is exact linear algebra.",
            k=X.shape[1],
        )

        res.add_panel(ctx.panel(
            "space3d", self._space_figure(ctx, X, y, y_hat, e, p),
            "labs.geom.figure.space", evidence=EvidenceType.GEOMETRIC_PROOF,
        ))
        if p["show_matrix_check"]:
            res.add_panel(ctx.panel(
                "matrices", self._matrix_figure(ctx, P_mat, M_mat),
                "labs.geom.figure.matrices", tab="math",
                evidence=EvidenceType.SYMBOLIC_DERIVATION,
            ))
        res.add_panel(ctx.panel(
            "decomposition", self._decomposition_figure(ctx, y, y_hat, e),
            "labs.geom.figure.decomposition", tab="compare",
            evidence=EvidenceType.GEOMETRIC_PROOF,
        ))

        checks = self._checks(X, P_mat, M_mat, y, y_hat, e, int(p["n_large"]), state.seed)
        res.metric("orthogonality", ctx.t("labs.geom.metric.orthogonality",
                                          "max |X' e| (should be exactly zero)"),
                   checks["orthogonality"], reference=0.0)
        res.metric("angle", ctx.t("labs.geom.metric.angle",
                                  "Angle between the residual and the column space"),
                   checks["angle_degrees"], reference=90.0,
                   note=ctx.t("labs.geom.metric.angle_note", "degrees"))
        res.metric("pythagoras", ctx.t("labs.geom.metric.pythagoras",
                                       "|y|^2 - |y_hat|^2 - |e|^2"),
                   checks["pythagoras"], reference=0.0)
        res.metric("idempotent_p", ctx.t("labs.geom.metric.idempotent_p",
                                         "max |P P - P|"), checks["idempotent_p"],
                   reference=0.0)
        res.metric("idempotent_m", ctx.t("labs.geom.metric.idempotent_m",
                                         "max |M M - M|"), checks["idempotent_m"],
                   reference=0.0)
        res.metric("pm_zero", ctx.t("labs.geom.metric.pm", "max |P M|"),
                   checks["pm"], reference=0.0)
        res.metric("trace_p", ctx.t("labs.geom.metric.trace",
                                    "trace(P) (equals the number of columns of X)"),
                   checks["trace_p"], reference=float(X.shape[1]))
        res.metric("large_sample_orthogonality",
                   ctx.t("labs.geom.metric.large",
                         "max |X' e| in an n = {n} regression", n=int(p["n_large"])),
                   checks["large_orthogonality"], reference=0.0,
                   note=ctx.t("labs.geom.metric.large_note",
                              "the identity is dimension-free; only the picture is 3-D"))

        res.assume("full_rank", ctx.t("labs.geom.assume.rank_label",
                                      "X has full column rank"),
                   bool(np.linalg.matrix_rank(X) == X.shape[1]),
                   detail=ctx.t("labs.geom.assume.rank",
                                "Without full rank the projection still exists and is "
                                "unique, but the coefficients that produce it are not."))
        res.assume("no_perfect_collinearity", ctx.t("assumptions.no_perfect_collinearity"),
                   abs(float(p["x_correlation"])) < 0.999 or k == 1)
        if k == 2 and p["include_intercept"]:
            res.warnings.append(ctx.t(
                "labs.geom.warn.capped",
                "With three observations, an intercept plus two regressors would span the "
                "whole space and leave a residual of exactly zero. The lab therefore draws "
                "the intercept and one regressor; switch the intercept off to see two "
                "regressors spanning a plane.",
            ))

        res.animations.append(self._animation(ctx, X, y, P_mat))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.regression.ols_geometry.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.regression.ols_geometry.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.regression.ols_geometry.math"), kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.geom.proof",
            "Claim: the least-squares residual is orthogonal to every column of X.\n"
            "1. b_hat minimises S(b) = (y - Xb)'(y - Xb).\n"
            "2. S is differentiable and convex, so the minimum satisfies dS/db = 0.\n"
            "3. dS/db = -2 X'(y - Xb), so at the optimum X'(y - X b_hat) = 0.\n"
            "4. Writing e = y - X b_hat gives X'e = 0 exactly - the normal equations.\n"
            "5. Hence e is orthogonal to the column space of X, and y = y_hat + e is an "
            "orthogonal decomposition. Pythagoras then gives |y|^2 = |y_hat|^2 + |e|^2, "
            "which is exactly the analysis-of-variance identity.\n"
            "Nothing here is approximate and nothing depends on a sample size: the "
            "numbers in the read-out are zero to machine precision.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.regression.ols_geometry.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.regression.ols_geometry.warning"), kind="warning")
        return res

    @staticmethod
    def _design(p, k):
        """Build a design matrix with at most two columns.

        With three observations, three columns would span the whole space and
        leave a zero residual - so the picture would have nothing to show. The
        lab therefore caps the column count at two and says so.
        """
        cols = []
        if p["include_intercept"]:
            cols.append(np.ones(3) / np.sqrt(3.0))
        base = np.array([1.0, 2.0, 3.0])
        cols.append(base / np.linalg.norm(base))
        room = 2 - len(cols)
        if k == 2 and room > 0:
            rho = float(p["x_correlation"])
            other = np.array([1.0, -1.0, 0.5])
            other = other - (other @ cols[-1]) * cols[-1]
            other = other / max(np.linalg.norm(other), 1e-12)
            mixed = rho * cols[-1] + np.sqrt(max(1 - rho**2, 1e-9)) * other
            cols.append(mixed / max(np.linalg.norm(mixed), 1e-12))
        return np.column_stack(cols[:2])

    @staticmethod
    def _checks(X, P_mat, M_mat, y, y_hat, e, n_large, seed):
        gen = rng(seed, "geom_large")
        Xl = np.column_stack([np.ones(n_large), gen.standard_normal((n_large, 3))])
        yl = Xl @ np.array([1.0, 2.0, -1.0, 0.5]) + gen.standard_normal(n_large)
        fl = LM.ols(yl, Xl, names=("const", "x1", "x2", "x3"))
        norm_e = float(np.linalg.norm(e))
        cosines = [] if norm_e < 1e-12 else [
            float(abs(X[:, j] @ e) / (np.linalg.norm(X[:, j]) * norm_e))
            for j in range(X.shape[1])
        ]
        angle = 90.0 if not cosines else float(np.degrees(np.arccos(
            np.clip(max(cosines), -1, 1))))
        return {
            "orthogonality": float(np.max(np.abs(X.T @ e))),
            "angle_degrees": angle,
            "pythagoras": float(y @ y - y_hat @ y_hat - e @ e),
            "idempotent_p": float(np.max(np.abs(P_mat @ P_mat - P_mat))),
            "idempotent_m": float(np.max(np.abs(M_mat @ M_mat - M_mat))),
            "pm": float(np.max(np.abs(P_mat @ M_mat))),
            "trace_p": float(np.trace(P_mat)),
            "large_orthogonality": float(np.max(np.abs(Xl.T @ fl.residuals))),
        }

    def _space_figure(self, ctx, X, y, y_hat, e, p):
        go = P.require_plotly()
        fig = ctx.figure("labs.geom.figure.space", height=520, showlegend=True)
        span = float(max(np.max(np.abs(y)), 1.0)) * 1.4

        if X.shape[1] == 1:
            direction = X[:, 0] / np.linalg.norm(X[:, 0])
            t = np.linspace(-span, span, 2)
            line = np.outer(t, direction)
            fig.add_trace(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines",
                line={"color": ctx.color("muted"), "width": 6},
                name=ctx.t("labs.geom.trace.colspace", "column space of X"),
            ))
        else:
            a = X[:, 0] / np.linalg.norm(X[:, 0])
            b = X[:, 1] - (X[:, 1] @ a) * a
            b = b / max(np.linalg.norm(b), 1e-12)
            u = np.linspace(-span, span, 12)
            v = np.linspace(-span, span, 12)
            U, V = np.meshgrid(u, v)
            pts = U[..., None] * a + V[..., None] * b
            fig.add_trace(go.Surface(
                x=pts[..., 0], y=pts[..., 1], z=pts[..., 2],
                opacity=0.28, showscale=False,
                colorscale=[[0, ctx.color("muted")], [1, ctx.color("muted")]],
                name=ctx.t("labs.geom.trace.colspace", "column space of X"),
            ))

        for j in range(X.shape[1]):
            P.add_arrow3d(fig, [0, 0, 0], X[:, j] * span * 0.6,
                          ctx.t("labs.geom.trace.column", "column {j} of X", j=j + 1),
                          "baseline", theme=ctx.theme)
        P.add_arrow3d(fig, [0, 0, 0], y, ctx.t("labs.geom.trace.y", "y"),
                      "primary", theme=ctx.theme)
        P.add_arrow3d(fig, [0, 0, 0], y_hat,
                      ctx.t("labs.geom.trace.yhat", "y_hat = P y (the projection)"),
                      "fitted", theme=ctx.theme)
        if p["show_residual"]:
            P.add_arrow3d(fig, y_hat, y,
                          ctx.t("labs.geom.trace.residual", "e = M y (the residual)"),
                          "residual", theme=ctx.theme, dash="dot")
        if p["show_right_angle"] and np.linalg.norm(e) > 1e-9:
            self._right_angle_marker(fig, ctx, y_hat, e, X)

        fig.update_layout(scene={
            "xaxis_title": "observation 1", "yaxis_title": "observation 2",
            "zaxis_title": "observation 3",
            "aspectmode": "cube",
        })
        P.add_legend_note(fig, ctx.t(
            "labs.geom.legend_space",
            "Each AXIS is an observation, and each VECTOR is a variable. The regression is "
            "the single closest point to y inside the shaded set - and the shortest route "
            "there is always perpendicular.",
        ), theme=ctx.theme)
        return fig

    @staticmethod
    def _right_angle_marker(fig, ctx, y_hat, e, X):
        go = P.require_plotly()
        u = X[:, 0] / np.linalg.norm(X[:, 0])
        v = e / np.linalg.norm(e)
        size = 0.18 * float(np.linalg.norm(e))
        corner = np.asarray(y_hat, dtype=float)
        pts = np.array([corner + size * u, corner + size * (u + v), corner + size * v])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
            line={"color": ctx.color("warning"), "width": 5},
            name=ctx.t("labs.geom.trace.right_angle", "exact right angle"),
        ))

    def _matrix_figure(self, ctx, P_mat, M_mat):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            "P = X(X'X)^-1 X'", "M = I - P", "P M",
        ))
        for i, mat in enumerate((P_mat, M_mat, P_mat @ M_mat)):
            fig.add_trace(go.Heatmap(
                z=mat, colorscale=ctx.theme.diverging_colorscale, zmid=0,
                showscale=i == 2,
                text=[[fmt(v, 3) for v in row] for row in mat],
                texttemplate="%{text}",
            ), row=1, col=i + 1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=320, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.geom.legend_matrices",
            "P projects onto the column space, M annihilates it, and their product is the "
            "zero matrix. Applying either one twice changes nothing - projections are "
            "idempotent.",
        ), theme=ctx.theme)
        return fig

    def _decomposition_figure(self, ctx, y, y_hat, e):
        fig = ctx.figure(
            "labs.geom.figure.decomposition",
            xaxis_title=ctx.t("labs.geom.axis.component", "Component"),
            yaxis_title=ctx.t("labs.geom.axis.squared", "Squared length"),
            height=340,
        )
        labels = [ctx.t("labs.geom.trace.total", "|y|^2 (total)"),
                  ctx.t("labs.geom.trace.explained", "|y_hat|^2 (explained)"),
                  ctx.t("labs.geom.trace.unexplained", "|e|^2 (residual)")]
        values = [float(y @ y), float(y_hat @ y_hat), float(e @ e)]
        P.add_bar(fig, labels, values, ctx.t("labs.geom.trace.lengths", "squared length"),
                  "primary", theme=ctx.theme,
                  text=[fmt(v, 3) for v in values])
        P.add_legend_note(fig, ctx.t(
            "labs.geom.legend_decomposition",
            "The second and third bars add up to the first, exactly. That is Pythagoras in "
            "n dimensions, and it is the whole content of the analysis-of-variance table.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, X, y, P_mat):
        go = P.require_plotly()
        y_hat = P_mat @ y
        ts = np.linspace(0.0, 1.0, 18)
        frames, steps = [], []
        for i, t in enumerate(ts):
            point = y + t * (y_hat - y)
            dist = float(np.linalg.norm(y - point))
            resid = float(np.linalg.norm(y - point) if t < 1 else np.linalg.norm(y - y_hat))
            orth = float(np.max(np.abs(X.T @ (y - point))))
            frames.append(go.Frame(name=f"{t:.2f}", data=[
                go.Scatter3d(x=[y[0], point[0]], y=[y[1], point[1]], z=[y[2], point[2]]),
                go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]]),
            ]))
            steps.append(AnimationStep(
                id=f"t_{i}", frame=i,
                title=ctx.t("labs.geom.anim.title", "{pct}% of the way to the projection",
                            pct=int(round(100 * t))),
                what_you_see=ctx.t("labs.geom.anim.see",
                                   "The outcome vector y and a moving candidate point "
                                   "travelling towards the column space."),
                what_changed=ctx.t("labs.geom.anim.changed",
                                   "The candidate moved to {pct}% of the way down.",
                                   pct=int(round(100 * t))),
                why=ctx.t("labs.geom.anim.why",
                          "Least squares looks for the closest point inside the shaded set, "
                          "so the search is literally a descent along the shortest route."),
                interpretation=ctx.t("labs.geom.anim.interpret",
                                     "Distance travelled leaves a residual of length {r}; "
                                     "max |X' (y - point)| is {o}.",
                                     r=fmt(resid, 4), o=fmt(orth, 6)),
                conclusion=ctx.t("labs.geom.anim.conclude",
                                 "Only at the endpoint does X'(y - point) become exactly "
                                 "zero. That condition IS the normal equations."),
                warning=ctx.t("labs.geom.anim.warn",
                              "The right angle is exact algebra, not an artefact of the "
                              "camera. Rotate the plot and it stays a right angle."),
                math="min_b |y - Xb|^2  <=>  X'(y - Xb) = 0",
                outputs={"progress": round(float(t), 3),
                         "max_abs_Xte": round(orth, 8),
                         "residual_norm": round(dist, 5)},
                active_assumptions=("full_rank",),
                highlighted=("residual_vector", "column_space"),
            ))
        fig = ctx.figure("labs.geom.figure.animation", height=440)
        fig.add_trace(go.Scatter3d(x=[y[0], y[0]], y=[y[1], y[1]], z=[y[2], y[2]],
                                   mode="lines",
                                   line={"color": ctx.color("residual"), "width": 6},
                                   name=ctx.t("labs.geom.trace.descent",
                                              "route from y to the projection")))
        fig.add_trace(go.Scatter3d(x=[y[0]], y=[y[1]], z=[y[2]], mode="markers",
                                   marker={"color": ctx.color("fitted"), "size": 6},
                                   name=ctx.t("labs.geom.trace.candidate",
                                              "candidate point")))
        if X.shape[1] == 1:
            d = X[:, 0] / np.linalg.norm(X[:, 0])
            span = float(np.linalg.norm(y)) * 1.3
            line = np.outer(np.linspace(-span, span, 2), d)
            fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2],
                                       mode="lines",
                                       line={"color": ctx.color("muted"), "width": 6},
                                       name=ctx.t("labs.geom.trace.colspace",
                                                  "column space of X")))
        fig.update_layout(scene={"aspectmode": "cube"})
        build_frames(fig, frames, duration=280, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.geom.slider", "progress"))
        return animation(
            "projection_descent", fig, steps,
            purpose=ctx.t("labs.geom.anim.purpose",
                          "Turn 'minimise the sum of squares' into 'walk perpendicularly "
                          "to the nearest point'."),
            summary=ctx.t(
                "labs.geom.anim.summary",
                "Minimising squared distance and meeting the subspace at a right angle are "
                "the same statement. The normal equations X'e = 0 are that right angle "
                "written in algebra, and they hold in any number of dimensions - the "
                "picture is limited to three, the theorem is not."),
            evidence=EvidenceType.GEOMETRIC_PROOF,
        )


LAB = GeometryLab(SPEC)
