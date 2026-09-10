"""Vectors, span and orthogonal projection - the geometry every later lab reuses."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...backends import linear as LM

__all__ = ["LAB", "SPEC"]


SPEC = make_spec(
    "math.projection",
    Domain.MATH,
    "linear_algebra",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.GEOMETRIC_PROOF,
    controls=(
        slider("y1", 3.0, -6.0, 6.0, 0.1, group="vector"),
        slider("y2", 4.0, -6.0, 6.0, 0.1, group="vector"),
        slider("y3", 2.0, -6.0, 6.0, 0.1, group="vector"),
        slider("a1", 1.0, -6.0, 6.0, 0.1, group="subspace"),
        slider("a2", 1.0, -6.0, 6.0, 0.1, group="subspace"),
        slider("a3", 0.0, -6.0, 6.0, 0.1, group="subspace"),
        slider("b1", 0.0, -6.0, 6.0, 0.1, group="subspace"),
        slider("b2", 1.0, -6.0, 6.0, 0.1, group="subspace"),
        slider("b3", 1.0, -6.0, 6.0, 0.1, group="subspace"),
        int_slider("dimension", 1, 1, 2, 1, group="subspace"),
        toggle("show_residual", True, group="views"),
        toggle("show_distance_curve", True, group="views"),
        toggle("show_gram_schmidt", False, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", dimension=1, y1=3.0, y2=4.0, y3=2.0),
        scenario("onto_a_plane", "compare_methods", dimension=2),
        scenario("already_in_the_span", "boundary", y1=2.0, y2=2.0, y3=0.0,
                 dimension=1),
        scenario("orthogonal_to_the_span", "boundary", y1=1.0, y2=-1.0, y3=0.0,
                 a1=1.0, a2=1.0, a3=0.0, dimension=1),
        scenario("nearly_parallel_basis", "violation", dimension=2, b1=1.0, b2=1.02,
                 b3=0.0),
        scenario("gram_schmidt", "compare_methods", dimension=2,
                 show_gram_schmidt=True),
        scenario("long_vector", "sensitivity", y1=6.0, y2=6.0, y3=6.0),
    ),
    related=("regression.ols_geometry", "multivariate.pca"),
    next_concepts=("regression.ols_geometry",),
    tags=("vector", "projection", "orthogonality", "span", "basis", "gram-schmidt",
          "inner product"),
    aliases=("orthogonal projection", "projection orthogonale", "الإسقاط المتعامد",
             "span", "linear algebra"),
    backends=("numpy",),
    proof_ids=("math.projection.shortest_distance",),
    references=(
        ref("Strang, G. (2016). Introduction to Linear Algebra.", kind="book"),
        ref("Axler, S. (2015). Linear Algebra Done Right.", kind="book"),
    ),
)


class ProjectionLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        y = np.array([float(p["y1"]), float(p["y2"]), float(p["y3"])])
        a = np.array([float(p["a1"]), float(p["a2"]), float(p["a3"])])
        b = np.array([float(p["b1"]), float(p["b2"]), float(p["b3"])])
        dim = int(p["dimension"])
        A = np.column_stack([a] if dim == 1 else [a, b])
        rank = int(np.linalg.matrix_rank(A))
        P_mat, M_mat = LM.projection_matrices(A)
        y_hat = P_mat @ y
        e = M_mat @ y

        res.dgp = ctx.t(
            "labs.proj.dgp",
            "Projecting y = ({y}) onto the span of {k} direction(s) in three dimensions. "
            "This is exact linear algebra - nothing is estimated.",
            y=", ".join(fmt(v, 2) for v in y), k=dim,
        )

        res.add_panel(ctx.panel(
            "space", self._space_figure(ctx, A, y, y_hat, e, dim, p),
            "labs.proj.figure.space", evidence=EvidenceType.GEOMETRIC_PROOF,
        ))
        if p["show_distance_curve"]:
            res.add_panel(ctx.panel(
                "distance", self._distance_figure(ctx, A, y, y_hat, dim),
                "labs.proj.figure.distance", tab="math",
                evidence=EvidenceType.GEOMETRIC_PROOF,
            ))
        if p["show_gram_schmidt"] and dim == 2:
            res.add_panel(ctx.panel(
                "gram_schmidt", self._gram_schmidt_figure(ctx, a, b),
                "labs.proj.figure.gram_schmidt", tab="compare",
                evidence=EvidenceType.SYMBOLIC_DERIVATION,
            ))
        res.add_panel(ctx.panel(
            "decomposition", self._decomposition_figure(ctx, y, y_hat, e),
            "labs.proj.figure.decomposition", tab="compare",
            evidence=EvidenceType.GEOMETRIC_PROOF,
        ))

        norm_e = float(np.linalg.norm(e))
        res.metric("projection", ctx.t("labs.proj.metric.projection",
                                       "Projection of y"),
                   ", ".join(fmt(v, 4) for v in y_hat))
        res.metric("residual", ctx.t("labs.proj.metric.residual", "Residual"),
                   ", ".join(fmt(v, 4) for v in e))
        res.metric("orthogonality", ctx.t("labs.proj.metric.orthogonality",
                                          "max |A' e| (should be exactly zero)"),
                   float(np.max(np.abs(A.T @ e))), reference=0.0)
        res.metric("angle", ctx.t("labs.proj.metric.angle",
                                  "Angle between the residual and the subspace (degrees)"),
                   90.0 if norm_e < 1e-12 else self._angle(A, e), reference=90.0)
        res.metric("pythagoras", ctx.t("labs.proj.metric.pythagoras",
                                       "|y|^2 - |y_hat|^2 - |e|^2"),
                   float(y @ y - y_hat @ y_hat - e @ e), reference=0.0)
        res.metric("distance", ctx.t("labs.proj.metric.distance",
                                     "Distance from y to the subspace"), norm_e)
        res.metric("share_kept", ctx.t("labs.proj.metric.share",
                                       "Share of |y|^2 kept by the projection"),
                   float(y_hat @ y_hat / max(y @ y, 1e-12)))
        res.metric("idempotent", ctx.t("labs.proj.metric.idempotent",
                                       "max |P P - P|"),
                   float(np.max(np.abs(P_mat @ P_mat - P_mat))), reference=0.0,
                   note=ctx.t("labs.proj.metric.idempotent_note",
                              "projecting twice is the same as projecting once"))
        res.metric("symmetric", ctx.t("labs.proj.metric.symmetric",
                                      "max |P - P'|"),
                   float(np.max(np.abs(P_mat - P_mat.T))), reference=0.0)
        res.metric("rank", ctx.t("labs.proj.metric.rank",
                                 "Rank of the direction matrix"), rank,
                   reference=dim)
        coeffs = np.linalg.pinv(A) @ y
        res.metric("coefficients", ctx.t("labs.proj.metric.coefficients",
                                         "Coefficients that produce the projection"),
                   ", ".join(fmt(v, 4) for v in np.atleast_1d(coeffs)))

        degenerate = rank < dim
        near_parallel = (dim == 2 and rank == 2
                         and float(np.linalg.cond(A)) > 50)
        res.assume("independent_directions",
                   ctx.t("labs.proj.assume.independent_label",
                         "The directions are linearly independent"), not degenerate,
                   detail=ctx.t("labs.proj.assume.independent",
                                "The projection itself is always unique. The coefficients "
                                "that produce it are unique only when the directions are "
                                "independent."),
                   consequence="" if not degenerate else ctx.t(
                       "labs.proj.assume.degenerate",
                       "These directions span only {r} dimension(s), so infinitely many "
                       "coefficient pairs give the same projected point.", r=rank))
        res.assume("well_conditioned",
                   ctx.t("labs.proj.assume.conditioned_label",
                         "The directions are not nearly parallel"), not near_parallel,
                   detail=ctx.t("labs.proj.assume.conditioned",
                                "Near-parallel directions still span a plane, but tiny "
                                "changes in y produce enormous changes in the coefficients. "
                                "This is exactly the geometry behind multicollinearity."))

        res.animations.append(self._animation(ctx, A, y, dim))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.math.projection.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.math.projection.intuition"))
        res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.math.projection.math"),
                    kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.proj.proof",
            "Claim: the closest point to y in a subspace S is the unique point whose "
            "residual is orthogonal to S.\n"
            "1. Let p be any point of S with y - p orthogonal to S, and let q be any other "
            "point of S.\n"
            "2. Write y - q = (y - p) + (p - q). The second term lies in S because S is "
            "closed under subtraction.\n"
            "3. Since y - p is orthogonal to everything in S, the two terms are "
            "orthogonal, so Pythagoras gives |y - q|^2 = |y - p|^2 + |p - q|^2.\n"
            "4. The second term is non-negative and is zero only when q = p. Therefore p "
            "is the unique closest point.\n"
            "5. Existence: take p = A(A'A)^-1 A' y. Then A'(y - p) = A'y - A'y = 0, so the "
            "residual is orthogonal to every column of A, as required.\n"
            "Everything later in the library - least squares, the Frisch-Waugh-Lovell "
            "theorem, two-stage least squares, principal components - is this one statement "
            "applied to a different subspace.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.math.projection.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.math.projection.warning"), kind="warning")

        if degenerate:
            res.warnings.append(ctx.t(
                "labs.proj.warn.degenerate",
                "The two directions are linearly dependent, so they span only a line. The "
                "projected point is still unique; the coefficients producing it are not.",
            ))
        if near_parallel:
            res.warnings.append(ctx.t(
                "labs.proj.warn.conditioned",
                "The condition number of the direction matrix is {c}. The projection is "
                "stable but the coefficients are not - which is precisely what a large "
                "variance inflation factor reports in a regression.",
                c=fmt(float(np.linalg.cond(A)), 1),
            ))
        return res

    @staticmethod
    def _angle(A, e):
        norm_e = float(np.linalg.norm(e))
        cosines = [abs(float(A[:, j] @ e)) /
                   max(float(np.linalg.norm(A[:, j])) * norm_e, 1e-12)
                   for j in range(A.shape[1])]
        return float(np.degrees(np.arccos(np.clip(max(cosines), -1, 1))))

    def _space_figure(self, ctx, A, y, y_hat, e, dim, p):
        go = P.require_plotly()
        fig = ctx.figure("labs.proj.figure.space", height=520)
        span = float(max(np.max(np.abs(y)), 1.0)) * 1.5
        if dim == 1:
            d = A[:, 0] / max(np.linalg.norm(A[:, 0]), 1e-12)
            line = np.outer(np.linspace(-span, span, 2), d)
            fig.add_trace(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines",
                line={"color": ctx.color("muted"), "width": 7},
                name=ctx.t("labs.proj.trace.span", "the span (a line)"),
            ))
        else:
            u = A[:, 0] / max(np.linalg.norm(A[:, 0]), 1e-12)
            v = A[:, 1] - (A[:, 1] @ u) * u
            nv = float(np.linalg.norm(v))
            v = v / nv if nv > 1e-12 else np.array([0.0, 0.0, 1.0])
            g = np.linspace(-span, span, 12)
            U, V = np.meshgrid(g, g)
            pts = U[..., None] * u + V[..., None] * v
            fig.add_trace(go.Surface(
                x=pts[..., 0], y=pts[..., 1], z=pts[..., 2], opacity=0.3,
                showscale=False,
                colorscale=[[0, ctx.color("muted")], [1, ctx.color("muted")]],
                name=ctx.t("labs.proj.trace.span_plane", "the span (a plane)"),
            ))
        for j in range(A.shape[1]):
            norm = float(np.linalg.norm(A[:, j])) or 1.0
            P.add_arrow3d(fig, [0, 0, 0], A[:, j] * span * 0.55 / norm,
                          ctx.t("labs.proj.trace.direction", "direction {j}", j=j + 1),
                          "baseline", theme=ctx.theme)
        P.add_arrow3d(fig, [0, 0, 0], y, "y", "primary", theme=ctx.theme)
        P.add_arrow3d(fig, [0, 0, 0], y_hat,
                      ctx.t("labs.proj.trace.projection", "projection of y"),
                      "fitted", theme=ctx.theme)
        if p["show_residual"]:
            P.add_arrow3d(fig, y_hat, y,
                          ctx.t("labs.proj.trace.residual", "residual"),
                          "residual", theme=ctx.theme, dash="dot")
        fig.update_layout(scene={"aspectmode": "cube",
                                 "xaxis_title": "e1", "yaxis_title": "e2",
                                 "zaxis_title": "e3"})
        P.add_legend_note(fig, ctx.t(
            "labs.proj.legend_space",
            "Rotate the plot: the residual meets the shaded set at a right angle from every "
            "viewpoint, because the orthogonality is algebraic rather than a trick of the "
            "camera.",
        ), theme=ctx.theme)
        return fig

    def _distance_figure(self, ctx, A, y, y_hat, dim):
        if dim == 1:
            d = A[:, 0] / max(np.linalg.norm(A[:, 0]), 1e-12)
            centre = float(y @ d)
            ts = np.linspace(centre - 4, centre + 4, 300)
            distances = [float(np.linalg.norm(y - t * d)) for t in ts]
            xaxis = ctx.t("labs.proj.axis.coefficient",
                          "Coefficient on the direction")
        else:
            u = A[:, 0] / max(np.linalg.norm(A[:, 0]), 1e-12)
            centre = float(y @ u)
            ts = np.linspace(centre - 4, centre + 4, 300)
            rest = y_hat - float(y_hat @ u) * u
            distances = [float(np.linalg.norm(y - (t * u + rest))) for t in ts]
            xaxis = ctx.t("labs.proj.axis.coefficient_first",
                          "Coefficient on the first direction")
        fig = ctx.figure(
            "labs.proj.figure.distance",
            xaxis_title=xaxis,
            yaxis_title=ctx.t("labs.proj.axis.distance", "Distance from y"),
            height=360,
        )
        P.add_curve(fig, ts, distances,
                    ctx.t("labs.proj.trace.distance",
                          "distance to each candidate point"),
                    "primary", theme=ctx.theme)
        best = ts[int(np.argmin(distances))]
        P.add_vline(fig, float(best),
                    ctx.t("labs.proj.trace.minimum",
                          "minimum, where the residual is orthogonal"),
                    "truth", theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.proj.legend_distance",
            "The curve is smooth and has exactly one minimum, so 'closest point' and "
            "'perpendicular residual' pick out the same point. Both descriptions define the "
            "projection.",
        ), theme=ctx.theme)
        return fig

    def _gram_schmidt_figure(self, ctx, a, b):
        go = P.require_plotly()
        u1 = a / max(np.linalg.norm(a), 1e-12)
        proj = (b @ u1) * u1
        w = b - proj
        u2 = w / max(np.linalg.norm(w), 1e-12)
        fig = ctx.figure("labs.proj.figure.gram_schmidt", height=460)
        P.add_arrow3d(fig, [0, 0, 0], a, "a", "baseline", theme=ctx.theme)
        P.add_arrow3d(fig, [0, 0, 0], b, "b", "secondary", theme=ctx.theme)
        P.add_arrow3d(fig, [0, 0, 0], proj,
                      ctx.t("labs.proj.trace.proj_b",
                            "the part of b already explained by a"),
                      "fitted", theme=ctx.theme, dash="dash")
        P.add_arrow3d(fig, proj, b,
                      ctx.t("labs.proj.trace.orthogonal_part",
                            "what is left: orthogonal to a"),
                      "positive", theme=ctx.theme)
        fig.update_layout(scene={"aspectmode": "cube"})
        P.add_legend_note(fig, ctx.t(
            "labs.proj.legend_gram_schmidt",
            "Gram-Schmidt is projection used as a tool: subtract from b whatever a already "
            "explains, and what remains is orthogonal by construction. The Frisch-Waugh-"
            "Lovell theorem is this same step applied to a regression. Residual angle to a: "
            "{deg} degrees.",
            deg=fmt(float(np.degrees(np.arccos(
                np.clip(abs((b - proj) @ u1) /
                        max(np.linalg.norm(b - proj), 1e-12), -1, 1)))), 1),
        ), theme=ctx.theme)
        return fig

    def _decomposition_figure(self, ctx, y, y_hat, e):
        labels = [ctx.t("labs.proj.trace.total", "|y|^2"),
                  ctx.t("labs.proj.trace.projected", "|projection|^2"),
                  ctx.t("labs.proj.trace.residual_sq", "|residual|^2")]
        values = [float(y @ y), float(y_hat @ y_hat), float(e @ e)]
        fig = ctx.figure(
            "labs.proj.figure.decomposition",
            xaxis_title=ctx.t("labs.proj.axis.part", "Component"),
            yaxis_title=ctx.t("labs.proj.axis.squared", "Squared length"),
            height=340,
        )
        P.add_bar(fig, labels, values,
                  ctx.t("labs.proj.trace.lengths", "squared length"),
                  "primary", theme=ctx.theme, text=[fmt(v, 3) for v in values])
        P.add_legend_note(fig, ctx.t(
            "labs.proj.legend_decomposition",
            "The last two bars sum to the first, exactly. In a regression the same three "
            "numbers are called total, explained and residual sum of squares - and their "
            "ratio is R-squared.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, A, y, dim):
        go = P.require_plotly()
        d = A[:, 0] / max(np.linalg.norm(A[:, 0]), 1e-12)
        centre = float(y @ d)
        ts = np.linspace(centre - 3.0, centre + 3.0, 25)
        frames, steps = [], []
        for i, t in enumerate(ts):
            point = t * d
            resid = y - point
            dist = float(np.linalg.norm(resid))
            dot = float(d @ resid)
            frames.append(go.Frame(name=f"{t:.2f}", data=[
                go.Scatter3d(x=[point[0], y[0]], y=[point[1], y[1]],
                             z=[point[2], y[2]]),
                go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]]),
            ]))
            steps.append(AnimationStep(
                id=f"t_{i}", frame=i,
                title=ctx.t("labs.proj.anim.title",
                            "candidate point at coefficient {t}", t=fmt(t, 2)),
                what_you_see=ctx.t("labs.proj.anim.see",
                                   "A candidate point sliding along the span, with the "
                                   "segment joining it to y."),
                what_changed=ctx.t("labs.proj.anim.changed",
                                   "The coefficient moved to {t}.", t=fmt(t, 2)),
                why=ctx.t("labs.proj.anim.why",
                          "Every point of the span is some multiple of the direction. "
                          "Sliding the coefficient walks through all of them."),
                interpretation=ctx.t("labs.proj.anim.interpret",
                                     "Distance to y is {d}; the residual's component along "
                                     "the direction is {o}.",
                                     d=fmt(dist, 4), o=fmt(dot, 5)),
                conclusion=ctx.t("labs.proj.anim.conclude",
                                 "The distance is smallest exactly where that component "
                                 "reaches zero - closest and perpendicular are the same "
                                 "condition."),
                warning=ctx.t("labs.proj.anim.warn",
                              "The right angle is a statement about the inner product, not "
                              "about how the picture happens to be drawn."),
                math="minimise |y - t a|  <=>  a'(y - t a) = 0  <=>  t = a'y / a'a",
                outputs={"coefficient": round(float(t), 4),
                         "distance": round(dist, 5),
                         "residual_dot_direction": round(dot, 8)},
                highlighted=("candidate_point", "residual_segment"),
            ))
        fig = ctx.figure("labs.proj.figure.animation", height=440)
        span = float(max(np.max(np.abs(y)), 1.0)) * 1.5
        line = np.outer(np.linspace(-span, span, 2), d)
        fig.add_trace(go.Scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2],
                                   mode="lines",
                                   line={"color": ctx.color("muted"), "width": 7},
                                   name=ctx.t("labs.proj.trace.span",
                                              "the span (a line)")))
        fig.add_trace(go.Scatter3d(x=[0, y[0]], y=[0, y[1]], z=[0, y[2]], mode="lines",
                                   line={"color": ctx.color("residual"), "width": 5,
                                         "dash": "dot"},
                                   name=ctx.t("labs.proj.trace.segment",
                                              "segment from the candidate to y")))
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                   marker={"color": ctx.color("fitted"), "size": 7},
                                   name=ctx.t("labs.proj.trace.candidate",
                                              "candidate point")))
        P.add_arrow3d(fig, [0, 0, 0], y, "y", "primary", theme=ctx.theme)
        fig.update_layout(scene={"aspectmode": "cube"})
        build_frames(fig, frames, duration=280, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.proj.slider", "coefficient"))
        return animation(
            "sliding_candidate", fig, steps,
            purpose=ctx.t("labs.proj.anim.purpose",
                          "Find the closest point by hand and discover the right angle."),
            summary=ctx.t(
                "labs.proj.anim.summary",
                "Sliding along the span, the distance to y falls, reaches a single minimum "
                "and rises again - and the minimum is precisely where the residual stops "
                "having any component along the direction. That single fact is reused, "
                "unchanged, by least squares, partialling out, instrumental variables and "
                "principal components."),
            evidence=EvidenceType.GEOMETRIC_PROOF,
        )


LAB = ProjectionLab(SPEC)
