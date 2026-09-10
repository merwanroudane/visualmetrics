"""Principal component analysis as a variance-maximizing projection."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...data.generators.ml import correlated_features

__all__ = ["LAB", "SPEC", "pca"]


def pca(X, standardize: bool = False):
    """Eigen-decomposition of the covariance (or correlation) matrix."""
    X = np.asarray(X, dtype=float)
    centred = X - X.mean(axis=0)
    if standardize:
        centred = centred / np.clip(centred.std(axis=0, ddof=1), 1e-12, None)
    cov = np.cov(centred, rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # sign convention: make the largest-magnitude loading positive
    for j in range(vecs.shape[1]):
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] *= -1
    scores = centred @ vecs
    total = float(np.sum(vals))
    return {"values": vals, "vectors": vecs, "scores": scores, "centred": centred,
            "explained": vals / total if total > 0 else vals,
            "cumulative": np.cumsum(vals) / total if total > 0 else vals,
            "cov": cov, "total": total}


SPEC = make_spec(
    "multivariate.pca",
    Domain.MULTIVARIATE,
    "dimension_reduction",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.GEOMETRIC_PROOF,
    controls=(
        slider("variance_ratio", 4.0, 1.0, 40.0, 0.1, group="data"),
        slider("rotation", 30.0, 0.0, 180.0, 1.0, group="data"),
        int_slider("n_features", 2, 2, 12, 1, group="data"),
        int_slider("n", 300, 20, 20000, 10, group="data"),
        slider("noise", 0.0, 0.0, 2.0, 0.05, group="data"),
        slider("scale_x1", 1.0, 0.1, 20.0, 0.1, group="data"),
        int_slider("components", 1, 1, 12, 1, group="projection"),
        toggle("standardize", False, group="projection"),
        toggle("show_reconstruction", True, group="views"),
        toggle("show_scree", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", variance_ratio=4.0, rotation=30.0),
        scenario("isotropic", "null", variance_ratio=1.0),
        scenario("extreme_elongation", "strong", variance_ratio=30.0),
        scenario("axis_aligned", "boundary", rotation=0.0, variance_ratio=6.0),
        scenario("forty_five_degrees", "canonical", rotation=45.0),
        scenario("scale_trap", "counterexample", scale_x1=15.0, variance_ratio=3.0,
                 standardize=False),
        scenario("scale_trap_fixed", "robustness", scale_x1=15.0, variance_ratio=3.0,
                 standardize=True),
        scenario("many_features", "compare_methods", n_features=8, components=3),
        scenario("noisy", "high_noise", noise=1.5),
        scenario("small_sample", "small_sample", n=25),
    ),
    related=("probability.bivariate", "regression.multicollinearity",
             "multivariate.clustering"),
    tags=("pca", "eigenvector", "explained variance", "projection", "svd",
          "dimension reduction", "scree plot"),
    aliases=("principal component analysis", "acp", "المكونات الرئيسية",
             "dimension reduction", "eigenvector"),
    backends=("numpy", "scikit-learn"),
    proof_ids=("multivariate.pca.variance_maximization",),
    references=(
        ref("Jolliffe, I. T. (2002). Principal Component Analysis.", kind="book"),
        ref("Strang, G. (2016). Introduction to Linear Algebra.", kind="book"),
    ),
    curriculum_tags=("dz.data_analysis1",),
)


class PCALab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = correlated_features(
            n=int(p["n"]), variance_ratio=float(p["variance_ratio"]),
            rotation=float(p["rotation"]), noise=float(p["noise"]),
            n_features=int(p["n_features"]), seed=state.seed,
        )
        k = int(p["n_features"])
        X = np.column_stack([data[f"x{j + 1}"] for j in range(k)])
        X[:, 0] = X[:, 0] * float(p["scale_x1"])
        model = pca(X, bool(p["standardize"]))
        n_comp = min(int(p["components"]), k)

        res.data = data
        res.dgp = data.dgp + (
            f";  feature 1 rescaled by {float(p['scale_x1']):g}"
            if float(p["scale_x1"]) != 1.0 else "")

        res.add_panel(ctx.panel(
            "projection", self._projection_figure(ctx, X, model, n_comp, p),
            "labs.pca.figure.projection", evidence=EvidenceType.GEOMETRIC_PROOF,
        ))
        if p["show_scree"]:
            res.add_panel(ctx.panel(
                "scree", self._scree_figure(ctx, model, n_comp),
                "labs.pca.figure.scree", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_reconstruction"]:
            res.add_panel(ctx.panel(
                "reconstruction", self._reconstruction_figure(ctx, X, model, n_comp),
                "labs.pca.figure.reconstruction", tab="diagnostics",
                evidence=EvidenceType.GEOMETRIC_PROOF,
            ))
        res.add_panel(ctx.panel(
            "variance_objective", self._objective_figure(ctx, model, k),
            "labs.pca.figure.objective", tab="math",
            evidence=EvidenceType.GEOMETRIC_PROOF,
        ))

        recon = self._reconstruct(model, n_comp)
        # divide by n - 1 so the identity matches the ddof=1 eigenvalues exactly
        dof = max(model["centred"].shape[0] - 1, 1)
        recon_error = float(np.sum((model["centred"] - recon) ** 2) / dof)
        discarded = float(np.sum(model["values"][n_comp:]))

        for j in range(min(k, 4)):
            res.metric(f"eigenvalue_{j + 1}",
                       ctx.t("labs.pca.metric.eigenvalue",
                             "Variance along component {j}", j=j + 1),
                       float(model["values"][j]))
            res.metric(f"explained_{j + 1}",
                       ctx.t("labs.pca.metric.explained",
                             "Share of variance in component {j}", j=j + 1),
                       float(model["explained"][j]))
        res.metric("cumulative", ctx.t("labs.pca.metric.cumulative",
                                       "Variance kept by the first {k} components",
                                       k=n_comp),
                   float(model["cumulative"][n_comp - 1]))
        res.metric("reconstruction_error", ctx.t("labs.pca.metric.reconstruction",
                                                 "Mean squared reconstruction error"),
                   recon_error, reference=discarded,
                   note=ctx.t("labs.pca.metric.reconstruction_note",
                              "equals exactly the variance of the discarded components"))
        res.metric("identity_error", ctx.t("labs.pca.metric.identity",
                                           "|reconstruction error - discarded variance|"),
                   abs(recon_error - discarded), reference=0.0)
        res.metric("first_direction", ctx.t("labs.pca.metric.direction",
                                            "Angle of the first component (degrees)"),
                   float(np.degrees(np.arctan2(model["vectors"][1, 0],
                                               model["vectors"][0, 0])) % 180),
                   reference=float(p["rotation"]) % 180 if float(p["scale_x1"]) == 1.0
                   else None)
        res.metric("condition_number", ctx.t("labs.pca.metric.condition",
                                             "Ratio of largest to smallest eigenvalue"),
                   float(model["values"][0] /
                         max(float(model["values"][-1]), 1e-12)))

        ratio = float(model["values"][0] / max(float(model["values"][1]), 1e-12))
        isotropic = ratio < 1.3
        rescaled = float(p["scale_x1"]) != 1.0 and not bool(p["standardize"])
        res.assume("scale_choice", ctx.t("labs.pca.assume.scale_label",
                                         "The variables are on comparable scales"),
                   not rescaled,
                   detail=ctx.t("labs.pca.assume.scale",
                                "PCA maximises variance, and variance depends on units. "
                                "Standardizing first means analysing the correlation matrix "
                                "instead - a different question, not a neutral cleanup."),
                   consequence="" if not rescaled else ctx.t(
                       "labs.pca.assume.scale_consequence",
                       "Feature 1 was multiplied by {s}, so it now dominates the first "
                       "component purely because of its units.",
                       s=fmt(p["scale_x1"], 1)))
        res.assume("meaningful_direction",
                   ctx.t("labs.pca.assume.direction_label",
                         "There is a dominant direction to find"), not isotropic,
                   detail=ctx.t("labs.pca.assume.direction",
                                "In an isotropic cloud every direction has the same "
                                "variance, so the components are determined by sampling "
                                "noise and are not reproducible."))
        res.assume("linearity", ctx.t("labs.pca.assume.linear_label",
                                      "The structure is linear"), True,
                   detail=ctx.t("labs.pca.assume.linear",
                                "PCA can only find flat subspaces. Curved structure - a "
                                "spiral, a ring - is invisible to it."))

        res.animations.append(self._animation(ctx, X, model))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.multivariate.pca.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.multivariate.pca.intuition"))
        res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.multivariate.pca.math"),
                    kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.pca.proof",
            "Claim: the first principal component maximises the variance of the projection, "
            "and equivalently minimises the squared reconstruction error.\n"
            "1. For a unit vector w, the projected data have variance w' S w, where S is "
            "the sample covariance matrix.\n"
            "2. Maximise w' S w subject to w'w = 1. The Lagrangian is w' S w - lambda "
            "(w'w - 1), and setting its derivative to zero gives S w = lambda w.\n"
            "3. So the optimum is an eigenvector of S, and the value attained is "
            "w' S w = lambda. The maximum is therefore the largest eigenvalue and its "
            "eigenvector.\n"
            "4. For the reconstruction view: total variance is fixed, and by Pythagoras "
            "each point splits into its projection plus a perpendicular residual. "
            "Maximising the variance kept is therefore identical to minimising the squared "
            "residual, which is why one procedure answers both questions.\n"
            "The read-out above verifies step 4 numerically: the mean squared "
            "reconstruction error equals the sum of the discarded eigenvalues to machine "
            "precision.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.multivariate.pca.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.multivariate.pca.warning"), kind="warning")

        if rescaled:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.pca.counterexample",
                "The first component now explains {e} of the variance, and its direction "
                "follows feature 1 - not because that feature matters, but because it was "
                "measured in smaller units. Switch on standardization and watch the "
                "component turn.", e=pct(float(model["explained"][0])),
            ), kind="warning")
        if isotropic:
            res.warnings.append(ctx.t(
                "labs.pca.warn.isotropic",
                "The eigenvalues are nearly equal, so the components are essentially "
                "arbitrary rotations. Re-run with another seed and the directions will "
                "change completely.",
            ))
        return res

    @staticmethod
    def _reconstruct(model, n_comp):
        W = model["vectors"][:, :n_comp]
        return model["scores"][:, :n_comp] @ W.T

    def _projection_figure(self, ctx, X, model, n_comp, p):
        centred = model["centred"]
        fig = ctx.figure(
            "labs.pca.figure.projection",
            xaxis_title="x1 (centred)", yaxis_title="x2 (centred)", height=460,
        )
        P.add_points(fig, centred[:, 0], centred[:, 1],
                     ctx.t("labs.pca.trace.data", "centred data"),
                     "primary", theme=ctx.theme, size=5, opacity=0.5)
        scale = float(np.sqrt(model["values"][0])) * 2.2
        for j in range(min(2, model["vectors"].shape[1])):
            v = model["vectors"][:2, j]
            length = float(np.sqrt(model["values"][j])) * 2.2
            P.add_arrow(fig, 0, 0, v[0] * length, v[1] * length,
                        ctx.t("labs.pca.trace.component",
                              "component {j}: {e} of the variance", j=j + 1,
                              e=pct(float(model["explained"][j]))),
                        "truth" if j == 0 else "secondary", theme=ctx.theme, width=4.0)
        recon = self._reconstruct(model, n_comp)
        if n_comp == 1 and centred.shape[1] >= 2:
            go = P.require_plotly()
            sub = min(centred.shape[0], 120)
            seg_x, seg_y = [], []
            for i in range(sub):
                seg_x += [centred[i, 0], recon[i, 0], None]
                seg_y += [centred[i, 1], recon[i, 1], None]
            fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines",
                                     line={"color": ctx.color("residual"), "width": 0.9,
                                           "dash": "dot"},
                                     name=ctx.t("labs.pca.trace.residual",
                                                "what the projection throws away"),
                                     hoverinfo="skip"))
            P.add_points(fig, recon[:, 0], recon[:, 1],
                         ctx.t("labs.pca.trace.projected", "projected points"),
                         "fitted", theme=ctx.theme, size=4, opacity=0.6)
        fig.update_xaxes(range=[-scale, scale], scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[-scale, scale])
        P.add_legend_note(fig, ctx.t(
            "labs.pca.legend_projection",
            "The dotted segments are exactly perpendicular to the first component. "
            "Minimising their total squared length and maximising the spread along the "
            "arrow are the same problem.",
        ), theme=ctx.theme)
        return fig

    def _scree_figure(self, ctx, model, n_comp):
        go = P.require_plotly()
        idx = np.arange(1, model["values"].size + 1)
        fig = ctx.figure(
            "labs.pca.figure.scree",
            xaxis_title=ctx.t("labs.common.axis.component"),
            yaxis_title=ctx.t("labs.common.axis.explained_var"),
            height=370,
        )
        fig.add_trace(go.Bar(x=idx, y=model["explained"],
                             marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.pca.trace.individual",
                                        "share per component"),
                             text=[pct(v) for v in model["explained"]],
                             textposition="outside"))
        P.add_curve(fig, idx, model["cumulative"],
                    ctx.t("labs.pca.trace.cumulative", "cumulative share"),
                    "secondary", theme=ctx.theme, mode="lines+markers")
        P.add_vline(fig, n_comp + 0.5,
                    ctx.t("labs.pca.trace.kept", "components kept"),
                    "warning", theme=ctx.theme, dash="dash")
        fig.update_yaxes(range=[0, 1.05])
        P.add_legend_note(fig, ctx.t(
            "labs.pca.legend_scree",
            "A sharp elbow means there really is a low-dimensional structure. A gentle "
            "slope means the choice of how many components to keep is arbitrary - and "
            "should be admitted as such.",
        ), theme=ctx.theme)
        return fig

    def _reconstruction_figure(self, ctx, X, model, n_comp):
        max_k = model["values"].size
        errors, discarded = [], []
        for k in range(1, max_k + 1):
            recon = self._reconstruct(model, k)
            dof = max(model["centred"].shape[0] - 1, 1)
            errors.append(float(np.sum((model["centred"] - recon) ** 2) / dof))
            discarded.append(float(np.sum(model["values"][k:])))
        ks = np.arange(1, max_k + 1)
        fig = ctx.figure(
            "labs.pca.figure.reconstruction",
            xaxis_title=ctx.t("labs.pca.axis.kept", "Components kept"),
            yaxis_title=ctx.t("labs.pca.axis.error", "Mean squared reconstruction error"),
            height=360,
        )
        P.add_curve(fig, ks, errors,
                    ctx.t("labs.pca.trace.measured", "measured reconstruction error"),
                    "primary", theme=ctx.theme, mode="lines+markers")
        P.add_curve(fig, ks, discarded,
                    ctx.t("labs.pca.trace.predicted",
                          "sum of the discarded eigenvalues"),
                    "truth", theme=ctx.theme, dash="dash", mode="lines+markers")
        P.add_vline(fig, n_comp, ctx.t("labs.common.trace.current"), "warning",
                    theme=ctx.theme)
        P.add_legend_note(fig, ctx.t(
            "labs.pca.legend_reconstruction",
            "The two curves lie exactly on top of each other. That coincidence is not a "
            "coincidence - it is the theorem, checked numerically.",
        ), theme=ctx.theme)
        return fig

    def _objective_figure(self, ctx, model, k):
        angles = np.linspace(0, np.pi, 361)
        cov2 = model["cov"][:2, :2]
        variances = [float(np.array([np.cos(a), np.sin(a)]) @ cov2
                           @ np.array([np.cos(a), np.sin(a)])) for a in angles]
        best = float(np.degrees(angles[int(np.argmax(variances))]))
        fig = ctx.figure(
            "labs.pca.figure.objective",
            xaxis_title=ctx.t("labs.pca.axis.angle", "Direction (degrees)"),
            yaxis_title=ctx.t("labs.pca.axis.projected_var",
                              "Variance of the projection"),
            height=350,
        )
        P.add_curve(fig, np.degrees(angles), variances,
                    ctx.t("labs.pca.trace.objective",
                          "w' S w over all unit directions"),
                    "primary", theme=ctx.theme)
        P.add_vline(fig, best,
                    ctx.t("labs.pca.trace.maximum",
                          "maximum at {a} degrees", a=fmt(best, 1)),
                    "truth", theme=ctx.theme)
        P.add_hline(fig, float(model["values"][0]),
                    ctx.t("labs.pca.trace.eigenvalue",
                          "largest eigenvalue = {v}", v=fmt(float(model["values"][0]), 3)),
                    "positive", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.pca.legend_objective",
            "The peak of this curve sits exactly at the largest eigenvalue, and the angle at "
            "which it peaks is exactly the first eigenvector. The optimisation and the "
            "eigen-decomposition are the same computation.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, X, model):
        go = P.require_plotly()
        centred = model["centred"]
        angles = np.linspace(0, np.pi, 25)
        cov2 = model["cov"][:2, :2]
        scale = float(np.sqrt(model["values"][0])) * 2.2
        frames, steps = [], []
        for i, a in enumerate(angles):
            w = np.array([np.cos(a), np.sin(a)])
            proj = (centred[:, :2] @ w)[:, None] * w[None, :]
            var = float(w @ cov2 @ w)
            resid = float(np.mean(np.sum((centred[:, :2] - proj) ** 2, axis=1)))
            frames.append(go.Frame(name=f"{np.degrees(a):.0f}", data=[
                go.Scatter(x=[-scale * w[0], scale * w[0]],
                           y=[-scale * w[1], scale * w[1]]),
                go.Scatter(x=proj[:, 0], y=proj[:, 1]),
            ]))
            steps.append(AnimationStep(
                id=f"angle_{i}", frame=i,
                title=ctx.t("labs.pca.anim.title", "direction at {a} degrees",
                            a=fmt(float(np.degrees(a)), 0)),
                what_you_see=ctx.t("labs.pca.anim.see",
                                   "A candidate direction and the data projected onto it."),
                what_changed=ctx.t("labs.pca.anim.changed",
                                   "The candidate direction rotated to {a} degrees.",
                                   a=fmt(float(np.degrees(a)), 0)),
                why=ctx.t("labs.pca.anim.why",
                          "Rotating the line changes how spread out the shadows are. Total "
                          "variance is fixed, so whatever the projection does not keep ends "
                          "up in the perpendicular residual."),
                interpretation=ctx.t("labs.pca.anim.interpret",
                                     "Projected variance {v}, squared residual {r} - and "
                                     "their sum is constant at {t}.",
                                     v=fmt(var, 4), r=fmt(resid, 4),
                                     t=fmt(var + resid, 4)),
                conclusion=ctx.t("labs.pca.anim.conclude",
                                 "The best direction maximises one and minimises the other "
                                 "simultaneously. It is the leading eigenvector."),
                warning=ctx.t("labs.pca.anim.warn",
                              "This objective depends on the units. Rescale a variable and "
                              "the winning direction moves."),
                math="maximise w' S w subject to w'w = 1  =>  S w = lambda w",
                outputs={"angle_degrees": round(float(np.degrees(a)), 1),
                         "projected_variance": round(var, 5),
                         "residual": round(resid, 5)},
                highlighted=("candidate_direction", "projections"),
            ))
        fig = ctx.figure(
            "labs.pca.figure.animation",
            xaxis_title="x1 (centred)", yaxis_title="x2 (centred)", height=430,
        )
        P.add_points(fig, centred[:, 0], centred[:, 1],
                     ctx.t("labs.pca.trace.data", "centred data"), "muted",
                     theme=ctx.theme, size=4, opacity=0.35)
        fig.add_trace(go.Scatter(x=[-scale, scale], y=[0, 0], mode="lines",
                                 line={"color": ctx.color("truth"), "width": 3.4},
                                 name=ctx.t("labs.pca.trace.candidate",
                                            "candidate direction")))
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("fitted"), "size": 5,
                                         "opacity": 0.7},
                                 name=ctx.t("labs.pca.trace.projected",
                                            "projected points")))
        fig.update_xaxes(range=[-scale, scale], scaleanchor="y", scaleratio=1)
        fig.update_yaxes(range=[-scale, scale])
        build_frames(fig, frames, duration=300, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.pca.slider", "angle"))
        return animation(
            "rotating_direction", fig, steps,
            purpose=ctx.t("labs.pca.anim.purpose",
                          "Search for the best direction by hand and find the eigenvector."),
            summary=ctx.t(
                "labs.pca.anim.summary",
                "Every candidate direction splits the data's total variance into a kept "
                "part and a discarded part, and the two always add to the same total. The "
                "first principal component is simply the rotation that wins that split - "
                "which is why maximising variance and minimising reconstruction error give "
                "the same answer."),
            evidence=EvidenceType.GEOMETRIC_PROOF,
        )


LAB = PCALab(SPEC)
