"""Ridge, lasso and the geometry that makes one sparse and the other not."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...backends import linear as LM
from ...data.generators.ml import sparse_regression

__all__ = ["LAB", "SPEC", "ridge_path", "lasso_coordinate_descent"]


def ridge_path(X, y, lambdas):
    """Closed-form ridge coefficients along a penalty path (intercept unpenalized)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.empty((len(lambdas), X.shape[1]))
    for i, lam in enumerate(lambdas):
        pen = lam * np.eye(X.shape[1])
        out[i] = np.linalg.solve(X.T @ X + pen, X.T @ y)
    return out


def lasso_coordinate_descent(X, y, lam, max_iter: int = 300, tol: float = 1e-7):
    """Lasso by cyclic coordinate descent with soft thresholding."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    norms = np.sum(X**2, axis=0)
    resid = y.copy()
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            if norms[j] < 1e-12:
                continue
            rho = float(X[:, j] @ resid) + norms[j] * beta[j]
            new = np.sign(rho) * max(abs(rho) - lam * n, 0.0) / norms[j]
            change = new - beta[j]
            if change != 0.0:
                resid -= change * X[:, j]
                beta[j] = new
                max_change = max(max_change, abs(change))
        if max_change < tol:
            break
    return beta


SPEC = make_spec(
    "ml.regularization",
    Domain.ML,
    "regularization",
    module=__name__,
    levels=("intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.GEOMETRIC_PROOF,
    controls=(
        slider("lambda", 0.1, 0.0, 5.0, 0.005, group="penalty"),
        select("penalty", "lasso", ("ridge", "lasso"), group="penalty"),
        int_slider("p", 20, 2, 200, 1, group="data"),
        int_slider("n_active", 3, 0, 20, 1, group="data"),
        int_slider("n", 120, 10, 5000, 10, group="data"),
        slider("signal", 2.0, 0.0, 6.0, 0.05, group="data"),
        slider("noise", 1.0, 0.05, 5.0, 0.05, group="data"),
        slider("correlation", 0.0, 0.0, 0.95, 0.01, group="data"),
        toggle("show_geometry", True, group="views"),
        toggle("show_path", True, group="views"),
        toggle("show_cv", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", penalty="lasso", p=20, n_active=3),
        scenario("no_penalty", "null", **{"lambda": 0.0}),
        scenario("ridge", "compare_methods", penalty="ridge", **{"lambda": 1.0}),
        scenario("lasso", "compare_methods", penalty="lasso", **{"lambda": 0.1}),
        scenario("heavy_penalty", "strong", **{"lambda": 3.0}),
        scenario("high_dimensional", "boundary", p=150, n=60, n_active=5),
        scenario("dense_truth", "misspecification", n_active=18, p=20),
        scenario("correlated_features", "violation", correlation=0.85, p=20, n_active=3),
        scenario("weak_signal", "weak", signal=0.4),
        scenario("small_sample", "small_sample", n=25, p=20),
        scenario("large_sample", "large_sample", n=2000, p=20),
    ),
    prerequisites=("ml.bias_variance",),
    related=("regression.multicollinearity", "ml.bias_variance"),
    tags=("ridge", "lasso", "elastic net", "shrinkage", "sparsity", "penalty",
          "coefficient path"),
    aliases=("ridge lasso", "regularisation", "الانكماش", "shrinkage",
             "penalized regression"),
    backends=("numpy", "scikit-learn"),
    proof_ids=("ml.lasso.sparsity_geometry",),
    references=(
        ref("Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. "
            "JRSS-B 58(1).", kind="paper", doi="10.1111/j.2517-6161.1996.tb02080.x"),
        ref("Hoerl, A. E. and Kennard, R. W. (1970). Ridge regression. Technometrics 12(1).",
            kind="paper", doi="10.1080/00401706.1970.10488634"),
    ),
    curriculum_tags=("stanford.stats315a",),
)


class RegularizationLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = self._generate(p, state.seed)
        X = data.meta["X"]
        y = data.meta["y"]
        beta_true = data.meta["beta"]
        lam = float(p["lambda"])
        penalty = str(p["penalty"])

        ols = self._ols(X, y)
        coefs = self._fit(X, y, lam, penalty)
        path, lambdas = self._path(X, y, penalty)
        cv = self._cv_curve(X, y, penalty, state.seed) if p["show_cv"] else None

        res.data = data
        res.dgp = data.dgp

        if p["show_geometry"]:
            res.add_panel(ctx.panel(
                "geometry", self._geometry_figure(ctx, X, y, lam, penalty),
                "labs.reg.figure.geometry", evidence=EvidenceType.GEOMETRIC_PROOF,
            ))
        res.add_panel(ctx.panel(
            "coefficients", self._coefficients_figure(ctx, coefs, ols, beta_true, p),
            "labs.reg.figure.coefficients",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_path"]:
            res.add_panel(ctx.panel(
                "path", self._path_figure(ctx, path, lambdas, beta_true, lam, penalty),
                "labs.reg.figure.path", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if cv is not None:
            res.add_panel(ctx.panel(
                "cv", self._cv_figure(ctx, cv, lam),
                "labs.reg.figure.cv", tab="diagnostics",
                evidence=EvidenceType.SIMULATION,
            ))

        nonzero = int(np.sum(np.abs(coefs) > 1e-8))
        true_nonzero = int(np.sum(np.abs(beta_true) > 1e-8))
        res.metric("nonzero", ctx.t("labs.reg.metric.nonzero",
                                    "Coefficients not exactly zero"), nonzero,
                   reference=true_nonzero,
                   note=ctx.t("labs.reg.metric.nonzero_note",
                              "out of {p} features", p=int(p["p"])))
        res.metric("true_nonzero", ctx.t("labs.reg.metric.true_nonzero",
                                         "Truly non-zero coefficients"), true_nonzero)
        res.metric("correct_selection", ctx.t("labs.reg.metric.selection",
                                              "Active features correctly identified"),
                   int(np.sum((np.abs(coefs) > 1e-8) & (np.abs(beta_true) > 1e-8))),
                   reference=true_nonzero)
        res.metric("false_selection", ctx.t("labs.reg.metric.false",
                                            "Irrelevant features kept"),
                   int(np.sum((np.abs(coefs) > 1e-8) & (np.abs(beta_true) <= 1e-8))))
        res.metric("coefficient_error", ctx.t("labs.reg.metric.error",
                                              "Squared error of the coefficients"),
                   float(np.sum((coefs - beta_true) ** 2)))
        res.metric("ols_error", ctx.t("labs.reg.metric.ols_error",
                                      "Squared coefficient error without a penalty"),
                   float(np.sum((ols - beta_true) ** 2))
                   if ols is not None else "not estimable",
                   note=ctx.t("labs.reg.metric.ols_note",
                              "least squares has no unique solution when p >= n"))
        res.metric("shrinkage", ctx.t("labs.reg.metric.shrinkage",
                                      "Total coefficient size (L1 norm)"),
                   float(np.sum(np.abs(coefs))))
        if cv is not None:
            best = min(cv, key=lambda row: row["mse"])
            res.metric("cv_best_lambda", ctx.t("labs.reg.metric.cv",
                                               "Penalty minimising cross-validated error"),
                       best["lambda"])
            res.metric("cv_error", ctx.t("labs.reg.metric.cv_error",
                                         "Cross-validated error at that penalty"),
                       best["mse"])

        res.assume("standardized", ctx.t("labs.reg.assume.scale_label",
                                         "Features are standardized"), True,
                   detail=ctx.t("labs.reg.assume.scale",
                                "Penalties are not scale invariant: doubling a feature's "
                                "units halves the penalty it feels. The generator "
                                "standardizes for this reason."))
        res.assume("sparsity", ctx.t("labs.reg.assume.sparsity_label",
                                     "The truth really is sparse"),
                   true_nonzero <= max(int(p["p"]) // 4, 1),
                   detail=ctx.t("labs.reg.assume.sparsity",
                                "The lasso's selection consistency requires that only a few "
                                "coefficients matter. With a dense truth it discards real "
                                "signal."))
        res.assume("no_causal_claim", ctx.t("labs.reg.assume.causal_label",
                                            "Selection is not identification"), True,
                   detail=ctx.t("labs.reg.assume.causal",
                                "A penalty trades bias for variance on purpose, so "
                                "penalized coefficients are biased by construction and "
                                "must not be read as causal effects. Post-selection "
                                "inference on them is also invalid without correction."))

        res.animations.append(self._animation(ctx, X, y, beta_true, penalty))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.ml.regularization.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.ml.regularization.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.ml.regularization.math"), kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.reg.proof",
            "Why the lasso produces exact zeros and ridge does not.\n"
            "1. Both problems can be written as: minimise the residual sum of squares "
            "subject to the coefficient vector lying inside a budget region.\n"
            "2. Level sets of the residual sum of squares are ellipses centred on the "
            "least-squares solution.\n"
            "3. The optimum is the first point at which a growing ellipse touches the "
            "budget region.\n"
            "4. Ridge's region is a disc: it is smooth everywhere, so the tangency point "
            "almost surely has both coordinates non-zero.\n"
            "5. The lasso's region is a diamond whose corners lie exactly on the axes. A "
            "corner is where the boundary is non-differentiable, and an ellipse arriving "
            "from a generic direction touches a corner for a whole range of orientations - "
            "an event of positive probability, not measure zero.\n"
            "6. Touching a corner means a coordinate is exactly zero. That is variable "
            "selection, and it comes from the geometry of the L1 ball rather than from any "
            "thresholding rule added afterwards.\n"
            "In the orthonormal case this becomes explicit: ridge multiplies each "
            "coefficient by 1/(1 + lambda) while the lasso applies soft thresholding, "
            "sign(b) * max(|b| - lambda, 0), which is exactly zero for small |b|.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.ml.regularization.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.ml.regularization.warning"), kind="warning")

        if penalty == "ridge" and lam > 0:
            res.explain("compare", ctx.t("labs.reg.ridge_note.title",
                                         "Why ridge selected nothing"), ctx.t(
                "labs.reg.ridge_note",
                "Ridge shrank every coefficient towards zero but set none of them exactly "
                "to zero: {k} of {p} are still non-zero. That is the geometric difference "
                "made numerical.", k=nonzero, p=int(p["p"]),
            ))
        if float(p["correlation"]) > 0.5 and penalty == "lasso":
            res.warnings.append(ctx.t(
                "labs.reg.warn.correlated",
                "With strongly correlated features the lasso tends to pick one from each "
                "correlated group almost arbitrarily. Which one it keeps can change with "
                "the seed, so the selected set is not a stable finding.",
            ))
        return res

    @staticmethod
    def _generate(p, seed):
        d = sparse_regression(n=int(p["n"]), p=int(p["p"]), n_active=int(p["n_active"]),
                              noise=float(p["noise"]),
                              correlation=float(p["correlation"]),
                              signal=float(p["signal"]), seed=seed)
        X = np.column_stack([d[f"x{j + 1}"] for j in range(int(p["p"]))])
        X = (X - X.mean(axis=0)) / np.clip(X.std(axis=0), 1e-9, None)
        y = d["y"] - d["y"].mean()
        return type(d)(columns={**d.columns, "y": y}, dgp=d.dgp, truth=d.truth,
                       notes=d.notes,
                       meta={**d.meta, "X": X, "y": y, "beta": d.meta["beta"]})

    @staticmethod
    def _ols(X, y):
        if X.shape[1] >= X.shape[0]:
            return None
        return np.linalg.pinv(X.T @ X) @ (X.T @ y)

    @staticmethod
    def _fit(X, y, lam, penalty):
        if lam <= 0:
            beta = np.linalg.pinv(X) @ y
            return beta
        if penalty == "ridge":
            return ridge_path(X, y, [lam * X.shape[0]])[0]
        return lasso_coordinate_descent(X, y, lam)

    def _path(self, X, y, penalty):
        points = 40 if X.shape[1] <= 60 else 18
        lambdas = np.geomspace(1e-3, 5.0, points)
        rows = []
        for lam in lambdas:
            rows.append(self._fit(X, y, float(lam), penalty))
        return np.asarray(rows), lambdas

    def _cv_curve(self, X, y, penalty, seed, folds: int = 5):
        n = X.shape[0]
        rng_ = np.random.default_rng(seed)
        idx = rng_.permutation(n)
        parts = np.array_split(idx, folds)
        lambdas = np.geomspace(1e-3, 3.0, 18 if X.shape[1] <= 60 else 10)
        rows = []
        for lam in lambdas:
            errs = []
            for f in range(folds):
                test = parts[f]
                train = np.concatenate([parts[g] for g in range(folds) if g != f])
                beta = self._fit(X[train], y[train], float(lam), penalty)
                errs.append(float(np.mean((y[test] - X[test] @ beta) ** 2)))
            rows.append({"lambda": float(lam), "mse": float(np.mean(errs)),
                         "se": float(np.std(errs, ddof=1) / np.sqrt(folds))})
        return rows

    def _geometry_figure(self, ctx, X, y, lam, penalty):
        go = P.require_plotly()
        X2 = X[:, :2]
        b_ols = np.linalg.pinv(X2.T @ X2) @ (X2.T @ y)
        span = max(float(np.max(np.abs(b_ols))) * 2.0, 1.0)
        g = np.linspace(-span, span, 140)
        B1, B2 = np.meshgrid(g, g)
        resid = (y[:, None, None] - B1[None] * X2[:, 0][:, None, None]
                 - B2[None] * X2[:, 1][:, None, None])
        sse = np.sum(resid**2, axis=0)
        fig = ctx.figure(
            "labs.reg.figure.geometry",
            xaxis_title="beta_1", yaxis_title="beta_2", height=450,
        )
        fig.add_trace(go.Contour(x=g, y=g, z=sse, colorscale=ctx.theme.colorscale,
                                 contours={"showlabels": False}, showscale=False,
                                 opacity=0.75,
                                 name=ctx.t("labs.reg.trace.sse",
                                            "residual sum of squares")))
        budget = float(np.sum(np.abs(self._fit(X2, y, lam, penalty)))) if lam > 0 else None
        if budget is None or budget < 1e-9:
            budget = float(np.sum(np.abs(b_ols))) * 0.6
        theta = np.linspace(0, 2 * np.pi, 300)
        if penalty == "ridge":
            r = float(np.linalg.norm(self._fit(X2, y, lam, penalty))) or budget
            P.add_curve(fig, r * np.cos(theta), r * np.sin(theta),
                        ctx.t("labs.reg.trace.ridge_ball",
                              "ridge budget: a smooth disc"),
                        "warning", theme=ctx.theme, width=3.0)
        else:
            t = np.linspace(0, 1, 100)
            corners = np.array([[budget, 0], [0, budget], [-budget, 0], [0, -budget],
                                [budget, 0]])
            xs, ys = [], []
            for i in range(4):
                xs.extend(corners[i, 0] + t * (corners[i + 1, 0] - corners[i, 0]))
                ys.extend(corners[i, 1] + t * (corners[i + 1, 1] - corners[i, 1]))
            P.add_curve(fig, xs, ys,
                        ctx.t("labs.reg.trace.lasso_ball",
                              "lasso budget: a diamond with corners on the axes"),
                        "warning", theme=ctx.theme, width=3.0)
        b_pen = self._fit(X2, y, lam, penalty)
        P.add_points(fig, [b_ols[0]], [b_ols[1]],
                     ctx.t("labs.reg.trace.ols", "least squares (no penalty)"),
                     "muted", theme=ctx.theme, size=12)
        P.add_points(fig, [b_pen[0]], [b_pen[1]],
                     ctx.t("labs.reg.trace.penalized", "penalized solution"),
                     "primary", theme=ctx.theme, size=13, symbol="star")
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        P.add_vline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_geometry",
            "The star is where a growing ellipse first touches the budget region. On a "
            "diamond that first contact is often a corner - and a corner means a "
            "coefficient of exactly zero. On a disc there are no corners.",
        ), theme=ctx.theme)
        return fig

    def _coefficients_figure(self, ctx, coefs, ols, beta_true, p):
        go = P.require_plotly()
        idx = np.arange(1, coefs.size + 1)
        fig = ctx.figure(
            "labs.reg.figure.coefficients",
            xaxis_title=ctx.t("labs.reg.axis.feature", "Feature"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=390,
        )
        fig.add_trace(go.Bar(x=idx, y=beta_true,
                             marker={"color": ctx.color("truth")},
                             name=ctx.t("labs.common.trace.truth"), opacity=0.55))
        if ols is not None:
            fig.add_trace(go.Bar(x=idx, y=ols, marker={"color": ctx.color("muted")},
                                 name=ctx.t("labs.reg.trace.ols_short", "least squares"),
                                 opacity=0.55))
        fig.add_trace(go.Bar(x=idx, y=coefs, marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.reg.trace.penalized_short",
                                        "{pen} at lambda = {l}",
                                        pen=str(p["penalty"]),
                                        l=fmt(p["lambda"], 3))))
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_coefficients",
            "Look for bars that vanish entirely. The lasso removes features; ridge only "
            "shortens them.",
        ), theme=ctx.theme)
        return fig

    def _path_figure(self, ctx, path, lambdas, beta_true, current, penalty):
        fig = ctx.figure(
            "labs.reg.figure.path",
            xaxis_title=ctx.t("labs.common.axis.penalty"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=390,
        )
        active = np.abs(beta_true) > 1e-8
        for j in range(path.shape[1]):
            P.add_curve(fig, lambdas, path[:, j],
                        (ctx.t("labs.reg.trace.active", "truly relevant features")
                         if active[j] else
                         ctx.t("labs.reg.trace.irrelevant", "irrelevant features")),
                        "positive" if active[j] else "muted", theme=ctx.theme,
                        width=2.4 if active[j] else 1.0,
                        opacity=1.0 if active[j] else 0.4,
                        showlegend=bool((j == int(np.argmax(active)) and active[j])
                                        or (j == int(np.argmax(~active))
                                            and not active[j])))
        P.add_vline(fig, current, ctx.t("labs.common.trace.current"), "highlight",
                    theme=ctx.theme)
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme, dash="dot")
        fig.update_xaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_path",
            "Reading right to left is variable selection in motion. A good penalty is one "
            "where the grey paths have collapsed to zero but the green ones have not.",
        ), theme=ctx.theme)
        return fig

    def _cv_figure(self, ctx, cv, current):
        lambdas = [row["lambda"] for row in cv]
        mses = [row["mse"] for row in cv]
        ses = [row["se"] for row in cv]
        best = min(cv, key=lambda row: row["mse"])
        threshold = best["mse"] + best["se"]
        one_se = next((row["lambda"] for row in reversed(cv)
                       if row["mse"] <= threshold), best["lambda"])
        fig = ctx.figure(
            "labs.reg.figure.cv",
            xaxis_title=ctx.t("labs.common.axis.penalty"),
            yaxis_title=ctx.t("labs.reg.axis.cv", "Cross-validated mean squared error"),
            height=360,
        )
        P.shade_between(fig, lambdas,
                        np.asarray(mses) - np.asarray(ses),
                        np.asarray(mses) + np.asarray(ses),
                        ctx.t("labs.reg.trace.cv_band", "+/- 1 standard error"),
                        "info", theme=ctx.theme, alpha=0.18)
        P.add_curve(fig, lambdas, mses,
                    ctx.t("labs.reg.trace.cv", "cross-validated error"),
                    "primary", theme=ctx.theme)
        P.add_vline(fig, best["lambda"],
                    ctx.t("labs.reg.trace.cv_min", "minimum"), "positive",
                    theme=ctx.theme)
        P.add_vline(fig, one_se,
                    ctx.t("labs.reg.trace.one_se", "one-standard-error rule"),
                    "warning", theme=ctx.theme, dash="dot")
        P.add_vline(fig, current, ctx.t("labs.common.trace.current"), "highlight",
                    theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.reg.legend_cv",
            "The one-standard-error rule deliberately picks a stronger penalty than the "
            "minimum: it buys a simpler model for an error that is statistically "
            "indistinguishable.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, X, y, beta_true, penalty):
        go = P.require_plotly()
        lambdas = np.geomspace(0.001, 3.0, 20)
        idx = np.arange(1, beta_true.size + 1)
        frames, steps = [], []
        for i, lam in enumerate(lambdas):
            coefs = self._fit(X, y, float(lam), penalty)
            nonzero = int(np.sum(np.abs(coefs) > 1e-8))
            frames.append(go.Frame(name=f"{lam:.4f}", data=[go.Bar(x=idx, y=coefs)]))
            steps.append(AnimationStep(
                id=f"lam_{i}", frame=i,
                title=ctx.t("labs.reg.anim.title", "lambda = {l}", l=fmt(lam, 4)),
                what_you_see=ctx.t("labs.reg.anim.see",
                                   "All {p} estimated coefficients at this penalty level.",
                                   p=int(beta_true.size)),
                what_changed=ctx.t("labs.reg.anim.changed",
                                   "The penalty strength moved to {l}.", l=fmt(lam, 4)),
                why=ctx.t("labs.reg.anim.why",
                          "A larger penalty shrinks the budget the coefficients must fit "
                          "inside, so every coefficient is pushed towards zero - and with "
                          "an L1 budget, some arrive there exactly."),
                interpretation=ctx.t("labs.reg.anim.interpret",
                                     "{k} coefficients are non-zero; the squared "
                                     "coefficient error is {e}.",
                                     k=nonzero,
                                     e=fmt(float(np.sum((coefs - beta_true) ** 2)), 4)),
                conclusion=ctx.t("labs.reg.anim.conclude",
                                 "There is a sweet spot: enough shrinkage to kill the noise "
                                 "features, not so much that the real signal dies too."),
                warning=ctx.t("labs.reg.anim.warn",
                              "Every coefficient here is biased towards zero on purpose. "
                              "That is the deal regularization offers, and it is why these "
                              "numbers are not effect estimates."),
                math=("ridge: b/(1 + lambda);  lasso: sign(b) * max(|b| - lambda, 0)"),
                outputs={"lambda": round(float(lam), 5), "nonzero": nonzero,
                         "coefficient_error":
                             round(float(np.sum((coefs - beta_true) ** 2)), 5)},
                highlighted=("coefficients",),
            ))
        fig = ctx.figure(
            "labs.reg.figure.animation",
            xaxis_title=ctx.t("labs.reg.axis.feature", "Feature"),
            yaxis_title=ctx.t("labs.common.axis.coefficient"),
            height=390,
        )
        fig.add_trace(go.Bar(x=idx, y=np.zeros_like(idx, dtype=float),
                             marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.reg.trace.penalized_short2",
                                        "penalized coefficients")))
        fig.add_trace(go.Scatter(x=idx, y=beta_true, mode="markers",
                                 marker={"color": ctx.color("truth"), "size": 8,
                                         "symbol": "star"},
                                 name=ctx.t("labs.common.trace.truth")))
        span = float(np.max(np.abs(beta_true))) * 1.6 + 0.5
        fig.update_yaxes(range=[-span, span])
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label="lambda")
        return animation(
            "shrinkage_sweep", fig, steps,
            purpose=ctx.t("labs.reg.anim.purpose",
                          "Watch a penalty select variables one at a time."),
            summary=ctx.t(
                "labs.reg.anim.summary",
                "Regularization buys a large reduction in variance for a small amount of "
                "bias. The lasso does it while also deleting features, and that deletion "
                "comes from the corners of the L1 ball rather than from any rule applied "
                "afterwards."),
            evidence=EvidenceType.GEOMETRIC_PROOF,
        )


LAB = RegularizationLab(SPEC)
