"""Bias-variance trade-off, decomposed exactly because the truth is known."""

from __future__ import annotations

from typing import Any

from ...data.generators.ml import nonlinear_target
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

TARGETS = ("sine", "polynomial", "step", "linear")


def _design(x, degree, ridge=0.0):
    return np.column_stack([np.asarray(x, dtype=float) ** k for k in range(degree + 1)])


def _fit_poly(x, y, degree, ridge=0.0):
    X = _design(x, degree)
    if ridge > 0:
        penalty = ridge * np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    else:
        beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return beta


SPEC = make_spec(
    "ml.bias_variance",
    Domain.ML,
    "generalization",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.SIMULATION,
    controls=(
        int_slider("degree", 3, 0, 15, 1, group="model"),
        slider("ridge", 0.0, 0.0, 5.0, 0.01, group="model"),
        int_slider("n", 40, 5, 500, 1, group="data"),
        slider("noise", 0.3, 0.0, 2.0, 0.01, group="data"),
        select("target", "sine", TARGETS, group="data"),
        int_slider("datasets", 60, 5, 500, 5, group="simulation", expensive=True),
        int_slider("max_degree", 12, 2, 20, 1, group="views"),
        toggle("show_fits", True, group="views"),
        toggle("show_learning_curve", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", degree=3, n=40, noise=0.3),
        scenario("underfitting", "misspecification", degree=0, n=40),
        scenario("well_fitted", "canonical", degree=3, n=40),
        scenario("overfitting", "misspecification", degree=13, n=25),
        scenario("interpolation", "boundary", degree=15, n=18, noise=0.3),
        scenario("no_noise", "low_noise", noise=0.0, degree=9),
        scenario("high_noise", "high_noise", noise=1.2),
        scenario("small_sample", "small_sample", n=12, degree=6),
        scenario("large_sample", "large_sample", n=400, degree=9),
        scenario("regularized", "robustness", degree=13, ridge=0.5, n=25),
        scenario("step_target", "compare_methods", target="step", degree=9),
        scenario("linear_truth", "null", target="linear", degree=1),
    ),
    related=("ml.regularization", "ml.cross_validation"),
    next_concepts=("ml.regularization",),
    tags=("bias", "variance", "overfitting", "underfitting", "generalization",
          "learning curve", "model complexity"),
    aliases=("bias variance tradeoff", "biais variance",
             "المفاضلة بين التحيز والتباين", "overfitting", "generalization"),
    backends=("numpy", "scikit-learn"),
    references=(
        ref("Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of "
            "Statistical Learning.", kind="book"),
        ref("Stanford STATS 315A syllabus", kind="course",
            url="https://web.stanford.edu/class/stats315a/syllabus.html"),
    ),
    curriculum_tags=("stanford.stats315a",),
)


class BiasVarianceLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        degree = int(p["degree"])
        n = int(p["n"])
        datasets = int(p["datasets"])
        grid = np.linspace(0.0, 1.0, 120)
        truth = self._truth(grid, str(p["target"]))

        fits = self._many_fits(p, n, datasets, state.seed, grid, degree)
        decomposition = self._decompose(fits, truth, float(p["noise"]))
        curve = self._complexity_curve(p, n, datasets, state.seed, grid, truth,
                                       int(p["max_degree"]))

        res.dgp = ctx.t(
            "labs.bv.dgp",
            "{k} independent training samples of size n = {n} from f(x) = {target} with "
            "noise sd {s}; a degree-{d} polynomial is fitted to each.",
            k=datasets, n=n, target=str(p["target"]), s=fmt(p["noise"], 2), d=degree,
        )

        if p["show_fits"]:
            res.add_panel(ctx.panel(
                "fits", self._fits_figure(ctx, grid, truth, fits, p),
                "labs.bv.figure.fits", evidence=EvidenceType.SIMULATION,
            ))
        res.add_panel(ctx.panel(
            "decomposition", self._decomposition_figure(ctx, grid, truth, fits,
                                                        decomposition),
            "labs.bv.figure.decomposition", evidence=EvidenceType.SIMULATION,
        ))
        res.add_panel(ctx.panel(
            "complexity", self._complexity_figure(ctx, curve, degree),
            "labs.bv.figure.complexity", tab="compare",
            evidence=EvidenceType.SIMULATION,
        ))
        if p["show_learning_curve"]:
            res.add_panel(ctx.panel(
                "learning", self._learning_figure(ctx, p, state.seed, grid, truth),
                "labs.bv.figure.learning", tab="diagnostics",
                evidence=EvidenceType.SIMULATION,
            ))

        res.metric("bias_squared", ctx.t("labs.bv.metric.bias", "Squared bias"),
                   decomposition["bias2"])
        res.metric("variance", ctx.t("labs.bv.metric.variance", "Variance"),
                   decomposition["variance"])
        res.metric("noise", ctx.t("labs.bv.metric.noise",
                                  "Irreducible noise"), decomposition["noise"],
                   note=ctx.t("labs.bv.metric.noise_note",
                              "no model can reduce this term"))
        res.metric("expected_error", ctx.t("labs.bv.metric.total",
                                           "Expected test error"),
                   decomposition["total"],
                   note=ctx.t("labs.bv.metric.total_note",
                              "bias squared + variance + noise"))
        res.metric("train_error", ctx.t("labs.bv.metric.train", "Average training error"),
                   decomposition["train"],
                   note=ctx.t("labs.bv.metric.train_note",
                              "falls monotonically with complexity - which is why it "
                              "cannot be used to choose a model"))
        best = min(curve, key=lambda row: row["total"])
        res.metric("best_degree", ctx.t("labs.bv.metric.best",
                                        "Complexity minimising expected test error"),
                   int(best["degree"]),
                   note=ctx.t("labs.bv.metric.best_note",
                              "at the current n and noise level"))
        res.metric("effective_parameters", ctx.t("labs.bv.metric.params",
                                                 "Fitted parameters"), degree + 1,
                   note=ctx.t("labs.bv.metric.params_note",
                              "out of {n} observations", n=n))

        interpolating = degree + 1 >= n
        res.assume("enough_data", ctx.t("labs.bv.assume.data_label",
                                        "More observations than parameters"),
                   not interpolating,
                   detail=ctx.t("labs.bv.assume.data",
                                "A degree-{d} polynomial has {k} parameters against {n} "
                                "observations.", d=degree, k=degree + 1, n=n),
                   consequence="" if not interpolating else ctx.t(
                       "labs.bv.assume.interpolate",
                       "The model can pass exactly through every training point, driving "
                       "the training error to zero while learning nothing generalizable."))
        res.assume("known_truth", ctx.t("labs.bv.assume.truth_label",
                                        "The true function is known"), True,
                   detail=ctx.t("labs.bv.assume.truth",
                                "The decomposition below requires f(x). On real data you "
                                "can estimate the total error but never split it into "
                                "these three parts."))

        res.animations.append(self._animation(ctx, p, n, datasets, state.seed, grid, truth))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.ml.bias_variance.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.ml.bias_variance.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.ml.bias_variance.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.ml.bias_variance.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.ml.bias_variance.warning"), kind="warning")

        if interpolating:
            res.warnings.append(ctx.t(
                "labs.bv.warn.interpolation",
                "With {k} parameters and {n} observations the model interpolates the "
                "training data exactly. Training error {tr} against an expected test error "
                "of {te} - the gap between those two numbers IS overfitting.",
                k=degree + 1, n=n, tr=fmt(decomposition["train"], 4),
                te=fmt(decomposition["total"], 4),
            ))
        return res

    @staticmethod
    def _truth(x, kind):
        if kind == "sine":
            return np.sin(2 * np.pi * x)
        if kind == "polynomial":
            return 1.5 * x**3 - 2.0 * x**2 + 0.5 * x
        if kind == "step":
            return np.where(x > 0.5, 1.0, -1.0)
        return x

    def _many_fits(self, p, n, datasets, seed, grid, degree):
        preds = np.empty((datasets, grid.size))
        train_err = np.empty(datasets)
        for k in range(datasets):
            d = nonlinear_target(n=n, noise=float(p["noise"]),
                                 seed=int(seed) * 65537 + k, kind=str(p["target"]))
            beta = _fit_poly(d["x"], d["y"], degree, float(p["ridge"]))
            preds[k] = _design(grid, degree) @ beta
            train_err[k] = float(np.mean((d["y"] - _design(d["x"], degree) @ beta) ** 2))
        return {"preds": preds, "train_error": train_err}

    @staticmethod
    def _decompose(fits, truth, noise):
        preds = fits["preds"]
        mean_pred = preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - truth) ** 2))
        variance = float(np.mean(np.var(preds, axis=0, ddof=1)))
        noise_term = float(noise**2)
        return {"bias2": bias2, "variance": variance, "noise": noise_term,
                "total": bias2 + variance + noise_term,
                "train": float(np.mean(fits["train_error"])),
                "mean_pred": mean_pred}

    def _complexity_curve(self, p, n, datasets, seed, grid, truth, max_degree):
        rows = []
        sub = min(datasets, 40)
        for d in range(max_degree + 1):
            fits = self._many_fits(p, n, sub, seed, grid, d)
            dec = self._decompose(fits, truth, float(p["noise"]))
            rows.append({"degree": d, "bias2": dec["bias2"], "variance": dec["variance"],
                         "noise": dec["noise"], "total": dec["total"],
                         "train": dec["train"]})
        return rows

    def _fits_figure(self, ctx, grid, truth, fits, p):
        fig = ctx.figure(
            "labs.bv.figure.fits",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=430,
        )
        preds = fits["preds"]
        for k in range(min(preds.shape[0], 30)):
            P.add_curve(fig, grid, preds[k],
                        ctx.t("labs.bv.trace.one_fit", "one training sample's fit"),
                        "muted", theme=ctx.theme, width=1.0, opacity=0.35,
                        showlegend=k == 0)
        P.add_curve(fig, grid, preds.mean(axis=0),
                    ctx.t("labs.bv.trace.average", "average fit across samples"),
                    "primary", theme=ctx.theme, width=3.2)
        P.add_curve(fig, grid, truth, ctx.t("labs.common.trace.true_function"),
                    "truth", theme=ctx.theme, width=3.0, dash="dash")
        span = float(np.percentile(np.abs(truth), 95)) * 3 + 1
        fig.update_yaxes(range=[-span, span])
        P.add_legend_note(fig, ctx.t(
            "labs.bv.legend_fits",
            "Distance from the thick blue line to the dashed black one is BIAS. Spread of "
            "the grey lines around the blue one is VARIANCE. Both are visible at once here "
            "and never on real data.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _decomposition_figure(self, ctx, grid, truth, fits, dec):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        preds = fits["preds"]
        pointwise_bias2 = (preds.mean(axis=0) - truth) ** 2
        pointwise_var = np.var(preds, axis=0, ddof=1)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], subplot_titles=(
            ctx.t("labs.bv.trace.pointwise", "where the error lives"),
            ctx.t("labs.bv.trace.totals", "expected error decomposition"),
        ))
        fig.add_trace(go.Scatter(x=grid, y=pointwise_bias2, mode="lines",
                                 line={"color": ctx.color("negative"), "width": 2.4},
                                 name=ctx.t("labs.bv.metric.bias", "Squared bias")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=grid, y=pointwise_var, mode="lines",
                                 line={"color": ctx.color("secondary"), "width": 2.4},
                                 name=ctx.t("labs.bv.metric.variance", "Variance")),
                      row=1, col=1)
        fig.add_trace(go.Bar(
            x=[ctx.t("labs.bv.metric.bias", "Squared bias"),
               ctx.t("labs.bv.metric.variance", "Variance"),
               ctx.t("labs.bv.metric.noise", "Irreducible noise")],
            y=[dec["bias2"], dec["variance"], dec["noise"]],
            marker={"color": [ctx.color("negative"), ctx.color("secondary"),
                              ctx.color("muted")]},
            text=[fmt(dec["bias2"], 4), fmt(dec["variance"], 4), fmt(dec["noise"], 4)],
            textposition="outside", showlegend=False), row=1, col=2)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=380, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.bv.legend_decomposition",
            "Variance is usually worst at the edges of the training range - which is "
            "exactly where flexible models are asked to extrapolate.",
        ), theme=ctx.theme)
        return fig

    def _complexity_figure(self, ctx, curve, current):
        degrees = [row["degree"] for row in curve]
        fig = ctx.figure(
            "labs.bv.figure.complexity",
            xaxis_title=ctx.t("labs.common.axis.complexity"),
            yaxis_title=ctx.t("labs.common.axis.error"),
            height=390,
        )
        P.add_curve(fig, degrees, [row["bias2"] for row in curve],
                    ctx.t("labs.bv.metric.bias", "Squared bias"), "negative",
                    theme=ctx.theme)
        P.add_curve(fig, degrees, [row["variance"] for row in curve],
                    ctx.t("labs.bv.metric.variance", "Variance"), "secondary",
                    theme=ctx.theme)
        P.add_curve(fig, degrees, [row["total"] for row in curve],
                    ctx.t("labs.bv.metric.total", "Expected test error"), "primary",
                    theme=ctx.theme, width=3.2)
        P.add_curve(fig, degrees, [row["train"] for row in curve],
                    ctx.t("labs.bv.metric.train", "Average training error"), "muted",
                    theme=ctx.theme, dash="dot")
        P.add_vline(fig, current, ctx.t("labs.common.trace.current"), "highlight",
                    theme=ctx.theme)
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.bv.legend_complexity",
            "The U-shape is the trade-off. The dotted training curve only ever goes down, "
            "which is why the point where it flattens tells you nothing about where to stop.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _learning_figure(self, ctx, p, seed, grid, truth):
        sizes = np.unique(np.round(np.geomspace(6, 400, 12)).astype(int))
        totals, biases, variances = [], [], []
        for ns in sizes:
            if int(p["degree"]) + 1 >= ns:
                totals.append(np.nan)
                biases.append(np.nan)
                variances.append(np.nan)
                continue
            fits = self._many_fits(p, int(ns), 30, seed, grid, int(p["degree"]))
            dec = self._decompose(fits, truth, float(p["noise"]))
            totals.append(dec["total"])
            biases.append(dec["bias2"])
            variances.append(dec["variance"])
        fig = ctx.figure(
            "labs.bv.figure.learning",
            xaxis_title=ctx.t("labs.common.axis.sample_size"),
            yaxis_title=ctx.t("labs.common.axis.error"),
            height=360,
        )
        P.add_curve(fig, sizes, totals,
                    ctx.t("labs.bv.metric.total", "Expected test error"),
                    "primary", theme=ctx.theme)
        P.add_curve(fig, sizes, biases, ctx.t("labs.bv.metric.bias", "Squared bias"),
                    "negative", theme=ctx.theme, dash="dash")
        P.add_curve(fig, sizes, variances, ctx.t("labs.bv.metric.variance", "Variance"),
                    "secondary", theme=ctx.theme, dash="dot")
        P.add_hline(fig, float(p["noise"]) ** 2,
                    ctx.t("labs.bv.metric.noise", "Irreducible noise"),
                    "truth", theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.bv.legend_learning",
            "More data kills variance and leaves bias untouched. A learning curve that "
            "flattens well above the noise floor is telling you the model is too rigid, not "
            "that you need more data.",
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, datasets, seed, grid, truth):
        go = P.require_plotly()
        degrees = list(range(0, min(int(p["max_degree"]), n - 2) + 1))
        frames, steps = [], []
        sub = min(datasets, 25)
        for i, d in enumerate(degrees):
            fits = self._many_fits(p, n, sub, seed, grid, d)
            dec = self._decompose(fits, truth, float(p["noise"]))
            data = [go.Scatter(x=grid, y=fits["preds"][k]) for k in range(min(sub, 12))]
            data.append(go.Scatter(x=grid, y=fits["preds"].mean(axis=0)))
            frames.append(go.Frame(name=str(d), data=data))
            steps.append(AnimationStep(
                id=f"deg_{d}", frame=i,
                title=ctx.t("labs.bv.anim.title", "polynomial degree {d}", d=d),
                what_you_see=ctx.t("labs.bv.anim.see",
                                   "Fits from {k} independent training samples, plus their "
                                   "average.", k=min(sub, 12)),
                what_changed=ctx.t("labs.bv.anim.changed",
                                   "Model complexity moved to degree {d} ({k} parameters).",
                                   d=d, k=d + 1),
                why=ctx.t("labs.bv.anim.why",
                          "A more flexible model can track the true function more closely "
                          "but also tracks the noise in whichever sample it happened to see."),
                interpretation=ctx.t("labs.bv.anim.interpret",
                                     "Squared bias {b}, variance {v}, expected test error {t}.",
                                     b=fmt(dec["bias2"], 4), v=fmt(dec["variance"], 4),
                                     t=fmt(dec["total"], 4)),
                conclusion=ctx.t("labs.bv.anim.conclude",
                                 "The curves converge on the truth and then start fanning "
                                 "apart. Where they fan is where you have gone too far."),
                warning=ctx.t("labs.bv.anim.warn",
                              "Training error is still falling in every frame. It will keep "
                              "falling all the way to zero."),
                math="E[(y - f_hat)^2] = Bias^2 + Var + sigma^2",
                outputs={"degree": d, "bias_squared": round(dec["bias2"], 5),
                         "variance": round(dec["variance"], 5),
                         "test_error": round(dec["total"], 5),
                         "train_error": round(dec["train"], 5)},
                violated_assumptions=("enough_data",) if d + 1 >= n else (),
                highlighted=("fits", "average_fit"),
            ))
        fig = ctx.figure(
            "labs.bv.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.x"),
            yaxis_title=ctx.t("labs.common.axis.y"),
            height=400,
        )
        for k in range(12):
            fig.add_trace(go.Scatter(x=grid, y=np.zeros_like(grid), mode="lines",
                                     line={"color": ctx.color("muted"), "width": 1.0},
                                     opacity=0.4, showlegend=k == 0,
                                     name=ctx.t("labs.bv.trace.one_fit",
                                                "one training sample's fit")))
        fig.add_trace(go.Scatter(x=grid, y=np.zeros_like(grid), mode="lines",
                                 line={"color": ctx.color("primary"), "width": 3.2},
                                 name=ctx.t("labs.bv.trace.average",
                                            "average fit across samples")))
        P.add_curve(fig, grid, truth, ctx.t("labs.common.trace.true_function"),
                    "truth", theme=ctx.theme, dash="dash", width=3.0)
        span = float(np.percentile(np.abs(truth), 95)) * 3 + 1
        fig.update_yaxes(range=[-span, span])
        build_frames(fig, frames, duration=520, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.bv.slider", "degree"))
        return animation(
            "complexity_sweep", fig, steps,
            purpose=ctx.t("labs.bv.anim.purpose",
                          "Watch bias turn into variance as flexibility increases."),
            summary=ctx.t(
                "labs.bv.anim.summary",
                "Bias and variance are two ways of being wrong, and buying less of one "
                "costs more of the other. The lowest total sits somewhere in between - and "
                "on real data, where the truth is unknown, cross-validation is how you look "
                "for it."),
            evidence=EvidenceType.SIMULATION,
        )


LAB = BiasVarianceLab(SPEC)
