"""Gradient descent: the step, the rate, and the ways it fails."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "SURFACES", "descend"]

SURFACES = ("quadratic", "ill_conditioned", "rosenbrock", "double_well", "plateau",
            "saddle")


def _surface(kind: str, condition: float):
    """Return (loss, gradient, optimum, plot window) for a named surface."""
    if kind == "quadratic":
        def loss(w):
            return 0.5 * (w[0] ** 2 + w[1] ** 2)

        def grad(w):
            return np.array([w[0], w[1]])

        return loss, grad, np.zeros(2), (-3.5, 3.5)
    if kind == "ill_conditioned":
        c = max(condition, 1.0)

        def loss(w):
            return 0.5 * (c * w[0] ** 2 + w[1] ** 2)

        def grad(w):
            return np.array([c * w[0], w[1]])

        return loss, grad, np.zeros(2), (-3.5, 3.5)
    if kind == "rosenbrock":
        def loss(w):
            return (1 - w[0]) ** 2 + 20.0 * (w[1] - w[0] ** 2) ** 2

        def grad(w):
            return np.array([
                -2 * (1 - w[0]) - 80.0 * w[0] * (w[1] - w[0] ** 2),
                40.0 * (w[1] - w[0] ** 2),
            ])

        return loss, grad, np.array([1.0, 1.0]), (-2.0, 2.5)
    if kind == "double_well":
        def loss(w):
            return 0.25 * (w[0] ** 2 - 4) ** 2 + 0.5 * w[1] ** 2 + 0.6 * w[0]

        def grad(w):
            return np.array([w[0] * (w[0] ** 2 - 4) + 0.6, w[1]])

        return loss, grad, np.array([-2.07, 0.0]), (-3.5, 3.5)
    if kind == "plateau":
        def loss(w):
            return np.tanh(0.6 * w[0]) ** 2 + np.tanh(0.6 * w[1]) ** 2

        def grad(w):
            return np.array([
                2 * np.tanh(0.6 * w[0]) * 0.6 * (1 - np.tanh(0.6 * w[0]) ** 2),
                2 * np.tanh(0.6 * w[1]) * 0.6 * (1 - np.tanh(0.6 * w[1]) ** 2),
            ])

        return loss, grad, np.zeros(2), (-6.0, 6.0)

    def loss(w):
        return w[0] ** 2 - w[1] ** 2

    def grad(w):
        return np.array([2 * w[0], -2 * w[1]])

    return loss, grad, np.zeros(2), (-3.0, 3.0)


def descend(kind, condition, start, lr, steps, momentum=0.0, noise=0.0, seed=0):
    """Run gradient descent and return the path, losses and gradient norms."""
    loss, grad, optimum, _ = _surface(kind, condition)
    gen = rng(seed, "gd")
    w = np.asarray(start, dtype=float).copy()
    velocity = np.zeros_like(w)
    path = [w.copy()]
    losses = [float(loss(w))]
    grads = [float(np.linalg.norm(grad(w)))]
    diverged = False
    for _ in range(int(steps)):
        g = grad(w)
        if noise > 0:
            g = g + noise * gen.standard_normal(w.size)
        velocity = momentum * velocity - lr * g
        w = w + velocity
        if not np.all(np.isfinite(w)) or float(np.max(np.abs(w))) > 1e6:
            diverged = True
            break
        path.append(w.copy())
        losses.append(float(loss(w)))
        grads.append(float(np.linalg.norm(grad(w))))
    return {"path": np.asarray(path), "losses": np.asarray(losses),
            "grad_norms": np.asarray(grads), "diverged": diverged,
            "optimum": optimum, "loss": loss, "grad": grad}


SPEC = make_spec(
    "ml.gradient_descent",
    Domain.ML,
    "optimization",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive",
           "code", "quiz", "references"),
    evidence=EvidenceType.VISUAL_DERIVATION,
    controls=(
        select("surface", "quadratic", SURFACES, group="problem"),
        slider("learning_rate", 0.2, 0.001, 2.5, 0.001, group="optimizer"),
        slider("momentum", 0.0, 0.0, 0.99, 0.01, group="optimizer"),
        slider("noise", 0.0, 0.0, 2.0, 0.01, group="optimizer"),
        int_slider("steps", 40, 1, 500, 1, group="optimizer"),
        slider("start_x", -2.5, -6.0, 6.0, 0.05, group="problem"),
        slider("start_y", 2.0, -6.0, 6.0, 0.05, group="problem"),
        slider("condition", 8.0, 1.0, 60.0, 0.5, group="problem",
               depends_on=("surface", ("ill_conditioned",))),
        toggle("show_loss_curve", True, group="views"),
        toggle("show_rate_sweep", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", surface="quadratic", learning_rate=0.2),
        scenario("too_small", "weak", learning_rate=0.01, steps=40),
        scenario("well_tuned", "canonical", learning_rate=0.6),
        scenario("oscillating", "boundary", learning_rate=1.85),
        scenario("diverging", "counterexample", learning_rate=2.2),
        scenario("ill_conditioned", "misspecification", surface="ill_conditioned",
                 condition=25.0, learning_rate=0.07, steps=120),
        scenario("momentum_helps", "robustness", surface="ill_conditioned",
                 condition=25.0, learning_rate=0.07, momentum=0.9, steps=120),
        scenario("non_convex", "violation", surface="double_well", start_x=2.5,
                 learning_rate=0.05, steps=80),
        scenario("escapes_with_noise", "robustness", surface="double_well", start_x=2.5,
                 learning_rate=0.05, noise=1.2, steps=200),
        scenario("plateau", "boundary", surface="plateau", start_x=5.0, start_y=4.0,
                 learning_rate=0.5, steps=200),
        scenario("saddle", "boundary", surface="saddle", start_x=1.5, start_y=0.0,
                 learning_rate=0.1, steps=60),
        scenario("saddle_escapes", "compare_methods", surface="saddle", start_x=1.5,
                 start_y=0.02, learning_rate=0.1, steps=60),
        scenario("rosenbrock", "boundary", surface="rosenbrock", start_x=-1.2,
                 start_y=1.0, learning_rate=0.004, steps=400),
        scenario("stochastic", "compare_methods", noise=0.8, steps=150),
    ),
    related=("deep_learning.backpropagation", "ml.regularization"),
    next_concepts=("deep_learning.backpropagation",),
    tags=("gradient descent", "learning rate", "momentum", "convergence", "sgd",
          "condition number", "saddle point"),
    aliases=("gradient descent", "descente de gradient", "الانحدار التدريجي",
             "learning rate", "optimization"),
    backends=("numpy",),
    references=(
        ref("Boyd, S. and Vandenberghe, L. (2004). Convex Optimization.", kind="book"),
        ref("Goodfellow, I., Bengio, Y. and Courville, A. (2016). Deep Learning.",
            kind="book"),
    ),
)


class GradientDescentLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        kind = str(p["surface"])
        lr = float(p["learning_rate"])
        start = np.array([float(p["start_x"]), float(p["start_y"])])
        run = descend(kind, float(p["condition"]), start, lr, int(p["steps"]),
                      float(p["momentum"]), float(p["noise"]), state.seed)
        loss, grad, optimum, window = _surface(kind, float(p["condition"]))

        res.dgp = ctx.t(
            "labs.gd.dgp",
            "Minimising the {s} surface from ({x}, {y}) with learning rate {lr}"
            "{mom}{noi}, for {n} steps.",
            s=kind, x=fmt(start[0], 2), y=fmt(start[1], 2), lr=fmt(lr, 3),
            mom=(f", momentum {fmt(p['momentum'], 2)}" if float(p["momentum"]) else ""),
            noi=(f", gradient noise {fmt(p['noise'], 2)}" if float(p["noise"]) else ""),
            n=int(p["steps"]),
        )

        res.add_panel(ctx.panel(
            "surface", self._surface_figure(ctx, run, loss, grad, window, optimum),
            "labs.gd.figure.surface", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        if p["show_loss_curve"]:
            res.add_panel(ctx.panel(
                "loss", self._loss_figure(ctx, run),
                "labs.gd.figure.loss", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_rate_sweep"]:
            res.add_panel(ctx.panel(
                "rates", self._rate_figure(ctx, p, start, state.seed),
                "labs.gd.figure.rates", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        final = run["path"][-1]
        res.metric("final_loss", ctx.t("labs.gd.metric.loss", "Final loss"),
                   float(run["losses"][-1]))
        res.metric("initial_loss", ctx.t("labs.gd.metric.initial", "Initial loss"),
                   float(run["losses"][0]))
        res.metric("distance_to_optimum", ctx.t("labs.gd.metric.distance",
                                                "Distance to the nearest known optimum"),
                   float(np.linalg.norm(final - optimum)))
        res.metric("final_gradient", ctx.t("labs.gd.metric.grad",
                                           "Gradient norm at the final point"),
                   float(run["grad_norms"][-1]),
                   note=ctx.t("labs.gd.metric.grad_note",
                              "near zero at minima, saddles and plateaus alike"))
        res.metric("steps_taken", ctx.t("labs.gd.metric.steps", "Steps taken"),
                   int(run["path"].shape[0] - 1))
        res.metric("diverged", ctx.t("labs.gd.metric.diverged", "Diverged"),
                   ctx.t("labs.gd.yes", "yes") if run["diverged"]
                   else ctx.t("labs.gd.no", "no"))
        stability = self._stability_limit(kind, float(p["condition"]))
        if np.isfinite(stability):
            res.metric("stability_limit", ctx.t("labs.gd.metric.limit",
                                                "Largest stable learning rate"),
                       stability,
                       note=ctx.t("labs.gd.metric.limit_note",
                                  "2 divided by the largest curvature"))
        if kind == "ill_conditioned":
            res.metric("condition_number", ctx.t("labs.gd.metric.condition",
                                                 "Condition number"),
                       float(p["condition"]),
                       note=ctx.t("labs.gd.metric.condition_note",
                                  "convergence slows in proportion to this"))

        convex = kind in ("quadratic", "ill_conditioned")
        stable = not run["diverged"] and (not np.isfinite(stability) or lr < stability)
        res.assume("convexity", ctx.t("labs.gd.assume.convex_label",
                                      "The objective is convex"), convex,
                   detail=ctx.t("labs.gd.assume.convex",
                                "On a convex surface any local minimum is global. On a "
                                "non-convex one, where you start decides where you land."))
        res.assume("stable_step", ctx.t("labs.gd.assume.stable_label",
                                        "The learning rate is inside the stable range"),
                   stable,
                   detail=ctx.t("labs.gd.assume.stable",
                                "For a quadratic with largest curvature L, gradient descent "
                                "converges only when the learning rate is below 2/L."),
                   consequence="" if stable else ctx.t(
                       "labs.gd.assume.diverge",
                       "Above that limit each step overshoots by more than the last, and "
                       "the loss increases without bound."))
        res.assume("gradient_informative",
                   ctx.t("labs.gd.assume.gradient_label",
                         "A vanishing gradient means a minimum"),
                   kind not in ("saddle", "plateau"),
                   detail=ctx.t("labs.gd.assume.gradient",
                                "Saddle points and flat plateaus also have near-zero "
                                "gradients. A small gradient is not evidence of success."))

        res.animations.append(self._animation(ctx, run, loss, window, optimum, p))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.ml.gradient_descent.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.ml.gradient_descent.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.ml.gradient_descent.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.ml.gradient_descent.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.ml.gradient_descent.warning"), kind="warning")

        growing = float(run["losses"][-1]) > float(run["losses"][0]) * 1.5
        if run["diverged"] or (not stable) or growing:
            res.warnings.append(ctx.t(
                "labs.gd.warn.diverged",
                "The loss went from {a} to {b}. With a learning rate of {lr} against a "
                "stability limit of {lim}, every step overshoots further than the last, so "
                "the iterates move away from the minimum rather than towards it.",
                a=fmt(float(run["losses"][0]), 4), b=fmt(float(run["losses"][-1]), 4),
                lr=fmt(lr, 3), lim=fmt(stability, 3),
            ))
        if kind == "saddle" and float(run["grad_norms"][-1]) < 0.05:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.gd.counterexample",
                "The gradient norm is {g}, which looks like convergence - but this is a "
                "saddle point, not a minimum. Move away in the second coordinate and the "
                "loss falls without limit.",
                g=fmt(float(run["grad_norms"][-1]), 5),
            ), kind="warning")
        return res

    @staticmethod
    def _stability_limit(kind, condition):
        if kind == "quadratic":
            return 2.0
        if kind == "ill_conditioned":
            return 2.0 / max(condition, 1e-9)
        return float("nan")

    def _surface_figure(self, ctx, run, loss, grad, window, optimum):
        go = P.require_plotly()
        lo, hi = window
        g = np.linspace(lo, hi, 140)
        A, B = np.meshgrid(g, g)
        Z = np.array([[loss(np.array([a, b])) for a in g] for b in g])
        fig = ctx.figure(
            "labs.gd.figure.surface",
            xaxis_title="w1", yaxis_title="w2", height=460,
        )
        fig.add_trace(go.Contour(x=g, y=g, z=Z, colorscale=ctx.theme.colorscale,
                                 contours={"showlabels": False}, showscale=False,
                                 opacity=0.8,
                                 name=ctx.t("labs.gd.trace.loss_surface", "loss surface")))
        path = run["path"]
        P.add_curve(fig, path[:, 0], path[:, 1],
                    ctx.t("labs.gd.trace.path", "optimizer path"),
                    "primary", theme=ctx.theme, mode="lines+markers", width=2.4)
        P.add_points(fig, [path[0, 0]], [path[0, 1]],
                     ctx.t("labs.gd.trace.start", "start"), "warning", theme=ctx.theme,
                     size=12)
        P.add_points(fig, [path[-1, 0]], [path[-1, 1]],
                     ctx.t("labs.gd.trace.end", "final point"), "positive",
                     theme=ctx.theme, size=12, symbol="star")
        P.add_points(fig, [optimum[0]], [optimum[1]],
                     ctx.t("labs.gd.trace.optimum", "known optimum"), "truth",
                     theme=ctx.theme, size=11, symbol="x")
        g0 = grad(path[0])
        norm = float(np.linalg.norm(g0)) or 1.0
        scale = 0.6 * (hi - lo) / 6.0
        P.add_arrow(fig, path[0, 0], path[0, 1],
                    path[0, 0] - scale * g0[0] / norm, path[0, 1] - scale * g0[1] / norm,
                    ctx.t("labs.gd.trace.direction", "first step direction"),
                    "negative", theme=ctx.theme)
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi], scaleanchor="x", scaleratio=1)
        P.add_legend_note(fig, ctx.t(
            "labs.gd.legend_surface",
            "Each step moves against the gradient, which is perpendicular to the contour "
            "line. That is why zig-zagging happens on elongated valleys: the steepest "
            "direction is not the direction of the minimum.",
        ), theme=ctx.theme)
        return fig

    def _loss_figure(self, ctx, run):
        it = np.arange(run["losses"].size)
        fig = ctx.figure(
            "labs.gd.figure.loss",
            xaxis_title=ctx.t("labs.common.axis.iteration"),
            yaxis_title=ctx.t("labs.common.axis.loss"),
            height=350,
        )
        P.add_curve(fig, it, run["losses"],
                    ctx.t("labs.gd.trace.loss_curve", "loss"), "primary",
                    theme=ctx.theme)
        P.add_curve(fig, it, run["grad_norms"],
                    ctx.t("labs.gd.trace.grad_curve", "gradient norm"), "secondary",
                    theme=ctx.theme, dash="dot")
        if np.all(run["losses"] > 0):
            fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.gd.legend_loss",
            "A straight line on a log scale means geometric convergence. A curve that "
            "flattens while the gradient stays large means the step size is too small; one "
            "that oscillates means it is too large.",
        ), theme=ctx.theme)
        return fig

    def _rate_figure(self, ctx, p, start, seed):
        rates = np.geomspace(0.001, 2.4, 26)
        finals, diverged = [], []
        for lr in rates:
            run = descend(str(p["surface"]), float(p["condition"]), start, float(lr),
                          int(p["steps"]), float(p["momentum"]), 0.0, seed)
            finals.append(float(run["losses"][-1]) if not run["diverged"] else np.nan)
            diverged.append(run["diverged"])
        fig = ctx.figure(
            "labs.gd.figure.rates",
            xaxis_title=ctx.t("labs.gd.axis.rate", "Learning rate"),
            yaxis_title=ctx.t("labs.gd.axis.final_loss",
                              "Loss after {n} steps", n=int(p["steps"])),
            height=360,
        )
        P.add_curve(fig, rates, finals,
                    ctx.t("labs.gd.trace.final_loss", "final loss"),
                    "primary", theme=ctx.theme, mode="lines+markers")
        limit = self._stability_limit(str(p["surface"]), float(p["condition"]))
        if np.isfinite(limit):
            P.add_vline(fig, limit,
                        ctx.t("labs.gd.trace.limit", "stability limit 2/L"),
                        "negative", theme=ctx.theme)
        P.add_vline(fig, float(p["learning_rate"]), ctx.t("labs.common.trace.current"),
                    "highlight", theme=ctx.theme, dash="dash")
        fig.update_xaxes(type="log")
        if np.any(np.isfinite(finals)) and np.nanmin(finals) > 0:
            fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.gd.legend_rates",
            "Missing points are rates at which the optimizer diverged. The best rate sits "
            "just below the cliff - which is why tuning it matters and why it is dangerous "
            "to push it.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, run, loss, window, optimum, p):
        go = P.require_plotly()
        path = run["path"]
        total = path.shape[0]
        idx = np.unique(np.round(np.linspace(1, total, min(total, 26))).astype(int))
        frames, steps = [], []
        for i, k in enumerate(idx):
            w = path[k - 1]
            frames.append(go.Frame(name=str(k), data=[
                go.Scatter(x=path[:k, 0], y=path[:k, 1]),
                go.Scatter(x=[w[0]], y=[w[1]]),
            ]))
            steps.append(AnimationStep(
                id=f"step_{k}", frame=i,
                title=ctx.t("labs.gd.anim.title", "after {k} steps", k=int(k) - 1),
                what_you_see=ctx.t("labs.gd.anim.see",
                                   "The optimizer's path across the loss contours."),
                what_changed=ctx.t("labs.gd.anim.changed",
                                   "The parameters moved to ({x}, {y}).",
                                   x=fmt(w[0], 3), y=fmt(w[1], 3)),
                why=ctx.t("labs.gd.anim.why",
                          "Each update subtracts the learning rate times the gradient. The "
                          "gradient points uphill, so the step goes downhill - by an amount "
                          "you chose, not an amount the surface chose."),
                interpretation=ctx.t("labs.gd.anim.interpret",
                                     "Loss {l}, gradient norm {g}.",
                                     l=fmt(float(run["losses"][k - 1]), 5),
                                     g=fmt(float(run["grad_norms"][k - 1]), 5)),
                conclusion=ctx.t("labs.gd.anim.conclude",
                                 "Progress slows as the gradient shrinks - the method takes "
                                 "smaller steps precisely where it is close to a stationary "
                                 "point."),
                warning=ctx.t("labs.gd.anim.warn",
                              "A small gradient does not distinguish a minimum from a saddle "
                              "or a plateau."),
                math="w <- w - eta * gradient(L(w))",
                outputs={"step": int(k) - 1, "w1": round(float(w[0]), 5),
                         "w2": round(float(w[1]), 5),
                         "loss": round(float(run["losses"][k - 1]), 6),
                         "gradient_norm": round(float(run["grad_norms"][k - 1]), 6)},
                violated_assumptions=("stable_step",) if run["diverged"] else (),
                highlighted=("optimizer_path", "current_point"),
            ))
        lo, hi = window
        g = np.linspace(lo, hi, 120)
        Z = np.array([[loss(np.array([a, b])) for a in g] for b in g])
        fig = ctx.figure(
            "labs.gd.figure.animation",
            xaxis_title="w1", yaxis_title="w2", height=420,
        )
        fig.add_trace(go.Contour(x=g, y=g, z=Z, colorscale=ctx.theme.colorscale,
                                 showscale=False, opacity=0.8,
                                 name=ctx.t("labs.gd.trace.loss_surface",
                                            "loss surface")))
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode="lines+markers",
                                 line={"color": ctx.color("primary"), "width": 2.4},
                                 name=ctx.t("labs.gd.trace.path", "optimizer path")))
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode="markers",
                                 marker={"color": ctx.color("positive"), "size": 13,
                                         "symbol": "star"},
                                 name=ctx.t("labs.gd.trace.current_point",
                                            "current parameters")))
        P.add_points(fig, [optimum[0]], [optimum[1]],
                     ctx.t("labs.gd.trace.optimum", "known optimum"), "truth",
                     theme=ctx.theme, size=11, symbol="x")
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(range=[lo, hi], scaleanchor="x", scaleratio=1)
        build_frames(fig, frames, duration=260, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.common.axis.iteration"))
        return animation(
            "descent", fig, steps,
            purpose=ctx.t("labs.gd.anim.purpose",
                          "Watch the update rule play out one step at a time."),
            summary=ctx.t(
                "labs.gd.anim.summary",
                "Gradient descent is one line of arithmetic repeated. Everything "
                "interesting - zig-zagging, divergence, getting stuck - comes from the "
                "interaction between that line and the shape of the surface, not from any "
                "cleverness in the method itself."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = GradientDescentLab(SPEC)
