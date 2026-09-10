"""Backpropagation: the chain rule with bookkeeping, on a graph you can read."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "forward", "backward", "ACTIVATIONS"]

ACTIVATIONS = ("relu", "sigmoid", "tanh", "linear")


def _activate(z, kind):
    if kind == "relu":
        return np.maximum(z, 0.0)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    if kind == "tanh":
        return np.tanh(z)
    return z


def _activate_grad(z, kind):
    if kind == "relu":
        return (z > 0).astype(float)
    if kind == "sigmoid":
        s = _activate(z, "sigmoid")
        return s * (1 - s)
    if kind == "tanh":
        return 1.0 - np.tanh(z) ** 2
    return np.ones_like(z)


def forward(weights, biases, x, activation):
    """Run the forward pass, keeping every intermediate value."""
    a = np.asarray(x, dtype=float)
    zs, activations = [], [a]
    for W, b in zip(weights, biases, strict=False):
        z = W @ a + b
        zs.append(z)
        a = _activate(z, activation)
        activations.append(a)
    # the output layer is linear
    activations[-1] = zs[-1]
    return zs, activations


def backward(weights, biases, zs, activations, target, activation):
    """Backpropagate a squared-error loss, returning every local quantity."""
    L = len(weights)
    output = activations[-1]
    loss = float(0.5 * np.sum((output - target) ** 2))
    deltas = [None] * L
    deltas[L - 1] = output - np.asarray(target, dtype=float)
    for layer in range(L - 2, -1, -1):
        deltas[layer] = (weights[layer + 1].T @ deltas[layer + 1]) * \
            _activate_grad(zs[layer], activation)
    grad_w = [np.outer(deltas[i], activations[i]) for i in range(L)]
    grad_b = list(deltas)
    return {"loss": loss, "deltas": deltas, "grad_w": grad_w, "grad_b": grad_b}


SPEC = make_spec(
    "deep_learning.backpropagation",
    Domain.DEEP_LEARNING,
    "training",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive", "prove",
           "code", "quiz", "references"),
    evidence=EvidenceType.SYMBOLIC_DERIVATION,
    controls=(
        select("activation", "tanh", ACTIVATIONS, group="network"),
        int_slider("hidden_layers", 2, 1, 8, 1, group="network"),
        int_slider("width", 3, 1, 6, 1, group="network"),
        slider("weight_scale", 1.0, 0.05, 3.0, 0.01, group="network"),
        slider("input_value", 1.0, -3.0, 3.0, 0.05, group="data"),
        slider("target", 0.5, -3.0, 3.0, 0.05, group="data"),
        slider("learning_rate", 0.1, 0.0, 1.0, 0.005, group="training"),
        int_slider("training_steps", 40, 0, 500, 1, group="training"),
        toggle("show_gradient_flow", True, group="views"),
        toggle("show_numeric_check", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", activation="tanh", hidden_layers=2, width=3),
        scenario("single_layer", "boundary", hidden_layers=1, width=2),
        scenario("deep_network", "compare_methods", hidden_layers=6, width=3),
        scenario("vanishing_gradient", "counterexample", activation="sigmoid",
                 hidden_layers=8, width=3, weight_scale=0.5, training_steps=0),
        scenario("exploding_gradient", "counterexample", activation="linear",
                 hidden_layers=8, width=3, weight_scale=2.2, training_steps=0),
        scenario("relu_survives", "robustness", activation="relu", hidden_layers=8,
                 weight_scale=1.4),
        scenario("dead_relu", "violation", activation="relu", weight_scale=0.05,
                 input_value=-2.0),
        scenario("no_training", "null", training_steps=0),
        scenario("well_trained", "positive", training_steps=200, learning_rate=0.15),
        scenario("learning_rate_too_high", "violation", learning_rate=0.9,
                 training_steps=100),
    ),
    prerequisites=("ml.gradient_descent",),
    related=("ml.gradient_descent", "ai.attention"),
    next_concepts=("ai.attention",),
    tags=("backpropagation", "chain rule", "computational graph", "gradients",
          "vanishing gradient", "exploding gradient"),
    aliases=("backprop", "retropropagation", "الانتشار العكسي", "chain rule",
             "computational graph"),
    backends=("numpy",),
    proof_ids=("deep_learning.backprop.chain_rule",),
    references=(
        ref("Rumelhart, D. E., Hinton, G. E. and Williams, R. J. (1986). Learning "
            "representations by back-propagating errors. Nature 323.", kind="paper",
            doi="10.1038/323533a0"),
        ref("Goodfellow, I., Bengio, Y. and Courville, A. (2016). Deep Learning.",
            kind="book"),
    ),
)


class BackpropLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        act = str(p["activation"])
        weights, biases = self._init_network(p, state.seed)
        x = np.array([float(p["input_value"])])
        target = np.array([float(p["target"])])

        history = self._train(weights, biases, x, target, act,
                              float(p["learning_rate"]), int(p["training_steps"]))
        weights, biases = history["weights"], history["biases"]
        zs, activations = forward(weights, biases, x, act)
        back = backward(weights, biases, zs, activations, target, act)
        numeric = (self._numeric_check(weights, biases, x, target, act)
                   if p["show_numeric_check"] else None)

        res.dgp = ctx.t(
            "labs.bp.dgp",
            "A {L}-hidden-layer network of width {w} with {a} activations, one input "
            "({x}) and one target ({t}); squared-error loss.",
            L=int(p["hidden_layers"]), w=int(p["width"]), a=act,
            x=fmt(p["input_value"], 2), t=fmt(p["target"], 2),
        )

        res.add_panel(ctx.panel(
            "graph", self._graph_figure(ctx, weights, activations, back, act),
            "labs.bp.figure.graph", evidence=EvidenceType.SYMBOLIC_DERIVATION,
        ))
        if p["show_gradient_flow"]:
            res.add_panel(ctx.panel(
                "flow", self._flow_figure(ctx, back, act),
                "labs.bp.figure.flow", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if history["losses"].size > 1:
            res.add_panel(ctx.panel(
                "training", self._training_figure(ctx, history),
                "labs.bp.figure.training", tab="simulate",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if numeric is not None:
            res.add_panel(ctx.panel(
                "check", self._check_figure(ctx, back, numeric),
                "labs.bp.figure.check", tab="math",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        norms = [float(np.linalg.norm(d)) for d in back["deltas"]]
        res.metric("loss", ctx.t("labs.bp.metric.loss", "Loss"), back["loss"])
        res.metric("output", ctx.t("labs.bp.metric.output", "Network output"),
                   float(activations[-1][0]), reference=float(p["target"]))
        res.metric("first_layer_gradient", ctx.t("labs.bp.metric.first",
                                                 "Gradient norm at the first layer"),
                   norms[0])
        res.metric("last_layer_gradient", ctx.t("labs.bp.metric.last",
                                                "Gradient norm at the last layer"),
                   norms[-1])
        res.metric("gradient_ratio", ctx.t("labs.bp.metric.ratio",
                                           "First layer divided by last layer"),
                   norms[0] / max(norms[-1], 1e-30),
                   note=ctx.t("labs.bp.metric.ratio_note",
                              "far below 1 is a vanishing gradient; far above 1 is an "
                              "exploding one"))
        res.metric("layers", ctx.t("labs.bp.metric.layers", "Weight layers"),
                   len(weights))
        res.metric("parameters", ctx.t("labs.bp.metric.parameters", "Parameters"),
                   int(sum(W.size for W in weights) + sum(b.size for b in biases)))
        if numeric is not None:
            res.metric("gradient_check", ctx.t("labs.bp.metric.check",
                                               "Largest disagreement with finite "
                                               "differences"),
                       numeric["max_error"], reference=0.0,
                       note=ctx.t("labs.bp.metric.check_note",
                                  "backpropagation is exact, so this is only rounding "
                                  "error"))
        if history["losses"].size > 1:
            res.metric("initial_loss", ctx.t("labs.bp.metric.initial_loss",
                                             "Loss before training"),
                       float(history["losses"][0]))
            res.metric("final_loss", ctx.t("labs.bp.metric.final_loss",
                                           "Loss after {n} steps",
                                           n=int(p["training_steps"])),
                       float(history["losses"][-1]))

        ratio = norms[0] / max(norms[-1], 1e-30)
        # a converged network has a vanishing gradient everywhere for a good reason
        converged = back["loss"] < 1e-10
        vanishing = bool(ratio < 0.01) and not converged
        exploding = bool(ratio > 20) and not converged
        dead = act == "relu" and all(float(np.max(np.abs(a))) < 1e-12
                                     for a in activations[1:-1])
        res.assume("gradient_flow", ctx.t("labs.bp.assume.flow_label",
                                          "Gradient reaches the early layers"),
                   not (vanishing or exploding),
                   detail=ctx.t("labs.bp.assume.flow",
                                "Backpropagation multiplies one factor per layer. Factors "
                                "below one shrink the product geometrically; factors above "
                                "one grow it."),
                   consequence="" if not (vanishing or exploding) else ctx.t(
                       "labs.bp.assume.flow_consequence",
                       "The early layers receive a gradient {r} times the size of the last "
                       "layer's, so they barely learn - or destabilise training entirely.",
                       r=fmt(ratio, 6)))
        res.assume("differentiable", ctx.t("labs.bp.assume.diff_label",
                                           "Every operation is differentiable"),
                   act != "relu" or not dead,
                   detail=ctx.t("labs.bp.assume.diff",
                                "ReLU has a kink at zero, and a unit whose input is always "
                                "negative outputs zero and receives zero gradient forever."))
        res.assume("exact_gradient", ctx.t("labs.bp.assume.exact_label",
                                           "The computed gradient is exact"), True,
                   detail=ctx.t("labs.bp.assume.exact",
                                "Backpropagation is not an approximation. It is the chain "
                                "rule evaluated in reverse order, and the numeric check "
                                "confirms it to rounding error."))

        res.animations.append(self._animation(ctx, weights, activations, back, act))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.deep_learning.backpropagation.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.deep_learning.backpropagation.intuition"))
        res.explain("math", ctx.t("tabs.math"),
                    ctx.t("concepts.deep_learning.backpropagation.math"), kind="math")
        res.explain("proof", ctx.t("tabs.proof"), ctx.t(
            "labs.bp.proof",
            "Backpropagation is the chain rule applied in reverse order.\n"
            "1. Write the network as a composition: L(a_K), a_k = f(z_k), z_k = W_k "
            "a_(k-1) + b_k.\n"
            "2. Define delta_k = dL/dz_k. At the output, delta_K = dL/da_K (times the "
            "output activation's derivative, which is one for a linear output).\n"
            "3. For any earlier layer, z_(k+1) depends on z_k only through a_k, so the "
            "chain rule gives delta_k = (W_(k+1)' delta_(k+1)) * f'(z_k).\n"
            "4. Since z_k = W_k a_(k-1) + b_k, we get dL/dW_k = delta_k a_(k-1)' and "
            "dL/db_k = delta_k.\n"
            "5. Computing the deltas from the output backwards reuses each partial result "
            "exactly once, which is why the cost of the whole gradient is about the cost of "
            "one forward pass rather than one pass per parameter.\n"
            "Step 3 also explains vanishing and exploding gradients directly: delta_1 is a "
            "product of K matrices and K activation derivatives, and a product of many "
            "small - or many large - numbers goes to zero or to infinity geometrically.",
        ), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.deep_learning.backpropagation.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.deep_learning.backpropagation.warning"), kind="warning")

        if vanishing:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.bp.counterexample_vanish",
                "The first layer's gradient is {a} while the last layer's is {b} - a ratio "
                "of {r}. With sigmoid activations the derivative never exceeds 0.25, so "
                "eight layers multiply at most 0.25^8, which is about 1 in 65000 before the "
                "weights are even considered.",
                a=fmt(norms[0], 8), b=fmt(norms[-1], 6), r=fmt(ratio, 8),
            ), kind="warning")
            res.warnings.append(ctx.t(
                "labs.bp.warn.vanishing",
                "The gradient has effectively vanished before reaching the first layer. "
                "Early layers cannot learn under these settings.",
            ))
        if exploding:
            res.warnings.append(ctx.t(
                "labs.bp.warn.exploding",
                "The gradient grows by a factor of {r} on its way back. Training will "
                "diverge unless the gradient is clipped or the weights rescaled.",
                r=fmt(ratio, 2),
            ))
        if dead:
            res.warnings.append(ctx.t(
                "labs.bp.warn.dead_relu",
                "Every hidden unit outputs zero for this input, so every gradient behind "
                "them is zero as well. These units are dead and no amount of training will "
                "revive them.",
            ))
        return res

    @staticmethod
    def _init_network(p, seed):
        gen = rng(seed, "backprop")
        width = int(p["width"])
        layers = [1] + [width] * int(p["hidden_layers"]) + [1]
        scale = float(p["weight_scale"])
        weights, biases = [], []
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            weights.append(gen.normal(0.0, scale / np.sqrt(fan_in),
                                      (layers[i + 1], layers[i])))
            biases.append(np.zeros(layers[i + 1]))
        return weights, biases

    @staticmethod
    def _train(weights, biases, x, target, act, lr, steps):
        W = [w.copy() for w in weights]
        B = [b.copy() for b in biases]
        losses = []
        for _ in range(int(steps) + 1):
            zs, activations = forward(W, B, x, act)
            back = backward(W, B, zs, activations, target, act)
            losses.append(back["loss"])
            if lr <= 0 or len(losses) > int(steps):
                break
            for i in range(len(W)):
                W[i] = W[i] - lr * back["grad_w"][i]
                B[i] = B[i] - lr * back["grad_b"][i]
            if not np.all(np.isfinite(W[0])):
                break
        return {"weights": W, "biases": B, "losses": np.asarray(losses)}

    @staticmethod
    def _numeric_check(weights, biases, x, target, act, eps: float = 1e-6):
        zs, activations = forward(weights, biases, x, act)
        analytic = backward(weights, biases, zs, activations, target, act)["grad_w"]
        errors = []
        for layer, W in enumerate(weights):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    up = [w.copy() for w in weights]
                    down = [w.copy() for w in weights]
                    up[layer][i, j] += eps
                    down[layer][i, j] -= eps
                    lu = forward(up, biases, x, act)[1][-1] - target
                    ld = forward(down, biases, x, act)[1][-1] - target
                    numeric = (0.5 * np.sum(lu**2) - 0.5 * np.sum(ld**2)) / (2 * eps)
                    errors.append(abs(float(numeric) - float(analytic[layer][i, j])))
        return {"max_error": float(np.max(errors)) if errors else 0.0,
                "errors": np.asarray(errors)}

    def _graph_figure(self, ctx, weights, activations, back, act):
        go = P.require_plotly()
        layers = [a.size for a in activations]
        fig = ctx.figure(
            "labs.bp.figure.graph", height=470, showlegend=True,
        )
        positions = {}
        for li, size in enumerate(layers):
            ys = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
            for ni in range(size):
                positions[(li, ni)] = (li, float(ys[ni]))
        finite = [float(np.max(np.abs(g[np.isfinite(g)]))) if np.any(np.isfinite(g))
                  else 0.0 for g in back["grad_w"]]
        max_grad = max(finite) if finite and max(finite) > 0 else 1.0
        for li, W in enumerate(weights):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    a = positions[(li, j)]
                    b = positions[(li + 1, i)]
                    raw = float(back["grad_w"][li][i, j])
                    strength = (abs(raw) / max_grad if np.isfinite(raw) else 1.0)
                    strength = float(np.clip(strength, 0.0, 1.0))
                    fig.add_trace(go.Scatter(
                        x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                        line={"color": P.rgba("residual", 0.15 + 0.85 * strength,
                                              ctx.theme),
                              "width": 0.8 + 4.0 * strength},
                        hovertemplate=(f"w[{li}][{i},{j}] = {W[i, j]:.3f}<br>"
                                       f"dL/dw = {back['grad_w'][li][i, j]:.5f}"
                                       "<extra></extra>"),
                        showlegend=False,
                    ))
        for (li, ni), (px, py) in positions.items():
            value = float(activations[li][ni])
            role = ("primary" if li == 0 else
                    ("positive" if li == len(layers) - 1 else "secondary"))
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode="markers+text",
                marker={"color": ctx.color(role), "size": 34},
                text=[fmt(value, 2)], textposition="middle center",
                textfont={"color": ctx.color("paper"), "size": 10},
                hovertemplate=f"activation = {value:.4f}<extra></extra>",
                showlegend=False,
            ))
        fig.update_xaxes(visible=False, range=[-0.6, len(layers) - 0.4])
        fig.update_yaxes(visible=False)
        P.add_legend_note(fig, ctx.t(
            "labs.bp.legend_graph",
            "Node labels are forward activations; edge thickness is the size of that "
            "weight's gradient. Thin edges near the input are exactly what a vanishing "
            "gradient looks like. Hover any edge for its weight and its derivative.",
        ), theme=ctx.theme)
        return fig

    def _flow_figure(self, ctx, back, act):
        norms = [float(np.linalg.norm(d)) for d in back["deltas"]]
        layers = np.arange(1, len(norms) + 1)
        fig = ctx.figure(
            "labs.bp.figure.flow",
            xaxis_title=ctx.t("labs.bp.axis.layer", "Layer (1 = closest to the input)"),
            yaxis_title=ctx.t("labs.bp.axis.norm", "Gradient norm"),
            height=350,
        )
        P.add_curve(fig, layers, norms,
                    ctx.t("labs.bp.trace.delta", "|delta| at each layer"),
                    "primary", theme=ctx.theme, mode="lines+markers")
        if min(norms) > 0:
            fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.bp.legend_flow",
            "On a log scale a straight line means the gradient is being multiplied by a "
            "constant factor at every layer. Sloping down is vanishing; sloping up is "
            "exploding. Activation in use: {a}.", a=act,
        ), theme=ctx.theme)
        return fig

    def _training_figure(self, ctx, history):
        losses = history["losses"]
        fig = ctx.figure(
            "labs.bp.figure.training",
            xaxis_title=ctx.t("labs.common.axis.iteration"),
            yaxis_title=ctx.t("labs.common.axis.loss"),
            height=340,
        )
        P.add_curve(fig, np.arange(losses.size), losses,
                    ctx.t("labs.bp.trace.loss", "loss"), "primary", theme=ctx.theme)
        if np.all(losses > 0):
            fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.bp.legend_training",
            "Backpropagation supplies the gradient; this curve is what the optimizer does "
            "with it. The two are separate steps and fail for separate reasons.",
        ), theme=ctx.theme)
        return fig

    def _check_figure(self, ctx, back, numeric):
        analytic = np.concatenate([g.ravel() for g in back["grad_w"]])
        errors = numeric["errors"]
        fig = ctx.figure(
            "labs.bp.figure.check",
            xaxis_title=ctx.t("labs.bp.axis.analytic",
                              "Gradient from backpropagation"),
            yaxis_title=ctx.t("labs.bp.axis.numeric",
                              "Gradient from finite differences"),
            height=370,
        )
        P.add_points(fig, analytic, analytic + np.sign(analytic) * errors,
                     ctx.t("labs.bp.trace.pairs", "one point per weight"),
                     "primary", theme=ctx.theme, size=8)
        lim = float(np.max(np.abs(analytic))) * 1.1 + 1e-9
        P.add_curve(fig, [-lim, lim], [-lim, lim],
                    ctx.t("labs.bp.trace.identity", "exact agreement"),
                    "truth", theme=ctx.theme, dash="dash")
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        P.add_legend_note(fig, ctx.t(
            "labs.bp.legend_check",
            "Every point sits on the diagonal to {e}. Backpropagation is the chain rule "
            "computed exactly, not a numerical approximation - and this is how you would "
            "verify a hand-written gradient in practice.",
            e=fmt(numeric["max_error"], 9),
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, weights, activations, back, act):
        go = P.require_plotly()
        layers = [a.size for a in activations]
        positions = {}
        for li, size in enumerate(layers):
            ys = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
            for ni in range(size):
                positions[(li, ni)] = (li, float(ys[ni]))
        total = len(layers) + len(weights)
        frames, steps = [], []
        for step in range(total):
            forward_phase = step < len(layers)
            if forward_phase:
                li = step
                xs = [positions[(li, n)][0] for n in range(layers[li])]
                ys = [positions[(li, n)][1] for n in range(layers[li])]
                colour = ctx.color("positive")
                title = ctx.t("labs.bp.anim.forward_title",
                              "forward pass: layer {i}", i=li)
                see = ctx.t("labs.bp.anim.forward_see",
                            "Values flowing left to right; the highlighted layer has just "
                            "been computed.")
                changed = ctx.t("labs.bp.anim.forward_changed",
                                "Layer {i} activations were computed as f(W a + b): {v}.",
                                i=li, v=", ".join(fmt(v, 3)
                                                  for v in np.atleast_1d(activations[li])))
                why = ctx.t("labs.bp.anim.forward_why",
                            "Each layer transforms the previous layer's output. Every "
                            "intermediate value is stored, because the backward pass needs "
                            "them.")
                interp = ctx.t("labs.bp.anim.forward_interpret",
                               "Nothing is being learned yet - this pass only produces the "
                               "prediction and the loss.")
                outputs = {"phase": "forward", "layer": li}
            else:
                li = total - step - 1
                xs = [positions[(li + 1, n)][0] for n in range(layers[li + 1])]
                ys = [positions[(li + 1, n)][1] for n in range(layers[li + 1])]
                colour = ctx.color("residual")
                title = ctx.t("labs.bp.anim.backward_title",
                              "backward pass: layer {i}", i=li + 1)
                see = ctx.t("labs.bp.anim.backward_see",
                            "Sensitivities flowing right to left; the highlighted layer's "
                            "delta has just been computed.")
                changed = ctx.t("labs.bp.anim.backward_changed",
                                "delta at layer {i} is {v}, with gradient norm {n}.",
                                i=li + 1,
                                v=", ".join(fmt(v, 4)
                                            for v in np.atleast_1d(back["deltas"][li])),
                                n=fmt(float(np.linalg.norm(back["deltas"][li])), 5))
                why = ctx.t("labs.bp.anim.backward_why",
                            "delta_k = (W_(k+1)' delta_(k+1)) times the activation "
                            "derivative. Each step back multiplies by one more matrix and "
                            "one more derivative.")
                interp = ctx.t("labs.bp.anim.backward_interpret",
                               "The weight gradients at this layer are delta times the "
                               "incoming activation - a single outer product.")
                outputs = {"phase": "backward", "layer": li + 1,
                           "delta_norm": round(float(np.linalg.norm(back["deltas"][li])), 6)}
            frames.append(go.Frame(name=str(step), data=[
                go.Scatter(x=xs, y=ys, marker={"color": colour}),
            ]))
            steps.append(AnimationStep(
                id=f"step_{step}", frame=step, title=title, what_you_see=see,
                what_changed=changed, why=why, interpretation=interp,
                conclusion=ctx.t("labs.bp.anim.conclude",
                                 "One forward sweep and one backward sweep produce the "
                                 "gradient for EVERY parameter - that reuse is the whole "
                                 "reason training deep networks is feasible."),
                warning=ctx.t("labs.bp.anim.warn",
                              "Backpropagation computes gradients. It does not update "
                              "anything; the optimizer does that."),
                math="delta_k = (W_(k+1)' delta_(k+1)) * f'(z_k);  dL/dW_k = delta_k a_(k-1)'",
                outputs=outputs,
                highlighted=("active_layer",),
            ))
        fig = ctx.figure("labs.bp.figure.animation", height=420, showlegend=True)
        for li, W in enumerate(weights):
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    a = positions[(li, j)]
                    b = positions[(li + 1, i)]
                    fig.add_trace(go.Scatter(x=[a[0], b[0]], y=[a[1], b[1]], mode="lines",
                                             line={"color": ctx.color("muted"),
                                                   "width": 0.8},
                                             showlegend=False, hoverinfo="skip"))
        all_x = [pos[0] for pos in positions.values()]
        all_y = [pos[1] for pos in positions.values()]
        fig.add_trace(go.Scatter(x=all_x, y=all_y, mode="markers",
                                 marker={"color": ctx.color("muted"), "size": 22},
                                 name=ctx.t("labs.bp.trace.units", "units")))
        fig.add_trace(go.Scatter(x=[positions[(0, 0)][0]], y=[positions[(0, 0)][1]],
                                 mode="markers",
                                 marker={"color": ctx.color("positive"), "size": 30},
                                 name=ctx.t("labs.bp.trace.active", "active layer")))
        fig.update_xaxes(visible=False, range=[-0.6, len(layers) - 0.4])
        fig.update_yaxes(visible=False)
        build_frames(fig, frames, duration=700, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.bp.slider", "pass"))
        return animation(
            "forward_backward", fig, steps,
            purpose=ctx.t("labs.bp.anim.purpose",
                          "Separate the forward pass from the backward pass and watch each "
                          "one do its own job."),
            summary=ctx.t(
                "labs.bp.anim.summary",
                "Forward computes values; backward computes sensitivities. The backward "
                "sweep multiplies one matrix and one activation derivative per layer, which "
                "is simultaneously why the whole gradient is cheap and why deep networks "
                "suffer from vanishing and exploding gradients."),
            evidence=EvidenceType.SYMBOLIC_DERIVATION,
        )


LAB = BackpropLab(SPEC)
