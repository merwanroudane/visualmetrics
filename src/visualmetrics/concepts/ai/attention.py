"""Attention as soft lookup: queries, keys, values and a softmax you can read."""

from __future__ import annotations

from typing import Any

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
    pct,
    ref,
    scenario,
    seed_control,
    select,
    slider,
    toggle,
)

__all__ = ["LAB", "SPEC", "attention", "SEQUENCES"]

SEQUENCES = {
    "simple": ("the", "cat", "sat", "on", "the", "mat"),
    "coreference": ("the", "animal", "did", "not", "cross", "because", "it", "was",
                    "tired"),
    "arithmetic": ("2", "+", "3", "=", "?"),
    "repeated": ("a", "b", "a", "b", "a", "b"),
    "long": ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
             "iota", "kappa"),
}


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-30, None)


def attention(Q, K, V, scale: bool = True, mask: bool = False,
              temperature: float = 1.0):
    """Scaled dot-product attention, returning every intermediate matrix."""
    d_k = Q.shape[1]
    scores = Q @ K.T
    if scale:
        scores = scores / np.sqrt(max(d_k, 1))
    scores = scores / max(temperature, 1e-6)
    raw = scores.copy()
    if mask:
        n = scores.shape[0]
        scores = np.where(np.tril(np.ones((n, n))) > 0, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return {"scores": raw, "masked_scores": scores, "weights": weights,
            "output": output, "d_k": d_k}


SPEC = make_spec(
    "ai.attention",
    Domain.AI,
    "transformers",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "derive",
           "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("sequence", "simple", tuple(SEQUENCES), group="input"),
        int_slider("d_model", 8, 2, 64, 1, group="model"),
        int_slider("d_k", 4, 1, 32, 1, group="model"),
        int_slider("query_position", 1, 0, 20, 1, group="views"),
        slider("temperature", 1.0, 0.05, 8.0, 0.01, group="model"),
        toggle("scale_by_sqrt_dk", True, group="model"),
        toggle("causal_mask", False, group="model"),
        toggle("show_qkv", True, group="views"),
        toggle("show_scaling_effect", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", sequence="simple", d_model=8, d_k=4),
        scenario("uniform_attention", "null", temperature=8.0),
        scenario("hard_attention", "strong", temperature=0.1),
        scenario("causal_masking", "compare_methods", causal_mask=True),
        scenario("no_scaling", "counterexample", scale_by_sqrt_dk=False, d_k=32,
                 d_model=32),
        scenario("with_scaling", "robustness", scale_by_sqrt_dk=True, d_k=32,
                 d_model=32),
        scenario("repeated_tokens", "boundary", sequence="repeated"),
        scenario("long_sequence", "large_sample", sequence="long", d_model=16),
        scenario("coreference", "compare_methods", sequence="coreference",
                 query_position=6),
        scenario("tiny_key_dimension", "small_sample", d_k=1),
    ),
    related=("deep_learning.backpropagation", "xai.shap"),
    tags=("attention", "self-attention", "softmax", "query key value", "transformer",
          "temperature", "causal mask"),
    aliases=("attention", "الانتباه", "mecanisme d'attention", "self attention",
             "scaled dot product"),
    backends=("numpy",),
    references=(
        ref("Vaswani, A. et al. (2017). Attention is all you need. NeurIPS 30.",
            kind="paper", url="https://arxiv.org/abs/1706.03762"),
        ref("Jain, S. and Wallace, B. C. (2019). Attention is not explanation. NAACL.",
            kind="paper", doi="10.18653/v1/N19-1357"),
    ),
)


class AttentionLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        tokens = SEQUENCES[str(p["sequence"])]
        n = len(tokens)
        d_model, d_k = int(p["d_model"]), int(p["d_k"])
        emb, Wq, Wk, Wv = self._build(tokens, d_model, d_k, state.seed)
        Q, K, V = emb @ Wq, emb @ Wk, emb @ Wv
        out = attention(Q, K, V, bool(p["scale_by_sqrt_dk"]), bool(p["causal_mask"]),
                        float(p["temperature"]))
        pos = min(int(p["query_position"]), n - 1)

        res.dgp = ctx.t(
            "labs.att.dgp",
            "A hand-built single attention head: {n} tokens, model dimension {d}, key "
            "dimension {k}. Every matrix below is small enough to read.",
            n=n, d=d_model, k=d_k,
        )

        res.add_panel(ctx.panel(
            "weights", self._weights_figure(ctx, tokens, out, p),
            "labs.att.figure.weights", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "row", self._row_figure(ctx, tokens, out, pos),
            "labs.att.figure.row", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_qkv"]:
            res.add_panel(ctx.panel(
                "qkv", self._qkv_figure(ctx, tokens, Q, K, V),
                "labs.att.figure.qkv", tab="math",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_scaling_effect"]:
            res.add_panel(ctx.panel(
                "scaling", self._scaling_figure(ctx, emb, Wq, Wk, p),
                "labs.att.figure.scaling", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        weights = out["weights"]
        row = weights[pos]
        entropy = float(-np.sum(row * np.log(np.clip(row, 1e-30, None))))
        max_entropy = float(np.log(n))
        res.metric("query_token", ctx.t("labs.att.metric.query", "Query token"),
                   f"'{tokens[pos]}' (position {pos})")
        res.metric("top_attended", ctx.t("labs.att.metric.top",
                                         "Token receiving the most attention"),
                   f"'{tokens[int(np.argmax(row))]}' ({pct(float(np.max(row)))})")
        res.metric("row_sum", ctx.t("labs.att.metric.row_sum",
                                    "Attention weights in this row sum to"),
                   float(np.sum(row)), reference=1.0,
                   note=ctx.t("labs.att.metric.row_sum_note",
                              "softmax guarantees this exactly"))
        res.metric("entropy", ctx.t("labs.att.metric.entropy",
                                    "Entropy of the attention row"), entropy,
                   reference=max_entropy,
                   note=ctx.t("labs.att.metric.entropy_note",
                              "0 means all attention on one token; {m} means perfectly "
                              "uniform", m=fmt(max_entropy, 3)))
        res.metric("concentration", ctx.t("labs.att.metric.concentration",
                                          "Share of attention on the top token"),
                   float(np.max(row)))
        res.metric("effective_tokens", ctx.t("labs.att.metric.effective",
                                             "Effective number of tokens attended"),
                   float(np.exp(entropy)))
        res.metric("score_std", ctx.t("labs.att.metric.score_std",
                                      "Standard deviation of the raw scores"),
                   float(np.std(out["scores"])),
                   note=ctx.t("labs.att.metric.score_std_note",
                              "grows like sqrt(d_k) without the scaling factor"))
        res.metric("output_norm", ctx.t("labs.att.metric.output",
                                        "Norm of this position's output vector"),
                   float(np.linalg.norm(out["output"][pos])))

        saturated = float(np.max(row)) > 0.995
        res.assume("scaling", ctx.t("labs.att.assume.scaling_label",
                                    "Scores are divided by sqrt(d_k)"),
                   bool(p["scale_by_sqrt_dk"]),
                   detail=ctx.t("labs.att.assume.scaling",
                                "Dot products of {d}-dimensional vectors have standard "
                                "deviation proportional to sqrt(d). Without the division "
                                "the softmax saturates and its gradient disappears.",
                                d=d_k),
                   consequence="" if bool(p["scale_by_sqrt_dk"]) else ctx.t(
                       "labs.att.assume.no_scaling",
                       "The scores have standard deviation {s}, which pushes the softmax "
                       "towards a hard maximum and kills the gradient.",
                       s=fmt(float(np.std(out["scores"])), 2)))
        res.assume("not_explanation", ctx.t("labs.att.assume.explain_label",
                                            "Attention weights are not explanations"),
                   False,
                   detail=ctx.t("labs.att.assume.explain",
                                "Different attention patterns can produce identical "
                                "predictions, so a high weight is not evidence that a token "
                                "caused the output. This is a documented result, not a "
                                "caution about small models."))
        res.assume("toy_model", ctx.t("labs.att.assume.toy_label",
                                      "This is a hand-built head, not a trained model"),
                   True,
                   detail=ctx.t("labs.att.assume.toy",
                                "The projection matrices here are random. The mechanism is "
                                "exact; the patterns it produces carry no linguistic "
                                "meaning."))

        res.animations.append(self._animation(ctx, tokens, out, V))

        res.explain("overview", ctx.t("tabs.overview"), ctx.t("concepts.ai.attention.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.ai.attention.intuition"))
        res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.ai.attention.math"),
                    kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.ai.attention.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.ai.attention.warning"),
                    kind="warning")

        if not bool(p["scale_by_sqrt_dk"]) and d_k >= 16:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.att.counterexample",
                "Without the 1/sqrt(d_k) factor the raw scores have standard deviation {s}, "
                "so the softmax puts {c} of the weight on a single token. Its gradient at "
                "that point is essentially zero, and the head stops learning. Switch the "
                "scaling on and compare.",
                s=fmt(float(np.std(out["scores"])), 2), c=pct(float(np.max(row))),
            ), kind="warning")
        if saturated:
            res.warnings.append(ctx.t(
                "labs.att.warn.saturated",
                "This attention row places {c} of its weight on one token. A saturated "
                "softmax has a vanishing gradient, so the head can no longer adjust what it "
                "attends to.", c=pct(float(np.max(row))),
            ))
        if bool(p["causal_mask"]):
            res.explain("compare", ctx.t("labs.att.mask.title", "What the mask does"),
                        ctx.t(
                "labs.att.mask",
                "The causal mask sets every score above the diagonal to negative infinity "
                "before the softmax, so each position can only attend to itself and "
                "earlier positions. That is what makes autoregressive generation possible: "
                "position {i} cannot see the future it is about to predict.", i=pos,
            ))
        return res

    @staticmethod
    def _build(tokens, d_model, d_k, seed):
        gen = rng(seed, "attention")
        vocab = sorted(set(tokens))
        table = {tok: gen.normal(0, 1.0, d_model) for tok in vocab}
        emb = np.array([table[t] for t in tokens])
        # positional signal so repeated tokens are distinguishable
        pos = np.arange(len(tokens))[:, None]
        dims = np.arange(d_model)[None, :]
        emb = emb + 0.5 * np.sin(pos / (10 ** (2 * (dims // 2) / max(d_model, 1))))
        Wq = gen.normal(0, 1.0 / np.sqrt(d_model), (d_model, d_k))
        Wk = gen.normal(0, 1.0 / np.sqrt(d_model), (d_model, d_k))
        Wv = gen.normal(0, 1.0 / np.sqrt(d_model), (d_model, d_k))
        return emb, Wq, Wk, Wv

    def _weights_figure(self, ctx, tokens, out, p):
        go = P.require_plotly()
        w = out["weights"]
        fig = ctx.figure(
            "labs.att.figure.weights",
            xaxis_title=ctx.t("labs.att.axis.key", "Key position (attended to)"),
            yaxis_title=ctx.t("labs.att.axis.query", "Query position (attending)"),
            height=440,
        )
        fig.add_trace(go.Heatmap(
            z=w, x=list(tokens), y=list(tokens), colorscale=ctx.theme.colorscale,
            zmin=0, zmax=1,
            text=[[fmt(v, 2) for v in row] for row in w], texttemplate="%{text}",
            colorbar={"title": ctx.t("labs.common.axis.weight")},
        ))
        fig.update_yaxes(autorange="reversed")
        P.add_legend_note(fig, ctx.t(
            "labs.att.legend_weights",
            "Every ROW sums to exactly one - each position distributes a fixed budget of "
            "attention. Columns have no such constraint.{mask}",
            mask=("  The upper triangle is zero because of the causal mask."
                  if bool(p["causal_mask"]) else ""),
        ), theme=ctx.theme)
        return fig

    def _row_figure(self, ctx, tokens, out, pos):
        row = out["weights"][pos]
        scores = out["masked_scores"][pos]
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.att.trace.scores", "similarity scores (query . key)"),
            ctx.t("labs.att.trace.weights", "after softmax"),
        ))
        finite = np.where(np.isfinite(scores), scores, np.nan)
        fig.add_trace(go.Bar(x=list(tokens), y=finite,
                             marker={"color": ctx.color("secondary")},
                             showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=list(tokens), y=row,
                             marker={"color": ctx.color("primary")},
                             text=[pct(v) for v in row], textposition="outside",
                             showlegend=False), row=1, col=2)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=370, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.att.legend_row",
            "Softmax turns arbitrary real scores into a distribution: it preserves the "
            "ordering, exaggerates the differences, and guarantees the weights are "
            "non-negative and sum to one. Query token: '{q}'.", q=tokens[pos],
        ), theme=ctx.theme)
        return fig

    def _qkv_figure(self, ctx, tokens, Q, K, V):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        fig = make_subplots(rows=1, cols=3, subplot_titles=(
            ctx.t("labs.att.trace.q", "queries Q"),
            ctx.t("labs.att.trace.k", "keys K"),
            ctx.t("labs.att.trace.v", "values V"),
        ))
        for i, M in enumerate((Q, K, V)):
            fig.add_trace(go.Heatmap(
                z=M, y=list(tokens),
                colorscale=ctx.theme.diverging_colorscale, zmid=0, showscale=i == 2,
            ), row=1, col=i + 1)
            fig.update_yaxes(autorange="reversed", row=1, col=i + 1)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=360, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.att.legend_qkv",
            "Three different linear projections of the SAME embeddings. The query asks, the "
            "key advertises, the value is what gets carried forward - and the separation "
            "between asking and offering is what lets a position attend to something "
            "unlike itself.",
        ), theme=ctx.theme)
        return fig

    def _scaling_figure(self, ctx, emb, Wq, Wk, p):
        dims = np.unique(np.round(np.geomspace(1, 64, 12)).astype(int))
        unscaled_std, scaled_std, unscaled_max, scaled_max = [], [], [], []
        gen = rng(int(p.get("seed", 42)), "attn_scale")
        d_model = emb.shape[1]
        for d in dims:
            wq = gen.normal(0, 1.0 / np.sqrt(d_model), (d_model, int(d)))
            wk = gen.normal(0, 1.0 / np.sqrt(d_model), (d_model, int(d)))
            q, k = emb @ wq, emb @ wk
            raw = q @ k.T
            unscaled_std.append(float(np.std(raw)))
            scaled_std.append(float(np.std(raw / np.sqrt(d))))
            unscaled_max.append(float(np.max(softmax(raw, axis=-1))))
            scaled_max.append(float(np.max(softmax(raw / np.sqrt(d), axis=-1))))
        make_subplots = P.SUBPLOT()
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.att.trace.score_spread", "spread of the raw scores"),
            ctx.t("labs.att.trace.saturation", "largest softmax weight"),
        ))
        go = P.require_plotly()
        for col, (a, b) in enumerate(((unscaled_std, scaled_std),
                                      (unscaled_max, scaled_max)), start=1):
            # Dash and marker as well as colour: the two curves must stay
            # distinguishable without colour vision.
            fig.add_trace(go.Scatter(x=dims, y=a, mode="lines+markers",
                                     line={"color": ctx.color("negative"), "dash": "dash"},
                                     marker={"symbol": "x"},
                                     name=ctx.t("labs.att.trace.unscaled", "no scaling"),
                                     showlegend=col == 1), row=1, col=col)
            fig.add_trace(go.Scatter(x=dims, y=b, mode="lines+markers",
                                     line={"color": ctx.color("positive"), "dash": "solid"},
                                     marker={"symbol": "circle"},
                                     name=ctx.t("labs.att.trace.scaled",
                                                "divided by sqrt(d_k)"),
                                     showlegend=col == 1), row=1, col=col)
            fig.update_xaxes(type="log", title_text="d_k", row=1, col=col)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=360, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.att.legend_scaling",
            "This is the entire reason for the 1/sqrt(d_k) in the formula: without it the "
            "scores grow with the key dimension and the softmax saturates, which stops the "
            "gradient. With it, the spread stays roughly constant.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, tokens, out, V):
        go = P.require_plotly()
        n = len(tokens)
        frames, steps = [], []
        for i in range(n):
            row = out["weights"][i]
            frames.append(go.Frame(name=str(i), data=[
                go.Bar(x=list(tokens), y=row),
            ]))
            top = int(np.argmax(row))
            entropy = float(-np.sum(row * np.log(np.clip(row, 1e-30, None))))
            steps.append(AnimationStep(
                id=f"pos_{i}", frame=i,
                title=ctx.t("labs.att.anim.title",
                            "position {i}: '{t}' is attending", i=i, t=tokens[i]),
                what_you_see=ctx.t("labs.att.anim.see",
                                   "How much of its attention budget this position gives "
                                   "to each token in the sequence."),
                what_changed=ctx.t("labs.att.anim.changed",
                                   "The query moved to position {i} ('{t}').",
                                   i=i, t=tokens[i]),
                why=ctx.t("labs.att.anim.why",
                          "This position's query vector is compared with every key by dot "
                          "product; softmax turns those similarities into weights that sum "
                          "to one."),
                interpretation=ctx.t("labs.att.anim.interpret",
                                     "Most attention goes to '{k}' ({w}); the row's entropy "
                                     "is {e}, so it is effectively spreading over {n} "
                                     "tokens.",
                                     k=tokens[top], w=pct(float(row[top])),
                                     e=fmt(entropy, 3), n=fmt(float(np.exp(entropy)), 1)),
                conclusion=ctx.t("labs.att.anim.conclude",
                                 "The output at this position is the weighted average of "
                                 "the VALUE vectors, using exactly these weights."),
                warning=ctx.t("labs.att.anim.warn",
                              "A large weight shows where information was mixed from. It is "
                              "not evidence that this token determined the output."),
                math="Attention(Q, K, V) = softmax(Q K' / sqrt(d_k)) V",
                outputs={"position": i, "token": tokens[i],
                         "top_token": tokens[top], "top_weight": round(float(row[top]), 4),
                         "entropy": round(entropy, 4)},
                active_assumptions=("toy_model",),
                highlighted=("attention_row",),
            ))
        fig = ctx.figure(
            "labs.att.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.token"),
            yaxis_title=ctx.t("labs.common.axis.weight"),
            height=390,
        )
        fig.add_trace(go.Bar(x=list(tokens), y=out["weights"][0],
                             marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.att.trace.weights_short",
                                        "attention weights")))
        fig.update_yaxes(range=[0, 1.05])
        build_frames(fig, frames, duration=700, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.att.slider", "query position"))
        return animation(
            "query_sweep", fig, steps,
            purpose=ctx.t("labs.att.anim.purpose",
                          "Walk the query along the sequence and watch the lookup change."),
            summary=ctx.t(
                "labs.att.anim.summary",
                "Self-attention runs the same lookup once per position. Each position asks "
                "its own question, gets its own distribution over the sequence, and mixes "
                "the values accordingly - which is how a token's representation comes to "
                "depend on its context."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = AttentionLab(SPEC)
