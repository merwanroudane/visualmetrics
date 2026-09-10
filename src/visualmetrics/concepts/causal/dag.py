"""Causal graphs: confounders, colliders, mediators and bad controls."""

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

__all__ = ["LAB", "SPEC", "STRUCTURES", "path_status"]

STRUCTURES = ("confounder", "chain_mediator", "collider", "m_bias", "front_door",
              "instrument", "descendant_of_collider")

#: structure -> (nodes with positions, directed edges, whether Z should be controlled)
GRAPHS: dict[str, dict[str, Any]] = {
    "confounder": {
        "nodes": {"D": (0, 0), "Y": (2, 0), "Z": (1, 1)},
        "edges": [("Z", "D"), ("Z", "Y"), ("D", "Y")],
        "control": True,
        "story": "Z is a common cause of both D and Y.",
    },
    "chain_mediator": {
        "nodes": {"D": (0, 0), "Z": (1, 0.7), "Y": (2, 0)},
        "edges": [("D", "Z"), ("Z", "Y"), ("D", "Y")],
        "control": False,
        "story": "Z lies on a causal path from D to Y.",
    },
    "collider": {
        "nodes": {"D": (0, 0), "Y": (2, 0), "Z": (1, -1)},
        "edges": [("D", "Z"), ("Y", "Z"), ("D", "Y")],
        "control": False,
        "story": "Z is a common effect of D and Y.",
    },
    "m_bias": {
        "nodes": {"D": (0, 0), "Y": (2, 0), "Z": (1, -0.9), "U1": (0, -1.6),
                  "U2": (2, -1.6)},
        "edges": [("U1", "D"), ("U1", "Z"), ("U2", "Z"), ("U2", "Y"), ("D", "Y")],
        "control": False,
        "story": "Z is a pre-treatment collider between two unobserved causes.",
    },
    "front_door": {
        "nodes": {"D": (0, 0), "Z": (1, 0.7), "Y": (2, 0), "U": (1, -1.1)},
        "edges": [("D", "Z"), ("Z", "Y"), ("U", "D"), ("U", "Y")],
        "control": True,
        "story": "An unobserved confounder plus a fully mediating Z.",
    },
    "instrument": {
        "nodes": {"Z": (-1, 0), "D": (0.5, 0), "Y": (2, 0), "U": (1.2, -1.1)},
        "edges": [("Z", "D"), ("D", "Y"), ("U", "D"), ("U", "Y")],
        "control": False,
        "story": "Z affects Y only through D.",
    },
    "descendant_of_collider": {
        "nodes": {"D": (0, 0), "Y": (2, 0), "C": (1, -1.0), "Z": (1, -1.9)},
        "edges": [("D", "C"), ("Y", "C"), ("C", "Z"), ("D", "Y")],
        "control": False,
        "story": "Z is a child of a collider, which is just as dangerous.",
    },
}


def path_status(structure: str, conditioning: bool) -> dict[str, Any]:
    """What conditioning does to the back-door paths in this structure."""
    g = GRAPHS[structure]
    should = g["control"]
    if structure == "confounder":
        return {"open_before": True, "open_after": False,
                "correct": conditioning == should,
                "role": "confounder"}
    if structure == "chain_mediator":
        return {"open_before": False, "open_after": False,
                "correct": conditioning == should, "role": "mediator"}
    if structure in ("collider", "m_bias", "descendant_of_collider"):
        return {"open_before": False, "open_after": True,
                "correct": conditioning == should, "role": "collider"}
    if structure == "front_door":
        return {"open_before": True, "open_after": True,
                "correct": False, "role": "front door"}
    return {"open_before": True, "open_after": True, "correct": conditioning == should,
            "role": "instrument"}


SPEC = make_spec(
    "causal.dag",
    Domain.CAUSAL,
    "graphical_models",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "diagnose",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.VISUAL_DERIVATION,
    controls=(
        select("structure", "confounder", STRUCTURES, group="graph"),
        toggle("condition_on_z", False, group="graph"),
        slider("effect", 1.0, -3.0, 3.0, 0.05, group="dgp"),
        slider("edge_strength", 1.2, 0.0, 3.0, 0.05, group="dgp"),
        int_slider("n", 800, 50, 20000, 50, group="dgp"),
        slider("noise", 1.0, 0.05, 5.0, 0.05, group="dgp"),
        toggle("show_scatter", True, group="views"),
        toggle("show_all_structures", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("confounder_uncontrolled", "violation", structure="confounder",
                 condition_on_z=False),
        scenario("confounder_controlled", "canonical", structure="confounder",
                 condition_on_z=True),
        scenario("mediator_uncontrolled", "canonical", structure="chain_mediator",
                 condition_on_z=False),
        scenario("mediator_controlled", "counterexample", structure="chain_mediator",
                 condition_on_z=True),
        scenario("collider_uncontrolled", "canonical", structure="collider",
                 condition_on_z=False),
        scenario("collider_controlled", "counterexample", structure="collider",
                 condition_on_z=True),
        scenario("m_bias", "counterexample", structure="m_bias", condition_on_z=True),
        scenario("collider_descendant", "counterexample",
                 structure="descendant_of_collider", condition_on_z=True),
        scenario("instrument", "compare_methods", structure="instrument"),
        scenario("front_door", "compare_methods", structure="front_door",
                 condition_on_z=True),
        scenario("null_effect", "null", effect=0.0, structure="collider",
                 condition_on_z=True),
    ),
    related=("econometrics.omitted_variable_bias", "causal.potential_outcomes",
             "econometrics.endogeneity_iv"),
    tags=("dag", "confounder", "collider", "mediator", "backdoor", "bad control",
          "d-separation"),
    aliases=("causal graph", "graphe causal", "الرسم السببي", "collider bias",
             "backdoor criterion"),
    backends=("numpy", "networkx"),
    required_extras=("causal",),
    references=(
        ref("Pearl, J. (2009). Causality: Models, Reasoning and Inference.", kind="book"),
        ref("Cinelli, C., Forney, A. and Pearl, J. (2024). A crash course in good and bad "
            "controls. Sociological Methods and Research 53(3).", kind="paper",
            doi="10.1177/00491241221099552"),
    ),
    curriculum_tags=("harvard.api114",),
)


class DAGLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        structure = str(p["structure"])
        conditioning = bool(p["condition_on_z"])
        n = int(p["n"])
        data = self._generate(structure, p, n, state.seed)
        est = self._estimate(data, conditioning, structure)
        status = path_status(structure, conditioning)

        res.dgp = ctx.t(
            "labs.dag.dgp",
            "{story} True effect of D on Y = {e}; edge strength {s}; n = {n}. "
            "Currently {cond} conditioning on Z.",
            story=GRAPHS[structure]["story"], e=fmt(p["effect"], 2),
            s=fmt(p["edge_strength"], 2), n=n,
            cond=("" if conditioning else "not"),
        )

        res.add_panel(ctx.panel(
            "graph", self._graph_figure(ctx, structure, conditioning, status),
            "labs.dag.figure.graph", evidence=EvidenceType.VISUAL_DERIVATION,
        ))
        res.add_panel(ctx.panel(
            "estimates", self._estimates_figure(ctx, est, p),
            "labs.dag.figure.estimates", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_scatter"] and structure in ("collider", "m_bias",
                                               "descendant_of_collider"):
            res.add_panel(ctx.panel(
                "collider_scatter", self._collider_figure(ctx, data),
                "labs.dag.figure.collider", tab="diagnostics",
                evidence=EvidenceType.COUNTEREXAMPLE,
            ))
        if p["show_all_structures"]:
            res.add_panel(ctx.panel(
                "all", self._all_structures_figure(ctx, p, state.seed),
                "labs.dag.figure.all", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("true_effect", ctx.t("labs.dag.metric.truth", "True effect of D on Y"),
                   float(p["effect"]))
        res.metric("uncontrolled", ctx.t("labs.dag.metric.uncontrolled",
                                         "Estimate NOT conditioning on Z"),
                   est["uncontrolled"], reference=float(p["effect"]))
        res.metric("controlled", ctx.t("labs.dag.metric.controlled",
                                       "Estimate conditioning on Z"),
                   est["controlled"], reference=float(p["effect"]))
        res.metric("current", ctx.t("labs.dag.metric.current",
                                    "Estimate with the current choice"),
                   est["current"], reference=float(p["effect"]))
        res.metric("current_bias", ctx.t("labs.dag.metric.bias",
                                         "Bias of the current choice"),
                   est["current"] - float(p["effect"]), reference=0.0)
        res.metric("role_of_z", ctx.t("labs.dag.metric.role", "Role of Z"),
                   status["role"])
        res.metric("correct_choice", ctx.t("labs.dag.metric.correct",
                                           "Is the current conditioning choice correct?"),
                   ctx.t("labs.dag.yes", "yes") if status["correct"]
                   else ctx.t("labs.dag.no", "no"))
        res.metric("should_control", ctx.t("labs.dag.metric.should",
                                           "Should you condition on Z here?"),
                   ctx.t("labs.dag.yes", "yes") if GRAPHS[structure]["control"]
                   else ctx.t("labs.dag.no", "no"))

        res.assume("exchangeability", ctx.t("assumptions.exchangeability"),
                   status["correct"] and structure not in ("front_door", "instrument"),
                   detail=ctx.t("labs.dag.assume.backdoor",
                                "The back-door criterion is satisfied only when the "
                                "conditioning set blocks every path from D to Y that "
                                "carries an arrow into D, and contains no descendant of D."))
        res.assume("graph_correct", ctx.t("labs.dag.assume.graph_label",
                                          "The graph itself is correct"), True,
                   detail=ctx.t("labs.dag.assume.graph",
                                "Everything here follows from the graph you supplied. No "
                                "amount of data can test the graph; it encodes what you "
                                "already believe about the world."))

        res.animations.append(self._animation(ctx, p, n, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.causal.dag.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.causal.dag.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"), ctx.t("concepts.causal.dag.math"),
                        kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.causal.dag.misconceptions"), kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"), ctx.t("concepts.causal.dag.warning"),
                    kind="warning")

        if not status["correct"]:
            res.warnings.append(ctx.t(
                "labs.dag.warn.wrong_control",
                "Z is a {role} here, so conditioning on it is {advice}. The current choice "
                "moves the estimate from {right} to {wrong} against a true effect of {t}.",
                role=status["role"],
                advice=("required" if GRAPHS[structure]["control"] else "a mistake"),
                right=fmt(est["controlled"] if GRAPHS[structure]["control"]
                          else est["uncontrolled"], 3),
                wrong=fmt(est["current"], 3), t=fmt(p["effect"], 3),
            ))
        return res

    @staticmethod
    def _generate(structure, p, n, seed):
        gen = rng(seed, "dag", structure)
        e = float(p["effect"])
        s = float(p["edge_strength"])
        noise = float(p["noise"])
        if structure == "confounder":
            z = gen.standard_normal(n)
            d = s * z + noise * gen.standard_normal(n)
            y = e * d + s * z + noise * gen.standard_normal(n)
        elif structure == "chain_mediator":
            d = gen.standard_normal(n)
            z = s * d + noise * gen.standard_normal(n)
            y = 0.5 * e * d + (0.5 * e / max(s, 0.05)) * z + noise * gen.standard_normal(n)
        elif structure == "collider":
            d = gen.standard_normal(n)
            y = e * d + noise * gen.standard_normal(n)
            z = s * d + s * y + noise * gen.standard_normal(n)
        elif structure == "m_bias":
            u1 = gen.standard_normal(n)
            u2 = gen.standard_normal(n)
            z = s * u1 + s * u2 + noise * gen.standard_normal(n)
            d = s * u1 + noise * gen.standard_normal(n)
            y = e * d + s * u2 + noise * gen.standard_normal(n)
        elif structure == "front_door":
            u = gen.standard_normal(n)
            d = s * u + noise * gen.standard_normal(n)
            z = s * d + noise * gen.standard_normal(n)
            y = (e / max(s, 0.05)) * z + s * u + noise * gen.standard_normal(n)
        elif structure == "descendant_of_collider":
            d = gen.standard_normal(n)
            y = e * d + noise * gen.standard_normal(n)
            c = s * d + s * y + noise * gen.standard_normal(n)
            z = c + 0.3 * gen.standard_normal(n)
        else:  # instrument
            u = gen.standard_normal(n)
            z = gen.standard_normal(n)
            d = s * z + s * u + noise * gen.standard_normal(n)
            y = e * d + s * u + noise * gen.standard_normal(n)
        return {"d": d, "y": y, "z": z}

    @staticmethod
    def _estimate(data, conditioning, structure):
        d, y, z = data["d"], data["y"], data["z"]
        n = y.size
        short = LM.ols(y, np.column_stack([np.ones(n), d]), names=("const", "d"))
        long = LM.ols(y, np.column_stack([np.ones(n), d, z]), names=("const", "d", "z"))
        out = {"uncontrolled": short.coef("d"), "controlled": long.coef("d")}
        out["current"] = out["controlled"] if conditioning else out["uncontrolled"]
        if structure == "instrument":
            Z = np.column_stack([np.ones(n), z])
            iv = LM.iv_2sls(y, np.column_stack([np.ones(n), d]), Z, endog_index=1,
                            names=("const", "d"))
            out["iv"] = iv.second_stage.coef("d")
        return out

    def _graph_figure(self, ctx, structure, conditioning, status):
        P.require_plotly()
        g = GRAPHS[structure]
        fig = ctx.figure("labs.dag.figure.graph", height=420, showlegend=False)
        for a, b in g["edges"]:
            (ax, ay), (bx, by) = g["nodes"][a], g["nodes"][b]
            dx, dy = bx - ax, by - ay
            norm = float(np.hypot(dx, dy)) or 1.0
            pad = 0.17
            role = "negative" if (a == "Z" or b == "Z") and conditioning else "baseline"
            if a == "D" and b == "Y":
                role = "positive"
            P.add_arrow(fig, ax + pad * dx / norm, ay + pad * dy / norm,
                        bx - pad * dx / norm, by - pad * dy / norm, "",
                        role, theme=ctx.theme, showlegend=False, width=2.6)
        for name, (nx, ny) in g["nodes"].items():
            unobserved = name.startswith("U")
            highlighted = name == "Z" and conditioning
            P.add_points(fig, [nx], [ny], name,
                         "warning" if highlighted
                         else ("muted" if unobserved else "primary"),
                         theme=ctx.theme, size=46,
                         symbol="square" if highlighted
                         else ("circle-open" if unobserved else "circle"),
                         showlegend=False)
            P.add_annotation(fig, nx, ny, name, theme=ctx.theme,
                             role="paper" if not unobserved else "foreground")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        note = ctx.t(
            "labs.dag.legend_graph",
            "{story} Z is a {role}. Open circles are unobserved. A square marks a variable "
            "you are conditioning on. Conditioning is {verdict} here.",
            story=g["story"], role=status["role"],
            verdict=(ctx.t("labs.dag.correct", "the right choice") if status["correct"]
                     else ctx.t("labs.dag.incorrect", "the wrong choice")),
        )
        P.add_legend_note(fig, note, theme=ctx.theme)
        return fig

    def _estimates_figure(self, ctx, est, p):
        go = P.require_plotly()
        labels = [ctx.t("labs.dag.trace.without", "without conditioning on Z"),
                  ctx.t("labs.dag.trace.with", "conditioning on Z")]
        values = [est["uncontrolled"], est["controlled"]]
        if "iv" in est:
            labels.append(ctx.t("labs.dag.trace.iv", "using Z as an instrument"))
            values.append(est["iv"])
        fig = ctx.figure(
            "labs.dag.figure.estimates",
            xaxis_title=ctx.t("labs.dag.axis.strategy", "Analysis choice"),
            yaxis_title=ctx.t("labs.dag.axis.estimate", "Estimated effect of D on Y"),
            height=380,
        )
        fig.add_trace(go.Bar(x=labels, y=values,
                             marker={"color": [ctx.color("primary"),
                                               ctx.color("secondary"),
                                               ctx.color("info")][: len(values)]},
                             text=[fmt(v, 3) for v in values], textposition="outside",
                             name=ctx.t("labs.common.axis.estimate")))
        P.add_hline(fig, float(p["effect"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.dag.legend_estimates",
            "Which bar lands on the dashed line is decided entirely by the graph, not by "
            "the data. Both analyses are computed from identical observations.",
        ), theme=ctx.theme)
        return fig

    def _collider_figure(self, ctx, data):
        d, y, z = data["d"], data["y"], data["z"]
        cut_lo, cut_hi = np.percentile(z, [40, 60])
        strat = (z > cut_lo) & (z < cut_hi)
        fig = ctx.figure(
            "labs.dag.figure.collider",
            xaxis_title="D", yaxis_title="Y", height=400,
        )
        P.add_points(fig, d, y,
                     ctx.t("labs.dag.trace.all", "all units: corr = {c}",
                           c=fmt(float(np.corrcoef(d, y)[0, 1]), 3)),
                     "muted", theme=ctx.theme, size=4, opacity=0.3)
        P.add_points(fig, d[strat], y[strat],
                     ctx.t("labs.dag.trace.stratum",
                           "units in a narrow band of Z: corr = {c}",
                           c=fmt(float(np.corrcoef(d[strat], y[strat])[0, 1]), 3)),
                     "negative", theme=ctx.theme, size=6, opacity=0.8)
        if strat.sum() > 5:
            f = LM.ols(y[strat], np.column_stack([np.ones(int(strat.sum())), d[strat]]))
            order = np.argsort(d[strat])
            P.add_curve(fig, d[strat][order], f.fitted_values[order],
                        ctx.t("labs.dag.trace.stratum_fit",
                              "slope within the band"),
                        "negative", theme=ctx.theme, width=3.0)
        P.add_legend_note(fig, ctx.t(
            "labs.dag.legend_collider",
            "Look at the coloured subset alone and a relationship appears that is not in "
            "the full cloud. Conditioning on a collider does not remove an association - "
            "it manufactures one.",
        ), theme=ctx.theme)
        return fig

    def _all_structures_figure(self, ctx, p, seed):
        go = P.require_plotly()
        rows_un, rows_ct, labels = [], [], []
        for structure in STRUCTURES:
            d = self._generate(structure, p, min(int(p["n"]), 3000), seed)
            e = self._estimate(d, False, structure)
            labels.append(structure.replace("_", " "))
            rows_un.append(e["uncontrolled"])
            rows_ct.append(e["controlled"])
        fig = ctx.figure(
            "labs.dag.figure.all",
            xaxis_title=ctx.t("labs.dag.axis.structure", "Graph structure"),
            yaxis_title=ctx.t("labs.dag.axis.estimate", "Estimated effect of D on Y"),
            height=400,
        )
        fig.add_trace(go.Bar(x=labels, y=rows_un,
                             marker={"color": ctx.color("primary")},
                             name=ctx.t("labs.dag.trace.without",
                                        "without conditioning on Z")))
        fig.add_trace(go.Bar(x=labels, y=rows_ct,
                             marker={"color": ctx.color("secondary")},
                             name=ctx.t("labs.dag.trace.with", "conditioning on Z")))
        P.add_hline(fig, float(p["effect"]), ctx.t("labs.common.trace.truth"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.dag.legend_all",
            "There is no universally safe rule. For a confounder you must condition; for a "
            "collider or a mediator you must not. 'Control for everything you have' is "
            "wrong in four of these seven graphs.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, p, n, seed):
        go = P.require_plotly()
        structure = str(p["structure"])
        quantiles = np.linspace(0.05, 0.95, 16)
        data = self._generate(structure, p, n, seed)
        d, y, z = data["d"], data["y"], data["z"]
        frames, steps = [], []
        width = 0.18
        for i, q in enumerate(quantiles):
            lo, hi = np.quantile(z, [max(q - width, 0.0), min(q + width, 1.0)])
            mask = (z >= lo) & (z <= hi)
            corr = (float(np.corrcoef(d[mask], y[mask])[0, 1])
                    if mask.sum() > 5 else float("nan"))
            frames.append(go.Frame(name=f"{q:.2f}", data=[
                go.Scatter(x=d[mask], y=y[mask]),
            ]))
            steps.append(AnimationStep(
                id=f"stratum_{i}", frame=i,
                title=ctx.t("labs.dag.anim.title",
                            "conditioning on Z near its {q} quantile", q=fmt(q, 2)),
                what_you_see=ctx.t("labs.dag.anim.see",
                                   "Only the units whose Z falls inside a narrow band."),
                what_changed=ctx.t("labs.dag.anim.changed",
                                   "The band moved to Z between {lo} and {hi}.",
                                   lo=fmt(lo, 2), hi=fmt(hi, 2)),
                why=ctx.t("labs.dag.anim.why",
                          "Conditioning means looking within a slice of Z. Whether that "
                          "blocks a path or opens one depends on where Z sits in the graph."),
                interpretation=ctx.t("labs.dag.anim.interpret",
                                     "Within this slice the D-Y correlation is {c}; over "
                                     "the whole sample it is {a}.",
                                     c=fmt(corr, 3),
                                     a=fmt(float(np.corrcoef(d, y)[0, 1]), 3)),
                conclusion=ctx.t("labs.dag.anim.conclude",
                                 "For a confounder the within-slice association is the "
                                 "causal one; for a collider it is an artefact of the "
                                 "slicing itself."),
                warning=ctx.t("labs.dag.anim.warn",
                              "Both cases look identical in the data. Only the graph "
                              "distinguishes them."),
                math="d-separation: a path is blocked at a collider unless you condition "
                     "on it or on its descendant",
                outputs={"quantile": round(float(q), 3),
                         "within_correlation": None if not np.isfinite(corr)
                         else round(corr, 4)},
                highlighted=("stratum",),
            ))
        fig = ctx.figure(
            "labs.dag.figure.animation",
            xaxis_title="D", yaxis_title="Y", height=390,
        )
        P.add_points(fig, d, y, ctx.t("labs.dag.trace.all_short", "all units"),
                     "muted", theme=ctx.theme, size=4, opacity=0.25)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                                 marker={"color": ctx.color("negative"), "size": 6},
                                 name=ctx.t("labs.dag.trace.stratum_short",
                                            "conditioned slice")))
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label="Z")
        return animation(
            "conditioning_slices", fig, steps,
            purpose=ctx.t("labs.dag.anim.purpose",
                          "Make 'conditioning on Z' a physical operation you can watch."),
            summary=ctx.t(
                "labs.dag.anim.summary",
                "Conditioning is slicing. Whether a slice reveals the causal relationship "
                "or invents one depends entirely on the graph - and the graph is an "
                "assumption you bring, not something the data can supply."),
            evidence=EvidenceType.VISUAL_DERIVATION,
        )


LAB = DAGLab(SPEC)
