"""Clustering: k-means, hierarchical linkage, and structure that is not there."""

from __future__ import annotations

from typing import Any

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...data.generators.ml import blobs
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "kmeans", "silhouette"]

SHAPES = ("blobs", "uniform", "elongated", "nested_rings", "unequal_sizes")


def kmeans(X, k, seed=0, iters=60, init="kmeans++"):
    """Lloyd's algorithm, returning the whole trajectory of centroids."""
    X = np.asarray(X, dtype=float)
    gen = rng(seed, "kmeans", k)
    n = X.shape[0]
    if init == "kmeans++" and k > 1:
        centres = [X[gen.integers(0, n)]]
        for _ in range(k - 1):
            d2 = np.min(np.sum((X[:, None, :] - np.array(centres)[None]) ** 2, axis=2),
                        axis=1)
            probs = d2 / max(float(d2.sum()), 1e-12)
            centres.append(X[gen.choice(n, p=probs)])
        centres = np.asarray(centres)
    else:
        centres = X[gen.choice(n, size=k, replace=False)]
    history = [centres.copy()]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = np.sum((X[:, None, :] - centres[None]) ** 2, axis=2)
        labels = np.argmin(d2, axis=1)
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                        else centres[j] for j in range(k)])
        history.append(new.copy())
        if np.allclose(new, centres, atol=1e-10):
            centres = new
            break
        centres = new
    d2 = np.sum((X[:, None, :] - centres[None]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)
    inertia = float(np.sum(np.min(d2, axis=1)))
    return {"centres": centres, "labels": labels, "inertia": inertia,
            "history": history, "iterations": len(history) - 1}


def silhouette(X, labels):
    """Mean silhouette coefficient (skipping degenerate single-point clusters)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return float("nan")
    d = np.sqrt(np.sum((X[:, None, :] - X[None]) ** 2, axis=2))
    scores = []
    for i in range(X.shape[0]):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = float(np.sum(d[i][own]) / (own.sum() - 1))
        b = min(float(np.mean(d[i][labels == c])) for c in uniq if c != labels[i])
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores)) if scores else float("nan")


SPEC = make_spec(
    "multivariate.clustering",
    Domain.MULTIVARIATE,
    "clustering",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "counterexample", "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("shape", "blobs", SHAPES, group="data"),
        int_slider("true_clusters", 3, 1, 8, 1, group="data"),
        int_slider("k", 3, 1, 10, 1, group="model"),
        int_slider("n", 300, 20, 5000, 10, group="data"),
        slider("spread", 1.0, 0.1, 5.0, 0.05, group="data"),
        slider("separation", 5.0, 0.0, 15.0, 0.1, group="data"),
        select("init", "kmeans++", ("kmeans++", "random"), group="model"),
        toggle("show_elbow", True, group="views"),
        toggle("show_silhouette", True, group="views"),
        toggle("show_restarts", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", shape="blobs", true_clusters=3, k=3),
        scenario("no_structure", "counterexample", shape="uniform", k=4),
        scenario("wrong_k", "misspecification", true_clusters=3, k=6),
        scenario("too_few_clusters", "misspecification", true_clusters=5, k=2),
        scenario("overlapping", "weak", separation=1.5, spread=1.5),
        scenario("well_separated", "strong", separation=12.0, spread=0.6),
        scenario("elongated", "violation", shape="elongated", k=2),
        scenario("nested_rings", "counterexample", shape="nested_rings", k=2),
        scenario("unequal_sizes", "violation", shape="unequal_sizes", k=3),
        scenario("bad_initialization", "sensitivity", init="random", k=5),
        scenario("small_sample", "small_sample", n=30),
    ),
    related=("multivariate.pca",),
    tags=("k-means", "hierarchical", "silhouette", "elbow", "inertia", "clustering"),
    aliases=("clustering", "classification automatique", "التصنيف العنقودي",
             "k means", "unsupervised"),
    backends=("numpy", "scikit-learn"),
    required_extras=("ai",),
    references=(
        ref("Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of "
            "Statistical Learning, chapter 14.", kind="book"),
        ref("Rousseeuw, P. J. (1987). Silhouettes. Journal of Computational and Applied "
            "Mathematics 20.", kind="paper", doi="10.1016/0377-0427(87)90125-7"),
    ),
    curriculum_tags=("dz.data_analysis2",),
)


class ClusteringLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        X, truth, dgp = self._generate(p, state.seed)
        k = int(p["k"])
        model = kmeans(X, k, state.seed, init=str(p["init"]))
        sil = silhouette(X, model["labels"]) if X.shape[0] <= 1200 else float("nan")

        res.dgp = dgp
        res.add_panel(ctx.panel(
            "clusters", self._cluster_figure(ctx, X, model, truth, p),
            "labs.clu.figure.clusters", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_elbow"]:
            elbow = self._elbow(X, state.seed, str(p["init"]))
            res.add_panel(ctx.panel(
                "elbow", self._elbow_figure(ctx, elbow, k, int(p["true_clusters"])),
                "labs.clu.figure.elbow", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
            best_sil = max((row for row in elbow if np.isfinite(row["silhouette"])),
                           key=lambda row: row["silhouette"], default=None)
            if best_sil:
                res.metric("best_k_silhouette",
                           ctx.t("labs.clu.metric.best_k",
                                 "k with the highest silhouette"), int(best_sil["k"]),
                           reference=int(p["true_clusters"]))
        if p["show_restarts"]:
            res.add_panel(ctx.panel(
                "restarts", self._restart_figure(ctx, X, k, str(p["init"]), state.seed),
                "labs.clu.figure.restarts", tab="diagnostics",
                evidence=EvidenceType.SIMULATION,
            ))

        res.metric("k", ctx.t("labs.clu.metric.k", "Clusters requested"), k,
                   reference=int(p["true_clusters"]) if truth is not None else None)
        res.metric("inertia", ctx.t("labs.clu.metric.inertia",
                                    "Within-cluster sum of squares"), model["inertia"],
                   note=ctx.t("labs.clu.metric.inertia_note",
                              "falls with every extra cluster by construction"))
        res.metric("silhouette", ctx.t("labs.clu.metric.silhouette",
                                       "Mean silhouette coefficient"), sil,
                   note=ctx.t("labs.clu.metric.silhouette_note",
                              "near 1 means tight and well separated; near 0 means the "
                              "partition is arbitrary"))
        res.metric("iterations", ctx.t("labs.clu.metric.iterations",
                                       "Iterations until convergence"),
                   model["iterations"])
        sizes = [int(np.sum(model["labels"] == j)) for j in range(k)]
        res.metric("smallest_cluster", ctx.t("labs.clu.metric.smallest",
                                             "Smallest cluster size"), min(sizes))
        res.metric("largest_cluster", ctx.t("labs.clu.metric.largest",
                                            "Largest cluster size"), max(sizes))
        if truth is not None:
            res.metric("agreement", ctx.t("labs.clu.metric.agreement",
                                          "Agreement with the true grouping"),
                       self._agreement(model["labels"], truth),
                       note=ctx.t("labs.clu.metric.agreement_note",
                                  "adjusted Rand index; 0 means no better than chance"))

        no_structure = str(p["shape"]) == "uniform"
        non_spherical = str(p["shape"]) in ("elongated", "nested_rings")
        res.assume("structure_exists", ctx.t("labs.clu.assume.structure_label",
                                             "There are real groups to find"),
                   not no_structure,
                   detail=ctx.t("labs.clu.assume.structure",
                                "k-means always returns k clusters. It cannot report that "
                                "the data have no grouping at all."),
                   consequence="" if not no_structure else ctx.t(
                       "labs.clu.assume.no_structure",
                       "The partition below is a slicing of a uniform cloud, presented with "
                       "exactly the same confidence as a real grouping."))
        res.assume("spherical", ctx.t("labs.clu.assume.spherical_label",
                                      "Clusters are roughly spherical and similar in size"),
                   not non_spherical and str(p["shape"]) != "unequal_sizes",
                   detail=ctx.t("labs.clu.assume.spherical",
                                "k-means assigns each point to the nearest centre, so its "
                                "boundaries are always straight lines. Elongated or nested "
                                "shapes cannot be recovered."),
                   consequence="" if not non_spherical else ctx.t(
                       "labs.clu.assume.non_spherical",
                       "The visible structure here cannot be captured by any partition into "
                       "convex cells, however good the diagnostics look."))
        res.assume("k_known", ctx.t("labs.clu.assume.k_label",
                                    "The number of clusters is chosen honestly"), True,
                   detail=ctx.t("labs.clu.assume.k",
                                "The elbow is not a test. Inertia always falls, and where "
                                "the bend appears is a matter of interpretation."))

        res.animations.append(self._animation(ctx, X, model, k))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.multivariate.clustering.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.multivariate.clustering.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.multivariate.clustering.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.multivariate.clustering.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.multivariate.clustering.warning"), kind="warning")

        if no_structure:
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.clu.counterexample",
                "These points are drawn uniformly at random - there are no groups at all. "
                "k-means still returned {k} tidy clusters with a silhouette of {s}. The "
                "algorithm cannot tell you that its answer is meaningless.",
                k=k, s=fmt(sil, 3),
            ), kind="warning")
            res.warnings.append(ctx.t(
                "labs.clu.warn.no_structure",
                "There is no real grouping in this data. Any interpretation of the clusters "
                "below would be describing the algorithm, not the world.",
            ))
        if non_spherical:
            res.warnings.append(ctx.t(
                "labs.clu.warn.shape",
                "The true structure here is not made of round blobs, so k-means cuts across "
                "it. This is a limitation of the distance-to-centre rule, not of the "
                "settings - a density- or graph-based method is needed instead.",
            ))
        return res

    @staticmethod
    def _generate(p, seed):
        shape = str(p["shape"])
        n = int(p["n"])
        gen = rng(seed, "cluster", shape)
        if shape == "blobs":
            d = blobs(n=n, n_clusters=int(p["true_clusters"]), spread=float(p["spread"]),
                      separation=float(p["separation"]), seed=seed)
            X = np.column_stack([d["x1"], d["x2"]])
            return X, d["label"].astype(int), d.dgp
        if shape == "uniform":
            X = gen.uniform(-6, 6, (n, 2))
            return X, None, (f"{n} points drawn uniformly at random on a square - "
                             "there is no cluster structure whatsoever.")
        if shape == "elongated":
            g = gen.standard_normal((n, 2)) * np.array([4.0, 0.35])
            offset = np.where(gen.random(n) < 0.5, -1.5, 1.5)
            X = g + np.column_stack([np.zeros(n), offset])
            return X, (offset > 0).astype(int), (
                f"Two long parallel bands separated vertically; n = {n}.")
        if shape == "nested_rings":
            theta = gen.uniform(0, 2 * np.pi, n)
            inner = gen.random(n) < 0.5
            r = np.where(inner, 1.5, 5.0) + gen.normal(0, 0.35, n)
            X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            return X, inner.astype(int), (
                f"Two concentric rings; n = {n}. The groups are obvious to the eye and "
                "invisible to any method that assigns points to the nearest centre.")
        sizes = np.array([0.7, 0.2, 0.1])
        counts = (sizes * n).astype(int)
        counts[0] += n - counts.sum()
        centres = np.array([[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]])
        spreads = np.array([2.4, 0.5, 0.5])
        parts, labels = [], []
        for j, c in enumerate(counts):
            parts.append(centres[j] + spreads[j] * gen.standard_normal((c, 2)))
            labels.append(np.full(c, j))
        return (np.vstack(parts), np.concatenate(labels),
                f"Three clusters of very unequal size ({counts.tolist()}) and spread; "
                f"n = {n}.")

    @staticmethod
    def _agreement(labels, truth):
        """Adjusted Rand index."""
        labels = np.asarray(labels)
        truth = np.asarray(truth)
        a = np.unique(labels)
        b = np.unique(truth)
        table = np.array([[np.sum((labels == i) & (truth == j)) for j in b] for i in a])
        n = labels.size

        def comb2(x):
            return x * (x - 1) / 2.0

        sum_ij = float(np.sum(comb2(table)))
        sum_i = float(np.sum(comb2(table.sum(axis=1))))
        sum_j = float(np.sum(comb2(table.sum(axis=0))))
        expected = sum_i * sum_j / max(comb2(n), 1e-12)
        maximum = 0.5 * (sum_i + sum_j)
        denom = maximum - expected
        return float((sum_ij - expected) / denom) if abs(denom) > 1e-12 else 0.0

    def _elbow(self, X, seed, init):
        rows = []
        for k in range(1, 11):
            m = kmeans(X, k, seed, init=init)
            sil = (silhouette(X, m["labels"])
                   if k > 1 and X.shape[0] <= 800 else float("nan"))
            rows.append({"k": k, "inertia": m["inertia"], "silhouette": sil})
        return rows

    def _cluster_figure(self, ctx, X, model, truth, p):
        go = P.require_plotly()
        fig = ctx.figure(
            "labs.clu.figure.clusters",
            xaxis_title="x1", yaxis_title="x2", height=450,
        )
        palette = ["primary", "secondary", "positive", "warning", "info", "negative",
                   "test", "validation", "training", "muted"]
        for j in range(int(p["k"])):
            mask = model["labels"] == j
            if mask.any():
                P.add_points(fig, X[mask, 0], X[mask, 1],
                             ctx.t("labs.clu.trace.cluster", "cluster {j}", j=j + 1),
                             palette[j % len(palette)], theme=ctx.theme, size=6,
                             opacity=0.65)
        P.add_points(fig, model["centres"][:, 0], model["centres"][:, 1],
                     ctx.t("labs.clu.trace.centres", "centroids"), "truth",
                     theme=ctx.theme, size=15, symbol="x")
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        note = ctx.t(
            "labs.clu.legend_clusters",
            "Every point belongs to its nearest centroid, so the boundaries are always "
            "straight lines. That single rule explains both what k-means does well and "
            "everything it cannot do.",
        )
        if truth is None:
            note += "  " + ctx.t("labs.clu.legend_no_truth",
                                 "There is no true grouping to compare against here.")
        P.add_legend_note(fig, note, theme=ctx.theme)
        return fig

    def _elbow_figure(self, ctx, elbow, current_k, true_k):
        make_subplots = P.SUBPLOT()
        go = P.require_plotly()
        ks = [row["k"] for row in elbow]
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            ctx.t("labs.clu.trace.elbow", "within-cluster sum of squares"),
            ctx.t("labs.clu.trace.sil", "mean silhouette"),
        ))
        fig.add_trace(go.Scatter(x=ks, y=[row["inertia"] for row in elbow],
                                 mode="lines+markers",
                                 line={"color": ctx.color("primary"), "width": 2.6},
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ks, y=[row["silhouette"] for row in elbow],
                                 mode="lines+markers",
                                 line={"color": ctx.color("secondary"), "width": 2.6},
                                 showlegend=False), row=1, col=2)
        for col in (1, 2):
            fig.add_vline(x=current_k, line={"color": ctx.color("warning"),
                                             "dash": "dash"}, row=1, col=col)
        layout = ctx.theme.plotly_layout(locale=ctx.locale)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        fig.update_layout(height=360, **layout)
        P.add_legend_note(fig, ctx.t(
            "labs.clu.legend_elbow",
            "The left curve always decreases, so its 'elbow' is a judgement call. The "
            "silhouette on the right can genuinely peak - but it peaks on random data too.",
        ), theme=ctx.theme)
        return fig

    def _restart_figure(self, ctx, X, k, init, seed):
        inertias, sils = [], []
        for r in range(25):
            m = kmeans(X, k, int(seed) * 131071 + r, init=init)
            inertias.append(m["inertia"])
            sils.append(silhouette(X, m["labels"]) if X.shape[0] <= 800 else np.nan)
        fig = ctx.figure(
            "labs.clu.figure.restarts",
            xaxis_title=ctx.t("labs.clu.axis.restart", "Random restart"),
            yaxis_title=ctx.t("labs.clu.metric.inertia",
                              "Within-cluster sum of squares"),
            height=340,
        )
        P.add_points(fig, np.arange(1, len(inertias) + 1), inertias,
                     ctx.t("labs.clu.trace.restart_result",
                           "result of one random restart"),
                     "primary", theme=ctx.theme, size=8)
        P.add_hline(fig, float(np.min(inertias)),
                    ctx.t("labs.clu.trace.best", "best found"), "positive",
                    theme=ctx.theme, dash="dash")
        spread = float(np.max(inertias) - np.min(inertias))
        P.add_legend_note(fig, ctx.t(
            "labs.clu.legend_restarts",
            "Lloyd's algorithm only finds a local optimum, so the answer depends on where "
            "it started. Spread across restarts here: {s}. If the points are not all at the "
            "same height, a single run is not a result.",
            s=fmt(spread, 3),
        ) + "  " + ctx.t("labs.common.note.simulation"), theme=ctx.theme)
        return fig

    def _animation(self, ctx, X, model, k):
        go = P.require_plotly()
        history = model["history"]
        frames, steps = [], []
        palette = [ctx.color(c) for c in ("primary", "secondary", "positive", "warning",
                                          "info", "negative", "test", "validation",
                                          "training", "muted")]
        for i, centres in enumerate(history):
            d2 = np.sum((X[:, None, :] - centres[None]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            inertia = float(np.sum(np.min(d2, axis=1)))
            frames.append(go.Frame(name=str(i), data=[
                go.Scatter(x=X[:, 0], y=X[:, 1],
                           marker={"color": [palette[j % len(palette)] for j in labels]}),
                go.Scatter(x=centres[:, 0], y=centres[:, 1]),
            ]))
            moved = (float(np.max(np.abs(centres - history[i - 1]))) if i else float("nan"))
            steps.append(AnimationStep(
                id=f"iter_{i}", frame=i,
                title=ctx.t("labs.clu.anim.title", "iteration {i}", i=i),
                what_you_see=ctx.t("labs.clu.anim.see",
                                   "Points coloured by their nearest centroid, and the "
                                   "centroids themselves."),
                what_changed=(ctx.t("labs.clu.anim.changed_init",
                                    "Initial centroids were placed.") if i == 0 else
                              ctx.t("labs.clu.anim.changed",
                                    "Centroids moved by up to {m}.", m=fmt(moved, 4))),
                why=ctx.t("labs.clu.anim.why",
                          "Lloyd's algorithm alternates two steps: assign every point to "
                          "its closest centre, then move each centre to the mean of its "
                          "points. Each step can only lower the objective."),
                interpretation=ctx.t("labs.clu.anim.interpret",
                                     "Within-cluster sum of squares is now {v}.",
                                     v=fmt(inertia, 3)),
                conclusion=ctx.t("labs.clu.anim.conclude",
                                 "The algorithm stops when nothing moves - at a LOCAL "
                                 "optimum that depends on where it started."),
                warning=ctx.t("labs.clu.anim.warn",
                              "Convergence is guaranteed. Correctness is not."),
                math="minimise sum over clusters of sum ||x - centre||^2",
                outputs={"iteration": i, "inertia": round(inertia, 4),
                         "max_centroid_move": None if not np.isfinite(moved)
                         else round(moved, 5)},
                highlighted=("assignments", "centroids"),
            ))
        fig = ctx.figure(
            "labs.clu.figure.animation",
            xaxis_title="x1", yaxis_title="x2", height=430,
        )
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker={"color": ctx.color("muted"), "size": 6,
                                         "opacity": 0.6},
                                 name=ctx.t("labs.clu.trace.points", "points")))
        fig.add_trace(go.Scatter(x=history[0][:, 0], y=history[0][:, 1], mode="markers",
                                 marker={"color": ctx.color("truth"), "size": 15,
                                         "symbol": "x"},
                                 name=ctx.t("labs.clu.trace.centres", "centroids")))
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        build_frames(fig, frames, duration=560, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.common.axis.iteration"))
        return animation(
            "lloyd", fig, steps,
            purpose=ctx.t("labs.clu.anim.purpose",
                          "Watch assignment and update alternate until nothing moves."),
            summary=ctx.t(
                "labs.clu.anim.summary",
                "k-means is two alternating steps that each reduce one objective, so it "
                "always stops. Where it stops depends on where it began, and whether the "
                "stopping point means anything depends on whether the data had groups in "
                "the first place - which the algorithm never checks."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = ClusteringLab(SPEC)
