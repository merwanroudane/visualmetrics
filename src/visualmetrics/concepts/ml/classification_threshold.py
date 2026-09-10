"""Classification thresholds, ROC, precision-recall, cost and calibration."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, pct, ref, scenario,
    seed_control, select, slider, toggle,
)
from ...data.generators.ml import classification_scores

__all__ = ["LAB", "SPEC", "confusion_at", "roc_curve", "pr_curve"]


def confusion_at(y, score, threshold):
    pred = np.asarray(score) >= threshold
    y = np.asarray(y) > 0
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def roc_curve(y, score):
    order = np.argsort(-np.asarray(score))
    y_sorted = (np.asarray(y) > 0)[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = np.concatenate([[0.0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0.0], fp / max(fp[-1], 1)])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pr_curve(y, score):
    order = np.argsort(-np.asarray(score))
    y_sorted = (np.asarray(y) > 0)[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, tp.size + 1)
    precision = tp / k
    recall = tp / max(tp[-1], 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


SPEC = make_spec(
    "ml.classification_threshold",
    Domain.ML,
    "classification",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "diagnose", "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        slider("threshold", 0.75, -4.0, 6.0, 0.01, group="decision"),
        slider("separation", 1.5, 0.0, 5.0, 0.05, group="data"),
        slider("positive_rate", 0.5, 0.005, 0.95, 0.005, group="data"),
        slider("noise_scale", 1.0, 0.2, 3.0, 0.05, group="data"),
        slider("miscalibration", 0.0, 0.0, 3.0, 0.05, group="data"),
        slider("cost_ratio", 1.0, 0.05, 20.0, 0.05, group="decision"),
        int_slider("n", 800, 40, 40000, 20, group="data"),
        toggle("show_roc", True, group="views"),
        toggle("show_pr", True, group="views"),
        toggle("show_calibration", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", separation=1.5, threshold=0.75),
        scenario("perfect_separation", "boundary", separation=5.0),
        scenario("no_signal", "null", separation=0.0),
        scenario("weak_signal", "weak", separation=0.6),
        scenario("rare_positives", "counterexample", positive_rate=0.02, n=4000),
        scenario("accuracy_trap", "counterexample", positive_rate=0.01, separation=1.0,
                 threshold=5.0, n=4000),
        scenario("costly_misses", "compare_methods", cost_ratio=10.0),
        scenario("costly_false_alarms", "compare_methods", cost_ratio=0.1),
        scenario("miscalibrated", "violation", miscalibration=2.0),
        scenario("small_sample", "small_sample", n=60),
        scenario("large_sample", "large_sample", n=20000),
    ),
    related=("econometrics.logit_probit", "ml.bias_variance"),
    tags=("roc", "auc", "precision", "recall", "confusion matrix", "calibration",
          "threshold", "class imbalance"),
    aliases=("roc curve", "seuil de classification", "عتبة التصنيف",
             "precision recall", "confusion matrix"),
    backends=("numpy", "scikit-learn"),
    references=(
        ref("Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition "
            "Letters 27(8).", kind="paper", doi="10.1016/j.patrec.2005.10.010"),
        ref("Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more "
            "informative than the ROC plot on imbalanced datasets. PLoS ONE 10(3).",
            kind="paper", doi="10.1371/journal.pone.0118432"),
    ),
    curriculum_tags=("stanford.stats202",),
)


class ThresholdLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        data = classification_scores(
            n=int(p["n"]), separation=float(p["separation"]),
            positive_rate=float(p["positive_rate"]),
            noise_scale=float(p["noise_scale"]),
            miscalibration=float(p["miscalibration"]), seed=state.seed,
        )
        y, score, prob = data["y"], data["score"], data["prob"]
        thr = float(p["threshold"])
        cm = confusion_at(y, score, thr)
        fpr, tpr, auc = roc_curve(y, score)
        recall, precision, ap = pr_curve(y, score)
        cost = self._cost_curve(y, score, float(p["cost_ratio"]))

        res.data = data
        res.dgp = data.dgp

        res.add_panel(ctx.panel(
            "scores", self._scores_figure(ctx, y, score, thr, cm),
            "labs.thr.figure.scores", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        res.add_panel(ctx.panel(
            "confusion", self._confusion_figure(ctx, cm),
            "labs.thr.figure.confusion", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_roc"]:
            res.add_panel(ctx.panel(
                "roc", self._roc_figure(ctx, fpr, tpr, auc, cm),
                "labs.thr.figure.roc", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_pr"]:
            res.add_panel(ctx.panel(
                "pr", self._pr_figure(ctx, recall, precision, ap, cm, float(np.mean(y))),
                "labs.thr.figure.pr", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        res.add_panel(ctx.panel(
            "cost", self._cost_figure(ctx, cost, thr, float(p["cost_ratio"])),
            "labs.thr.figure.cost", tab="diagnostics",
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_calibration"]:
            res.add_panel(ctx.panel(
                "calibration", self._calibration_figure(ctx, y, prob),
                "labs.thr.figure.calibration", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        base_rate = float(np.mean(y))
        res.metric("base_rate", ctx.t("labs.thr.metric.base",
                                      "Share of positives in the data"), base_rate)
        for key, label in (("tp", "True positives"), ("fp", "False positives"),
                           ("fn", "False negatives"), ("tn", "True negatives")):
            res.metric(key, ctx.t(f"labs.thr.metric.{key}", label), cm[key])
        res.metric("accuracy", ctx.t("labs.thr.metric.accuracy", "Accuracy"),
                   cm["accuracy"],
                   note=ctx.t("labs.thr.metric.accuracy_note",
                              "predicting the majority class always would give {v}",
                              v=pct(max(base_rate, 1 - base_rate))))
        res.metric("precision", ctx.t("labs.thr.metric.precision", "Precision"),
                   cm["precision"])
        res.metric("recall", ctx.t("labs.thr.metric.recall", "Recall"), cm["recall"])
        res.metric("specificity", ctx.t("labs.thr.metric.specificity", "Specificity"),
                   cm["specificity"])
        res.metric("f1", ctx.t("labs.thr.metric.f1", "F1 score"), cm["f1"])
        res.metric("auc", ctx.t("labs.thr.metric.auc", "Area under the ROC curve"), auc,
                   note=ctx.t("labs.thr.metric.auc_note",
                              "threshold-free, and blind to calibration"))
        res.metric("average_precision", ctx.t("labs.thr.metric.ap",
                                              "Average precision (area under PR)"), ap,
                   note=ctx.t("labs.thr.metric.ap_note",
                              "a random classifier scores {v} here, not 0.5",
                              v=fmt(base_rate, 3)))
        best_cost = min(cost, key=lambda row: row["cost"])
        res.metric("optimal_threshold", ctx.t("labs.thr.metric.optimal",
                                              "Cost-minimising threshold"),
                   best_cost["threshold"])
        res.metric("current_cost", ctx.t("labs.thr.metric.cost",
                                         "Expected cost at the current threshold"),
                   self._cost_at(y, score, thr, float(p["cost_ratio"])))
        res.metric("brier", ctx.t("labs.thr.metric.brier",
                                  "Brier score of the predicted probabilities"),
                   float(np.mean((prob - y) ** 2)),
                   note=ctx.t("labs.thr.metric.brier_note",
                              "lower is better; it penalises miscalibration"))
        res.metric("calibration_error", ctx.t("labs.thr.metric.calibration",
                                              "Mean absolute calibration error"),
                   self._calibration_error(y, prob))

        calibrated = bool(data.meta["calibrated"])
        imbalanced = base_rate < 0.1 or base_rate > 0.9
        res.assume("calibrated", ctx.t("labs.thr.assume.calibrated_label",
                                       "Predicted probabilities are calibrated"),
                   calibrated,
                   detail=ctx.t("labs.thr.assume.calibrated",
                                "A model can rank perfectly (high AUC) while its "
                                "probabilities are systematically wrong. Ranking and "
                                "calibration are different properties."))
        res.assume("balanced", ctx.t("labs.thr.assume.balanced_label",
                                     "Classes are not severely imbalanced"),
                   not imbalanced,
                   detail=ctx.t("labs.thr.assume.balanced",
                                "With {v} positives, accuracy and the ROC curve both look "
                                "flattering. The precision-recall curve does not.",
                                v=pct(base_rate)))
        res.assume("costs_known", ctx.t("labs.thr.assume.costs_label",
                                        "The relative cost of the two errors is known"),
                   True,
                   detail=ctx.t("labs.thr.assume.costs",
                                "There is no statistically optimal threshold. The optimum "
                                "depends on what a false alarm costs relative to a missed "
                                "case, which is a decision, not an estimate."))

        res.animations.append(self._animation(ctx, y, score, float(p["cost_ratio"])))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.ml.classification_threshold.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.ml.classification_threshold.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.ml.classification_threshold.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.ml.classification_threshold.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.ml.classification_threshold.warning"), kind="warning")

        if imbalanced:
            trivial = max(base_rate, 1 - base_rate)
            res.explain("counterexample", ctx.t("modes.counterexample"), ctx.t(
                "labs.thr.counterexample",
                "Accuracy here is {a}. A model that ignores every input and always predicts "
                "the majority class scores {t}. Under imbalance, accuracy measures the base "
                "rate far more than it measures the model.",
                a=pct(cm["accuracy"]), t=pct(trivial),
            ), kind="warning")
            res.warnings.append(ctx.t(
                "labs.thr.warn.imbalance",
                "With {v} positives, read the precision-recall panel rather than the ROC "
                "panel: the ROC curve barely reacts to false positives when negatives are "
                "abundant.", v=pct(base_rate),
            ))
        if not calibrated:
            res.warnings.append(ctx.t(
                "labs.thr.warn.calibration",
                "AUC is {a} but the calibration curve is off the diagonal. The model ranks "
                "well and its probabilities are still not usable as probabilities.",
                a=fmt(auc, 3),
            ))
        return res

    @staticmethod
    def _cost_at(y, score, threshold, cost_ratio):
        cm = confusion_at(y, score, threshold)
        return float(cm["fp"] + cost_ratio * cm["fn"]) / max(len(y), 1)

    def _cost_curve(self, y, score, cost_ratio):
        grid = np.linspace(float(np.min(score)) - 0.1, float(np.max(score)) + 0.1, 160)
        return [{"threshold": float(t),
                 "cost": self._cost_at(y, score, float(t), cost_ratio)} for t in grid]

    @staticmethod
    def _calibration_error(y, prob, bins: int = 10):
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
        errs, weights = [], []
        for b in range(bins):
            m = idx == b
            if m.sum() > 3:
                errs.append(abs(float(np.mean(prob[m])) - float(np.mean(y[m]))))
                weights.append(int(m.sum()))
        if not errs:
            return float("nan")
        return float(np.average(errs, weights=weights))

    def _scores_figure(self, ctx, y, score, thr, cm):
        fig = ctx.figure(
            "labs.thr.figure.scores",
            xaxis_title=ctx.t("labs.thr.axis.score", "Model score"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=430,
        )
        P.add_histogram(fig, score[y == 0],
                        ctx.t("labs.thr.trace.negatives", "actual negatives"),
                        "control", theme=ctx.theme, nbins=55, opacity=0.55)
        P.add_histogram(fig, score[y > 0],
                        ctx.t("labs.thr.trace.positives", "actual positives"),
                        "treatment", theme=ctx.theme, nbins=55, opacity=0.55)
        P.add_vline(fig, thr,
                    ctx.t("labs.thr.trace.threshold", "threshold = {t}", t=fmt(thr, 2)),
                    "warning", theme=ctx.theme, dash="solid")
        P.add_legend_note(fig, ctx.t(
            "labs.thr.legend_scores",
            "Everything right of the line is predicted positive. Negatives to the right are "
            "false alarms ({fp}); positives to the left are missed cases ({fn}). Moving the "
            "line trades one for the other and can never reduce both.",
            fp=cm["fp"], fn=cm["fn"],
        ), theme=ctx.theme)
        return fig

    def _confusion_figure(self, ctx, cm):
        go = P.require_plotly()
        z = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig = ctx.figure(
            "labs.thr.figure.confusion",
            xaxis_title=ctx.t("labs.thr.axis.predicted", "Predicted"),
            yaxis_title=ctx.t("labs.thr.axis.actual", "Actual"),
            height=380,
        )
        fig.add_trace(go.Heatmap(
            z=z,
            x=[ctx.t("labs.thr.trace.pred_neg", "negative"),
               ctx.t("labs.thr.trace.pred_pos", "positive")],
            y=[ctx.t("labs.thr.trace.act_neg", "negative"),
               ctx.t("labs.thr.trace.act_pos", "positive")],
            colorscale=ctx.theme.colorscale,
            text=[[f"TN = {cm['tn']}", f"FP = {cm['fp']}"],
                  [f"FN = {cm['fn']}", f"TP = {cm['tp']}"]],
            texttemplate="%{text}", showscale=False,
        ))
        P.add_legend_note(fig, ctx.t(
            "labs.thr.legend_confusion",
            "Precision reads down the predicted-positive column ({p}); recall reads across "
            "the actual-positive row ({r}). Every headline metric is a different ratio "
            "taken from these four numbers.",
            p=pct(cm["precision"]), r=pct(cm["recall"]),
        ), theme=ctx.theme)
        return fig

    def _roc_figure(self, ctx, fpr, tpr, auc, cm):
        fig = ctx.figure(
            "labs.thr.figure.roc",
            xaxis_title=ctx.t("labs.common.axis.false_positive"),
            yaxis_title=ctx.t("labs.common.axis.true_positive"),
            height=390,
        )
        P.add_curve(fig, fpr, tpr,
                    ctx.t("labs.thr.trace.roc", "ROC curve (AUC = {a})", a=fmt(auc, 3)),
                    "primary", theme=ctx.theme)
        P.add_curve(fig, [0, 1], [0, 1],
                    ctx.t("labs.thr.trace.chance", "a coin flip"),
                    "baseline", theme=ctx.theme, dash="dash")
        P.add_points(fig, [cm["fpr"]], [cm["tpr"]],
                     ctx.t("labs.thr.trace.current_point", "current threshold"),
                     "warning", theme=ctx.theme, size=13)
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1.02], scaleanchor="x", scaleratio=1)
        P.add_legend_note(fig, ctx.t(
            "labs.thr.legend_roc",
            "The curve is the model; the marked point is your decision. AUC summarises the "
            "curve and says nothing about which point you should pick.",
        ), theme=ctx.theme)
        return fig

    def _pr_figure(self, ctx, recall, precision, ap, cm, base_rate):
        fig = ctx.figure(
            "labs.thr.figure.pr",
            xaxis_title=ctx.t("labs.common.axis.recall"),
            yaxis_title=ctx.t("labs.common.axis.precision"),
            height=390,
        )
        P.add_curve(fig, recall, precision,
                    ctx.t("labs.thr.trace.pr",
                          "precision-recall curve (average precision = {a})",
                          a=fmt(ap, 3)),
                    "secondary", theme=ctx.theme)
        P.add_hline(fig, base_rate,
                    ctx.t("labs.thr.trace.pr_chance",
                          "a coin flip: precision = the base rate"),
                    "baseline", theme=ctx.theme, dash="dash")
        P.add_points(fig, [cm["recall"]], [cm["precision"]],
                     ctx.t("labs.thr.trace.current_point", "current threshold"),
                     "warning", theme=ctx.theme, size=13)
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1.02])
        P.add_legend_note(fig, ctx.t(
            "labs.thr.legend_pr",
            "The baseline here is the base rate, not 0.5. Under imbalance this curve reveals "
            "problems the ROC curve smooths away, because it never counts true negatives.",
        ), theme=ctx.theme)
        return fig

    def _cost_figure(self, ctx, cost, thr, ratio):
        thresholds = [row["threshold"] for row in cost]
        costs = [row["cost"] for row in cost]
        best = min(cost, key=lambda row: row["cost"])
        fig = ctx.figure(
            "labs.thr.figure.cost",
            xaxis_title=ctx.t("labs.common.axis.threshold"),
            yaxis_title=ctx.t("labs.thr.axis.cost", "Expected cost per observation"),
            height=360,
        )
        P.add_curve(fig, thresholds, costs,
                    ctx.t("labs.thr.trace.cost",
                          "cost when a miss is {r} times a false alarm", r=fmt(ratio, 2)),
                    "primary", theme=ctx.theme)
        P.add_vline(fig, best["threshold"],
                    ctx.t("labs.thr.trace.optimum", "cost-minimising threshold"),
                    "positive", theme=ctx.theme)
        P.add_vline(fig, thr, ctx.t("labs.common.trace.current"), "warning",
                    theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.thr.legend_cost",
            "Change the cost ratio and the optimum moves. This is the honest statement of "
            "the problem: the threshold is a decision about consequences, not a property "
            "of the model.",
        ), theme=ctx.theme)
        return fig

    def _calibration_figure(self, ctx, y, prob, bins: int = 12):
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
        xs, ys, ns = [], [], []
        for b in range(bins):
            m = idx == b
            if m.sum() > 3:
                xs.append(float(np.mean(prob[m])))
                ys.append(float(np.mean(y[m])))
                ns.append(int(m.sum()))
        fig = ctx.figure(
            "labs.thr.figure.calibration",
            xaxis_title=ctx.t("labs.common.axis.predicted_prob"),
            yaxis_title=ctx.t("labs.common.axis.observed_freq"),
            height=370,
        )
        P.add_points(fig, xs, ys,
                     ctx.t("labs.thr.trace.calibration", "observed vs predicted"),
                     "primary", theme=ctx.theme,
                     size=[max(6, min(24, 6 + n / 40)) for n in ns])
        P.add_curve(fig, [0, 1], [0, 1],
                    ctx.t("labs.thr.trace.perfect", "perfect calibration"),
                    "truth", theme=ctx.theme, dash="dash")
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
        P.add_legend_note(fig, ctx.t(
            "labs.thr.legend_calibration",
            "Points below the diagonal mean the model is over-confident: when it says 80%, "
            "the event happens less often than that. Ranking can be excellent and "
            "calibration still poor.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, y, score, cost_ratio):
        go = P.require_plotly()
        thresholds = np.linspace(float(np.max(score)) + 0.2,
                                 float(np.min(score)) - 0.2, 24)
        fpr, tpr, auc = roc_curve(y, score)
        frames, steps = [], []
        for i, t in enumerate(thresholds):
            cm = confusion_at(y, score, float(t))
            frames.append(go.Frame(name=f"{t:.2f}", data=[
                go.Scatter(x=[cm["fpr"]], y=[cm["tpr"]]),
            ]))
            steps.append(AnimationStep(
                id=f"thr_{i}", frame=i,
                title=ctx.t("labs.thr.anim.title", "threshold = {t}", t=fmt(t, 2)),
                what_you_see=ctx.t("labs.thr.anim.see",
                                   "The ROC curve with the point produced by the current "
                                   "threshold."),
                what_changed=ctx.t("labs.thr.anim.changed",
                                   "The decision threshold moved to {t}.", t=fmt(t, 2)),
                why=ctx.t("labs.thr.anim.why",
                          "Lowering the threshold labels more cases positive, so both the "
                          "true-positive rate and the false-positive rate rise together. "
                          "They cannot be separated."),
                interpretation=ctx.t("labs.thr.anim.interpret",
                                     "Recall {r}, precision {p}, {fp} false alarms and {fn} "
                                     "missed cases; cost {c} per observation.",
                                     r=pct(cm["recall"]), p=pct(cm["precision"]),
                                     fp=cm["fp"], fn=cm["fn"],
                                     c=fmt(self._cost_at(y, score, float(t), cost_ratio), 4)),
                conclusion=ctx.t("labs.thr.anim.conclude",
                                 "Sliding along the curve is all a threshold can do. Moving "
                                 "the curve itself requires a better model."),
                warning=ctx.t("labs.thr.anim.warn",
                              "AUC is unchanged in every frame ({a}) - it is a property of "
                              "the ranking, not of the decision.", a=fmt(auc, 3)),
                math="TPR = TP/(TP+FN), FPR = FP/(FP+TN); optimal odds = "
                     "cost_FP (1-p) / (cost_FN p)",
                outputs={"threshold": round(float(t), 4), "tpr": round(cm["tpr"], 4),
                         "fpr": round(cm["fpr"], 4),
                         "precision": round(cm["precision"], 4),
                         "cost": round(self._cost_at(y, score, float(t), cost_ratio), 5)},
                highlighted=("roc_point",),
            ))
        fig = ctx.figure(
            "labs.thr.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.false_positive"),
            yaxis_title=ctx.t("labs.common.axis.true_positive"),
            height=400,
        )
        P.add_curve(fig, fpr, tpr,
                    ctx.t("labs.thr.trace.roc", "ROC curve (AUC = {a})", a=fmt(auc, 3)),
                    "primary", theme=ctx.theme)
        P.add_curve(fig, [0, 1], [0, 1], ctx.t("labs.thr.trace.chance", "a coin flip"),
                    "baseline", theme=ctx.theme, dash="dash")
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                 marker={"color": ctx.color("warning"), "size": 15},
                                 name=ctx.t("labs.thr.trace.current_point",
                                            "current threshold")))
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1.02], scaleanchor="x", scaleratio=1)
        build_frames(fig, frames, duration=380, reduced_motion=ctx.reduced_motion,
                     slider_label=ctx.t("labs.common.axis.threshold"))
        return animation(
            "threshold_sweep", fig, steps,
            purpose=ctx.t("labs.thr.anim.purpose",
                          "Separate what the model gives you from what you decide."),
            summary=ctx.t(
                "labs.thr.anim.summary",
                "A classifier outputs a ranking; a threshold turns it into decisions. The "
                "curve is fixed by the model, the point on it is chosen by you, and the "
                "right choice depends on costs that no dataset contains."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = ThresholdLab(SPEC)
