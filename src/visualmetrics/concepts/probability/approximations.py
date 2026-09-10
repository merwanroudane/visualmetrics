"""Distribution approximations, with the error plotted rather than assumed away."""

from __future__ import annotations

from typing import Any

from scipy import stats

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

__all__ = ["LAB", "SPEC", "CASES"]

CASES = ("binomial_to_normal", "binomial_to_poisson", "poisson_to_normal",
         "t_to_normal", "chi2_to_normal", "hypergeometric_to_binomial")


SPEC = make_spec(
    "probability.approximations",
    Domain.PROBABILITY,
    "convergence_of_distributions",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "code", "quiz", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("case", "binomial_to_normal", CASES, group="approximation"),
        int_slider("n", 30, 1, 2000, 1, group="parameters"),
        slider("p", 0.35, 0.001, 0.999, 0.001, group="parameters"),
        slider("lam", 5.0, 0.05, 200.0, 0.05, group="parameters"),
        slider("df", 5.0, 1.0, 300.0, 1.0, group="parameters"),
        int_slider("population", 200, 10, 5000, 10, group="parameters"),
        toggle("continuity_correction", True, group="approximation"),
        toggle("show_error", True, group="views"),
        toggle("show_tail", True, group="views"),
        slider("tail_probability", 0.05, 0.0005, 0.5, 0.0005, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", case="binomial_to_normal", n=30, p=0.35),
        scenario("rule_of_thumb_ok", "positive", n=100, p=0.5),
        scenario("rule_of_thumb_fails", "counterexample", n=30, p=0.02),
        scenario("no_continuity_correction", "compare_methods", continuity_correction=False,
                 n=20, p=0.4),
        scenario("binomial_to_poisson", "compare_methods", case="binomial_to_poisson",
                 n=200, p=0.02),
        scenario("poisson_to_normal", "compare_methods", case="poisson_to_normal", lam=25.0),
        scenario("poisson_small_lambda", "boundary", case="poisson_to_normal", lam=1.0),
        scenario("t_small_df", "small_sample", case="t_to_normal", df=3.0),
        scenario("t_large_df", "large_sample", case="t_to_normal", df=200.0),
        scenario("chi2_skewed", "high_noise", case="chi2_to_normal", df=3.0),
        scenario("sampling_fraction_large", "violation",
                 case="hypergeometric_to_binomial", population=40, n=20, p=0.4),
        scenario("sampling_fraction_small", "positive",
                 case="hypergeometric_to_binomial", population=4000, n=20, p=0.4),
    ),
    prerequisites=("probability.distributions",),
    related=("inference.clt",),
    next_concepts=("inference.clt",),
    tags=("approximation", "continuity correction", "convergence", "limit"),
    aliases=("binomial normal approximation", "التقريب", "approximation",
             "continuity correction"),
    backends=("scipy",),
    references=(
        ref("Feller, W. (1968). An Introduction to Probability Theory and Its "
            "Applications, volume 1.", kind="book"),
    ),
    curriculum_tags=("dz.stat3",),
)


class ApproximationLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        case = str(p["case"])
        exact, approx, x, discrete, label = self._pair(case, p)
        cc = bool(p["continuity_correction"]) and discrete

        exact_cdf = exact.cdf(x)
        approx_cdf = approx.cdf(x + 0.5) if cc else approx.cdf(x)
        error = approx_cdf - exact_cdf
        max_err = float(np.max(np.abs(error)))

        res.dgp = ctx.t(
            "labs.approx.dgp",
            "Exact distribution compared with its {label} approximation over the whole "
            "support{cc}.", label=label,
            cc=(", with a continuity correction" if cc else ""),
        )

        res.add_panel(ctx.panel(
            "overlay", self._overlay_figure(ctx, exact, approx, x, discrete, cc, label),
            "labs.approx.figure.overlay", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_error"]:
            res.add_panel(ctx.panel(
                "error", self._error_figure(ctx, x, error, max_err),
                "labs.approx.figure.error", tab="diagnostics",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))
        if p["show_tail"]:
            res.add_panel(ctx.panel(
                "tail", self._tail_figure(ctx, exact, approx, discrete, cc,
                                          float(p["tail_probability"])),
                "labs.approx.figure.tail", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        res.metric("max_cdf_error", ctx.t("labs.approx.metric.max_error",
                                          "Largest CDF error over the support"), max_err)
        res.metric("mean_abs_error", ctx.t("labs.approx.metric.mean_error",
                                           "Average absolute CDF error"),
                   float(np.mean(np.abs(error))))
        tp = float(p["tail_probability"])
        exact_q = float(exact.ppf(1 - tp))
        approx_tail = float(approx.sf(exact_q - (0.5 if cc else 0.0)))
        res.metric("tail_exact", ctx.t("labs.approx.metric.tail_exact",
                                       "Exact tail probability above {q}", q=fmt(exact_q, 2)),
                   float(exact.sf(exact_q)))
        res.metric("tail_approx", ctx.t("labs.approx.metric.tail_approx",
                                        "Approximated tail probability"), approx_tail)
        res.metric("tail_relative_error",
                   ctx.t("labs.approx.metric.tail_rel", "Relative error in the tail"),
                   float(abs(approx_tail - exact.sf(exact_q)) /
                         max(float(exact.sf(exact_q)), 1e-12)),
                   note=ctx.t("labs.approx.metric.tail_note",
                              "this is what a p-value or a critical value actually uses"))

        rule = self._rule_of_thumb(case, p)
        res.assume("rule_of_thumb", ctx.t("labs.approx.assume.rule_label",
                                          "The usual rule of thumb is satisfied"),
                   rule["ok"], detail=rule["text"],
                   consequence="" if rule["ok"] else ctx.t(
                       "labs.approx.assume.rule_consequence",
                       "Outside the rule, the approximation can be badly wrong exactly where "
                       "you need it - in the tail."))
        res.assume("continuity_correction",
                   ctx.t("labs.approx.assume.cc_label",
                         "A continuity correction is applied where appropriate"),
                   cc or not discrete,
                   detail=ctx.t("labs.approx.assume.cc",
                                "Approximating a discrete distribution by a continuous one "
                                "without the half-unit correction systematically misplaces "
                                "probability at every jump."))

        res.animations.append(self._animation(ctx, case, p))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.probability.approximations.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.probability.approximations.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.probability.approximations.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.probability.approximations.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.probability.approximations.warning"), kind="warning")
        res.explain("interpretation", ctx.t("ui.conclusion"), ctx.t(
            "labs.approx.verdict",
            "The largest CDF error is {e}. In the tail the relative error is {r} - and the "
            "tail is where critical values and p-values live, so that is the number that "
            "decides whether this approximation is usable.",
            e=fmt(max_err, 5),
            r=pct(float(abs(approx_tail - exact.sf(exact_q)) /
                        max(float(exact.sf(exact_q)), 1e-12))),
        ))
        if not rule["ok"]:
            res.warnings.append(rule["text"])
        return res

    @staticmethod
    def _pair(case, p):
        n, prob = int(p["n"]), float(p["p"])
        lam, df = float(p["lam"]), float(p["df"])
        N = int(p["population"])
        if case == "binomial_to_normal":
            exact = stats.binom(n, prob)
            approx = stats.norm(n * prob, np.sqrt(n * prob * (1 - prob)))
            x = np.arange(0, n + 1)
            return exact, approx, x, True, "normal"
        if case == "binomial_to_poisson":
            exact = stats.binom(n, prob)
            approx = stats.poisson(n * prob)
            x = np.arange(0, min(n, int(n * prob + 10 * np.sqrt(max(n * prob, 1)))) + 1)
            return exact, approx, x, False, "Poisson"
        if case == "poisson_to_normal":
            exact = stats.poisson(lam)
            approx = stats.norm(lam, np.sqrt(lam))
            x = np.arange(0, int(lam + 6 * np.sqrt(lam)) + 2)
            return exact, approx, x, True, "normal"
        if case == "t_to_normal":
            exact = stats.t(df)
            approx = stats.norm(0, 1)
            x = np.linspace(-5, 5, 600)
            return exact, approx, x, False, "normal"
        if case == "chi2_to_normal":
            exact = stats.chi2(df)
            approx = stats.norm(df, np.sqrt(2 * df))
            x = np.linspace(max(df - 5 * np.sqrt(2 * df), 0.01), df + 6 * np.sqrt(2 * df), 600)
            return exact, approx, x, False, "normal"
        K = int(max(1, min(round(prob * N), N)))
        draws = int(min(n, N))
        exact = stats.hypergeom(N, K, draws)
        approx = stats.binom(draws, K / N)
        x = np.arange(0, draws + 1)
        return exact, approx, x, False, "binomial"

    @staticmethod
    def _rule_of_thumb(case, p):
        n, prob = int(p["n"]), float(p["p"])
        lam = float(p["lam"])
        df = float(p["df"])
        N = int(p["population"])
        if case == "binomial_to_normal":
            ok = n * prob >= 5 and n * (1 - prob) >= 5
            return {"ok": ok,
                    "text": f"np = {n * prob:.2f} and n(1-p) = {n * (1 - prob):.2f}; "
                            "the usual requirement is that both exceed 5."}
        if case == "binomial_to_poisson":
            ok = n >= 20 and prob <= 0.05
            return {"ok": ok,
                    "text": f"n = {n} and p = {prob:.3f}; the Poisson limit needs a large n "
                            "with a small p at roughly constant np."}
        if case == "poisson_to_normal":
            return {"ok": lam >= 10,
                    "text": f"lambda = {lam:.2f}; the normal approximation is usually only "
                            "recommended above 10."}
        if case == "t_to_normal":
            return {"ok": df >= 30,
                    "text": f"df = {df:.0f}; the t distribution is close to normal above "
                            "about 30 degrees of freedom, but its tails stay heavier."}
        if case == "chi2_to_normal":
            return {"ok": df >= 30,
                    "text": f"df = {df:.0f}; the chi-square is strongly right-skewed for "
                            "small degrees of freedom."}
        frac = min(int(p["n"]), N) / max(N, 1)
        return {"ok": frac <= 0.1,
                "text": f"the sampling fraction is {frac:.1%}; the binomial approximation "
                        "needs it below roughly 10% so that the draws are nearly independent."}

    def _overlay_figure(self, ctx, exact, approx, x, discrete, cc, label):
        fig = ctx.figure(
            "labs.approx.figure.overlay",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.probability"),
            height=420,
        )
        if discrete or hasattr(exact, "pmf"):
            try:
                pmf = exact.pmf(x)
                P.add_bar(fig, x, pmf, ctx.t("labs.approx.trace.exact", "exact"),
                          "primary", theme=ctx.theme, opacity=0.65)
            except (AttributeError, ValueError):
                P.add_curve(fig, x, exact.pdf(x),
                            ctx.t("labs.approx.trace.exact", "exact"),
                            "primary", theme=ctx.theme)
        else:
            P.add_curve(fig, x, exact.pdf(x), ctx.t("labs.approx.trace.exact", "exact"),
                        "primary", theme=ctx.theme)
        if hasattr(approx, "pmf") and not isinstance(approx.dist, type(stats.norm)):
            P.add_bar(fig, x, approx.pmf(x),
                      ctx.t("labs.approx.trace.approx", "{label} approximation", label=label),
                      "secondary", theme=ctx.theme, opacity=0.5)
        else:
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 500)
            P.add_curve(fig, grid, approx.pdf(grid),
                        ctx.t("labs.approx.trace.approx",
                              "{label} approximation", label=label),
                        "secondary", theme=ctx.theme, dash="dash")
        P.add_legend_note(fig, ctx.t(
            "labs.approx.legend",
            "Two curves that look alike here can still disagree by an order of magnitude in "
            "the tail. Read the error panel before trusting the picture.",
        ), theme=ctx.theme)
        return fig

    def _error_figure(self, ctx, x, error, max_err):
        fig = ctx.figure(
            "labs.approx.figure.error",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.approx.axis.error", "Approximate CDF minus exact CDF"),
            height=340,
        )
        P.add_curve(fig, x, error, ctx.t("labs.approx.trace.error", "approximation error"),
                    "negative", theme=ctx.theme)
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme)
        idx = int(np.argmax(np.abs(error)))
        P.add_points(fig, [x[idx]], [error[idx]],
                     ctx.t("labs.approx.trace.worst", "worst point"),
                     "warning", theme=ctx.theme, size=11)
        P.add_legend_note(fig, ctx.t(
            "labs.approx.legend_error",
            "The largest error is {e}, at value {v}. Side-by-side densities hide this; a "
            "difference plot does not.", e=fmt(max_err, 5), v=fmt(float(x[idx]), 2),
        ), theme=ctx.theme)
        return fig

    def _tail_figure(self, ctx, exact, approx, discrete, cc, tail_p):
        levels = np.array([0.20, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
        exact_q = exact.ppf(1 - levels)
        approx_tail = approx.sf(exact_q - (0.5 if cc else 0.0))
        fig = ctx.figure(
            "labs.approx.figure.tail",
            xaxis_title=ctx.t("labs.approx.axis.nominal", "Nominal tail probability"),
            yaxis_title=ctx.t("labs.approx.axis.actual", "Probability the approximation gives"),
            height=360,
        )
        P.add_curve(fig, levels, approx_tail,
                    ctx.t("labs.approx.trace.tail", "approximated tail probability"),
                    "secondary", theme=ctx.theme, mode="lines+markers")
        P.add_curve(fig, levels, levels,
                    ctx.t("labs.approx.trace.perfect", "perfect agreement"),
                    "truth", theme=ctx.theme, dash="dash")
        P.add_vline(fig, tail_p, ctx.t("labs.common.trace.current"), "highlight",
                    theme=ctx.theme)
        fig.update_xaxes(type="log", autorange="reversed")
        fig.update_yaxes(type="log")
        P.add_legend_note(fig, ctx.t(
            "labs.approx.legend_tail",
            "Points below the dashed line mean the approximation understates the tail, so a "
            "test built on it would reject too often. This is the panel that decides whether "
            "an approximation is safe for inference.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, case, p):
        go = P.require_plotly()
        key = {"binomial_to_normal": "n", "binomial_to_poisson": "n",
               "poisson_to_normal": "lam", "t_to_normal": "df",
               "chi2_to_normal": "df", "hypergeometric_to_binomial": "population"}[case]
        if key == "n":
            values = np.unique(np.round(np.geomspace(3, 400, 18)).astype(int))
        elif key == "lam":
            values = np.round(np.geomspace(0.3, 80, 18), 2)
        elif key == "df":
            values = np.unique(np.round(np.geomspace(1, 200, 18)).astype(int))
        else:
            values = np.unique(np.round(np.geomspace(30, 4000, 18)).astype(int))

        frames, steps = [], []
        for i, v in enumerate(values):
            q = dict(p)
            q[key] = float(v)
            exact, approx, x, discrete, label = self._pair(case, q)
            cc = bool(p["continuity_correction"]) and discrete
            err = approx.cdf(x + (0.5 if cc else 0.0)) - exact.cdf(x)
            frames.append(go.Frame(name=f"{v:g}", data=[go.Scatter(x=x, y=err)]))
            steps.append(AnimationStep(
                id=f"{key}_{i}", frame=i,
                title=ctx.t("labs.approx.anim.title", "{key} = {v}", key=key, v=fmt(v, 2)),
                what_you_see=ctx.t("labs.approx.anim.see",
                                   "The signed difference between the approximate CDF and "
                                   "the exact one, across the whole support."),
                what_changed=ctx.t("labs.approx.anim.changed",
                                   "The parameter {key} moved to {v}.", key=key, v=fmt(v, 2)),
                why=ctx.t("labs.approx.anim.why",
                          "The approximation is a limit statement: the error term shrinks as "
                          "this parameter grows, at a rate the theorem specifies."),
                interpretation=ctx.t("labs.approx.anim.interpret",
                                     "Largest error is now {e}.",
                                     e=fmt(float(np.max(np.abs(err))), 5)),
                conclusion=ctx.t("labs.approx.anim.conclude",
                                 "The error curve flattening towards zero is what "
                                 "'the approximation improves' actually means."),
                warning=ctx.t("labs.approx.anim.warn",
                              "Watch where the error is largest, not just how big it is. "
                              "Error concentrated in the tail matters far more than error "
                              "in the middle."),
                math="", outputs={key: float(v),
                                  "max_abs_error": round(float(np.max(np.abs(err))), 6)},
                highlighted=("error_curve",),
            ))
        fig = ctx.figure(
            "labs.approx.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.approx.axis.error",
                              "Approximate CDF minus exact CDF"),
            height=380,
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                                 line={"color": ctx.color("negative"), "width": 2.6},
                                 name=ctx.t("labs.approx.trace.error",
                                            "approximation error")))
        P.add_hline(fig, 0.0, "", "baseline", theme=ctx.theme)
        build_frames(fig, frames, duration=460, reduced_motion=ctx.reduced_motion,
                     slider_label=key)
        return animation(
            "error_shrinks", fig, steps,
            purpose=ctx.t("labs.approx.anim.purpose",
                          "Show convergence as an error curve collapsing, not as two "
                          "curves that happen to look similar."),
            summary=ctx.t(
                "labs.approx.anim.summary",
                "Approximation quality is a claim about error, so the honest way to display "
                "it is to plot the error. Rules of thumb such as np > 5 are conventions "
                "chosen to keep this curve small in the middle - they say nothing about "
                "the far tail."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = ApproximationLab(SPEC)
