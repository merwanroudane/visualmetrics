"""Distribution explorer: 22 families, PMF/PDF, CDF, quantiles, moments and sampling."""

from __future__ import annotations

from typing import Any

from scipy import stats

from .._kit import (
    AnimationStep, Domain, EvidenceType, LabBase, LabResult, LabState, P, animation,
    build_frames, context, fmt, int_slider, make_spec, np, ref, scenario, seed_control,
    select, slider, toggle,
)
from ...simulation.random import rng

__all__ = ["LAB", "SPEC", "FAMILIES", "build_distribution"]


class Family:
    def __init__(self, name, discrete, params, factory, support=None):
        self.name = name
        self.discrete = discrete
        self.params = params  # (control_name, default, min, max) triples
        self.factory = factory
        self.support = support


FAMILIES: dict[str, Family] = {
    "bernoulli": Family("bernoulli", True, (("p", 0.4, 0.01, 0.99),),
                        lambda p, **k: stats.bernoulli(p)),
    "binomial": Family("binomial", True, (("n_trials", 20, 1, 200), ("p", 0.4, 0.01, 0.99)),
                       lambda n_trials, p, **k: stats.binom(int(n_trials), p)),
    "geometric": Family("geometric", True, (("p", 0.3, 0.01, 0.99),),
                        lambda p, **k: stats.geom(p)),
    "negative_binomial": Family("negative_binomial", True,
                                (("r", 5.0, 1.0, 50.0), ("p", 0.4, 0.01, 0.99)),
                                lambda r, p, **k: stats.nbinom(int(r), p)),
    "hypergeometric": Family("hypergeometric", True,
                             (("N", 50, 2, 500), ("K", 20, 1, 200), ("n_draws", 10, 1, 100)),
                             lambda N, K, n_draws, **k: stats.hypergeom(
                                 int(N), min(int(K), int(N)), min(int(n_draws), int(N)))),
    "poisson": Family("poisson", True, (("lam", 3.0, 0.05, 60.0),),
                      lambda lam, **k: stats.poisson(lam)),
    "discrete_uniform": Family("discrete_uniform", True, (("a", 1.0, -50.0, 50.0),
                                                          ("b", 6.0, -50.0, 100.0)),
                               lambda a, b, **k: stats.randint(int(a), int(max(b, a + 1)) + 1)),
    "uniform": Family("uniform", False, (("a", 0.0, -50.0, 50.0), ("b", 1.0, -50.0, 100.0)),
                      lambda a, b, **k: stats.uniform(a, max(b - a, 1e-6))),
    "normal": Family("normal", False, (("mu", 0.0, -50.0, 50.0), ("sigma", 1.0, 0.01, 30.0)),
                     lambda mu, sigma, **k: stats.norm(mu, sigma)),
    "standard_normal": Family("standard_normal", False, (),
                              lambda **k: stats.norm(0, 1)),
    "lognormal": Family("lognormal", False, (("mu", 0.0, -5.0, 5.0),
                                             ("sigma", 0.6, 0.01, 3.0)),
                        lambda mu, sigma, **k: stats.lognorm(sigma, scale=np.exp(mu))),
    "exponential": Family("exponential", False, (("rate", 1.0, 0.01, 20.0),),
                          lambda rate, **k: stats.expon(scale=1.0 / rate)),
    "gamma": Family("gamma", False, (("shape", 2.0, 0.05, 30.0), ("rate", 1.0, 0.01, 20.0)),
                    lambda shape, rate, **k: stats.gamma(shape, scale=1.0 / rate)),
    "beta": Family("beta", False, (("alpha_p", 2.0, 0.05, 30.0), ("beta_p", 3.0, 0.05, 30.0)),
                   lambda alpha_p, beta_p, **k: stats.beta(alpha_p, beta_p)),
    "chi_square": Family("chi_square", False, (("df", 4.0, 1.0, 60.0),),
                         lambda df, **k: stats.chi2(df)),
    "student_t": Family("student_t", False, (("df", 5.0, 1.0, 200.0),),
                        lambda df, **k: stats.t(df)),
    "fisher_f": Family("fisher_f", False, (("df1", 5.0, 1.0, 100.0), ("df2", 10.0, 1.0, 200.0)),
                       lambda df1, df2, **k: stats.f(df1, df2)),
    "cauchy": Family("cauchy", False, (("mu", 0.0, -20.0, 20.0), ("scale", 1.0, 0.05, 20.0)),
                     lambda mu, scale, **k: stats.cauchy(mu, scale)),
    "logistic": Family("logistic", False, (("mu", 0.0, -20.0, 20.0),
                                           ("scale", 1.0, 0.05, 20.0)),
                       lambda mu, scale, **k: stats.logistic(mu, scale)),
    "weibull": Family("weibull", False, (("shape", 1.5, 0.1, 15.0), ("scale", 1.0, 0.05, 20.0)),
                      lambda shape, scale, **k: stats.weibull_min(shape, scale=scale)),
    "pareto": Family("pareto", False, (("shape", 2.5, 0.2, 20.0),),
                     lambda shape, **k: stats.pareto(shape)),
    "laplace": Family("laplace", False, (("mu", 0.0, -20.0, 20.0), ("scale", 1.0, 0.05, 20.0)),
                      lambda mu, scale, **k: stats.laplace(mu, scale)),
}


def build_distribution(family: str, params: dict[str, Any]):
    fam = FAMILIES[family]
    kwargs = {name: params.get(name, default) for name, default, _, _ in fam.params}
    return fam, fam.factory(**kwargs)


_PARAM_CONTROLS = {}
for _fam in FAMILIES.values():
    for _name, _default, _lo, _hi in _fam.params:
        if _name not in _PARAM_CONTROLS:
            _PARAM_CONTROLS[_name] = (float(_default), float(_lo), float(_hi))


SPEC = make_spec(
    "probability.distributions",
    Domain.PROBABILITY,
    "distributions",
    module=__name__,
    levels=("beginner", "intermediate", "advanced", "phd"),
    modes=("learn", "visualize", "animate", "experiment", "compare", "simulate",
           "code", "quiz", "data", "references"),
    evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
    controls=(
        select("family", "normal", tuple(FAMILIES), group="family"),
        *[slider(name, default, lo, hi,
                 0.01 if hi - lo < 60 else 1.0, group="parameters")
          for name, (default, lo, hi) in sorted(_PARAM_CONTROLS.items())],
        int_slider("sample_size", 400, 0, 20000, 10, group="sampling"),
        slider("quantile", 0.9, 0.001, 0.999, 0.001, group="views"),
        toggle("show_cdf", True, group="views"),
        toggle("show_sample", True, group="views"),
        toggle("show_quantile", True, group="views"),
        seed_control(),
    ),
    scenarios=(
        scenario("canonical", "canonical", family="normal", mu=0.0, sigma=1.0),
        scenario("shape_explorer", "sensitivity", family="beta", alpha_p=2.0, beta_p=5.0),
        scenario("symmetric_vs_skewed", "compare_methods", family="gamma", shape=1.5,
                 rate=1.0),
        scenario("rare_event", "weak", family="poisson", lam=0.3),
        scenario("thin_vs_heavy_tail", "boundary", family="student_t", df=2.0),
        scenario("no_variance", "counterexample", family="cauchy"),
        scenario("small_df", "small_sample", family="chi_square", df=1.0),
        scenario("large_df", "large_sample", family="student_t", df=150.0),
        scenario("sampling_without_replacement", "compare_methods",
                 family="hypergeometric", N=50, K=20, n_draws=10),
        scenario("memoryless", "positive", family="exponential", rate=0.5),
        scenario("bounded_support", "boundary", family="uniform", a=0.0, b=1.0),
        scenario("power_law", "high_noise", family="pareto", shape=1.2),
        scenario("binomial_basics", "positive", family="binomial", n_trials=20, p=0.4),
    ),
    related=("probability.approximations", "probability.bivariate", "inference.clt"),
    next_concepts=("probability.approximations", "inference.sampling_distributions"),
    tags=("distribution", "pdf", "pmf", "cdf", "quantile", "moments", "support"),
    aliases=("distribution explorer", "loi de probabilite", "التوزيعات",
             "probability distribution", "density"),
    backends=("scipy",),
    references=(
        ref("Johnson, N. L., Kotz, S. and Balakrishnan, N. (1994). Continuous Univariate "
            "Distributions.", kind="book"),
        ref("SciPy statistical functions", kind="docs",
            url="https://docs.scipy.org/doc/scipy/reference/stats.html"),
    ),
    curriculum_tags=("dz.stat3", "ksu.econ416"),
)


class DistributionLab(LabBase):
    def compute(self, p: dict[str, Any], state: LabState) -> LabResult:
        ctx = context(state)
        res = LabResult()
        fam, dist = build_distribution(str(p["family"]), p)
        n = int(p["sample_size"])
        q = float(p["quantile"])

        gen = rng(state.seed, "dist", fam.name)
        sample = dist.rvs(size=n, random_state=gen) if n else np.array([])
        x, pdf = self._grid(fam, dist)

        param_text = ", ".join(f"{name} = {fmt(p.get(name, d), 3)}"
                               for name, d, _, _ in fam.params) or "no free parameters"
        res.dgp = ctx.t(
            "labs.dist.dgp",
            "{family} distribution with {params}; {n} independent draws for the sample view.",
            family=fam.name, params=param_text, n=n,
        )

        res.add_panel(ctx.panel(
            "density", self._density_figure(ctx, fam, dist, x, pdf, sample, q, p),
            "labs.dist.figure.density", evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        ))
        if p["show_cdf"]:
            res.add_panel(ctx.panel(
                "cdf", self._cdf_figure(ctx, fam, dist, x, sample, q, p),
                "labs.dist.figure.cdf", tab="compare",
                evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
            ))

        stats_dict = self._moments(dist, fam)
        for key, label_key, default in (
            ("mean", "labs.dist.metric.mean", "Mean"),
            ("variance", "labs.dist.metric.variance", "Variance"),
            ("sd", "labs.dist.metric.sd", "Standard deviation"),
            ("skewness", "labs.dist.metric.skewness", "Skewness"),
            ("kurtosis", "labs.dist.metric.kurtosis", "Excess kurtosis"),
        ):
            value = stats_dict[key]
            emp = self._empirical(sample, key) if sample.size > 3 else None
            res.metric(key, ctx.t(label_key, default),
                       value if np.isfinite(value) else "does not exist",
                       reference=emp,
                       note=ctx.t("labs.dist.metric.sample_note", "sample value shown as "
                                  "reference") if emp is not None else "")
        res.metric("quantile", ctx.t("labs.dist.metric.quantile",
                                     "Quantile at probability {q}", q=fmt(q, 3)),
                   float(dist.ppf(q)))
        res.metric("support", ctx.t("labs.dist.metric.support", "Support"),
                   f"[{fmt(dist.support()[0], 3)}, {fmt(dist.support()[1], 3)}]")

        finite_var = np.isfinite(stats_dict["variance"])
        res.assume("finite_variance", ctx.t("assumptions.finite_variance"), finite_var,
                   detail=ctx.t("labs.dist.assume.variance",
                                "Several later results - the central limit theorem, standard "
                                "errors, least squares - need a finite variance."))
        res.animations.append(self._animation(ctx, fam, p, state.seed))

        res.explain("overview", ctx.t("tabs.overview"),
                    ctx.t("concepts.probability.distributions.summary"))
        res.explain("intuition", ctx.t("tabs.intuition"),
                    ctx.t("concepts.probability.distributions.intuition"))
        if ctx.show_math():
            res.explain("math", ctx.t("tabs.math"),
                        ctx.t("concepts.probability.distributions.math"), kind="math")
        res.explain("misconceptions", ctx.t("tabs.misconceptions"),
                    ctx.t("concepts.probability.distributions.misconceptions"),
                    kind="misconception")
        res.explain("warning", ctx.t("tabs.warning"),
                    ctx.t("concepts.probability.distributions.warning"), kind="warning")

        if fam.name == "cauchy":
            res.warnings.append(ctx.t(
                "labs.dist.warn.cauchy",
                "The Cauchy distribution has no mean and no variance. Its sample mean is "
                "not an estimate of anything - it has the same distribution as a single draw.",
            ))
        if fam.name == "hypergeometric":
            res.explain("intuition", ctx.t("labs.dist.hyper.title",
                                           "Sampling without replacement"), ctx.t(
                "labs.dist.hyper.body",
                "Each draw changes what is left in the urn, so the draws are dependent. As "
                "the population grows relative to the sample, that dependence fades and the "
                "hypergeometric approaches the binomial.",
            ))
        if fam.name == "exponential":
            res.explain("intuition", ctx.t("labs.dist.exp.title", "Memorylessness"), ctx.t(
                "labs.dist.exp.body",
                "P(X > s + t | X > s) = P(X > t) exactly. Having waited already tells you "
                "nothing about how much longer you must wait - a property no other "
                "continuous distribution has.",
            ))
        return res

    @staticmethod
    def _grid(fam, dist):
        if fam.discrete:
            lo = int(max(dist.ppf(1e-6), -1e6))
            hi = int(min(dist.ppf(1 - 1e-6), lo + 400))
            x = np.arange(lo, hi + 1)
            return x, dist.pmf(x)
        lo, hi = float(dist.ppf(0.0005)), float(dist.ppf(0.9995))
        if not np.isfinite(lo):
            lo = float(dist.ppf(0.01))
        if not np.isfinite(hi):
            hi = float(dist.ppf(0.99))
        x = np.linspace(lo, hi, 900)
        return x, dist.pdf(x)

    @staticmethod
    def _moments(dist, fam):
        with np.errstate(all="ignore"):
            m, v, s, k = dist.stats(moments="mvsk")
        return {
            "mean": float(m), "variance": float(v),
            "sd": float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan"),
            "skewness": float(s), "kurtosis": float(k),
        }

    @staticmethod
    def _empirical(sample, key):
        if key == "mean":
            return float(np.mean(sample))
        if key == "variance":
            return float(np.var(sample, ddof=1))
        if key == "sd":
            return float(np.std(sample, ddof=1))
        if key == "skewness":
            return float(stats.skew(sample))
        return float(stats.kurtosis(sample, fisher=True))

    def _density_figure(self, ctx, fam, dist, x, pdf, sample, q, p):
        fig = ctx.figure(
            "labs.dist.figure.density",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=(ctx.t("labs.common.axis.probability") if fam.discrete
                         else ctx.t("labs.common.axis.density")),
            height=430,
        )
        if p["show_sample"] and sample.size:
            if fam.discrete:
                vals, counts = np.unique(sample, return_counts=True)
                P.add_bar(fig, vals, counts / sample.size,
                          ctx.t("labs.dist.trace.sample", "sample relative frequency"),
                          "secondary", theme=ctx.theme, opacity=0.45)
            else:
                P.add_histogram(fig, sample,
                                ctx.t("labs.dist.trace.sample",
                                      "sample relative frequency"),
                                "secondary", theme=ctx.theme, nbins=50, opacity=0.4)
        if fam.discrete:
            P.add_curve(fig, x, pdf, ctx.t("labs.dist.trace.pmf", "PMF"),
                        "primary", theme=ctx.theme, mode="lines+markers")
        else:
            P.add_curve(fig, x, pdf, ctx.t("labs.dist.trace.pdf", "PDF"),
                        "primary", theme=ctx.theme)
        if p["show_quantile"]:
            qv = float(dist.ppf(q))
            mask = x <= qv
            if fam.discrete:
                P.add_bar(fig, x[mask], pdf[mask],
                          ctx.t("labs.dist.trace.tail", "probability up to the quantile"),
                          "info", theme=ctx.theme, opacity=0.35)
            else:
                P.shade_tail(fig, x, pdf, mask,
                             ctx.t("labs.dist.trace.tail",
                                   "probability up to the quantile"),
                             "info", theme=ctx.theme, alpha=0.25)
            P.add_vline(fig, qv, ctx.t("labs.dist.trace.quantile",
                                       "quantile at {q} = {v}", q=fmt(q, 3), v=fmt(qv, 3)),
                        "warning", theme=ctx.theme)
        mean = float(dist.mean())
        if np.isfinite(mean):
            P.add_vline(fig, mean, ctx.t("labs.common.trace.expectation"),
                        "truth", theme=ctx.theme, dash="dot")
        P.add_legend_note(fig, ctx.t(
            "labs.dist.legend",
            "For a continuous variable the height is a density, not a probability - only "
            "areas are probabilities, which is why the curve may rise above one.",
        ) if not fam.discrete else ctx.t(
            "labs.dist.legend_discrete",
            "For a discrete variable each bar height IS a probability, and all of them sum "
            "to exactly one.",
        ), theme=ctx.theme)
        return fig

    def _cdf_figure(self, ctx, fam, dist, x, sample, q, p):
        fig = ctx.figure(
            "labs.dist.figure.cdf",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.cumulative"),
            height=360,
        )
        P.add_curve(fig, x, dist.cdf(x), ctx.t("labs.dist.trace.cdf", "CDF"),
                    "primary", theme=ctx.theme,
                    mode="lines+markers" if fam.discrete else "lines")
        if p["show_sample"] and sample.size:
            s = np.sort(sample)
            ecdf = np.arange(1, s.size + 1) / s.size
            P.add_curve(fig, s, ecdf,
                        ctx.t("labs.dist.trace.ecdf", "empirical CDF of the sample"),
                        "secondary", theme=ctx.theme, dash="dot")
        qv = float(dist.ppf(q))
        P.add_hline(fig, q, "", "warning", theme=ctx.theme, dash="dot")
        P.add_vline(fig, qv, ctx.t("labs.dist.trace.quantile_read",
                                   "read the quantile here"), "warning", theme=ctx.theme)
        fig.update_yaxes(range=[0, 1.03])
        P.add_legend_note(fig, ctx.t(
            "labs.dist.legend_cdf",
            "The quantile function is this curve read backwards: pick a height, drop down to "
            "the axis. Every critical value in every test is exactly that operation.",
        ), theme=ctx.theme)
        return fig

    def _animation(self, ctx, fam, p, seed):
        go = P.require_plotly()
        if not fam.params:
            key, lo, hi = "sample_size", 10, 2000
            values = np.unique(np.round(np.geomspace(lo, hi, 16)).astype(int))
            frames, steps = [], []
            _, dist = build_distribution(fam.name, p)
            x, pdf = self._grid(fam, dist)
            for i, ns in enumerate(values):
                gen = rng(seed, "dist_anim", int(ns))
                s = dist.rvs(size=int(ns), random_state=gen)
                hist, edges = np.histogram(s, bins=40, range=(x[0], x[-1]), density=True)
                centres = 0.5 * (edges[1:] + edges[:-1])
                frames.append(go.Frame(name=str(ns), data=[go.Bar(x=centres, y=hist)]))
                steps.append(AnimationStep(
                    id=f"n_{ns}", frame=i,
                    title=ctx.t("labs.dist.anim.sample_title", "{n} draws", n=int(ns)),
                    what_you_see=ctx.t("labs.dist.anim.sample_see",
                                       "A histogram of draws with the exact density on top."),
                    what_changed=ctx.t("labs.dist.anim.sample_changed",
                                       "The sample now has {n} observations.", n=int(ns)),
                    why=ctx.t("labs.dist.anim.sample_why",
                              "Relative frequencies converge to probabilities as the sample "
                              "grows - that is the law of large numbers at work."),
                    interpretation=ctx.t("labs.dist.anim.sample_interpret",
                                         "Departures from the curve are sampling noise, not "
                                         "a wrong formula."),
                    conclusion=ctx.t("labs.dist.anim.sample_conclude",
                                     "The distribution is the limit the histogram is aiming at."),
                    warning="", math="",
                    outputs={"n": int(ns)},
                    highlighted=("histogram",),
                ))
        else:
            name, default, lo, hi = fam.params[0]
            values = np.linspace(lo, min(hi, lo + (hi - lo) * 0.6), 18)
            frames, steps = [], []
            _, base = build_distribution(fam.name, p)
            xb, _ = self._grid(fam, base)
            for i, v in enumerate(values):
                q = dict(p)
                q[name] = float(v)
                _, dist = build_distribution(fam.name, q)
                y = dist.pmf(xb) if fam.discrete else dist.pdf(xb)
                frames.append(go.Frame(name=f"{v:.3g}", data=[go.Scatter(x=xb, y=y)]))
                mean = float(dist.mean())
                var = float(dist.var())
                steps.append(AnimationStep(
                    id=f"{name}_{i}", frame=i,
                    title=ctx.t("labs.dist.anim.title", "{name} = {v}", name=name,
                                v=fmt(v, 3)),
                    what_you_see=ctx.t("labs.dist.anim.see",
                                       "The density of the {family} family at this parameter "
                                       "value.", family=fam.name),
                    what_changed=ctx.t("labs.dist.anim.changed",
                                       "The parameter {name} moved to {v}.", name=name,
                                       v=fmt(v, 3)),
                    why=ctx.t("labs.dist.anim.why",
                              "Parameters are not decoration: they enter the formula for the "
                              "density and therefore reshape mass, mean and spread together."),
                    interpretation=ctx.t("labs.dist.anim.interpret",
                                         "Mean is now {m}, variance {v2}.",
                                         m=fmt(mean, 4), v2=fmt(var, 4)),
                    conclusion=ctx.t("labs.dist.anim.conclude",
                                     "Watching which features move and which stay fixed is "
                                     "how you learn what a parameter actually controls."),
                    warning="", math="",
                    outputs={name: round(float(v), 4), "mean": round(mean, 4),
                             "variance": round(var, 4)},
                    highlighted=("density_curve",),
                ))

        fig = ctx.figure(
            "labs.dist.figure.animation",
            xaxis_title=ctx.t("labs.common.axis.value"),
            yaxis_title=ctx.t("labs.common.axis.density"),
            height=380,
        )
        _, dist = build_distribution(fam.name, p)
        x, pdf = self._grid(fam, dist)
        if fam.params:
            fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines",
                                     line={"color": ctx.color("primary"), "width": 2.8},
                                     name=ctx.t("labs.dist.trace.pdf", "PDF")))
        else:
            centres = np.linspace(x[0], x[-1], 40)
            fig.add_trace(go.Bar(x=centres, y=np.zeros_like(centres),
                                 marker={"color": ctx.color("secondary")}, opacity=0.6,
                                 name=ctx.t("labs.dist.trace.sample", "sample")))
            P.add_curve(fig, x, pdf, ctx.t("labs.dist.trace.pdf", "PDF"), "primary",
                        theme=ctx.theme)
        build_frames(fig, frames, duration=420, reduced_motion=ctx.reduced_motion,
                     slider_label=fam.params[0][0] if fam.params else "n")
        return animation(
            "parameter_sweep", fig, steps,
            purpose=ctx.t("labs.dist.anim.purpose",
                          "Connect a parameter to the shape it produces."),
            summary=ctx.t(
                "labs.dist.anim.summary",
                "A distribution family is a shape with dials. Learning a family means "
                "knowing which dial moves the centre, which moves the spread, and which "
                "changes the shape itself."),
            evidence=EvidenceType.NUMERICAL_DEMONSTRATION,
        )


LAB = DistributionLab(SPEC)
