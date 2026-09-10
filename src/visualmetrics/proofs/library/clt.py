"""Central limit theorem via the moment generating function."""

from __future__ import annotations

from .._kit import (
    EvidenceType,
    Level,
    ProofSpec,
    Reference,
    StepKind,
    VisualAction,
    assume,
    step,
)

PROOF = ProofSpec(
    id="inference.clt.moment_generating",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.ADVANCED,
    title="Central limit theorem: why the normal limit is inevitable",
    claim=(
        "If X_1, X_2, ... are independent and identically distributed with mean mu and "
        "finite variance sigma^2 > 0, then the standardised mean converges in "
        "distribution to the standard normal law."
    ),
    intuition=(
        "Standardising shrinks each observation's contribution to order 1/sqrt(n). At "
        "that scale only the first two moments survive the expansion - everything "
        "particular to the original distribution is pushed into terms that vanish. Since "
        "the limit depends on nothing but the mean and the variance, every distribution "
        "with a finite variance must land on the same one."
    ),
    assumptions=(
        assume(
            "iid",
            "The observations are independent and identically distributed.",
            if_violated=(
                "Dependence changes the scaling: for a stationary series with summable "
                "autocovariances the limit variance is the long-run variance, not sigma^2, "
                "which is exactly why HAC standard errors exist. With strong long-memory "
                "dependence the normalisation itself changes."
            ),
        ),
        assume(
            "finite_variance",
            "The variance sigma^2 is finite and strictly positive.",
            if_violated=(
                "With infinite variance the limit is a non-normal stable law and the "
                "scaling is no longer sqrt(n) - the Cauchy distribution's sample mean, for "
                "instance, has the same distribution as a single observation, so averaging "
                "achieves nothing at all."
            ),
        ),
        assume(
            "mgf_exists",
            "The moment generating function of the standardised variable is finite in a "
            "neighbourhood of the origin.",
            essential=False,
            if_violated=(
                "The theorem is still true - this assumption is an artefact of the proof "
                "technique, not of the result. The general proof replaces the moment "
                "generating function with the characteristic function, which always exists, "
                "and uses Levy's continuity theorem. Every step below then goes through "
                "unchanged with it replaced by E[exp(i t Y)]."
            ),
        ),
    ),
    steps=(
        step(
            "standardise",
            "Standardise so the limit has a chance of being non-degenerate.",
            equation=r"Y_i = \frac{X_i - \mu}{\sigma}, \qquad Z_n = \frac{1}{\sqrt n}\sum_{i=1}^n Y_i "
            r"= \frac{\sqrt n(\bar X_n - \mu)}{\sigma}",
            why=(
                "E[Y] = 0 and Var(Y) = 1. Dividing by n instead of sqrt(n) would give the "
                "law of large numbers and a constant limit; sqrt(n) is the exact rate that "
                "keeps the variance at one."
            ),
            kind=StepKind.SETUP,
            uses=("finite_variance",),
        ),
        step(
            "mgf_definition",
            "Work with the moment generating function, which turns sums into products.",
            equation=r"M_Y(t) = E[e^{tY}], \qquad M_Y(0) = 1, \ M_Y'(0) = E[Y] = 0, \ M_Y''(0) = E[Y^2] = 1",
            why=(
                "Finiteness near the origin lets the expectation and the derivatives be "
                "exchanged, so the derivatives at zero are the moments."
            ),
            kind=StepKind.DEFINITION,
            uses=("mgf_exists", "standardise"),
        ),
        step(
            "factorise",
            "Independence turns the moment generating function of the sum into a power.",
            equation=r"M_{Z_n}(t) = E\Big[e^{\frac{t}{\sqrt n}\sum_i Y_i}\Big] "
            r"= \prod_{i=1}^n E\big[e^{\frac{t}{\sqrt n}Y_i}\big] = \Big[M_Y\!\big(t/\sqrt n\big)\Big]^n",
            why=(
                "The expectation of a product of independent variables factorises; "
                "identical distribution makes all n factors the same. This is the step that "
                "uses assumption iid, and it is the only place it is used."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("iid", "mgf_definition"),
        ),
        step(
            "taylor",
            "Expand around zero. The argument t/sqrt(n) is shrinking, so the expansion is legitimate.",
            equation=r"M_Y(s) = 1 + M_Y'(0)s + \tfrac12 M_Y''(0)s^2 + o(s^2) = 1 + \tfrac{s^2}{2} + o(s^2)",
            why=(
                "The first-order term vanishes because the mean is zero, and the "
                "second-order coefficient is one half because the variance is one. Nothing "
                "else about the distribution appears at this order - this is where the "
                "particulars are lost."
            ),
            kind=StepKind.CALCULUS,
            uses=("mgf_definition",),
            visual=VisualAction.HIGHLIGHT_TERM,
        ),
        step(
            "substitute",
            "Substitute s = t/sqrt(n) into the n-fold product.",
            equation=r"M_{Z_n}(t) = \Big[1 + \frac{t^2}{2n} + o\!\big(\tfrac1n\big)\Big]^{n}",
            why="Combining the two previous steps.",
            kind=StepKind.SUBSTITUTION,
            uses=("factorise", "taylor"),
        ),
        step(
            "logarithm",
            "Take logarithms to convert the power into a product, then expand once more.",
            equation=r"\log M_{Z_n}(t) = n\log\Big[1 + \frac{t^2}{2n} + o\!\big(\tfrac1n\big)\Big] "
            r"= n\Big[\frac{t^2}{2n} + o\!\big(\tfrac1n\big)\Big] \longrightarrow \frac{t^2}{2}",
            why=(
                "log(1 + x) = x + O(x^2) for small x, and the bracket is of order 1/n. "
                "Multiplying by n leaves t^2/2 plus a term that still vanishes."
            ),
            kind=StepKind.CALCULUS,
            uses=("substitute",),
        ),
        step(
            "identify",
            "The limit is the moment generating function of the standard normal.",
            equation=r"M_{Z_n}(t) \to e^{t^2/2} = M_{N(0,1)}(t) \quad \text{for every } t \text{ near } 0",
            why="Exponentiating the previous limit; the normal moment generating function is exp(t^2/2).",
            kind=StepKind.KEY_INSIGHT,
            uses=("logarithm",),
            visual=VisualAction.SHADE_REGION,
        ),
        step(
            "continuity_theorem",
            "Pointwise convergence of the transforms transfers to convergence in distribution.",
            equation=r"Z_n \xrightarrow{d} N(0, 1), \qquad \text{equivalently} \quad "
            r"\bar X_n \approx N\!\Big(\mu, \frac{\sigma^2}{n}\Big)",
            why=(
                "The continuity theorem: if the moment generating functions converge in a "
                "neighbourhood of the origin to one that is finite there, the distributions "
                "converge. This step is what makes the argument a proof rather than a "
                "manipulation of formulas."
            ),
            kind=StepKind.CONCLUSION,
            uses=("identify", "mgf_exists"),
        ),
    ),
    conclusion=(
        "Whatever the shape of the underlying distribution, as long as its variance is "
        "finite the standardised sample mean has a standard normal limit. This is why "
        "confidence intervals and t-tests remain approximately valid far outside the "
        "normal model - and the reason is visible in the proof: after standardisation "
        "only the first two moments survive."
    ),
    limitations=(
        "This is a statement about a limit, not about any particular sample size. It gives "
        "no error bound: how large n must be depends on the skewness and the tail weight of "
        "the underlying distribution, and for strongly skewed or heavy-tailed data the "
        "approximation can still be poor at n in the hundreds. Berry-Esseen supplies a rate "
        "of order 1/sqrt(n); this proof supplies none. The theorem concerns the centre of "
        "the distribution, so tail probabilities and extreme quantiles converge far more "
        "slowly than the middle. It is about the mean of the sampling distribution of an "
        "average, never about the data themselves becoming normal - a common and consequential "
        "misreading. And it says nothing when the variance is infinite or the observations "
        "are dependent."
    ),
    prerequisites=(),
    concept_ids=("inference.clt", "inference.lln", "inference.sampling_distributions"),
    references=(
        Reference("Billingsley, P. (1995). Probability and Measure, 3rd ed., section 27.", kind="book"),
        Reference(
            "Casella, G. and Berger, R. L. (2002). Statistical Inference, 2nd ed., section 5.5.",
            kind="book",
        ),
    ),
)
