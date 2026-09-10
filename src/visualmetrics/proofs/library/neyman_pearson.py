"""The Neyman-Pearson lemma."""

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
    id="inference.neyman_pearson.lemma",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.ADVANCED,
    title="Neyman-Pearson lemma: the likelihood ratio test is most powerful",
    claim=(
        "For a simple null H0: X ~ f0 against a simple alternative H1: X ~ f1, the test "
        "that rejects when f1(x) > k f0(x) and has size exactly alpha is at least as "
        "powerful as any other test of level alpha."
    ),
    intuition=(
        "You have a fixed budget of false alarms - alpha. Spend it on the sample points "
        "that give you the most alternative-density per unit of null-density. That "
        "exchange rate is the likelihood ratio, so ranking points by the likelihood "
        "ratio is the optimal way to spend the budget."
    ),
    assumptions=(
        assume(
            "simple_hypotheses",
            "Both hypotheses are simple: f0 and f1 are fully specified, with no unknown "
            "parameters.",
            if_violated=(
                "With a composite alternative there is no single most powerful test in "
                "general. A uniformly most powerful test exists only under extra "
                "structure, such as a monotone likelihood ratio in a one-sided problem."
            ),
        ),
        assume(
            "common_dominating_measure",
            "f0 and f1 are densities with respect to the same dominating measure "
            "(both continuous, or both probability mass functions).",
            if_violated="The ratio f1/f0 is not well defined and the integrals below do not compare.",
        ),
        assume(
            "exact_size",
            "The critical value k, together with randomisation on the boundary, is "
            "chosen so that the test has size exactly alpha.",
            essential=True,
            if_violated=(
                "With a discrete statistic no non-randomised k attains alpha exactly. "
                "A conservative test with size below alpha is still valid but no longer "
                "optimal - part of the budget goes unspent."
            ),
        ),
    ),
    steps=(
        step(
            "test_function",
            "Represent a test by its rejection probability, allowing randomisation.",
            equation=r"\varphi(x) \in [0,1], \quad \text{size} = E_0[\varphi(X)], \quad "
            r"\text{power} = E_1[\varphi(X)]",
            why=(
                "Writing tests as functions rather than regions lets a single algebraic "
                "argument cover randomised and non-randomised tests at once. Both "
                "expectations are well defined only because each hypothesis fixes one "
                "specific density."
            ),
            kind=StepKind.SETUP,
            uses=("simple_hypotheses",),
        ),
        step(
            "define_star",
            "Define the likelihood ratio test.",
            equation=(
                r"\varphi^*(x) = \begin{cases} 1 & f_1(x) > k f_0(x) \\ "
                r"\gamma & f_1(x) = k f_0(x) \\ 0 & f_1(x) < k f_0(x)\end{cases}, \quad "
                r"E_0[\varphi^*] = \alpha"
            ),
            why="Assumption exact_size fixes k and gamma.",
            kind=StepKind.DEFINITION,
            uses=("exact_size",),
            visual=VisualAction.SHADE_TAIL,
        ),
        step(
            "competitor",
            "Let phi be any competing test of level alpha.",
            equation=r"E_0[\varphi] \le \alpha",
            why="This is what 'level alpha' means; the claim must hold against all such tests.",
            kind=StepKind.SETUP,
        ),
        step(
            "pointwise",
            "The key pointwise inequality: the two factors always share a sign.",
            equation=r"\big(\varphi^*(x) - \varphi(x)\big)\big(f_1(x) - k f_0(x)\big) \ge 0 "
            r"\quad \text{for every } x",
            why=(
                "Case analysis. Where f1 > k f0 the test phi* equals 1, its maximum, so "
                "phi* - phi >= 0 and both factors are non-negative. Where f1 < k f0 the "
                "test phi* equals 0, its minimum, so phi* - phi <= 0 and both factors are "
                "non-positive. Where f1 = k f0 the second factor is zero. A product of "
                "two like-signed numbers is never negative."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("define_star", "common_dominating_measure"),
            visual=VisualAction.COMPARE_REGIONS,
        ),
        step(
            "integrate",
            "Integrate the inequality over the sample space.",
            equation=r"\int (\varphi^* - \varphi) f_1 \, d\mu \;\ge\; k \int (\varphi^* - \varphi) f_0 \, d\mu",
            why="A non-negative function has a non-negative integral; the integral is linear.",
            kind=StepKind.PROBABILITY,
            uses=("pointwise",),
        ),
        step(
            "recognise_moments",
            "Recognise the two integrals as a power difference and a size difference.",
            equation=r"E_1[\varphi^*] - E_1[\varphi] \;\ge\; k\big(E_0[\varphi^*] - E_0[\varphi]\big) "
            r"= k\big(\alpha - E_0[\varphi]\big)",
            why="By definition of expectation under each hypothesis, and E0[phi*] = alpha.",
            kind=StepKind.SUBSTITUTION,
            uses=("integrate", "define_star"),
        ),
        step(
            "sign",
            "The right-hand side is non-negative.",
            equation=r"k \ge 0 \ \text{ and } \ E_0[\varphi] \le \alpha \implies k(\alpha - E_0[\varphi]) \ge 0",
            why="Assumption competitor gives the level constraint; k is a ratio of densities, hence non-negative.",
            kind=StepKind.ALGEBRA,
            uses=("competitor", "recognise_moments"),
        ),
        step(
            "conclude",
            "Therefore the likelihood ratio test is at least as powerful.",
            equation=r"E_1[\varphi^*] \ge E_1[\varphi]",
            why="Chaining the two previous inequalities.",
            kind=StepKind.CONCLUSION,
            uses=("sign",),
        ),
    ),
    conclusion=(
        "No level-alpha test of a simple null against a simple alternative can beat the "
        "likelihood ratio test. The optimal rejection region is a level set of the "
        "likelihood ratio, which is why so many standard tests - the z test, the t test, "
        "the F test in the normal linear model - turn out to be monotone functions of a "
        "likelihood ratio."
    ),
    limitations=(
        "Simple against simple only. It gives no most-powerful test for a composite "
        "alternative, where uniform optimality generally fails and one must fall back on "
        "monotone-likelihood-ratio families, unbiasedness or invariance restrictions. It "
        "also assumes the model is correct: the lemma optimises power within the assumed "
        "pair of densities and offers no protection when neither f0 nor f1 generated the "
        "data. Finally it is a statement about a fixed alpha, not a recommendation for "
        "which alpha to use, and it says nothing about how to interpret a p-value."
    ),
    prerequisites=(),
    concept_ids=("inference.neyman_pearson", "inference.hypothesis_testing", "inference.power"),
    references=(
        Reference(
            "Neyman, J. and Pearson, E. S. (1933). On the problem of the most efficient "
            "tests of statistical hypotheses. Philosophical Transactions of the Royal "
            "Society A 231, 289-337.",
            kind="paper",
            doi="10.1098/rsta.1933.0009",
        ),
        Reference(
            "Lehmann, E. L. and Romano, J. P. (2005). Testing Statistical Hypotheses, "
            "3rd ed., ch. 3.",
            kind="book",
        ),
    ),
)
