"""The Cramer-Rao lower bound."""

from __future__ import annotations

from .._kit import (
    EvidenceType,
    Level,
    ProofSpec,
    Reference,
    StepKind,
    VisualAction,
    assume,
    np,
    numeric_check,
    step,
)


def _check_bound() -> tuple[float, str]:
    """For the normal mean the sample mean attains the bound exactly: sigma^2/n."""
    rng = np.random.default_rng(53)
    n, sigma, reps = 40, 2.0, 20000
    samples = rng.normal(loc=1.0, scale=sigma, size=(reps, n))
    empirical = float(np.var(samples.mean(axis=1), ddof=1))
    bound = sigma**2 / n
    # Monte Carlo error of a variance estimate is about sqrt(2/reps) relative.
    tolerance = 4.0 * np.sqrt(2.0 / reps)
    relative_gap = abs(empirical - bound) / bound
    shortfall = max(relative_gap - tolerance, 0.0)
    return shortfall, (
        f"Cramer-Rao bound sigma^2/n = {bound:.6f}; sampling variance of the mean over "
        f"{reps} replications = {empirical:.6f} (relative gap {relative_gap:.4f}, "
        f"Monte Carlo tolerance {tolerance:.4f})"
    )


PROOF = ProofSpec(
    id="inference.cramer_rao.bound",
    kind=EvidenceType.SYMBOLIC_DERIVATION,
    level=Level.ADVANCED,
    title="Cramer-Rao: no unbiased estimator can beat the inverse information",
    claim=(
        "Under regularity conditions, any unbiased estimator T of theta satisfies "
        "Var(T) >= 1 / I(theta), where I(theta) is the Fisher information. Equality "
        "holds only when the score is a linear function of T."
    ),
    intuition=(
        "The score measures how sharply the likelihood reacts when theta moves. An "
        "unbiased estimator must track theta exactly, so it has to be correlated with "
        "that reaction. A correlation cannot exceed one, and writing that fact down is "
        "the entire bound: a flat likelihood - little information - forces a large "
        "variance."
    ),
    assumptions=(
        assume(
            "regularity",
            "The support of the density does not depend on theta, and differentiation "
            "under the integral sign is permitted.",
            if_violated=(
                "The bound simply does not apply. For a uniform distribution on [0, theta] "
                "the support moves with theta, and the maximum order statistic converges at "
                "rate 1/n - far faster than any information bound would suggest."
            ),
        ),
        assume(
            "unbiased",
            "T is unbiased for theta at the value considered: E[T] = theta for all theta "
            "in a neighbourhood.",
            essential=True,
            if_violated=(
                "Biased estimators are not bounded below by 1/I(theta) and can have much "
                "smaller variance - shrinkage and ridge estimators buy their lower mean "
                "squared error precisely by giving up unbiasedness."
            ),
        ),
        assume(
            "positive_information",
            "0 < I(theta) < infinity.",
            if_violated="With zero information theta is not identified and the bound is vacuous.",
        ),
    ),
    steps=(
        step(
            "score",
            "Define the score and record that it has mean zero.",
            equation=r"S(\theta) = \frac{\partial}{\partial\theta}\log f(X;\theta), "
            r"\qquad E[S(\theta)] = 0",
            why=(
                "Differentiating the identity that the density integrates to one, and "
                "exchanging derivative and integral, gives E[S] = 0. This is where the "
                "regularity assumption does its work."
            ),
            kind=StepKind.DEFINITION,
            uses=("regularity",),
        ),
        step(
            "information",
            "Define the Fisher information as the variance of the score.",
            equation=r"I(\theta) = \operatorname{Var}(S) = E[S^2] "
            r"= -E\Big[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\Big]",
            why=(
                "The variance equals the second moment because the mean is zero; the "
                "second form follows by differentiating the score identity once more, "
                "again under regularity."
            ),
            kind=StepKind.DEFINITION,
            uses=("score", "positive_information"),
            visual=VisualAction.DRAW_LEVEL_CURVES,
        ),
        step(
            "differentiate_unbiasedness",
            "Differentiate the unbiasedness condition with respect to theta.",
            equation=r"\frac{\partial}{\partial\theta}\int T(x)f(x;\theta)\,dx "
            r"= \int T(x)\,\frac{\partial f}{\partial\theta}\,dx = 1",
            why="E[T] = theta holds identically in theta, so both sides may be differentiated.",
            kind=StepKind.CALCULUS,
            uses=("unbiased", "regularity"),
        ),
        step(
            "covariance",
            "Rewrite that derivative as a covariance between T and the score.",
            equation=r"\int T\,\frac{\partial f}{\partial\theta}\,dx "
            r"= \int T\,S\,f\,dx = E[TS] = \operatorname{Cov}(T, S) = 1",
            why=(
                "The log-derivative trick: (df/dtheta) = S f. The expectation equals the "
                "covariance because the score has mean zero. So an unbiased estimator must "
                "have unit covariance with the score - it cannot ignore it."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("differentiate_unbiasedness", "score"),
            visual=VisualAction.HIGHLIGHT_TERM,
        ),
        step(
            "cauchy_schwarz",
            "Apply the Cauchy-Schwarz inequality to that covariance.",
            equation=r"1 = \operatorname{Cov}(T, S)^2 \le \operatorname{Var}(T)\,\operatorname{Var}(S) "
            r"= \operatorname{Var}(T)\,I(\theta)",
            why=(
                "Equivalently: the squared correlation between T and S cannot exceed one. "
                "Everything before this step was setting up the two quantities; this is "
                "the inequality itself."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("covariance", "information"),
        ),
        step(
            "bound",
            "Rearrange.",
            equation=r"\operatorname{Var}(T) \ge \frac{1}{I(\theta)}",
            why="Dividing by the information, which is strictly positive and finite.",
            kind=StepKind.CONCLUSION,
            uses=("cauchy_schwarz", "positive_information"),
        ),
        step(
            "equality",
            "Equality requires the score to be a linear function of the estimator.",
            equation=r"\operatorname{Var}(T) = \frac{1}{I(\theta)} \iff "
            r"S(\theta) = a(\theta)\,\big(T - \theta\big)",
            why=(
                "Cauchy-Schwarz is tight only for perfectly correlated variables. This "
                "condition holds exactly in the exponential family with T as the natural "
                "sufficient statistic, which is why efficient estimators are rare outside it."
            ),
            kind=StepKind.CONCLUSION,
            uses=("cauchy_schwarz",),
        ),
        step(
            "iid_case",
            "For n independent observations the information adds, so the bound falls like 1/n.",
            equation=r"I_n(\theta) = n\,I_1(\theta) \implies "
            r"\operatorname{Var}(T) \ge \frac{1}{n\,I_1(\theta)}",
            why=(
                "The score of a sample is the sum of the individual scores, and independent "
                "mean-zero terms have variances that add. This is the origin of the "
                "familiar root-n rate."
            ),
            kind=StepKind.CONCLUSION,
            uses=("information", "bound"),
        ),
    ),
    conclusion=(
        "The inverse Fisher information is a floor on the variance of every unbiased "
        "estimator. It sets the standard that 'efficiency' is measured against, explains "
        "why the maximum likelihood estimator's asymptotic variance is 1/I(theta), and "
        "shows why a flat likelihood and an imprecise estimate are the same statement."
    ),
    limitations=(
        "The bound constrains unbiased estimators only, and a biased estimator can have "
        "smaller mean squared error - the James-Stein estimator dominates the sample mean "
        "in three dimensions or more despite this bound. Regularity is essential and fails "
        "whenever the support depends on the parameter, where estimators can converge much "
        "faster than 1/sqrt(n). Attaining the bound in finite samples is special to the "
        "exponential family; elsewhere it is only approached asymptotically. The bound also "
        "assumes the model is correct: under misspecification the information matrix "
        "equality fails and the relevant variance is the sandwich form, not 1/I. Finally it "
        "is a statement about variance at one parameter value, not about how the estimator "
        "behaves across the whole parameter space."
    ),
    prerequisites=(),
    concept_ids=("inference.cramer_rao", "inference.mle"),
    references=(
        Reference(
            "Rao, C. R. (1945). Information and the accuracy attainable in the estimation "
            "of statistical parameters. Bulletin of the Calcutta Mathematical Society 37, "
            "81-91.",
            kind="paper",
        ),
        Reference(
            "Lehmann, E. L. and Casella, G. (1998). Theory of Point Estimation, 2nd ed., "
            "ch. 2.",
            kind="book",
        ),
    ),
    checks=(
        numeric_check(
            "normal_mean_attains_the_bound",
            "Simulate the sampling distribution of the sample mean and compare its "
            "variance with sigma^2/n, the Cramer-Rao bound for a normal mean.",
            _check_bound,
            tol=1e-12,
        ),
    ),
)
