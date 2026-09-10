"""The Gauss-Markov theorem."""

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


def _check_variance_gap() -> tuple[float, str]:
    """The algebraic core: Var(b~) - Var(b) must equal sigma^2 D D', which is PSD."""
    rng = np.random.default_rng(31)
    n, k = 50, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    XtX_inv = np.linalg.inv(X.T @ X)
    A = XtX_inv @ X.T

    # Build an arbitrary linear unbiased competitor: C = A + D with DX = 0.
    M = np.eye(n) - X @ A
    D = rng.normal(size=(k, n)) @ M  # rows of D lie in the null space of X'
    C = A + D
    unbiased = float(np.max(np.abs(C @ X - np.eye(k))))

    gap = C @ C.T - XtX_inv  # equals D D' if the cross terms vanish
    predicted = D @ D.T
    identity_error = float(np.max(np.abs(gap - predicted)))
    min_eig = float(np.min(np.linalg.eigvalsh((gap + gap.T) / 2)))
    excess = unbiased + identity_error + max(-min_eig, 0.0)
    return excess, (
        f"unbiasedness residual max|CX - I| = {unbiased:.3e}; "
        f"max|CC' - (X'X)^-1 - DD'| = {identity_error:.3e}; "
        f"smallest eigenvalue of the variance gap = {min_eig:.3e} (must be >= 0)"
    )


PROOF = ProofSpec(
    id="inference.gauss_markov.blue",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.ADVANCED,
    title="Gauss-Markov: OLS is the best linear unbiased estimator",
    claim=(
        "If the errors have zero conditional mean, constant variance and no correlation, "
        "then among all estimators that are linear in y and unbiased, ordinary least "
        "squares has the smallest variance - and the difference is positive "
        "semi-definite, so the statement holds for every linear combination of the "
        "coefficients at once."
    ),
    intuition=(
        "Any competing linear unbiased estimator can be written as OLS plus a detour. "
        "Unbiasedness forces the detour to be orthogonal to the regressors, which is "
        "exactly the condition that makes it uncorrelated with the OLS estimator - so it "
        "adds its own variance without cancelling any. A detour can only cost."
    ),
    assumptions=(
        assume(
            "linearity",
            "y = X beta + u with X of full column rank and treated as fixed (or conditioned on).",
            if_violated="With rank deficiency beta is not identified; with a nonlinear model the class of linear estimators is the wrong comparison set.",
        ),
        assume(
            "zero_mean",
            "E[u | X] = 0.",
            if_violated=(
                "OLS becomes biased and inconsistent. Minimum variance within a biased "
                "class is no consolation - this is the endogeneity problem, and it is not "
                "fixed by any amount of efficiency."
            ),
        ),
        assume(
            "spherical",
            "Var(u | X) = sigma^2 I: constant variance and zero correlation across observations.",
            if_violated=(
                "OLS stays unbiased but is no longer best: generalised least squares with "
                "weights Omega^{-1} is. In practice one keeps OLS and repairs the standard "
                "errors instead (robust, cluster or HAC), trading efficiency for honesty."
            ),
        ),
        assume(
            "linear_class",
            "The comparison is restricted to estimators of the form b~ = Cy with C a "
            "function of X alone.",
            essential=True,
            if_violated=(
                "Nonlinear estimators are outside the theorem entirely. Under normality "
                "OLS is in fact best among all unbiased estimators, but that is a "
                "different, stronger result requiring a distributional assumption."
            ),
        ),
    ),
    steps=(
        step(
            "ols_is_linear",
            "OLS is itself a linear estimator, with a matrix that reproduces beta exactly.",
            equation=r"b = Ay, \quad A = (X'X)^{-1}X', \quad AX = I_k",
            why="Definition of OLS under full column rank.",
            kind=StepKind.SETUP,
            uses=("linearity",),
        ),
        step(
            "unbiased_ols",
            "It is unbiased.",
            equation=r"E[b \mid X] = A\,E[y \mid X] = AX\beta = \beta",
            why="Assumption zero_mean, then AX = I.",
            kind=StepKind.PROBABILITY,
            uses=("zero_mean", "ols_is_linear"),
        ),
        step(
            "variance_ols",
            "And its variance is the familiar sandwich collapsed by sphericity.",
            equation=r"\operatorname{Var}(b \mid X) = A\,\operatorname{Var}(y \mid X)A' = \sigma^2 AA' = \sigma^2 (X'X)^{-1}",
            why="Assumption spherical; then A A' = (X'X)^{-1}X'X(X'X)^{-1} = (X'X)^{-1}.",
            kind=StepKind.PROBABILITY,
            uses=("spherical", "ols_is_linear"),
        ),
        step(
            "competitor",
            "Write an arbitrary linear competitor as OLS plus a detour.",
            equation=r"\tilde b = Cy, \qquad C = A + D",
            why="Assumption linear_class; D is defined as C - A, so nothing is lost in generality.",
            kind=StepKind.SETUP,
            uses=("linear_class", "ols_is_linear"),
            visual=VisualAction.DECOMPOSE_VECTOR,
        ),
        step(
            "unbiasedness_constraint",
            "Requiring unbiasedness for every beta forces the detour to annihilate X.",
            equation=r"E[\tilde b \mid X] = CX\beta = \beta \ \ \forall \beta \implies CX = I \implies DX = 0",
            why=(
                "CX = I must hold for all beta, not just one, so it is a matrix identity. "
                "Subtracting AX = I leaves DX = 0. This is the constraint that does all "
                "the work."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("competitor", "zero_mean", "ols_is_linear"),
            visual=VisualAction.MARK_RIGHT_ANGLE,
        ),
        step(
            "cross_term",
            "That constraint kills the cross term.",
            equation=r"AD' = (X'X)^{-1}X'D' = (X'X)^{-1}(DX)' = 0",
            why="Directly from DX = 0. Geometrically, the detour is orthogonal to the column space of X.",
            kind=StepKind.ALGEBRA,
            uses=("unbiasedness_constraint",),
        ),
        step(
            "variance_competitor",
            "So the competitor's variance is the OLS variance plus a non-negative term.",
            equation=r"\operatorname{Var}(\tilde b \mid X) = \sigma^2 CC' = \sigma^2(AA' + AD' + DA' + DD') "
            r"= \sigma^2 (X'X)^{-1} + \sigma^2 DD'",
            why="Expanding CC' and using the vanishing cross terms, then the OLS variance.",
            kind=StepKind.PROBABILITY,
            uses=("cross_term", "variance_ols", "spherical"),
        ),
        step(
            "psd",
            "The extra term is positive semi-definite.",
            equation=r"c'(DD')c = \|D'c\|^2 \ge 0 \quad \text{for every } c \in \mathbb{R}^k",
            why="D D' is a Gram matrix, so its quadratic form is a squared length.",
            kind=StepKind.ALGEBRA,
            uses=("variance_competitor",),
        ),
        step(
            "conclude",
            "Hence OLS is best, and best for every linear combination simultaneously.",
            equation=r"\operatorname{Var}(c'\tilde b) - \operatorname{Var}(c'b) = \sigma^2\|D'c\|^2 \ge 0",
            why=(
                "A positive semi-definite difference of variance matrices is exactly the "
                "statement that no linear combination of the coefficients is estimated "
                "more precisely by the competitor. Equality requires D'c = 0."
            ),
            kind=StepKind.CONCLUSION,
            uses=("psd", "variance_competitor"),
        ),
    ),
    conclusion=(
        "Under the classical error assumptions, no linear unbiased estimator beats OLS, "
        "and the optimality extends to every linear combination of the coefficients. "
        "Notably, normality is nowhere used."
    ),
    limitations=(
        "Best within a class, and the class is narrow twice over. Restricting to unbiased "
        "estimators is a real restriction: ridge regression is biased and can have "
        "strictly smaller mean squared error, which is why the theorem does not make OLS "
        "the best choice for prediction. Restricting to linear estimators is another: "
        "under non-normal errors nonlinear estimators can do better, and under heavy tails "
        "robust estimators usually do. The theorem is silent when sphericity fails, which "
        "is the normal state of affairs in applied work. Above all it is a statement about "
        "efficiency, not about identification: if E[u | X] is not zero the theorem does not "
        "apply at all, and no efficiency argument can turn a biased estimate into a causal "
        "one."
    ),
    prerequisites=("regression.ols.residual_orthogonality",),
    concept_ids=("regression.gauss_markov", "regression.ols_geometry", "econometrics.heteroskedasticity"),
    references=(
        Reference("Greene, W. H. (2018). Econometric Analysis, 8th ed., ch. 4.", kind="book"),
        Reference(
            "Hansen, B. E. (2022). Econometrics, ch. 4 (the Gauss-Markov theorem).", kind="book"
        ),
    ),
    checks=(
        numeric_check(
            "variance_gap_is_psd",
            "Construct an arbitrary linear unbiased competitor and confirm the variance "
            "gap equals sigma^2 D D' and has no negative eigenvalue.",
            _check_variance_gap,
            tol=1e-8,
        ),
    ),
)
