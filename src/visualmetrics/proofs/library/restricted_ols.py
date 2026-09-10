"""Restricted least squares and the F statistic."""

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


def _check_f_identity() -> tuple[float, str]:
    """The two standard F formulas - sums of squares and Wald - must agree exactly."""
    rng = np.random.default_rng(67)
    n, k, q = 120, 4, 2
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    y = X @ np.array([1.0, 2.0, -0.5, 0.8]) + rng.normal(scale=1.5, size=n)
    R = np.zeros((q, k))
    R[0, 1] = 1.0
    R[1, 2] = 1.0
    r = np.array([2.0, 0.0])

    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    s2 = float(e @ e) / (n - k)

    # Wald form
    middle = np.linalg.inv(R @ XtX_inv @ R.T)
    diff = R @ b - r
    wald = float(diff @ middle @ diff) / (q * s2)

    # Restricted least squares, then the sum-of-squares form
    lam = middle @ diff
    b_r = b - XtX_inv @ R.T @ lam
    e_r = y - X @ b_r
    ssr_r, ssr_u = float(e_r @ e_r), float(e @ e)
    ssq = (ssr_r - ssr_u) / q / s2

    gap = abs(wald - ssq)
    restriction_error = float(np.max(np.abs(R @ b_r - r)))
    return gap + restriction_error, (
        f"Wald form F = {wald:.8f}, sum-of-squares form F = {ssq:.8f} (gap {gap:.3e}); "
        f"the restricted estimator satisfies Rb = r to {restriction_error:.3e}"
    )


PROOF = ProofSpec(
    id="regression.restricted.f_statistic",
    kind=EvidenceType.SYMBOLIC_DERIVATION,
    level=Level.ADVANCED,
    title="Restricted least squares and the two faces of the F statistic",
    claim=(
        "Minimising the sum of squares subject to R b = r gives b_R = b - (X'X)^-1 R' "
        "[R (X'X)^-1 R']^-1 (R b - r). The resulting F statistic can be written either "
        "as a scaled increase in the residual sum of squares or as a Wald quadratic form, "
        "and the two expressions are algebraically identical."
    ),
    intuition=(
        "A restriction confines the coefficient vector to a smaller subspace. The fit can "
        "only get worse, and the amount it worsens measures how far the data had to be "
        "pushed to satisfy the restriction. Comparing that push with ordinary noise is the "
        "F test."
    ),
    assumptions=(
        assume(
            "full_rank_design",
            "X has full column rank k.",
            if_violated="(X'X)^-1 does not exist and neither estimator is defined.",
        ),
        assume(
            "independent_restrictions",
            "R is q x k with rank q: the restrictions are not redundant or contradictory.",
            if_violated=(
                "R (X'X)^-1 R' is singular and the multiplier cannot be solved for. "
                "Redundant restrictions must be dropped before testing."
            ),
        ),
        assume(
            "spherical_normal",
            "Errors are independent, homoskedastic and normal, given X.",
            essential=True,
            if_violated=(
                "The estimator algebra is unaffected, but the exact F distribution is lost. "
                "Under heteroskedasticity or clustering only the Wald form survives, with "
                "the middle matrix replaced by a robust covariance estimate - which is why "
                "the sum-of-squares formula must not be used with robust standard errors."
            ),
        ),
    ),
    steps=(
        step(
            "lagrangian",
            "Write the constrained problem with a multiplier.",
            equation=r"\mathcal{L}(b, \lambda) = (y - Xb)'(y - Xb) + 2\lambda'(Rb - r)",
            why="The factor 2 is cosmetic and cancels the 2 from differentiating the quadratic form.",
            kind=StepKind.SETUP,
            uses=("full_rank_design", "independent_restrictions"),
            visual=VisualAction.DRAW_CONSTRAINT_SET,
        ),
        step(
            "first_order",
            "Set both derivatives to zero.",
            equation=r"-2X'(y - Xb_R) + 2R'\lambda = 0, \qquad Rb_R - r = 0",
            why="Stationarity in b and feasibility in lambda; the objective is convex, so this is the minimum.",
            kind=StepKind.CALCULUS,
            uses=("lagrangian",),
        ),
        step(
            "solve_for_b",
            "Solve the first condition for the restricted estimator in terms of the multiplier.",
            equation=r"b_R = (X'X)^{-1}X'y - (X'X)^{-1}R'\lambda = b - (X'X)^{-1}R'\lambda",
            why="Recognising the unrestricted estimator b in the first term.",
            kind=StepKind.ALGEBRA,
            uses=("first_order", "full_rank_design"),
        ),
        step(
            "solve_for_lambda",
            "Impose feasibility to pin the multiplier down.",
            equation=r"Rb_R = r \implies \lambda = \big[R(X'X)^{-1}R'\big]^{-1}(Rb - r)",
            why=(
                "Substituting the previous line into Rb_R = r. The inverse exists because "
                "R has full row rank and (X'X)^-1 is positive definite."
            ),
            kind=StepKind.SUBSTITUTION,
            uses=("solve_for_b", "independent_restrictions"),
        ),
        step(
            "restricted_estimator",
            "Combine: the restricted estimator is the unrestricted one, corrected in "
            "proportion to how badly it violated the restriction.",
            equation=r"b_R = b - (X'X)^{-1}R'\big[R(X'X)^{-1}R'\big]^{-1}(Rb - r)",
            why=(
                "Substituting lambda back. If b already satisfies the restriction the "
                "correction is exactly zero and the two estimators coincide."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("solve_for_b", "solve_for_lambda"),
            visual=VisualAction.PROJECT_ONTO_SUBSPACE,
        ),
        step(
            "ssr_gap",
            "Compute how much the fit deteriorates.",
            equation=r"\text{SSR}_R - \text{SSR}_U = (b - b_R)'X'X(b - b_R) "
            r"= (Rb - r)'\big[R(X'X)^{-1}R'\big]^{-1}(Rb - r)",
            why=(
                "Expand SSR_R = (y - Xb_R)'(y - Xb_R) around b, using X'(y - Xb) = 0 to kill "
                "the cross term - the orthogonality of the unrestricted residual is what "
                "makes this exact. Then substitute the correction term."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("restricted_estimator",),
            visual=VisualAction.HIGHLIGHT_TRIANGLE,
        ),
        step(
            "two_forms",
            "Divide by q and by the unrestricted variance estimate: the two familiar "
            "formulas are the same expression.",
            equation=r"F = \frac{(\text{SSR}_R - \text{SSR}_U)/q}{\text{SSR}_U/(n-k)} "
            r"= \frac{(Rb - r)'\big[R(X'X)^{-1}R'\big]^{-1}(Rb - r)}{q\,s^2}",
            why="Immediate from the previous line; nothing distributional has been used yet.",
            kind=StepKind.CONCLUSION,
            uses=("ssr_gap",),
        ),
        step(
            "distribution",
            "Under normality the two quadratic forms are independent chi-squares, so the "
            "ratio has an exact F distribution.",
            equation=r"F \sim F_{q,\,n-k} \quad \text{under } H_0: Rb = r",
            why=(
                "The numerator is a chi-square with q degrees of freedom, the denominator a "
                "chi-square with n - k, and they are independent because they are quadratic "
                "forms in orthogonal projections of the same normal vector."
            ),
            kind=StepKind.CONCLUSION,
            uses=("two_forms", "spherical_normal"),
            visual=VisualAction.SHADE_TAIL,
        ),
        step(
            "t_special_case",
            "One restriction reproduces the square of the t statistic.",
            equation=r"q = 1 \implies F = t^2, \qquad F_{1,\,n-k} = \big(t_{n-k}\big)^2",
            why="With a single row the quadratic form collapses to a squared standardised coefficient.",
            kind=StepKind.CONCLUSION,
            uses=("two_forms",),
        ),
    ),
    conclusion=(
        "Restricted least squares is the unrestricted estimator pulled onto the "
        "restriction, by an amount proportional to the violation. The F statistic measures "
        "that pull, and the sum-of-squares and Wald formulations are two ways of writing "
        "one quantity - as long as the classical error assumptions hold."
    ),
    limitations=(
        "The exact F distribution needs normal, homoskedastic, independent errors; without "
        "them the statistic is only asymptotically chi-square after scaling, and the "
        "sum-of-squares form is no longer valid at all - with robust or clustered "
        "covariance only the Wald form applies. The test says whether the data are "
        "compatible with the restriction, never that the restriction is true: failing to "
        "reject is not evidence for it, especially in small samples where power is low. "
        "Imposing a false restriction biases every coefficient, while imposing a true one "
        "lowers variance - the test alone cannot tell you which case you are in. And a "
        "restriction that is statistically rejected may still be economically negligible."
    ),
    prerequisites=("regression.ols.residual_orthogonality",),
    concept_ids=("regression.restricted", "regression.gauss_markov", "inference.hypothesis_testing"),
    references=(
        Reference("Greene, W. H. (2018). Econometric Analysis, 8th ed., ch. 5.", kind="book"),
        Reference("Davidson, R. and MacKinnon, J. G. (2004). Econometric Theory and Methods, ch. 4.",
                  kind="book"),
    ),
    checks=(
        numeric_check(
            "wald_and_sum_of_squares_agree",
            "Compute the F statistic both ways on simulated data and confirm the "
            "restricted estimator satisfies the restriction exactly.",
            _check_f_identity,
            tol=1e-8,
        ),
    ),
)
