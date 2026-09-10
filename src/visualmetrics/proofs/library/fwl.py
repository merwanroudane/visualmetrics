"""The Frisch-Waugh-Lovell theorem."""

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


def _check_fwl() -> tuple[float, str]:
    rng = np.random.default_rng(7)
    n = 120
    X1 = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    X2 = (X1[:, 1:] @ np.array([[0.6, -0.3], [0.2, 0.8]])) + rng.normal(scale=1.0, size=(n, 2))
    X = np.column_stack([X1, X2])
    y = X @ np.array([1.0, 0.5, -1.0, 2.0, -0.7]) + rng.normal(scale=1.5, size=n)

    full = np.linalg.lstsq(X, y, rcond=None)[0][X1.shape[1]:]
    P1 = X1 @ np.linalg.pinv(X1)
    M1 = np.eye(n) - P1
    partial = np.linalg.lstsq(M1 @ X2, M1 @ y, rcond=None)[0]
    gap = float(np.max(np.abs(full - partial)))
    return gap, (
        f"full-regression coefficients {np.round(full, 6).tolist()} vs partialled-out "
        f"{np.round(partial, 6).tolist()}; max gap {gap:.3e}"
    )


PROOF = ProofSpec(
    id="regression.fwl.theorem",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.ADVANCED,
    title="Frisch-Waugh-Lovell: partialling out reproduces the multiple-regression coefficient",
    claim=(
        "In the regression y = X1 b1 + X2 b2 + e, the least squares estimate b2 is "
        "identical to the estimate obtained by regressing the residuals of y on X1 "
        "against the residuals of X2 on X1. The residual vector e is identical too."
    ),
    intuition=(
        "'Controlling for X1' is not a vague adjustment. It literally means: strip out "
        "of both y and X2 everything X1 can explain, then look at what is left. The "
        "multiple regression does exactly this in one step."
    ),
    assumptions=(
        assume(
            "full_rank",
            "The stacked design X = [X1 X2] has full column rank.",
            if_violated=(
                "X2 lies partly inside the span of X1, so M1 X2 loses rank and b2 is "
                "not identified - the partialled-out regressor is degenerate."
            ),
        ),
        assume(
            "same_sample",
            "Both regressions are computed on exactly the same observations.",
            essential=True,
            if_violated=(
                "Dropping different rows (for example through differing missing-value "
                "handling) breaks the identity immediately."
            ),
        ),
    ),
    steps=(
        step(
            "partition",
            "Write the normal equations for the partitioned design.",
            equation=(
                r"\begin{pmatrix} X_1'X_1 & X_1'X_2 \\ X_2'X_1 & X_2'X_2 \end{pmatrix}"
                r"\begin{pmatrix} b_1 \\ b_2 \end{pmatrix} = "
                r"\begin{pmatrix} X_1'y \\ X_2'y \end{pmatrix}"
            ),
            why=(
                "X'Xb = X'y written block by block, for one fixed set of observations - "
                "both regressions compared later must use exactly these rows."
            ),
            kind=StepKind.SETUP,
            uses=("full_rank", "same_sample"),
        ),
        step(
            "annihilator",
            "Define the residual-maker for X1.",
            equation=r"P_1 = X_1(X_1'X_1)^{-1}X_1', \quad M_1 = I - P_1",
            why="Definition; M1 z is the residual from regressing z on X1.",
            kind=StepKind.DEFINITION,
            visual=VisualAction.DRAW_SUBSPACE,
            subspace="col(X1)",
        ),
        step(
            "properties",
            "M1 is symmetric, idempotent, and annihilates X1.",
            equation=r"M_1 = M_1' = M_1'M_1, \qquad M_1X_1 = 0",
            why=(
                "P1 is an orthogonal projection, so I - P1 is one too; and P1 X1 = X1 "
                "because the columns of X1 are already in their own span."
            ),
            kind=StepKind.GEOMETRY,
            uses=("annihilator",),
            visual=VisualAction.MARK_RIGHT_ANGLE,
        ),
        step(
            "premultiply",
            "Solve the first block for b1 and substitute it into the second block. "
            "Equivalently: premultiply the fitted equation by M1.",
            equation=r"M_1y = M_1X_1b_1 + M_1X_2b_2 + M_1e = M_1X_2b_2 + e",
            why=(
                "M1 X1 b1 vanishes by the annihilation property, and M1 e = e because "
                "e is already orthogonal to X1, hence untouched by the projection."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("properties", "partition"),
            visual=VisualAction.DECOMPOSE_VECTOR,
        ),
        step(
            "orthogonal_residual",
            "e is orthogonal to X2 as well, so e is the least squares residual of the "
            "partialled-out regression.",
            equation=r"X_2'e = 0 \implies (M_1X_2)'\,(M_1y - M_1X_2 b_2) = X_2'M_1e = X_2'e = 0",
            why=(
                "Orthogonality of OLS residuals to all regressors, plus idempotency "
                "and symmetry of M1."
            ),
            kind=StepKind.ALGEBRA,
            uses=("premultiply", "properties"),
        ),
        step(
            "normal_equation_short",
            "That is exactly the normal equation of a regression of M1 y on M1 X2.",
            equation=r"(M_1X_2)'(M_1X_2)\,b_2 = (M_1X_2)'(M_1y)",
            why="Rearranging the previous line.",
            kind=StepKind.SUBSTITUTION,
            uses=("orthogonal_residual",),
        ),
        step(
            "solve",
            "Its solution is unique, so it must be b2 itself.",
            equation=r"b_2 = (X_2'M_1X_2)^{-1}X_2'M_1y",
            why=(
                "X2'M1X2 is invertible under full_rank, and M1'M1 = M1 collapses the "
                "two projections into one."
            ),
            kind=StepKind.CONCLUSION,
            uses=("normal_equation_short", "full_rank"),
        ),
        step(
            "geometry_summary",
            "Geometrically: the multiple-regression slope on X2 is the slope in the "
            "subspace orthogonal to X1.",
            equation=r"b_2 = \frac{\langle M_1X_2,\; M_1y\rangle}{\|M_1X_2\|^2} \ \ (k_2 = 1)",
            why="The scalar case of the previous line.",
            kind=StepKind.GEOMETRY,
            uses=("solve",),
            visual=VisualAction.PROJECT_ONTO_SUBSPACE,
        ),
    ),
    conclusion=(
        "b2 and e from the full regression coincide exactly with those from regressing "
        "M1 y on M1 X2. This is what gives added-variable plots their meaning, explains "
        "why the within transformation reproduces fixed-effects estimates, and underlies "
        "the partialling-out step of double machine learning."
    ),
    limitations=(
        "The identity is about point estimates and residuals only. Standard errors from "
        "the two-step regression are wrong unless the degrees of freedom are corrected "
        "for the columns of X1 that were partialled out, since the short regression "
        "believes it estimated fewer parameters than it did. The theorem is pure linear "
        "algebra: it does not make b2 causal, unbiased or consistent, and it says nothing "
        "about what happens if a relevant variable is missing from both X1 and X2."
    ),
    prerequisites=("regression.ols.residual_orthogonality",),
    concept_ids=("regression.fwl", "panel.fixed_vs_random", "causal.dml"),
    references=(
        Reference(
            "Frisch, R. and Waugh, F. V. (1933). Partial time regressions as compared "
            "with individual trends. Econometrica 1(4), 387-401.",
            kind="paper",
            doi="10.2307/1907330",
        ),
        Reference(
            "Lovell, M. C. (1963). Seasonal adjustment of economic time series. "
            "Journal of the American Statistical Association 58(304), 993-1010.",
            kind="paper",
            doi="10.1080/01621459.1963.10480682",
        ),
    ),
    checks=(
        numeric_check(
            "coefficient_equality",
            "Compare b2 from the full regression with b2 from the partialled-out regression.",
            _check_fwl,
            tol=1e-8,
        ),
    ),
)
