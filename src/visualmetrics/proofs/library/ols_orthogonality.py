"""OLS residuals are orthogonal to the regressors."""

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


def _check_orthogonality() -> tuple[float, str]:
    rng = np.random.default_rng(11)
    n, k = 60, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k))])
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=2.0, size=n)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    worst = float(np.max(np.abs(X.T @ e)))
    return worst, f"max |X'e| = {worst:.3e} over {n} observations and {X.shape[1]} columns"


PROOF = ProofSpec(
    id="regression.ols.residual_orthogonality",
    kind=EvidenceType.GEOMETRIC_PROOF,
    level=Level.INTERMEDIATE,
    title="OLS residuals are orthogonal to every regressor",
    claim=(
        "If X has full column rank and b is the ordinary least squares coefficient "
        "vector, then the residual vector e = y - Xb satisfies X'e = 0 exactly, in "
        "every sample."
    ),
    intuition=(
        "Least squares picks the point of the regressor plane closest to y. If the "
        "residual still leaned in any direction lying in that plane, you could slide "
        "along that direction and get closer - so at the optimum it leans in none of "
        "them: it stands at a right angle to the whole plane."
    ),
    assumptions=(
        assume(
            "full_rank",
            "X (n x k) has rank k, so X'X is invertible and the minimiser is unique.",
            if_violated=(
                "The projection of y still exists and is unique, but the coefficient "
                "vector b is not: infinitely many b give the same fitted values. "
                "Orthogonality survives; identification does not."
            ),
        ),
        assume(
            "euclidean",
            "Fit is measured by the Euclidean (sum of squares) norm.",
            if_violated=(
                "Under an absolute-error or quantile loss the fitted point is a "
                "different projection and X'e = 0 no longer holds."
            ),
        ),
    ),
    steps=(
        step(
            "objective",
            "Write the least squares problem as squared Euclidean distance from y to "
            "the column space of X.",
            equation=r"S(b) = \|y - Xb\|^2 = (y - Xb)'(y - Xb)",
            why="Definition of ordinary least squares; the norm is the Euclidean one.",
            kind=StepKind.SETUP,
            uses=("euclidean",),
            visual=VisualAction.DRAW_SUBSPACE,
            subspace="col(X)",
            vector="y",
        ),
        step(
            "expand",
            "Expand the quadratic form.",
            equation=r"S(b) = y'y - 2b'X'y + b'X'Xb",
            why="b'X'y is a scalar, so it equals its own transpose y'Xb.",
            kind=StepKind.ALGEBRA,
        ),
        step(
            "gradient",
            "Differentiate with respect to b.",
            equation=r"\nabla_b S(b) = -2X'y + 2X'Xb",
            why=(
                "S is a convex quadratic in b, so its stationary point is the global "
                "minimum - no second-order check beyond X'X being positive "
                "semi-definite is required."
            ),
            kind=StepKind.CALCULUS,
        ),
        step(
            "normal_equations",
            "Set the gradient to zero: the normal equations.",
            equation=r"X'Xb = X'y",
            why="First-order condition at the minimiser.",
            kind=StepKind.KEY_INSIGHT,
            uses=("full_rank", "gradient"),
        ),
        step(
            "solve",
            "Full column rank makes X'X invertible, so the minimiser is unique.",
            equation=r"b = (X'X)^{-1}X'y",
            why="Assumption full_rank.",
            kind=StepKind.SUBSTITUTION,
            uses=("full_rank", "normal_equations"),
            visual=VisualAction.PROJECT_ONTO_SUBSPACE,
        ),
        step(
            "residual",
            "Substitute the residual definition into the normal equations.",
            equation=r"X'e = X'(y - Xb) = X'y - X'Xb = X'y - X'y = 0",
            why="Directly from the normal equations - nothing statistical is used.",
            kind=StepKind.SUBSTITUTION,
            uses=("normal_equations",),
            visual=VisualAction.MARK_RIGHT_ANGLE,
        ),
        step(
            "geometry",
            "Every column of X is therefore perpendicular to e, so e is perpendicular "
            "to the whole column space, and Xb = Py is the orthogonal projection of y.",
            equation=r"P = X(X'X)^{-1}X', \quad M = I - P, \quad e = My, \quad PM = 0",
            why="P and M are symmetric and idempotent, and their ranges are orthogonal complements.",
            kind=StepKind.GEOMETRY,
            uses=("residual", "solve"),
            visual=VisualAction.DECOMPOSE_VECTOR,
        ),
        step(
            "pythagoras",
            "Orthogonality splits the total sum of squares exactly.",
            equation=r"\|y\|^2 = \|Py\|^2 + \|My\|^2",
            why="Pythagoras applied to y = Py + My with the two parts orthogonal.",
            kind=StepKind.CONCLUSION,
            uses=("geometry",),
            visual=VisualAction.HIGHLIGHT_TRIANGLE,
        ),
    ),
    conclusion=(
        "X'e = 0 is an algebraic identity of the least squares fit. It holds in every "
        "sample, for every data set, and is what makes the sum-of-squares "
        "decomposition (and therefore R-squared) exact."
    ),
    limitations=(
        "This is orthogonality in the sample, not exogeneity in the population. X'e = 0 "
        "holds by construction even when the model is badly misspecified and even when "
        "E[u | X] is not zero, so it can never be used as evidence that the regressors "
        "are exogenous. If the regression includes an intercept, the residuals also sum "
        "to zero for the same mechanical reason - again, not a diagnostic. It also says "
        "nothing about the residuals being independent, homoskedastic or normal."
    ),
    prerequisites=("math.projection.shortest_distance",),
    concept_ids=("regression.ols_geometry", "regression.simple_linear"),
    references=(
        Reference("Greene, W. H. (2018). Econometric Analysis, 8th ed., ch. 3.", kind="book"),
        Reference("Strang, G. (2016). Introduction to Linear Algebra, ch. 4.", kind="book"),
    ),
    checks=(
        numeric_check(
            "orthogonality_residual",
            "Fit OLS on simulated data and measure the largest element of X'e.",
            _check_orthogonality,
            tol=1e-9,
        ),
    ),
)
