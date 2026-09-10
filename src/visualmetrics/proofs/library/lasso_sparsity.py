"""Why the lasso produces exact zeros and ridge does not."""

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


def _check_soft_threshold() -> tuple[float, str]:
    """Coordinate descent on an orthonormal design must reproduce soft-thresholding."""
    rng = np.random.default_rng(3)
    n, k = 200, 5
    Q = np.linalg.qr(rng.normal(size=(n, k)))[0]  # orthonormal columns: Q'Q = I
    y = Q @ np.array([2.0, -1.2, 0.15, 0.0, 0.6]) + rng.normal(scale=0.4, size=n)
    z = Q.T @ y  # the OLS coefficients, since (Q'Q)^-1 = I
    lam = 0.5

    closed_form = np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

    # Independent numerical minimisation of the lasso objective, coordinate by
    # coordinate on a fine grid, to confirm the closed form really is the argmin.
    grid = np.linspace(-3.0, 3.0, 240001)
    numeric = np.array(
        [grid[np.argmin(0.5 * (grid - zj) ** 2 + lam * np.abs(grid))] for zj in z]
    )
    gap = float(np.max(np.abs(closed_form - numeric)))
    zeros = int(np.sum(closed_form == 0.0))
    return gap, (
        f"soft-threshold vs grid minimiser: max gap {gap:.3e}; "
        f"{zeros} of {k} coefficients set exactly to zero at lambda = {lam}"
    )


PROOF = ProofSpec(
    id="ml.lasso.sparsity_geometry",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.ADVANCED,
    title="The lasso sets coefficients exactly to zero; ridge never does",
    claim=(
        "With an orthonormal design, the lasso solution is the soft-thresholded least "
        "squares coefficient, which is exactly zero whenever the least squares "
        "coefficient is no larger than lambda in magnitude. The ridge solution is a "
        "uniform shrinkage that is zero only when the least squares coefficient is zero."
    ),
    intuition=(
        "The absolute value has a kink at zero: its slope jumps from -lambda to +lambda "
        "without passing through the intermediate values. If the pull from the data is "
        "weaker than lambda, the kink can absorb it entirely and the coefficient stays "
        "pinned at zero. The squared penalty has no kink - its slope at zero is zero - "
        "so any non-zero pull moves the coefficient off zero."
    ),
    assumptions=(
        assume(
            "orthonormal",
            "The design satisfies X'X = I (orthonormal columns), so the objective "
            "separates into one independent problem per coefficient.",
            essential=False,
            if_violated=(
                "The coordinates no longer separate and there is no closed form. The "
                "sparsity conclusion still holds - it follows from the same subgradient "
                "condition applied coordinate-wise - but it must then be established "
                "through the KKT conditions of the full problem rather than by this "
                "one-line argument."
            ),
        ),
        assume(
            "lambda_positive",
            "lambda > 0.",
            if_violated="At lambda = 0 both penalties vanish and the solution is ordinary least squares.",
        ),
        assume(
            "convexity",
            "Both objectives are convex, so a point satisfying the first-order condition "
            "is a global minimiser.",
            if_violated="For non-convex penalties (for example the L0 or SCAD penalty) the "
            "stationarity argument only characterises local minima.",
        ),
    ),
    steps=(
        step(
            "objective",
            "Write the lasso objective and expand the quadratic term.",
            equation=r"L(b) = \tfrac12\|y - Xb\|^2 + \lambda\|b\|_1",
            why="Definition of the lasso.",
            kind=StepKind.SETUP,
        ),
        step(
            "separate",
            "With X'X = I the quadratic term separates: write z = X'y for the least "
            "squares coefficients.",
            equation=r"L(b) = \text{const} + \sum_{j=1}^{k}\Big[\tfrac12 (b_j - z_j)^2 + \lambda|b_j|\Big]",
            why=(
                "Expanding gives -b'X'y + (1/2)b'X'Xb = -b'z + (1/2)b'b, which is a sum "
                "of one-dimensional terms. Assumption orthonormal is what removes the "
                "cross terms."
            ),
            kind=StepKind.ALGEBRA,
            uses=("orthonormal", "objective"),
        ),
        step(
            "subgradient",
            "The absolute value is not differentiable at zero, so use its subdifferential.",
            equation=r"\partial |b| = \begin{cases}\{\operatorname{sign}(b)\} & b \neq 0 \\ "
            r"[-1, 1] & b = 0\end{cases}",
            why=(
                "A convex function is minimised exactly where zero belongs to its "
                "subdifferential. At the kink the subdifferential is a whole interval, "
                "and that interval is the entire mechanism behind sparsity."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("convexity",),
            visual=VisualAction.MARK_CORNER,
        ),
        step(
            "stationarity",
            "Optimality condition for coordinate j.",
            equation=r"0 \in (b_j - z_j) + \lambda\,\partial|b_j|",
            why="Zero must lie in the subdifferential of the separated objective.",
            kind=StepKind.CALCULUS,
            uses=("separate", "subgradient"),
        ),
        step(
            "zero_case",
            "Ask when b_j = 0 satisfies it.",
            equation=r"0 \in -z_j + \lambda[-1,1] \iff |z_j| \le \lambda",
            why=(
                "Substituting b_j = 0 leaves -z_j plus the interval [-lambda, lambda]. "
                "Zero lies in that shifted interval precisely when |z_j| does not exceed "
                "lambda. So the coefficient is exactly zero, not merely small, whenever "
                "the data's pull is weaker than the penalty's kink."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("stationarity", "lambda_positive"),
            visual=VisualAction.MARK_CORNER,
        ),
        step(
            "nonzero_case",
            "Otherwise the objective is differentiable at the optimum and shrinks by lambda.",
            equation=r"b_j - z_j + \lambda\operatorname{sign}(b_j) = 0 \implies "
            r"b_j = z_j - \lambda\operatorname{sign}(z_j)",
            why=(
                "When |z_j| > lambda the solution has the same sign as z_j, so the "
                "subdifferential is the single value sign(z_j)."
            ),
            kind=StepKind.ALGEBRA,
            uses=("stationarity",),
        ),
        step(
            "soft_threshold",
            "Combine both cases into the soft-thresholding operator.",
            equation=r"\hat b_j^{\text{lasso}} = \operatorname{sign}(z_j)\,(|z_j| - \lambda)_+",
            why="The two cases above are exactly the two branches of this formula.",
            kind=StepKind.CONCLUSION,
            uses=("zero_case", "nonzero_case"),
            visual=VisualAction.TRACE_PATH,
        ),
        step(
            "ridge_contrast",
            "Repeat with the squared penalty: the kink is gone, and so is the sparsity.",
            equation=r"0 = (b_j - z_j) + 2\lambda b_j \implies "
            r"\hat b_j^{\text{ridge}} = \frac{z_j}{1 + 2\lambda}",
            why=(
                "The derivative of b^2 at zero is zero, so it cannot absorb any pull. "
                "The ridge estimate is zero only if z_j is exactly zero - an event of "
                "probability zero for continuous data."
            ),
            kind=StepKind.CONCLUSION,
            uses=("convexity",),
            visual=VisualAction.DRAW_LEVEL_CURVES,
        ),
        step(
            "geometry",
            "Geometric reading of the same fact via the constrained form.",
            equation=r"\min_b \tfrac12\|y - Xb\|^2 \ \text{ s.t. } \ \|b\|_1 \le t",
            why=(
                "The elliptical level sets of the squared-error loss expand until they "
                "first touch the constraint set. The L1 ball is a cross-polytope whose "
                "vertices sit on the axes - a coefficient is zero there - and a vertex "
                "is a positive-measure target for the direction of first contact. The L2 "
                "ball is smooth, so contact at an axis point requires an exactly aligned "
                "ellipse: possible, but of measure zero."
            ),
            kind=StepKind.GEOMETRY,
            uses=("soft_threshold", "ridge_contrast"),
            visual=VisualAction.DRAW_CONSTRAINT_SET,
            penalty="l1_vs_l2",
        ),
    ),
    conclusion=(
        "The kink of the L1 penalty at zero, expressed as an interval-valued "
        "subdifferential, is what allows a coefficient to be held exactly at zero. This "
        "makes the lasso a selection method and not merely a shrinkage method. Ridge "
        "shrinks every coefficient by the same multiplicative factor and selects nothing."
    ),
    limitations=(
        "The closed form is proved only for an orthonormal design; with correlated "
        "regressors the lasso path has no closed form and must be computed numerically. "
        "Selection consistency is a separate and much stronger claim that requires "
        "additional conditions (the irrepresentable condition) and is not established "
        "here. Nothing here says the selected variables are the causally relevant ones, "
        "or that the non-zero coefficients are unbiased - they are biased toward zero by "
        "construction, which is why post-selection inference needs its own machinery. "
        "The geometric step is an illustration of the algebra, not an independent proof."
    ),
    prerequisites=("math.projection.shortest_distance",),
    concept_ids=("ml.regularization", "ml.bias_variance"),
    references=(
        Reference(
            "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. "
            "Journal of the Royal Statistical Society B 58(1), 267-288.",
            kind="paper",
            doi="10.1111/j.2517-6161.1996.tb02080.x",
        ),
        Reference(
            "Hastie, T., Tibshirani, R. and Wainwright, M. (2015). Statistical Learning "
            "with Sparsity, ch. 2.",
            kind="book",
        ),
    ),
    checks=(
        numeric_check(
            "soft_threshold_matches_argmin",
            "Minimise the separated lasso objective numerically on a grid and compare "
            "with the soft-thresholding formula.",
            _check_soft_threshold,
            tol=1e-4,
        ),
    ),
)
