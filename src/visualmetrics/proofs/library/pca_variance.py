"""The first principal component maximises variance."""

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


def _check_maximiser() -> tuple[float, str]:
    """No random unit vector may beat the leading eigenvector."""
    rng = np.random.default_rng(19)
    k = 6
    A = rng.normal(size=(k, k))
    S = A @ A.T  # symmetric positive semi-definite, like a covariance matrix
    vals, vecs = np.linalg.eigh(S)
    lam1 = float(vals[-1])
    v1 = vecs[:, -1]

    directions = rng.normal(size=(20000, k))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    best_random = float(np.max(np.einsum("ij,jk,ik->i", directions, S, directions)))

    attained = float(v1 @ S @ v1)
    excess = max(best_random - lam1, 0.0) + abs(attained - lam1)
    return excess, (
        f"leading eigenvalue {lam1:.6f}; quadratic form at its eigenvector {attained:.6f}; "
        f"best of 20000 random unit directions {best_random:.6f} (never exceeds it)"
    )


PROOF = ProofSpec(
    id="multivariate.pca.variance_maximization",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.INTERMEDIATE,
    title="The first principal component is the leading eigenvector",
    claim=(
        "Among all unit-length directions a, the projected variance a'Sa is maximised by "
        "the eigenvector of S belonging to its largest eigenvalue, and the maximum value "
        "is that eigenvalue."
    ),
    intuition=(
        "A symmetric matrix stretches space by different amounts along a set of "
        "perpendicular axes. Any direction is a blend of those axes, so the stretch it "
        "experiences is a weighted average of the axis stretches - and a weighted average "
        "can never exceed the largest ingredient. Equality needs all the weight on the "
        "most-stretched axis."
    ),
    assumptions=(
        assume(
            "symmetric_psd",
            "S is a symmetric positive semi-definite matrix (a covariance or correlation matrix).",
            if_violated=(
                "Without symmetry the spectral theorem does not apply: eigenvalues may be "
                "complex and eigenvectors need not be orthogonal."
            ),
        ),
        assume(
            "unit_norm",
            "The direction is constrained to unit length, ||a|| = 1.",
            essential=True,
            if_violated=(
                "Without the constraint the objective is unbounded - simply scale a up. "
                "The normalisation is what makes the problem well posed."
            ),
        ),
    ),
    steps=(
        step(
            "problem",
            "State the optimisation problem.",
            equation=r"\max_{a} \ a'Sa \quad \text{subject to} \quad a'a = 1",
            why="Projecting the data onto direction a gives variance a'Sa.",
            kind=StepKind.SETUP,
            uses=("unit_norm",),
            visual=VisualAction.DRAW_LEVEL_CURVES,
        ),
        step(
            "spectral",
            "Apply the spectral theorem.",
            equation=r"S = Q\Lambda Q', \quad Q'Q = I, \quad "
            r"\Lambda = \operatorname{diag}(\lambda_1 \ge \cdots \ge \lambda_k \ge 0)",
            why="Assumption symmetric_psd: a real symmetric matrix has an orthonormal eigenbasis "
            "and real eigenvalues, non-negative when it is positive semi-definite.",
            kind=StepKind.DEFINITION,
            uses=("symmetric_psd",),
            visual=VisualAction.ROTATE_AXES,
        ),
        step(
            "rotate",
            "Change coordinates into the eigenbasis. The constraint is unchanged.",
            equation=r"c = Q'a \implies c'c = a'QQ'a = a'a = 1",
            why="Q is orthogonal, so it preserves length: rotating the direction does not resize it.",
            kind=StepKind.SUBSTITUTION,
            uses=("spectral", "unit_norm"),
            visual=VisualAction.ROTATE_AXES,
        ),
        step(
            "diagonal_form",
            "In those coordinates the objective is a weighted average of the eigenvalues.",
            equation=r"a'Sa = c'\Lambda c = \sum_{i=1}^{k} \lambda_i c_i^2, \qquad \sum_i c_i^2 = 1",
            why="Substituting the spectral decomposition; the weights c_i^2 are non-negative and sum to one.",
            kind=StepKind.KEY_INSIGHT,
            uses=("rotate",),
        ),
        step(
            "bound",
            "Bound the weighted average by its largest ingredient.",
            equation=r"\sum_i \lambda_i c_i^2 \le \lambda_1 \sum_i c_i^2 = \lambda_1",
            why="Every lambda_i is at most lambda_1 and every weight is non-negative.",
            kind=StepKind.ALGEBRA,
            uses=("diagonal_form",),
        ),
        step(
            "attained",
            "The bound is attained, so it is the maximum rather than merely an upper limit.",
            equation=r"a = q_1 \implies c = e_1 \implies a'Sa = \lambda_1",
            why=(
                "Choosing the leading eigenvector puts all the weight on lambda_1. A bound "
                "that is reached is a maximum."
            ),
            kind=StepKind.CONCLUSION,
            uses=("bound", "spectral"),
            visual=VisualAction.DRAW_VECTOR,
        ),
        step(
            "uniqueness",
            "The maximiser is unique up to sign when the leading eigenvalue is simple.",
            equation=r"\lambda_1 > \lambda_2 \implies a^* = \pm q_1",
            why=(
                "Equality in the bound forces c_i = 0 for every i with lambda_i < lambda_1. "
                "If lambda_1 is repeated, any unit vector in its eigenspace attains the "
                "maximum and the component direction is genuinely not identified."
            ),
            kind=StepKind.CONCLUSION,
            uses=("bound", "attained"),
        ),
        step(
            "deflation",
            "Later components follow by the same argument restricted to the orthogonal complement.",
            equation=r"\max_{a'a=1,\ a \perp q_1,\ldots,q_{m-1}} a'Sa = \lambda_m, \quad a^* = q_m",
            why="Repeat the argument with the already-used eigen-directions removed from the sum.",
            kind=StepKind.CONCLUSION,
            uses=("attained",),
        ),
    ),
    conclusion=(
        "Principal components are eigenvectors and the variance they explain is the "
        "corresponding eigenvalue. This is why the eigenvalues, divided by their total, "
        "give the proportion of variance explained, and why discarding the smallest ones "
        "costs exactly their sum in reconstruction error."
    ),
    limitations=(
        "Maximal variance is not the same as maximal usefulness: a direction can carry most "
        "of the variance and none of the signal you care about, and PCA is entirely blind "
        "to any outcome variable. The result is scale-dependent - rescaling one variable "
        "changes S and therefore changes the components, which is why standardisation is a "
        "substantive choice and not a formality. It captures only linear structure and only "
        "second moments. In practice S is estimated from a finite sample, so the leading "
        "eigenvalue is biased upward and the estimated direction is unstable when the top "
        "eigenvalues are close together. Finally, components are mathematical artefacts and "
        "carry no guarantee of being interpretable as real factors."
    ),
    prerequisites=("math.projection.shortest_distance",),
    concept_ids=("multivariate.pca",),
    references=(
        Reference(
            "Hotelling, H. (1933). Analysis of a complex of statistical variables into "
            "principal components. Journal of Educational Psychology 24(6), 417-441.",
            kind="paper",
            doi="10.1037/h0071325",
        ),
        Reference("Jolliffe, I. T. (2002). Principal Component Analysis, 2nd ed., ch. 1-2.", kind="book"),
    ),
    checks=(
        numeric_check(
            "no_direction_beats_lambda1",
            "Evaluate the quadratic form at 20000 random unit directions and confirm none "
            "exceeds the leading eigenvalue, which the eigenvector attains.",
            _check_maximiser,
            tol=1e-9,
        ),
    ),
)
