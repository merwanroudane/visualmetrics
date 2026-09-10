"""The orthogonal projection is the closest point in a subspace."""

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


def _check_closest() -> tuple[float, str]:
    rng = np.random.default_rng(23)
    n, k = 40, 3
    X = rng.normal(size=(n, k))
    y = rng.normal(size=n)
    P = X @ np.linalg.pinv(X)
    p = P @ y
    d_star = float(np.linalg.norm(y - p))

    # Random points of the subspace, plus small perturbations around p itself.
    coefs = rng.normal(size=(5000, k)) * rng.uniform(0.01, 3.0, size=(5000, 1))
    pts = coefs @ X.T  # every row is a point of col(X)
    dists = np.linalg.norm(y - pts, axis=1)
    closer = float(np.min(dists) - d_star)
    # Pythagoras must hold exactly for each candidate.
    pyth = float(
        np.max(np.abs(dists**2 - (d_star**2 + np.linalg.norm(pts - p, axis=1) ** 2)))
    )
    excess = max(-closer, 0.0) + pyth
    return excess, (
        f"projection distance {d_star:.6f}; closest of 5000 random subspace points "
        f"{np.min(dists):.6f} (never smaller); max Pythagoras residual {pyth:.3e}"
    )


PROOF = ProofSpec(
    id="math.projection.shortest_distance",
    kind=EvidenceType.GEOMETRIC_PROOF,
    level=Level.BEGINNER,
    title="The orthogonal projection is the unique closest point in a subspace",
    claim=(
        "Let S be a subspace of a Euclidean space and let p be the orthogonal projection "
        "of y onto S. Then for every v in S, ||y - v|| >= ||y - p||, with equality only "
        "when v = p."
    ),
    intuition=(
        "Walking away from the foot of the perpendicular can only add a sideways leg to "
        "your journey, and the hypotenuse of a right triangle is always longer than "
        "either leg. There is nowhere to go inside the subspace that does not cost you."
    ),
    assumptions=(
        assume(
            "subspace",
            "S is a linear subspace: closed under addition and scalar multiplication.",
            if_violated=(
                "For a general non-convex set the nearest point may not be unique and the "
                "decomposition below fails. For a closed convex set a unique nearest point "
                "still exists, but it is characterised by a variational inequality rather "
                "than by exact orthogonality."
            ),
        ),
        assume(
            "inner_product",
            "Distance comes from an inner product: ||z||^2 = <z, z>.",
            essential=True,
            if_violated=(
                "Under the L1 or the maximum norm there is no Pythagorean identity. The "
                "closest point need not be unique and need not be the orthogonal "
                "projection - this is exactly why quantile regression and least absolute "
                "deviations behave differently from least squares."
            ),
        ),
        assume(
            "orthogonal_p",
            "p in S satisfies <y - p, s> = 0 for every s in S.",
            if_violated="p is then simply not the orthogonal projection; the claim is about that particular point.",
        ),
    ),
    steps=(
        step(
            "split",
            "Split the error at any candidate v into two pieces through p.",
            equation=r"y - v = (y - p) + (p - v)",
            why="Add and subtract p. No assumption used yet.",
            kind=StepKind.SETUP,
            visual=VisualAction.DECOMPOSE_VECTOR,
        ),
        step(
            "membership",
            "The second piece lies inside the subspace.",
            equation=r"p \in S, \ v \in S \implies p - v \in S",
            why="Assumption subspace: S is closed under subtraction.",
            kind=StepKind.GEOMETRY,
            uses=("subspace",),
            visual=VisualAction.DRAW_SUBSPACE,
        ),
        step(
            "orthogonality",
            "The first piece is perpendicular to everything in the subspace, so in "
            "particular to the second piece.",
            equation=r"\langle y - p,\ p - v\rangle = 0",
            why="Assumption orthogonal_p applied to the element p - v of S.",
            kind=StepKind.KEY_INSIGHT,
            uses=("orthogonal_p", "membership"),
            visual=VisualAction.MARK_RIGHT_ANGLE,
        ),
        step(
            "expand",
            "Expand the squared length; the cross term vanishes.",
            equation=r"\|y - v\|^2 = \|y - p\|^2 + 2\langle y-p,\ p-v\rangle + \|p - v\|^2 "
            r"= \|y - p\|^2 + \|p - v\|^2",
            why="Bilinearity of the inner product, then the previous step. This is Pythagoras.",
            kind=StepKind.ALGEBRA,
            uses=("split", "orthogonality", "inner_product"),
            visual=VisualAction.HIGHLIGHT_TRIANGLE,
        ),
        step(
            "inequality",
            "A squared length is never negative, so dropping it can only decrease the total.",
            equation=r"\|y - v\|^2 \ge \|y - p\|^2 \quad \text{for every } v \in S",
            why="||p - v||^2 >= 0.",
            kind=StepKind.CONCLUSION,
            uses=("expand",),
        ),
        step(
            "uniqueness",
            "Equality pins v down exactly.",
            equation=r"\|y - v\| = \|y - p\| \iff \|p - v\|^2 = 0 \iff v = p",
            why="An inner-product norm is zero only for the zero vector, so the closest point is unique.",
            kind=StepKind.CONCLUSION,
            uses=("expand", "inner_product"),
        ),
        step(
            "converse",
            "The converse also holds, which is why least squares produces orthogonality "
            "rather than merely being compatible with it.",
            equation=r"\|y - p\| \le \|y - v\| \ \forall v \in S \implies \langle y - p, s\rangle = 0 \ \forall s \in S",
            why=(
                "If some s in S had <y - p, s> = c not equal to 0, then moving to "
                "v = p + (c/||s||^2)s would give ||y - v||^2 = ||y - p||^2 - c^2/||s||^2, "
                "which is strictly smaller - contradicting minimality."
            ),
            kind=StepKind.CONCLUSION,
            uses=("subspace", "inner_product"),
            visual=VisualAction.TRACE_PATH,
        ),
    ),
    conclusion=(
        "Minimising Euclidean distance to a subspace and imposing orthogonality of the "
        "residual are the same condition. This single fact is what makes ordinary least "
        "squares a projection, what makes the sum-of-squares decomposition exact, and "
        "what gives conditional expectation its interpretation as the best mean-square "
        "predictor."
    ),
    limitations=(
        "Everything here depends on the norm coming from an inner product; under other "
        "norms the statement is false. It presumes the subspace is fixed and given - it "
        "says nothing about how to choose which subspace (that is, which regressors) to "
        "project onto, and the closest point in a badly chosen subspace is still a poor "
        "description of y. In infinite dimensions the subspace must additionally be closed "
        "for the projection to exist at all. Finally this is geometry, not statistics: it "
        "carries no notion of sampling error, bias or causality."
    ),
    prerequisites=(),
    concept_ids=("math.projection", "regression.ols_geometry"),
    references=(
        Reference("Strang, G. (2016). Introduction to Linear Algebra, 5th ed., ch. 4.", kind="book"),
        Reference("Luenberger, D. G. (1969). Optimization by Vector Space Methods, ch. 3.", kind="book"),
    ),
    checks=(
        numeric_check(
            "no_point_is_closer",
            "Sample 5000 points of the subspace and confirm none is closer than the "
            "projection, and that Pythagoras holds exactly for each.",
            _check_closest,
            tol=1e-9,
        ),
    ),
)
