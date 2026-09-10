"""The omitted variable bias formula."""

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


def _check_ovb() -> tuple[float, str]:
    """b_short = b_long + delta * gamma must hold exactly, in any sample."""
    rng = np.random.default_rng(41)
    n = 250
    x1 = rng.normal(size=n)
    x2 = 0.7 * x1 + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + rng.normal(size=n)
    ones = np.ones(n)
    short = np.linalg.lstsq(np.column_stack([ones, x1]), y, rcond=None)[0][1]
    long_fit = np.linalg.lstsq(np.column_stack([ones, x1, x2]), y, rcond=None)[0]
    delta = np.linalg.lstsq(np.column_stack([ones, x1]), x2, rcond=None)[0][1]
    predicted = long_fit[1] + delta * long_fit[2]
    gap = float(abs(short - predicted))
    return gap, (
        f"short regression slope {short:.8f} vs long slope plus bias term "
        f"{predicted:.8f} (delta = {delta:.4f}, gamma = {long_fit[2]:.4f}); gap {gap:.3e}"
    )


PROOF = ProofSpec(
    id="econometrics.ovb.formula",
    kind=EvidenceType.SYMBOLIC_DERIVATION,
    level=Level.INTERMEDIATE,
    title="Omitted variable bias: the short regression equals the long one plus a bias term",
    claim=(
        "If the long regression is y = b0 + b1 x1 + b2 x2 + e and x2 is omitted, the "
        "slope from the short regression satisfies b1_short = b1_long + delta * b2, "
        "where delta is the slope of x2 regressed on x1. In the population the same "
        "identity gives the asymptotic bias."
    ),
    intuition=(
        "Leaving out x2 does not delete its effect on y; it reroutes it. Whatever part "
        "of x2 moves together with x1 gets charged to x1's coefficient. The size of the "
        "misattribution is 'how much x2 moves with x1' times 'how much y responds to x2' - "
        "so it takes both a relationship with the regressor and an effect on the outcome "
        "to create bias."
    ),
    assumptions=(
        assume(
            "long_model",
            "The long regression is the model of interest: y = b0 + b1 x1 + b2 x2 + e "
            "with E[e | x1, x2] = 0.",
            if_violated=(
                "If the long model is itself misspecified, the identity below still holds "
                "algebraically but neither coefficient answers a causal question. The "
                "formula tells you the gap between two regressions, not the gap from truth."
            ),
        ),
        assume(
            "full_rank",
            "x1 and x2 are not perfectly collinear, so both regressions are well defined.",
            if_violated="delta is not identified and the long regression cannot be estimated at all.",
        ),
    ),
    steps=(
        step(
            "auxiliary",
            "Regress the omitted variable on the included one and split it into two parts.",
            equation=r"x_2 = \gamma_0 + \delta x_1 + v, \qquad \hat v \perp x_1",
            why=(
                "This is a definition, not an assumption: the auxiliary regression exists "
                "for any pair of variables, and by construction its residual is orthogonal "
                "to x1."
            ),
            kind=StepKind.DEFINITION,
            uses=("full_rank",),
            visual=VisualAction.DECOMPOSE_VECTOR,
        ),
        step(
            "substitute",
            "Substitute that decomposition into the long model.",
            equation=r"y = b_0 + b_1x_1 + b_2(\gamma_0 + \delta x_1 + v) + e "
            r"= (b_0 + b_2\gamma_0) + (b_1 + \delta b_2)x_1 + (b_2 v + e)",
            why="Expanding and collecting the terms that multiply x1.",
            kind=StepKind.SUBSTITUTION,
            uses=("auxiliary", "long_model"),
            visual=VisualAction.HIGHLIGHT_TERM,
        ),
        step(
            "orthogonal_error",
            "The new error term is orthogonal to x1, so this rearrangement *is* the short "
            "regression rather than merely resembling it.",
            equation=r"x_1'(b_2\hat v + \hat e) = b_2\,x_1'\hat v + x_1'\hat e = 0",
            why=(
                "The auxiliary residual is orthogonal to x1 by construction, and the long "
                "regression's residual is orthogonal to both regressors. A regression is "
                "identified by exactly this orthogonality condition."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("substitute", "auxiliary"),
            visual=VisualAction.MARK_RIGHT_ANGLE,
        ),
        step(
            "read_off",
            "Read the short-regression slope off the coefficient on x1.",
            equation=r"b_1^{\text{short}} = b_1 + \delta\,b_2",
            why="Uniqueness of the least squares coefficient given the orthogonality condition.",
            kind=StepKind.CONCLUSION,
            uses=("orthogonal_error",),
        ),
        step(
            "bias_term",
            "Name the two ingredients of the bias.",
            equation=r"\text{bias} = \delta \times b_2 = "
            r"\underbrace{\frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}}"
            r"_{\text{relation to the regressor}} \times "
            r"\underbrace{b_2}_{\text{effect on the outcome}}",
            why=(
                "delta is the auxiliary slope, which equals the covariance ratio. The bias "
                "vanishes if either factor is zero: an omitted variable unrelated to x1 is "
                "harmless, and so is one with no effect on y."
            ),
            kind=StepKind.CONCLUSION,
            uses=("read_off",),
        ),
        step(
            "sign",
            "The sign of the bias is therefore predictable from two judgements.",
            equation=r"\operatorname{sign}(\text{bias}) = \operatorname{sign}(\delta)\cdot"
            r"\operatorname{sign}(b_2)",
            why=(
                "Both factors have a substantive meaning, so a researcher who cannot "
                "measure x2 can still often argue which way the estimate is pulled - the "
                "standard 'the bias goes against me, so my estimate is a lower bound' "
                "argument, which is only as good as the two sign judgements behind it."
            ),
            kind=StepKind.CONCLUSION,
            uses=("bias_term",),
        ),
        step(
            "population",
            "The same algebra in the population gives the inconsistency.",
            equation=r"\operatorname{plim}\hat b_1^{\text{short}} = b_1 + "
            r"\frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}\,b_2",
            why=(
                "Replace sample moments by their limits. Note that this does not shrink "
                "with the sample size: more data estimates the wrong number more precisely."
            ),
            kind=StepKind.CONCLUSION,
            uses=("bias_term", "long_model"),
            visual=VisualAction.TRACE_PATH,
        ),
    ),
    conclusion=(
        "The short regression is not a noisy version of the long one; it is the long one "
        "plus a specific, computable term. Bias requires both correlation with the "
        "included regressor and an effect on the outcome, which is exactly why a "
        "confounder must satisfy both conditions and why controlling for a variable "
        "affected by the treatment causes trouble of its own."
    ),
    limitations=(
        "The identity compares two regressions; it does not certify that the long one is "
        "causal. If a third variable is missing from both, the long regression is biased "
        "too and the formula only describes the gap between two flawed estimates. The "
        "clean two-factor reading assumes a single omitted variable: with several, the "
        "bias terms add and can cancel, so a small measured bias does not mean small "
        "individual biases. Nothing here helps you find the omitted variable, and the sign "
        "argument is only as reliable as the two judgements it rests on. Finally, "
        "controlling for a variable that lies on the causal path from x1 to y removes part "
        "of the effect you wanted to measure - the same formula, applied where it should "
        "not be."
    ),
    prerequisites=("regression.fwl.theorem",),
    concept_ids=(
        "econometrics.omitted_variable_bias",
        "causal.dag",
        "econometrics.endogeneity_iv",
    ),
    references=(
        Reference(
            "Wooldridge, J. M. (2019). Introductory Econometrics: A Modern Approach, "
            "7th ed., section 3.3.",
            kind="book",
        ),
        Reference(
            "Angrist, J. D. and Pischke, J.-S. (2009). Mostly Harmless Econometrics, "
            "section 3.2.",
            kind="book",
        ),
    ),
    checks=(
        numeric_check(
            "short_equals_long_plus_bias",
            "Estimate all three regressions on simulated data and compare the short slope "
            "with the long slope plus delta times gamma.",
            _check_ovb,
            tol=1e-9,
        ),
    ),
)
