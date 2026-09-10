"""Consistency of the instrumental variables estimator."""

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
    id="econometrics.iv.consistency",
    kind=EvidenceType.FORMAL_PROOF,
    level=Level.ADVANCED,
    title="Instrumental variables are consistent exactly where OLS is not",
    claim=(
        "When a regressor is correlated with the error, OLS converges to the wrong value, "
        "and the size of the error is the inverse regressor second-moment matrix times the "
        "regressor-error covariance. An instrument that is uncorrelated with the error and "
        "correlated with the regressor delivers an estimator that converges to the true "
        "coefficient."
    ),
    intuition=(
        "OLS attributes to X everything that moves with X - including the part of the "
        "error that moves with it. An instrument supplies a source of variation in X that "
        "has no channel to the error, so the movement it induces in y can only have come "
        "through X. Using only that slice of the variation restores the right answer."
    ),
    assumptions=(
        assume(
            "model",
            "y = X beta + u, with the sample second moments converging: X'X/n -> Q_xx, "
            "invertible.",
            if_violated="Without a stable limiting second-moment matrix the law of large numbers step fails.",
        ),
        assume(
            "endogeneity",
            "plim X'u/n = q, not zero: at least one regressor is correlated with the error.",
            essential=False,
            if_violated="If q = 0 then OLS is already consistent and instrumenting only costs efficiency.",
        ),
        assume(
            "exogeneity",
            "plim Z'u/n = 0: the instruments are uncorrelated with the error.",
            if_violated=(
                "The IV estimator is inconsistent too, and its asymptotic bias is divided "
                "by the instrument's relevance - so a slightly invalid but weak instrument "
                "can be far worse than OLS."
            ),
        ),
        assume(
            "relevance",
            "plim Z'X/n = Q_zx has full rank: the instruments genuinely move the regressors.",
            if_violated=(
                "Q_zx is singular or nearly so. The estimator is then not identified, or "
                "only weakly: the asymptotic argument below still holds in the limit but "
                "describes finite samples badly, and the estimator is biased toward OLS "
                "with badly wrong confidence intervals."
            ),
        ),
        assume(
            "order_condition",
            "There are at least as many instruments as endogenous regressors.",
            if_violated="Q_zx cannot have full column rank; beta is not identified by these instruments.",
        ),
    ),
    steps=(
        step(
            "ols_error",
            "Write the OLS sampling error in terms of second moments.",
            equation=r"\hat\beta_{OLS} = \beta + \Big(\frac{X'X}{n}\Big)^{-1}\frac{X'u}{n}",
            why="Substitute y = X beta + u into (X'X)^{-1}X'y and divide numerator and denominator by n.",
            kind=StepKind.SETUP,
            uses=("model",),
        ),
        step(
            "ols_limit",
            "Take probability limits: the bias term does not vanish.",
            equation=r"\operatorname{plim}\hat\beta_{OLS} = \beta + Q_{xx}^{-1}q \neq \beta",
            why=(
                "By the law of large numbers each sample moment converges, and by the "
                "continuous mapping theorem and Slutsky's theorem the inverse and the "
                "product converge to the corresponding functions of the limits. "
                "Assumption endogeneity makes q non-zero, so the second term survives."
            ),
            kind=StepKind.PROBABILITY,
            uses=("ols_error", "endogeneity", "model"),
            visual=VisualAction.HIGHLIGHT_TERM,
        ),
        step(
            "no_rescue",
            "More data does not help: the inconsistency is a limit, not a variance.",
            equation=r"\hat\beta_{OLS} \xrightarrow{p} \beta + Q_{xx}^{-1}q \quad \text{as } n \to \infty",
            why=(
                "The confidence interval shrinks around the wrong number, so a large "
                "sample makes a biased conclusion look more precise rather than less wrong. "
                "This is the qualitative difference between endogeneity and noise."
            ),
            kind=StepKind.KEY_INSIGHT,
            uses=("ols_limit",),
            visual=VisualAction.TRACE_PATH,
        ),
        step(
            "iv_definition",
            "Define the just-identified IV estimator by imposing the moment condition in "
            "the sample.",
            equation=r"Z'(y - X\hat\beta_{IV}) = 0 \implies \hat\beta_{IV} = (Z'X)^{-1}Z'y",
            why=(
                "The population condition E[Z'u] = 0 is the identifying restriction; the "
                "estimator imposes its sample analogue exactly."
            ),
            kind=StepKind.DEFINITION,
            uses=("exogeneity", "relevance", "order_condition"),
        ),
        step(
            "iv_error",
            "Its sampling error has the same shape, with Z in place of the first X.",
            equation=r"\hat\beta_{IV} = \beta + \Big(\frac{Z'X}{n}\Big)^{-1}\frac{Z'u}{n}",
            why="Substitute the model into the definition and divide by n.",
            kind=StepKind.ALGEBRA,
            uses=("iv_definition", "model"),
        ),
        step(
            "iv_limit",
            "Now the second factor converges to zero while the first stays invertible.",
            equation=r"\operatorname{plim}\hat\beta_{IV} = \beta + Q_{zx}^{-1}\cdot 0 = \beta",
            why=(
                "Assumption exogeneity kills the numerator; assumption relevance keeps the "
                "inverse well behaved. Slutsky's theorem then delivers the product."
            ),
            kind=StepKind.CONCLUSION,
            uses=("iv_error", "exogeneity", "relevance"),
        ),
        step(
            "two_stage",
            "With more instruments than endogenous regressors, project first: this is 2SLS.",
            equation=r"\hat X = P_Z X, \quad P_Z = Z(Z'Z)^{-1}Z', \quad "
            r"\hat\beta_{2SLS} = (\hat X'\hat X)^{-1}\hat X'y = (X'P_ZX)^{-1}X'P_Zy",
            why=(
                "The first stage keeps only the part of X spanned by the instruments - the "
                "exogenous part - and discards the rest along with its correlation to u. "
                "The same limit argument applies with X replaced by its projection."
            ),
            kind=StepKind.CONCLUSION,
            uses=("iv_limit", "order_condition"),
            visual=VisualAction.PROJECT_ONTO_SUBSPACE,
        ),
        step(
            "weak_warning",
            "The relevance assumption is quantitative, not binary.",
            equation=r"\frac{\text{bias}(\hat\beta_{IV})}{\text{bias}(\hat\beta_{OLS})} "
            r"\approx \frac{1}{E[F_1]} \quad \text{(first-stage } F \text{ approximation)}",
            why=(
                "When the first-stage signal is weak, the inverse of Q_zx amplifies both "
                "the remaining correlation and the sampling noise. Consistency is a "
                "statement about the limit and gives no protection at any finite n."
            ),
            kind=StepKind.CONCLUSION,
            uses=("relevance", "iv_limit"),
        ),
    ),
    conclusion=(
        "Under instrument exogeneity and relevance, IV and 2SLS converge to the true "
        "coefficient while OLS converges to beta plus Q_xx^{-1}q. The correction comes "
        "entirely from replacing the endogenous variation in X with variation induced by Z."
    ),
    limitations=(
        "Consistency is asymptotic and says nothing about finite samples: with weak "
        "instruments the IV estimator is biased toward OLS, its distribution is not "
        "approximately normal, and conventional confidence intervals under-cover badly - "
        "an F statistic in the first stage below roughly 10 is the usual warning sign, and "
        "in that situation IV can easily be worse than the OLS it was meant to repair. "
        "Exogeneity of the instrument is untestable in the just-identified case; the Sargan "
        "and Hansen tests require over-identification and even then test the joint validity "
        "of the whole set, not any single instrument. IV is also less efficient than OLS "
        "when OLS happens to be consistent. Finally, with heterogeneous effects the "
        "estimand is a local average treatment effect for the subpopulation the instrument "
        "actually moves, which is not the average effect in the population and is not "
        "identified by this argument."
    ),
    prerequisites=("regression.ols.residual_orthogonality",),
    concept_ids=(
        "econometrics.endogeneity_iv",
        "econometrics.omitted_variable_bias",
        "econometrics.measurement_error",
    ),
    references=(
        Reference(
            "Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel "
            "Data, 2nd ed., ch. 5.",
            kind="book",
        ),
        Reference(
            "Stock, J. H. and Yogo, M. (2005). Testing for weak instruments in linear IV "
            "regression. In Identification and Inference for Econometric Models.",
            kind="paper",
            doi="10.1017/CBO9780511614491.006",
        ),
    ),
)
