"""Learning objectives per concept (blueprint section 62).

Objectives are written as things a learner can *do* and that the lab can
actually support - "predict what happens to the standard error when the
instrument weakens" rather than "understand instrumental variables". Each one
names the lab controls or scenarios that let the learner check the answer, so
an objective is never a slogan.

Coverage is reported honestly: :func:`coverage` says which concepts still have
none, rather than the module pretending to be complete.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Objective", "OBJECTIVES", "for_concept", "coverage", "all_objectives"]


@dataclass(frozen=True)
class Objective:
    """One thing the learner should be able to do after the lab."""

    id: str
    concept_id: str
    statement: str
    level: str = "intermediate"
    check: str = ""
    """How the learner verifies it inside the lab."""

    @property
    def label_key(self) -> str:
        return f"education.objective.{self.id}"


def _o(concept: str, index: int, statement: str, check: str = "",
       level: str = "intermediate") -> Objective:
    return Objective(
        id=f"{concept}.{index}", concept_id=concept, statement=statement,
        check=check, level=level,
    )


_RAW: dict[str, tuple[tuple[str, str, str], ...]] = {
    # -- mathematical foundations ----------------------------------------
    "math.projection": (
        ("Explain why the closest point in a subspace is the one whose error is perpendicular to it.",
         "Move the target vector and watch the right-angle marker survive every position.", "beginner"),
        ("Decompose a vector into a fitted part and a residual part and state what each represents.",
         "Read the two lengths and confirm they satisfy Pythagoras.", "beginner"),
        ("Predict what changes when the subspace gains a dimension.",
         "Add a basis vector and watch the residual shrink but never grow.", "intermediate"),
    ),
    "math.optimization": (
        ("Distinguish a stationary point from a minimum.",
         "Run the saddle scenario and see the gradient vanish where no minimum exists.", "intermediate"),
        ("Predict the effect of the step size on convergence.",
         "Raise the learning rate until the path diverges, then find the boundary.", "intermediate"),
        ("Explain why convexity makes the first-order condition sufficient.",
         "Compare the convex and non-convex scenarios from several starting points.", "advanced"),
    ),
    # -- probability ------------------------------------------------------
    "probability.distributions": (
        ("Match a distribution's parameters to the shape of its density.",
         "Move each parameter and describe which feature of the curve it controls.", "beginner"),
        ("Distinguish the density from the distribution function.",
         "Read a probability off both panels for the same interval.", "beginner"),
        ("Explain when a normal approximation to a discrete distribution is adequate.",
         "Compare the exact and approximate tail probabilities as n grows.", "intermediate"),
    ),
    "probability.bivariate": (
        ("Distinguish independence from zero correlation.",
         "Run the dependent-but-uncorrelated scenario and read both measures.", "intermediate"),
        ("Read a conditional distribution off a joint density.",
         "Slice the joint surface and compare with the plotted conditional.", "intermediate"),
        ("Predict how the correlation reshapes the joint density's contours.",
         "Sweep rho from -0.9 to 0.9 and watch the ellipse rotate and narrow.", "beginner"),
    ),
    "probability.inequalities": (
        ("State what Chebyshev's inequality guarantees without any distributional assumption.",
         "Compare the bound with the exact tail probability for several distributions.", "intermediate"),
        ("Explain why a distribution-free bound is necessarily loose.",
         "Find the distribution that makes the bound tight and see how unusual it is.", "advanced"),
        ("Use Markov's inequality to bound a tail from a mean alone.",
         "Change the threshold and watch the bound tighten and loosen.", "intermediate"),
    ),
    "probability.order_statistics": (
        ("Predict how the distribution of the maximum changes with the sample size.",
         "Raise n and watch the maximum's density move right and narrow.", "intermediate"),
        ("Explain why the median is less affected by contamination than the mean.",
         "Add an outlier and compare the two sampling distributions.", "beginner"),
        ("Derive the density of a general order statistic from first principles.",
         "Compare the simulated histogram with the plotted beta-form density.", "advanced"),
    ),
    "probability.approximations": (
        ("Decide when a Poisson approximation to a binomial is appropriate.",
         "Vary n and p at a fixed mean and watch the approximation error.", "intermediate"),
        ("Quantify the error of a normal approximation in the tails versus the centre.",
         "Compare exact and approximate probabilities at both.", "intermediate"),
        ("Explain why a continuity correction helps a discrete-to-continuous approximation.",
         "Toggle the correction and watch the approximation error fall.", "intermediate"),
    ),
    # -- descriptive ------------------------------------------------------
    "descriptive.explorer": (
        ("Choose a summary that suits the shape of the data.",
         "Run the skewed and bimodal scenarios and compare mean, median and trimmed mean.", "beginner"),
        ("Explain what a single-number summary hides in a bimodal distribution.",
         "Read the mode count and see where the mean falls relative to the peaks.", "beginner"),
        ("Show that the mean has a breakdown point of zero and the median does not.",
         "Drag one observation toward infinity and watch each summary.", "intermediate"),
    ),
    # -- inference --------------------------------------------------------
    "inference.sampling_distributions": (
        ("Distinguish the distribution of the data from the distribution of a statistic.",
         "Compare the two histograms in the same run.", "beginner"),
        ("Predict how the standard error responds to the sample size.",
         "Quadruple n and confirm the spread halves.", "beginner"),
        ("Explain why the sampling distribution is the object inference is about.",
         "Locate the observed statistic inside its own sampling distribution.", "intermediate"),
    ),
    "inference.lln": (
        ("Distinguish convergence of a proportion from compensation of counts.",
         "Watch the running proportion settle while the count difference grows.", "beginner"),
        ("State what the law of large numbers does and does not promise about a finite run.",
         "Compare several seeds at the same n.", "intermediate"),
        ("Explain why a simulation illustrates the law but cannot prove it.",
         "Read the evidence badge on the animation.", "intermediate"),
    ),
    "inference.clt": (
        ("Distinguish the shape of the data from the shape of the sampling distribution of the mean.",
         "Run the skewed scenario and compare the two panels.", "beginner"),
        ("Judge how large n must be for a given population shape.",
         "Increase n until the normal overlay fits, for skewed and heavy-tailed cases.", "intermediate"),
        ("Explain why the theorem fails when the variance is infinite.",
         "Run the heavy-tailed scenario and watch the mean refuse to settle.", "advanced"),
    ),
    "inference.convergence_modes": (
        ("Distinguish convergence in probability from convergence in distribution.",
         "Compare the two panels for the same sequence.", "advanced"),
        ("Give a sequence that converges in distribution but not in probability.",
         "Run the counterexample scenario.", "advanced"),
        ("Explain why almost-sure convergence is stronger than convergence in probability.",
         "Follow individual paths rather than the marginal distribution.", "advanced"),
    ),
    "inference.hypothesis_testing": (
        ("State what a p-value is a probability of.",
         "Run the null scenario and read the distribution of p-values.", "beginner"),
        ("Distinguish statistical significance from practical importance.",
         "Raise n at a negligible effect and watch p fall.", "intermediate"),
        ("Explain why failing to reject is not evidence for the null.",
         "Run an underpowered true-effect scenario and count the misses.", "intermediate"),
    ),
    "inference.confidence_intervals": (
        ("State the repeated-sampling meaning of a confidence level.",
         "Draw many samples and count the intervals that cover.", "beginner"),
        ("Predict the effect of the sample size and the confidence level on the width.",
         "Change each control and watch the interval.", "beginner"),
        ("Recognise a procedure whose actual coverage falls short of its nominal level.",
         "Compare the Wald and Wilson intervals for a small proportion.", "intermediate"),
    ),
    "inference.power": (
        ("Compute the power of a design before collecting data.",
         "Set the effect size, n and alpha, then verify against the simulation.", "intermediate"),
        ("Describe the trade-off between the two error types at a fixed sample size.",
         "Lower alpha and watch power fall.", "intermediate"),
        ("Explain why power computed after the fact from the observed effect is uninformative.",
         "Run the post-hoc scenario and compare with the p-value.", "advanced"),
    ),
    "inference.multiple_testing": (
        ("Predict the number of false positives among many independent tests.",
         "Run 100 null tests and count rejections at alpha = 0.05.", "intermediate"),
        ("Distinguish the family-wise error rate from the false discovery rate.",
         "Compare Bonferroni and Benjamini-Hochberg on the same set.", "intermediate"),
        ("Judge the cost in power of a correction.",
         "Apply each correction to a set containing real effects.", "advanced"),
    ),
    "inference.mle": (
        ("Read an estimate off the shape of a likelihood function.",
         "Move the parameter and watch the log-likelihood.", "intermediate"),
        ("Connect the curvature of the log-likelihood to the precision of the estimate.",
         "Compare a flat and a sharp likelihood at the same estimate.", "advanced"),
        ("Explain what maximum likelihood estimates when the model is wrong.",
         "Run the misspecified scenario and compare with the true parameter.", "advanced"),
    ),
    "inference.cramer_rao": (
        ("State what the Cramer-Rao bound constrains and what it leaves free.",
         "Compare the bound with the variance of an unbiased and a biased estimator.", "advanced"),
        ("Connect Fisher information to the sample size.",
         "Raise n and watch the bound fall like 1/n.", "advanced"),
        ("Recognise a case where the bound does not apply.",
         "Run the moving-support scenario.", "advanced"),
    ),
    "inference.neyman_pearson": (
        ("Explain why the likelihood ratio orders the sample space optimally.",
         "Compare the power of the likelihood ratio test with an arbitrary region of the same size.", "advanced"),
        ("Locate the two error probabilities on the same picture.",
         "Move the critical value and watch both regions change.", "intermediate"),
        ("State why the lemma does not extend to a composite alternative.",
         "Run the composite scenario and compare optimal regions across alternatives.", "advanced"),
    ),
    "inference.bootstrap": (
        ("Explain what the bootstrap resamples and what it estimates.",
         "Compare the bootstrap distribution with the true sampling distribution.", "intermediate"),
        ("Judge when the bootstrap fails.",
         "Run the maximum-statistic scenario and see the resampled distribution break down.", "advanced"),
        ("Compare percentile and bias-corrected intervals in terms of coverage.",
         "Read the measured coverage of each.", "advanced"),
    ),
    "inference.bayesian_updating": (
        ("Predict how the prior and the likelihood combine into the posterior.",
         "Move the prior and watch the posterior travel between prior and data.", "intermediate"),
        ("Describe how the influence of the prior changes with the sample size.",
         "Raise n and watch the posterior converge on the likelihood.", "intermediate"),
        ("Distinguish a credible interval from a confidence interval.",
         "Read both for the same data and state what each claims.", "advanced"),
    ),
    # -- regression -------------------------------------------------------
    "regression.simple_linear": (
        ("Interpret the slope as a conditional difference, not a change over time.",
         "Read the fitted line at two x values and compare.", "beginner"),
        ("Explain why a high R-squared does not certify the model.",
         "Run the nonlinear-truth scenario and see the fit stay high.", "intermediate"),
        ("Recognise that prediction outside the observed range rests on an assumption.",
         "Extend the line past the data and compare with the true curve.", "intermediate"),
    ),
    "regression.ols_geometry": (
        ("Describe least squares as a projection onto the column space.",
         "Rotate the 3-D view until the right angle is visible.", "intermediate"),
        ("Explain why the residual is orthogonal to every regressor in every sample.",
         "Read the X'e panel and change the data.", "intermediate"),
        ("State why orthogonality is not evidence of exogeneity.",
         "Run the endogenous scenario and see X'e stay zero.", "advanced"),
    ),
    "regression.gauss_markov": (
        ("State the four conditions under which OLS is best linear unbiased.",
         "Break each one in turn and watch which conclusion fails.", "intermediate"),
        ("Explain what 'best' means and what it excludes.",
         "Compare OLS with a biased estimator on mean squared error.", "advanced"),
        ("Show that normality is not needed for the theorem.",
         "Run a non-normal error scenario and check unbiasedness and variance.", "advanced"),
    ),
    "regression.fwl": (
        ("Explain 'controlling for' as removing what the controls explain.",
         "Compare the multiple-regression slope with the partialled-out slope.", "intermediate"),
        ("Read an added-variable plot correctly.",
         "Match the plotted slope to the coefficient in the full regression.", "intermediate"),
        ("Explain why the two-step standard error needs a degrees-of-freedom correction.",
         "Compare the two reported standard errors.", "advanced"),
    ),
    "regression.multicollinearity": (
        ("Distinguish an imprecise coefficient from a wrong one.",
         "Raise the correlation and watch the standard errors, not the bias.", "intermediate"),
        ("Interpret a variance inflation factor.",
         "Compare the VIF with the width of the confidence interval.", "intermediate"),
        ("Explain why a joint test can reject when no single coefficient is significant.",
         "Run the collinear scenario and compare the t and F results.", "advanced"),
    ),
    "regression.restricted": (
        ("Explain why imposing a restriction can only worsen the fit.",
         "Compare the two residual sums of squares.", "intermediate"),
        ("Compute an F statistic from two residual sums of squares.",
         "Check the reported value against the Wald form.", "intermediate"),
        ("Describe the cost of imposing a false restriction.",
         "Run the false-restriction scenario and inspect the bias.", "advanced"),
    ),
    # -- econometrics -----------------------------------------------------
    "econometrics.heteroskedasticity": (
        ("State what heteroskedasticity does and does not break.",
         "Compare the coefficient bias with the standard error bias.", "intermediate"),
        ("Judge whether robust standard errors repair the inference.",
         "Compare classical and HC1 rejection rates under the null.", "intermediate"),
        ("Recognise a case where weighting is worth the extra assumption.",
         "Compare OLS and WLS variances under a known variance function.", "advanced"),
    ),
    "econometrics.autocorrelation": (
        ("Explain why autocorrelated errors deflate conventional standard errors.",
         "Compare the reported standard error with the sampling spread.", "intermediate"),
        ("Choose an appropriate lag length for a HAC covariance.",
         "Vary the bandwidth and watch the standard error stabilise.", "advanced"),
        ("Distinguish serial correlation in the errors from a dynamic misspecification.",
         "Compare the lagged-dependent-variable scenario with the AR-error one.", "advanced"),
    ),
    "econometrics.omitted_variable_bias": (
        ("Predict the direction of the bias from two substantive judgements.",
         "Set the signs of both channels and check the estimate.", "intermediate"),
        ("Explain why an omitted variable unrelated to the regressor is harmless.",
         "Set the auxiliary slope to zero and watch the bias vanish.", "intermediate"),
        ("Distinguish a confounder from a mediator in terms of what to control for.",
         "Compare the two scenarios and their correct specifications.", "advanced"),
    ),
    "econometrics.endogeneity_iv": (
        ("State the two conditions an instrument must satisfy.",
         "Break each in turn and watch the estimator fail differently.", "intermediate"),
        ("Explain why a weak instrument can be worse than no instrument.",
         "Lower the first-stage strength and compare bias with OLS.", "advanced"),
        ("Interpret a first-stage F statistic as a warning device.",
         "Read the F alongside the confidence interval width.", "advanced"),
    ),
    "econometrics.measurement_error": (
        ("Predict the direction of attenuation from classical measurement error.",
         "Raise the noise in the regressor and watch the slope shrink.", "intermediate"),
        ("Explain why error in the dependent variable is less damaging.",
         "Move the noise from x to y and compare.", "intermediate"),
        ("Judge when an instrument repairs measurement error.",
         "Run the IV-repair scenario.", "advanced"),
    ),
    "econometrics.simultaneous_equations": (
        ("Explain why a simultaneous system makes OLS inconsistent.",
         "Compare the OLS estimate with the structural parameter.", "advanced"),
        ("Check the order condition for identification.",
         "Remove an excluded instrument and watch identification fail.", "advanced"),
        ("Distinguish a structural parameter from a reduced-form one.",
         "Read both in the same run.", "advanced"),
    ),
    "econometrics.logit_probit": (
        ("Explain why a coefficient in a nonlinear model is not a marginal effect.",
         "Compare the coefficient with the average marginal effect.", "intermediate"),
        ("Compare logit and probit on the same data.",
         "Overlay the fitted probability curves.", "intermediate"),
        ("Recognise that a linear probability model can predict outside [0, 1].",
         "Run the extreme-covariate scenario.", "intermediate"),
    ),
    "econometrics.quantile_regression": (
        ("Explain what a quantile regression coefficient measures.",
         "Compare the slopes at several quantiles.", "advanced"),
        ("Recognise data where the mean regression misses the story.",
         "Run the heteroskedastic scenario and compare the quantile fan.", "advanced"),
        ("Describe the robustness of the quantile fit to outliers.",
         "Add an extreme point and compare with OLS.", "intermediate"),
    ),
    "econometrics.gmm": (
        ("Express OLS and IV as special cases of a moment condition.",
         "Compare the GMM estimate with each.", "advanced"),
        ("Explain what over-identification buys and what it costs.",
         "Add instruments and read the J statistic.", "advanced"),
        ("Interpret a rejection of the over-identifying restrictions.",
         "Run the invalid-instrument scenario.", "advanced"),
    ),
    # -- time series ------------------------------------------------------
    "timeseries.stationarity": (
        ("Distinguish a trend-stationary series from a unit-root series.",
         "Compare how each responds to a shock.", "intermediate"),
        ("Interpret the result of a unit-root test, including its power limits.",
         "Run the near-unit-root scenario and watch the test fail to reject.", "advanced"),
        ("Explain why differencing a trend-stationary series is harmful.",
         "Compare the two corrections on the same series.", "advanced"),
    ),
    "timeseries.arma": (
        ("Identify an AR and an MA signature from the ACF and PACF.",
         "Switch between the AR and MA scenarios and read both plots.", "intermediate"),
        ("Explain why information criteria must be compared on the same sample.",
         "Change the maximum lag and watch the effective sample.", "advanced"),
        ("Relate the AR coefficients to the persistence of a shock.",
         "Read the impulse response.", "intermediate"),
    ),
    "timeseries.var": (
        ("Read an impulse response function.",
         "Trace a shock through the system.", "advanced"),
        ("Explain why the ordering matters for orthogonalised shocks.",
         "Swap the ordering and compare responses.", "advanced"),
        ("Distinguish Granger causality from causality.",
         "Run the common-cause scenario and read the test.", "advanced"),
    ),
    "timeseries.ardl": (
        ("Distinguish a short-run from a long-run coefficient.",
         "Read both from the same fitted model.", "advanced"),
        ("Interpret an error-correction coefficient.",
         "Change the adjustment speed and watch the response.", "advanced"),
        ("Judge when an ARDL bounds test is inconclusive.",
         "Run the borderline scenario.", "advanced"),
    ),
    "timeseries.cointegration": (
        ("Recognise a spurious regression between independent random walks.",
         "Run the spurious scenario and read the t statistic and R-squared.", "advanced"),
        ("Distinguish a spurious relation from a cointegrating one.",
         "Compare the residual stationarity in both scenarios.", "advanced"),
        ("Explain why the usual critical values do not apply to the residual test.",
         "Compare the standard and Engle-Granger critical values.", "advanced"),
    ),
    "timeseries.garch": (
        ("Distinguish predictable volatility from predictable returns.",
         "Read the two assumption rows in the same run.", "advanced"),
        ("Explain what persistence close to one implies for forecasts.",
         "Run the integrated scenario and read the warning.", "advanced"),
        ("Explain why GARCH generates heavy tails from normal innovations.",
         "Compare the kurtosis of the returns and of the standardised residuals.", "advanced"),
    ),
    # -- panel ------------------------------------------------------------
    "panel.fixed_vs_random": (
        ("State what the within transformation removes and what it costs.",
         "Compare pooled, fixed and random effects on the same panel.", "advanced"),
        ("Decide between fixed and random effects from the correlation of the effect with the regressors.",
         "Run both scenarios and read the Hausman test.", "advanced"),
        ("Explain why time-invariant regressors drop out of a fixed-effects model.",
         "Add one and watch it vanish.", "intermediate"),
    ),
    "panel.dynamic_gmm": (
        ("Explain why a lagged dependent variable biases the within estimator.",
         "Compare the fixed-effects estimate with the truth as T changes.", "advanced"),
        ("Describe what differencing does to the fixed effect and to the instruments.",
         "Read the instrument set in the difference GMM scenario.", "advanced"),
        ("Recognise instrument proliferation.",
         "Raise the lag depth and watch the instrument count and the J test.", "advanced"),
    ),
    # -- causal -----------------------------------------------------------
    "causal.potential_outcomes": (
        ("State the fundamental problem of causal inference.",
         "Reveal both potential outcomes in the simulation and compare with what is observed.", "intermediate"),
        ("Explain why randomisation identifies the average effect.",
         "Compare the randomised and self-selected scenarios.", "intermediate"),
        ("Distinguish the average treatment effect from the effect on the treated.",
         "Read both when the effect is heterogeneous.", "advanced"),
    ),
    "causal.dag": (
        ("Classify a variable as a confounder, a mediator or a collider.",
         "Run each scenario and compare the adjusted and unadjusted estimates.", "intermediate"),
        ("Explain why conditioning on a collider creates bias.",
         "Adjust for the collider and watch a null association appear.", "advanced"),
        ("Justify a control set from a diagram rather than from fit.",
         "Compare the correct adjustment set with the kitchen-sink regression.", "advanced"),
    ),
    "causal.did": (
        ("State the identifying assumption of difference-in-differences.",
         "Run the violated-trends scenario and see the bias.", "intermediate"),
        ("Explain why parallel pre-trends are neither necessary nor sufficient.",
         "Compare the pre-trend plot with the true counterfactual.", "advanced"),
        ("Recognise the problem with staggered adoption and heterogeneous effects.",
         "Run the staggered scenario and compare estimators.", "advanced"),
    ),
    "causal.rdd": (
        ("Explain what a regression discontinuity identifies and for whom.",
         "Compare the estimate at the cutoff with the global effect.", "advanced"),
        ("Judge the sensitivity of the estimate to bandwidth and polynomial order.",
         "Vary both and watch the estimate move.", "advanced"),
        ("Detect manipulation of the running variable.",
         "Run the manipulated scenario and read the density test.", "advanced"),
    ),
    "causal.synthetic_control": (
        ("Explain how a weighted combination of units builds a counterfactual.",
         "Read the donor weights and the pre-treatment fit.", "advanced"),
        ("Judge a synthetic control by its pre-treatment fit, not its post-treatment gap.",
         "Compare a good and a poor pre-fit.", "advanced"),
        ("Interpret a placebo distribution.",
         "Run the placebo test and locate the treated unit.", "advanced"),
    ),
    "causal.dml": (
        ("Explain why naive machine learning of the nuisance functions biases the effect.",
         "Compare the naive and orthogonalised estimates.", "advanced"),
        ("Describe what cross-fitting protects against.",
         "Turn it off and watch the bias return.", "advanced"),
        ("Connect double machine learning to the partialling-out theorem.",
         "Compare with the linear Frisch-Waugh-Lovell result.", "advanced"),
    ),
    # -- multivariate, ML, deep learning, AI -------------------------------
    "multivariate.pca": (
        ("Explain what the first component maximises.",
         "Rotate the direction manually and confirm no direction beats it.", "intermediate"),
        ("Relate the discarded eigenvalues to the reconstruction error.",
         "Drop components and read the error.", "advanced"),
        ("Explain why standardisation changes the components.",
         "Toggle scaling and compare the loadings.", "intermediate"),
    ),
    "multivariate.clustering": (
        ("Explain why k-means finds spherical clusters.",
         "Run the elongated-cluster scenario and watch it fail.", "intermediate"),
        ("Judge the number of clusters from more than one criterion.",
         "Compare the elbow and silhouette panels.", "intermediate"),
        ("Recognise that clustering returns groups even when none exist.",
         "Run the no-structure scenario.", "intermediate"),
    ),
    "ml.bias_variance": (
        ("Decompose prediction error into bias, variance and noise.",
         "Read the three components as the model flexes.", "intermediate"),
        ("Predict which way each component moves with model complexity.",
         "Sweep the complexity control.", "intermediate"),
        ("Explain why irreducible noise sets a floor on accuracy.",
         "Raise the noise and watch the floor rise.", "beginner"),
    ),
    "ml.cross_validation": (
        ("Explain why the training error understates the test error.",
         "Compare the two curves.", "intermediate"),
        ("Describe how leakage inflates a cross-validated score.",
         "Run the leaky-preprocessing scenario.", "advanced"),
        ("Judge the variance of a k-fold estimate.",
         "Change k and repeat with different seeds.", "intermediate"),
    ),
    "ml.regularization": (
        ("Explain why the lasso can set a coefficient to exactly zero and ridge cannot.",
         "Compare the two coefficient paths at the same penalty.", "advanced"),
        ("Describe the bias-variance trade the penalty makes.",
         "Sweep lambda and read both components.", "intermediate"),
        ("Recognise that selected variables are not necessarily causal.",
         "Run the correlated-predictors scenario and watch the selection switch.", "advanced"),
    ),
    "ml.gradient_descent": (
        ("Predict the effect of the learning rate on the path.",
         "Raise it until divergence and find the boundary.", "intermediate"),
        ("Explain why a saddle point stalls a first-order method.",
         "Run the saddle scenario and read the gradient norm.", "advanced"),
        ("Describe the effect of feature scaling on convergence.",
         "Compare the scaled and unscaled paths.", "intermediate"),
    ),
    "ml.classification_threshold": (
        ("Explain why accuracy is misleading under class imbalance.",
         "Compare accuracy with the confusion matrix at 95% prevalence.", "intermediate"),
        ("Choose a threshold from the cost of each error type.",
         "Move the threshold and read precision and recall.", "intermediate"),
        ("Distinguish discrimination from calibration.",
         "Compare the ROC curve with the calibration plot.", "advanced"),
    ),
    "deep_learning.backpropagation": (
        ("Describe backpropagation as the chain rule with cached sensitivities.",
         "Follow the delta values layer by layer.", "advanced"),
        ("Explain why gradients vanish or explode with depth.",
         "Run both scenarios and read the per-layer gradient norms.", "advanced"),
        ("Verify an analytic gradient against finite differences.",
         "Read the comparison panel.", "advanced"),
    ),
    "ai.tokenization": (
        ("Explain why token counts differ across scripts for the same meaning.",
         "Tokenise the same sentence in English and Arabic and compare.", "intermediate"),
        ("Describe how sub-word tokenisation handles an unseen word.",
         "Enter an invented word and read the pieces.", "beginner"),
        ("Relate token count to context limits and cost.",
         "Read the running token total.", "beginner"),
    ),
    "ai.attention": (
        ("Explain attention as a similarity-weighted lookup.",
         "Read the query-key similarity matrix and the resulting weights.", "advanced"),
        ("Explain why attention weights are not explanations.",
         "Compare two weight patterns that give the same output.", "advanced"),
        ("Describe the effect of the temperature-like scaling on the weights.",
         "Change the scale and watch the weights sharpen or flatten.", "advanced"),
    ),
    "xai.shap": (
        ("State the property that makes SHAP values add up to the prediction.",
         "Read the efficiency check in the metrics panel.", "advanced"),
        ("Distinguish attribution from causation.",
         "Run the correlated-features scenario and compare with the true model.", "advanced"),
        ("Explain how correlated features split an attribution between them.",
         "Raise the correlation and watch the split.", "advanced"),
    ),
    "xai.partial_dependence": (
        ("Explain what a partial dependence plot averages over.",
         "Compare it with the individual conditional expectation curves.", "advanced"),
        ("Recognise when averaging hides heterogeneity.",
         "Run the interaction scenario and compare the two panels.", "advanced"),
        ("Explain why extrapolation makes a partial dependence plot unreliable.",
         "Inspect the plot where the feature combination never occurs.", "advanced"),
    ),
    "spatial.autocorrelation": (
        ("Interpret Moran's I as a correlation between a value and its neighbours.",
         "Compare the clustered and random scenarios.", "advanced"),
        ("Explain why the spatial weights matrix is a modelling choice.",
         "Switch the neighbour definition and watch the statistic move.", "advanced"),
        ("Distinguish spatial dependence from spatial heterogeneity.",
         "Compare the two scenarios.", "advanced"),
    ),
}

OBJECTIVES: dict[str, tuple[Objective, ...]] = {
    concept: tuple(
        _o(concept, index + 1, statement, check, level)
        for index, (statement, check, level) in enumerate(rows)
    )
    for concept, rows in _RAW.items()
}


def for_concept(concept_id: str) -> tuple[Objective, ...]:
    """Objectives for one concept - empty when none have been written yet."""
    return OBJECTIVES.get(concept_id, ())


def all_objectives() -> tuple[Objective, ...]:
    return tuple(o for group in OBJECTIVES.values() for o in group)


def coverage() -> dict[str, list[str]]:
    """Which implemented concepts have objectives and which do not.

    Reported rather than hidden: a concept with no objectives is listed under
    ``"missing"`` so the gap is visible instead of being papered over.
    """
    from ..core.registry import registry

    implemented = [s.id for s in registry.specs(implemented_only=True)]
    return {
        "covered": sorted(c for c in implemented if OBJECTIVES.get(c)),
        "missing": sorted(c for c in implemented if not OBJECTIVES.get(c)),
    }
