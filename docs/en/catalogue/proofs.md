# Proofs

13 proofs, each with named assumptions,
per-step justifications and an explicit statement of its limits.

Read one in the GUI at `/proof/<id>`, or from the API:

```python
import visualmetrics as vm

print(vm.proof("regression.fwl.theorem").to_text())
```

### OLS residuals are orthogonal to every regressor

`regression.ols.residual_orthogonality` — **geometric proof**, 8 steps, 2 assumptions, with a numerical check

If X has full column rank and b is the ordinary least squares coefficient vector, then the residual vector e = y - Xb satisfies X'e = 0 exactly, in every sample.

*What it does not establish:* This is orthogonality in the sample, not exogeneity in the population. X'e = 0 holds by construction even when the model is badly misspecified and even when E[u | X] is not zero, so it can never be used as evidence that the regressors are exogenous. If the regression includes an intercept, the residuals also sum to zero for the same mechanical reason - again, not a diagnostic. It also says nothing about the residuals being independent, homoskedastic or normal.

### Frisch-Waugh-Lovell: partialling out reproduces the multiple-regression coefficient

`regression.fwl.theorem` — **formal proof**, 8 steps, 2 assumptions, with a numerical check

In the regression y = X1 b1 + X2 b2 + e, the least squares estimate b2 is identical to the estimate obtained by regressing the residuals of y on X1 against the residuals of X2 on X1. The residual vector e is identical too.

*What it does not establish:* The identity is about point estimates and residuals only. Standard errors from the two-step regression are wrong unless the degrees of freedom are corrected for the columns of X1 that were partialled out, since the short regression believes it estimated fewer parameters than it did. The theorem is pure linear algebra: it does not make b2 causal, unbiased or consistent, and it says nothing about what happens if a relevant variable is missing from both X1 and X2.

### Neyman-Pearson lemma: the likelihood ratio test is most powerful

`inference.neyman_pearson.lemma` — **formal proof**, 8 steps, 3 assumptions

For a simple null H0: X ~ f0 against a simple alternative H1: X ~ f1, the test that rejects when f1(x) > k f0(x) and has size exactly alpha is at least as powerful as any other test of level alpha.

*What it does not establish:* Simple against simple only. It gives no most-powerful test for a composite alternative, where uniform optimality generally fails and one must fall back on monotone-likelihood-ratio families, unbiasedness or invariance restrictions. It also assumes the model is correct: the lemma optimises power within the assumed pair of densities and offers no protection when neither f0 nor f1 generated the data. Finally it is a statement about a fixed alpha, not a recommendation for which alpha to use, and it says nothing about how to interpret a p-value.

### The lasso sets coefficients exactly to zero; ridge never does

`ml.lasso.sparsity_geometry` — **formal proof**, 9 steps, 3 assumptions, with a numerical check

With an orthonormal design, the lasso solution is the soft-thresholded least squares coefficient, which is exactly zero whenever the least squares coefficient is no larger than lambda in magnitude. The ridge solution is a uniform shrinkage that is zero only when the least squares coefficient is zero.

*What it does not establish:* The closed form is proved only for an orthonormal design; with correlated regressors the lasso path has no closed form and must be computed numerically. Selection consistency is a separate and much stronger claim that requires additional conditions (the irrepresentable condition) and is not established here. Nothing here says the selected variables are the causally relevant ones, or that the non-zero coefficients are unbiased - they are biased toward zero by construction, which is why post-selection inference needs its own machinery. The geometric step is an illustration of the algebra, not an independent proof.

### The first principal component is the leading eigenvector

`multivariate.pca.variance_maximization` — **formal proof**, 8 steps, 2 assumptions, with a numerical check

Among all unit-length directions a, the projected variance a'Sa is maximised by the eigenvector of S belonging to its largest eigenvalue, and the maximum value is that eigenvalue.

*What it does not establish:* Maximal variance is not the same as maximal usefulness: a direction can carry most of the variance and none of the signal you care about, and PCA is entirely blind to any outcome variable. The result is scale-dependent - rescaling one variable changes S and therefore changes the components, which is why standardisation is a substantive choice and not a formality. It captures only linear structure and only second moments. In practice S is estimated from a finite sample, so the leading eigenvalue is biased upward and the estimated direction is unstable when the top eigenvalues are close together. Finally, components are mathematical artefacts and carry no guarantee of being interpretable as real factors.

### Backpropagation: the chain rule applied once, in reverse

`deep_learning.backprop.chain_rule` — **symbolic derivation**, 9 steps, 3 assumptions, with a numerical check

For a feedforward network, the gradient of the loss with respect to every weight is obtained from a single backward sweep of the recursion delta^(l) = (W^(l+1)' delta^(l+1)) * sigma'(z^(l)), at a total cost proportional to one forward pass.

*What it does not establish:* This derives the gradient, nothing more. It gives no guarantee that gradient descent converges, that it reaches a global minimum, or that the resulting network generalises - the loss surface of a deep network is non-convex and this argument is silent about its shape. It assumes exact arithmetic: in float32 the product of many small factors underflows, which is a numerical failure on top of the mathematical vanishing described above. The differentiability assumption is violated at the ReLU kink. Nothing here explains why depth helps.

### The orthogonal projection is the unique closest point in a subspace

`math.projection.shortest_distance` — **geometric proof**, 7 steps, 3 assumptions, with a numerical check

Let S be a subspace of a Euclidean space and let p be the orthogonal projection of y onto S. Then for every v in S, ||y - v|| >= ||y - p||, with equality only when v = p.

*What it does not establish:* Everything here depends on the norm coming from an inner product; under other norms the statement is false. It presumes the subspace is fixed and given - it says nothing about how to choose which subspace (that is, which regressors) to project onto, and the closest point in a badly chosen subspace is still a poor description of y. In infinite dimensions the subspace must additionally be closed for the projection to exist at all. Finally this is geometry, not statistics: it carries no notion of sampling error, bias or causality.

### Gauss-Markov: OLS is the best linear unbiased estimator

`inference.gauss_markov.blue` — **formal proof**, 9 steps, 4 assumptions, with a numerical check

If the errors have zero conditional mean, constant variance and no correlation, then among all estimators that are linear in y and unbiased, ordinary least squares has the smallest variance - and the difference is positive semi-definite, so the statement holds for every linear combination of the coefficients at once.

*What it does not establish:* Best within a class, and the class is narrow twice over. Restricting to unbiased estimators is a real restriction: ridge regression is biased and can have strictly smaller mean squared error, which is why the theorem does not make OLS the best choice for prediction. Restricting to linear estimators is another: under non-normal errors nonlinear estimators can do better, and under heavy tails robust estimators usually do. The theorem is silent when sphericity fails, which is the normal state of affairs in applied work. Above all it is a statement about efficiency, not about identification: if E[u | X] is not zero the theorem does not apply at all, and no efficiency argument can turn a biased estimate into a causal one.

### Central limit theorem: why the normal limit is inevitable

`inference.clt.moment_generating` — **formal proof**, 8 steps, 3 assumptions

If X_1, X_2, ... are independent and identically distributed with mean mu and finite variance sigma^2 > 0, then the standardised mean converges in distribution to the standard normal law.

*What it does not establish:* This is a statement about a limit, not about any particular sample size. It gives no error bound: how large n must be depends on the skewness and the tail weight of the underlying distribution, and for strongly skewed or heavy-tailed data the approximation can still be poor at n in the hundreds. Berry-Esseen supplies a rate of order 1/sqrt(n); this proof supplies none. The theorem concerns the centre of the distribution, so tail probabilities and extreme quantiles converge far more slowly than the middle. It is about the mean of the sampling distribution of an average, never about the data themselves becoming normal - a common and consequential misreading. And it says nothing when the variance is infinite or the observations are dependent.

### Instrumental variables are consistent exactly where OLS is not

`econometrics.iv.consistency` — **formal proof**, 8 steps, 5 assumptions

When a regressor is correlated with the error, OLS converges to the wrong value, and the size of the error is the inverse regressor second-moment matrix times the regressor-error covariance. An instrument that is uncorrelated with the error and correlated with the regressor delivers an estimator that converges to the true coefficient.

*What it does not establish:* Consistency is asymptotic and says nothing about finite samples: with weak instruments the IV estimator is biased toward OLS, its distribution is not approximately normal, and conventional confidence intervals under-cover badly - an F statistic in the first stage below roughly 10 is the usual warning sign, and in that situation IV can easily be worse than the OLS it was meant to repair. Exogeneity of the instrument is untestable in the just-identified case; the Sargan and Hansen tests require over-identification and even then test the joint validity of the whole set, not any single instrument. IV is also less efficient than OLS when OLS happens to be consistent. Finally, with heterogeneous effects the estimand is a local average treatment effect for the subpopulation the instrument actually moves, which is not the average effect in the population and is not identified by this argument.

### Omitted variable bias: the short regression equals the long one plus a bias term

`econometrics.ovb.formula` — **symbolic derivation**, 7 steps, 2 assumptions, with a numerical check

If the long regression is y = b0 + b1 x1 + b2 x2 + e and x2 is omitted, the slope from the short regression satisfies b1_short = b1_long + delta * b2, where delta is the slope of x2 regressed on x1. In the population the same identity gives the asymptotic bias.

*What it does not establish:* The identity compares two regressions; it does not certify that the long one is causal. If a third variable is missing from both, the long regression is biased too and the formula only describes the gap between two flawed estimates. The clean two-factor reading assumes a single omitted variable: with several, the bias terms add and can cancel, so a small measured bias does not mean small individual biases. Nothing here helps you find the omitted variable, and the sign argument is only as reliable as the two judgements it rests on. Finally, controlling for a variable that lies on the causal path from x1 to y removes part of the effect you wanted to measure - the same formula, applied where it should not be.

### Cramer-Rao: no unbiased estimator can beat the inverse information

`inference.cramer_rao.bound` — **symbolic derivation**, 8 steps, 3 assumptions, with a numerical check

Under regularity conditions, any unbiased estimator T of theta satisfies Var(T) >= 1 / I(theta), where I(theta) is the Fisher information. Equality holds only when the score is a linear function of T.

*What it does not establish:* The bound constrains unbiased estimators only, and a biased estimator can have smaller mean squared error - the James-Stein estimator dominates the sample mean in three dimensions or more despite this bound. Regularity is essential and fails whenever the support depends on the parameter, where estimators can converge much faster than 1/sqrt(n). Attaining the bound in finite samples is special to the exponential family; elsewhere it is only approached asymptotically. The bound also assumes the model is correct: under misspecification the information matrix equality fails and the relevant variance is the sandwich form, not 1/I. Finally it is a statement about variance at one parameter value, not about how the estimator behaves across the whole parameter space.

### Restricted least squares and the two faces of the F statistic

`regression.restricted.f_statistic` — **symbolic derivation**, 9 steps, 3 assumptions, with a numerical check

Minimising the sum of squares subject to R b = r gives b_R = b - (X'X)^-1 R' [R (X'X)^-1 R']^-1 (R b - r). The resulting F statistic can be written either as a scaled increase in the residual sum of squares or as a Wald quadratic form, and the two expressions are algebraically identical.

*What it does not establish:* The exact F distribution needs normal, homoskedastic, independent errors; without them the statistic is only asymptotically chi-square after scaling, and the sum-of-squares form is no longer valid at all - with robust or clustered covariance only the Wald form applies. The test says whether the data are compatible with the restriction, never that the restriction is true: failing to reject is not evidence for it, especially in small samples where power is low. Imposing a false restriction biases every coefficient, while imposing a true one lowers variance - the test alone cannot tell you which case you are in. And a restriction that is statistically rejected may still be economically negligible.

