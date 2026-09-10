# Learning paths

Curated orders through the catalogue, for readers who want a route rather than
a list.

```python
import visualmetrics as vm

vm.learning_path("causal.did")        # how to get to a concept
vm.prerequisites("panel.dynamic_gmm") # what it assumes
```

## Foundations of statistical inference

From a probability distribution to sampling variability, intervals, tests and power.

1. `probability.distributions`
2. `inference.sampling_distributions`
3. `inference.lln`
4. `inference.clt`
5. `inference.confidence_intervals`
6. `inference.hypothesis_testing`
7. `inference.power`

## Estimation theory

Method of moments, maximum likelihood, information, efficiency bounds and resampling.

1. `probability.distributions`
2. `inference.mle`
3. `inference.cramer_rao`
4. `inference.bootstrap`
5. `inference.bayesian_updating`

## Regression from scratch

Projection geometry, least squares, partialling out and coefficient stability.

1. `math.projection`
2. `regression.simple_linear`
3. `regression.ols_geometry`
4. `regression.fwl`
5. `regression.multicollinearity`
6. `regression.restricted`

## Diagnostics and remedies

Detect a violated assumption, measure the damage, apply the remedy, see what it does not fix.

1. `regression.simple_linear`
2. `econometrics.heteroskedasticity`
3. `econometrics.autocorrelation`
4. `econometrics.omitted_variable_bias`
5. `regression.multicollinearity`

## Identification and causality

From omitted-variable bias to graphs, instruments and quasi-experimental designs.

1. `econometrics.omitted_variable_bias`
2. `causal.dag`
3. `causal.potential_outcomes`
4. `econometrics.endogeneity_iv`
5. `causal.did`
6. `causal.rdd`

## Time-series econometrics

Stationarity, ARMA modelling, cointegration, VAR dynamics and volatility.

1. `timeseries.stationarity`
2. `timeseries.arma`
3. `timeseries.cointegration`
4. `timeseries.var`
5. `timeseries.garch`

## Machine-learning essentials

Generalization, regularization, thresholds, optimization and dimension reduction.

1. `ml.bias_variance`
2. `ml.regularization`
3. `ml.classification_threshold`
4. `ml.gradient_descent`
5. `multivariate.pca`

## From gradients to transformers

Gradient descent, backpropagation, attention and post-hoc explanation.

1. `ml.gradient_descent`
2. `deep_learning.backpropagation`
3. `ai.attention`
4. `xai.shap`

