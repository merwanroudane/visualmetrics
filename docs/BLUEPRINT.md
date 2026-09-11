# VisualMetrics — Master Blueprint

> **Multilingual Interactive Visual Laboratory for Statistics, Statistical Inference, Econometrics, Causal Inference, Machine Learning and AI**
>
> مستند مرجعي شامل لتصميم وبناء مكتبة **VisualMetrics** بوصفها مختبرًا بصريًا وتعليميًا وبحثيًا، وليس مجرد مكتبة للرسم البياني.

---

## 0. Project Metadata

- **Project name:** `visualmetrics`
- **GitHub repository:** https://github.com/merwanroudane/visualmetrics
- **Maintainer email:** `merwanroudane920@gmail.com`
- **Primary package name:** `visualmetrics`
- **Primary CLI command:** `visualmetrics`
- **Primary GUI launch commands:**
  - `visualmetrics`
  - `visualmetrics gui`
  - `python -m visualmetrics`
- **Initial target runtime:** Python 3.11 as the reference development runtime.
- **Compatibility goal:** Python 3.11–3.13 where dependencies support it; keep the architecture ready to widen support later.
- **Initial version:** `0.1.0`
- **Recommended code license:** MIT License.
- **Recommended educational-content license:** CC BY 4.0 for original explanatory/proof/tutorial content, unless the maintainer chooses a single-license policy.
- **Repository state at planning time:** empty, allowing a clean architecture from the first commit.

---

# 1. Vision

VisualMetrics must become a **visual knowledge engine** for quantitative disciplines. Its purpose is not merely to produce plots from data. Its purpose is to transform difficult theoretical ideas into things that can be:

1. **Seen** — Visualization.
2. **Moved through time** — Animation.
3. **Manipulated** — Interactive controls.
4. **Compared** — Side-by-side scenarios.
5. **Derived** — Step-by-step mathematics.
6. **Proved when appropriate** — Visual or symbolic proof.
7. **Experimented with** — Simulation and sensitivity analysis.
8. **Diagnosed** — Assumption violations and edge cases.
9. **Learned without coding** — No-code GUI.
10. **Reproduced with code** — generated Python equivalent.
11. **Used in Arabic, English, and French**.
12. **Used by students, instructors, researchers, and developers**.

The project should be understood as:

> **A multilingual interactive laboratory for understanding why statistical, econometric and AI methods work, when they fail, what their assumptions mean, and how results change when the data-generating process changes.**

It should not be marketed merely as:

> “A Python plotting library.”

---

# 2. Non-Negotiable Principles

These requirements are authoritative and should shape all design and implementation decisions.

## 2.1 End-to-End Execution

A development agent must execute the complete requested scope in one continuous pass whenever technically possible.

**Do not use a gated workflow such as:**

- “I completed Part 1; confirm before Part 2.”
- “Should I continue?”
- “Please approve the architecture before implementation.”
- “Phase 1 is finished; I will wait for permission.”

Instead:

- inspect the repository;
- establish the architecture;
- implement the complete requested scope;
- run tests;
- repair failures;
- update docs;
- verify packaging;
- provide one final consolidated report.

Only stop for user input if a truly blocking issue exists that cannot be resolved by a safe and reasonable default.

## 2.2 GUI-First for Beginners, API-First for Reproducibility

Every major concept should be usable without writing code through a graphical interface, but every GUI action should correspond to reproducible Python state.

The user should be able to:

```text
GUI interaction -> inspect generated Python -> save configuration -> reproduce analysis
```

## 2.3 Scenario Completeness

Each concept must show more than one canonical example.

A concept lab should seek to expose:

- normal/canonical case;
- positive case;
- negative case;
- zero/null case;
- weak/strong cases;
- boundary cases;
- alternative functional forms;
- assumption violations;
- counterexamples;
- misleading appearances;
- small-sample vs large-sample behavior;
- noisy vs clean data;
- sensitivity to parameter changes;
- multiple data-generating processes where relevant;
- empirical and theoretical perspectives;
- comparison between competing methods where relevant.

## 2.4 Scientific Honesty

Never call a simulation a proof.

The UI and documentation must distinguish among:

- **Formal Proof**
- **Symbolic Derivation**
- **Geometric Proof**
- **Visual Derivation**
- **Visual Intuition**
- **Simulation Evidence**
- **Numerical Demonstration**
- **Counterexample**
- **Empirical Illustration**

This distinction is essential for the scientific credibility of the project.

## 2.5 Multilingual by Architecture, Not Translation Patch

Arabic, English and French must be supported from the first architecture. No scientific text should be hard-coded in GUI classes.

## 2.6 Dependency Reuse

Do not reimplement mature statistical/econometric/AI engines unless there is a specific pedagogical reason.

VisualMetrics should add value mainly through:

- pedagogy;
- visualization;
- visual proofs;
- scenario design;
- interaction;
- comparison;
- explanation;
- multilingual presentation;
- integration.

## 2.7 Local-First and Privacy-Respecting

The default GUI should run locally. No telemetry or cloud upload by default. User datasets remain local unless the user explicitly exports or connects an external service in a future extension.

---

# 3. Target Users

## 3.1 Beginner Student

Needs:

- no-code interaction;
- definitions;
- intuition;
- guided controls;
- warnings against misconceptions;
- immediate visual feedback;
- quizzes;
- bilingual terminology.

## 3.2 Intermediate Student

Needs:

- formulas;
- assumptions;
- examples and counterexamples;
- comparisons;
- diagnostics;
- generated code.

## 3.3 Advanced / Master Student

Needs:

- derivations;
- asymptotics;
- model comparison;
- estimator properties;
- identification;
- sensitivity analysis;
- real data.

## 3.4 PhD Student / Researcher

Needs:

- formal assumptions;
- advanced inference;
- identification maps;
- modern econometric estimators;
- causal ML;
- robust inference;
- code reproducibility;
- exportable teaching/research visuals.

## 3.5 Lecturer / Instructor

Needs:

- presentation mode;
- large-font classroom view;
- predefined demonstrations;
- animation speed control;
- freeze/pause/step-through;
- export PNG/SVG/HTML/GIF/MP4 where supported;
- concept playlists;
- ready-made lesson flows;
- instructor notes;
- challenge/quiz mode.

---

# 4. Core Learning Modes

Every concept should implement as many of the following modes as scientifically useful.

| Mode | Purpose |
|---|---|
| `Learn` | concise theory, definition, context, prerequisites |
| `Visualize` | static or reactive visual representation |
| `Animate` | time-based explanation |
| `Experiment` | sliders, toggles and parameter manipulation |
| `Compare` | compare methods, assumptions, scenarios or models |
| `Prove` | formal/geometric proof where genuinely available |
| `Derive` | mathematical derivation step by step |
| `Simulate` | Monte Carlo or data-generating experiment |
| `Diagnose` | detect assumptions/violations/failure modes |
| `Counterexample` | demonstrate when an intuitive claim fails |
| `Quiz` | formative assessment |
| `Code` | equivalent Python code |
| `Data` | inspect or replace dataset |
| `References` | books, papers, course links and package docs |

A lab must not be required to implement all modes. The `ConceptSpec` declares exactly which modes are valid.

---

# 5. Pedagogical Flow

Recommended default sequence for a concept:

```text
Why it matters
   -> What it means
   -> Visual intuition
   -> Interactive experiment
   -> Assumptions
   -> What changes when assumptions fail
   -> Mathematics
   -> Proof / derivation if applicable
   -> Real-data example
   -> Generated code
   -> Misconceptions
   -> Quiz
   -> References
```

A beginner can stop after the visual layer. An advanced learner can open the mathematics and derivation layers.

---

# 6. Curriculum Scope

The scope must be broad enough to function as a visual reference across Statistics, Econometrics and AI curricula.

## 6.1 Mathematical Foundations

### Algebra and functions

- variables and parameters;
- functions and mappings;
- linear vs nonlinear functions;
- slopes and intercepts;
- powers, exponentials and logarithms;
- transformations;
- inverse functions;
- systems of equations;
- inequalities;
- sequences and series;
- limits and continuity.

### Calculus

- derivatives;
- partial derivatives;
- gradients;
- Hessians;
- Taylor approximation;
- optimization;
- constrained optimization;
- Lagrange multipliers;
- convexity and concavity;
- integration;
- numerical differentiation/integration where educationally useful.

### Linear algebra

- vectors;
- matrices;
- matrix multiplication;
- transpose;
- rank;
- inverse;
- determinant;
- linear independence;
- basis and span;
- orthogonality;
- projections;
- eigenvalues/eigenvectors;
- positive definiteness;
- quadratic forms;
- SVD;
- geometric interpretation of least squares.

These foundations are critical because many econometric and ML “proofs” become dramatically clearer geometrically.

---

# 7. Probability Foundations

## 7.1 Core Probability

- random experiments;
- sample spaces;
- events;
- unions/intersections/complements;
- mutually exclusive events;
- independence;
- conditional probability;
- law of total probability;
- Bayes theorem;
- odds and likelihood ratios.

## 7.2 Random Variables

- discrete vs continuous;
- PMF;
- PDF;
- CDF;
- quantiles;
- transformations of random variables;
- joint distributions;
- marginal distributions;
- conditional distributions;
- independence of random variables.

## 7.3 Moments

- expectation;
- variance;
- standard deviation;
- covariance;
- correlation;
- higher moments;
- skewness;
- kurtosis;
- moment-generating intuition;
- law of iterated expectations;
- law of total variance.

## 7.4 Core Distributions

At minimum:

- Bernoulli;
- Binomial;
- Geometric;
- Negative Binomial;
- Hypergeometric;
- Poisson;
- Discrete Uniform;
- Continuous Uniform;
- Normal;
- Standard Normal;
- Lognormal;
- Exponential;
- Gamma;
- Beta;
- Chi-square;
- Student t;
- F;
- Cauchy;
- Logistic;
- Weibull;
- Pareto;
- Multivariate Normal.

Each distribution lab should support parameter sliders, PDF/PMF, CDF, quantiles, random sampling, moments and comparisons.


### 7.4.1 Core probability-distribution coverage

VisualMetrics must include comprehensive probability-distribution coverage validated against university syllabi from Algeria, the Arab world, and leading international programs. Local course labels are evidence sources only and must not define the library taxonomy.

The profile must cover, at minimum:

**Discrete distributions**

- Bernoulli;
- Binomial;
- Poisson;
- Geometric;
- Hypergeometric.

**Continuous distributions**

- Normal;
- Uniform;
- Exponential;
- Gamma;
- Beta;
- Chi-square;
- Student t;
- Fisher F.

**Distribution-approximation / convergence cases**

- Binomial -> Normal;
- Binomial -> Poisson;
- Poisson -> Normal where the approximation is appropriate;
- Student t -> Normal as degrees of freedom increase;
- visual comparison of exact and approximate probabilities;
- continuity-correction controls where appropriate;
- approximation-error visualization rather than only side-by-side curves.

**Bivariate random-variable material**

- discrete bivariate random variables;
- continuous bivariate random variables;
- joint PMF/PDF;
- marginal distributions;
- conditional distributions;
- independence;
- covariance and correlation;
- functions/transformations of two random variables;
- joint support visualization;
- probability tables, heatmaps, surfaces and contour views.

Every distribution lab should provide pedagogically relevant presets such as:

```text
Shape explorer
Parameter sensitivity
Rare-event regime
Symmetric vs skewed
Thin vs heavy tail
Small vs large degrees of freedom
Exact vs approximation
Sampling experiment
Mean/variance decomposition
CDF/PDF/PMF link
Quantile explorer
```

For the Hypergeometric lab, explicitly visualize **sampling without replacement** and contrast it with Binomial sampling with replacement/approximately independent draws. For Exponential, visualize memorylessness. For Student t, Chi-square and F, animate the effect of degrees of freedom and connect these distributions to later confidence-interval and testing labs.

## 7.5 Inequalities and Bounds

- Markov inequality;
- Chebyshev inequality;
- Jensen inequality;
- union bound;
- concentration intuition where appropriate.

---

# 8. Limit Theory and Asymptotics

This is a flagship area for VisualMetrics.

Topics:

- sequences of random variables;
- convergence in probability;
- almost sure convergence;
- convergence in distribution;
- mean-square convergence;
- relationships among convergence concepts;
- Weak Law of Large Numbers;
- Strong Law of Large Numbers;
- Central Limit Theorem;
- Slutsky theorem;
- Continuous Mapping Theorem;
- Delta Method;
- asymptotic bias;
- consistency;
- asymptotic normality;
- asymptotic variance;
- asymptotic efficiency.

Visual design must distinguish **finite-sample behavior** from **asymptotic statements**.


### 8.1 Limit-theory visual proof requirements

Limit-theory labs must deliberately separate four layers:

1. **definition** — what the mode of convergence formally means;
2. **visual intuition** — what a learner can see in a finite display;
3. **simulation** — finite Monte Carlo evidence;
4. **proof/derivation** — only when a mathematically valid proof or derivation is actually implemented.

Required comparison labs include:

- convergence in probability vs almost-sure convergence;
- convergence in distribution vs convergence in probability;
- WLLN vs SLLN;
- CLT vs LLN;
- finite-sample sampling distribution vs limiting normal distribution;
- Slutsky composition of convergent sequences;
- continuous-mapping transformations;
- delta-method local linearization.

The GUI must explicitly warn that an animation of sample paths or Monte Carlo histograms is **not itself a proof** of an asymptotic theorem.

---

# 9. Descriptive Statistics and Exploratory Data Analysis

- population vs sample;
- mean, median, mode;
- variance and standard deviation;
- range and IQR;
- quantiles and percentiles;
- skewness and kurtosis;
- robust summaries;
- histograms;
- density plots;
- ECDF;
- boxplots;
- violin plots;
- scatter plots;
- correlation matrices;
- outliers;
- transformations;
- missingness visualization;
- categorical summaries;
- grouped statistics;
- long vs wide data where useful for applied tutorials.

---

# 10. Sampling Theory

- population, sample and sampling frame;
- simple random sampling;
- stratified sampling;
- cluster sampling;
- systematic sampling;
- multistage sampling;
- sampling with/without replacement;
- sampling bias;
- nonresponse;
- design effects;
- sampling distributions;
- standard error;
- finite population correction;
- bootstrap;
- jackknife;
- permutation/resampling intuition.

## 10.1 Sampling Theory core

The **Sampling Theory** module must include dedicated labs for the sampling distributions that repeatedly appear across inferential-statistics curricula:

- sample mean from one population;
- sample proportion from one population;
- sample variance;
- difference between two sample means;
- difference between two sample proportions;
- ratio of two sample variances;
- standardized sample statistics;
- order statistics;
- with-replacement vs without-replacement sampling;
- finite-population correction;
- exhaustive sampling-distribution construction for small finite populations;
- standard error as repeated-sampling variability;
- sampling bias vs sampling variability;
- effect of sample size on sampling distributions.

### Required Sampling Distribution Builder

The GUI should allow the learner to choose:

```text
Population distribution
Population parameters
Finite / conceptual infinite population
Sampling scheme
With / without replacement
Sample size n
Number of repeated samples B
Statistic T(X)
One-sample / two-sample design
Equal / unequal group sizes
```

The user should then be able to switch between:

- population view;
- one realized sample;
- all possible samples when computationally feasible;
- repeated-sampling animation;
- empirical sampling distribution;
- theoretical sampling distribution;
- approximation overlay;
- standard-error decomposition.

---

# 11. Statistical Estimation

## 11.1 Estimator Properties

- estimator vs estimate;
- unbiasedness;
- bias;
- variance;
- MSE;
- consistency;
- efficiency;
- asymptotic normality;
- sufficiency;
- minimal sufficiency;
- completeness;
- ancillary statistics;
- Rao–Blackwell theorem;
- Lehmann–Scheffé intuition;
- Fisher information;
- Cramér–Rao lower bound;
- efficiency relative to the information bound.

## 11.2 Estimation Methods

- Method of Moments;
- Maximum Likelihood Estimation;
- least squares as an estimation principle;
- Bayesian point estimation;
- MAP;
- posterior mean/median/mode;
- numerical optimization of objective functions.

## 11.3 Estimation Theory coverage and comparison labs

The inference curriculum must include the full progression from **point estimation** to **interval estimation** and from finite-sample properties to asymptotic properties.

Required point-estimation labs:

- Method of Moments vs Maximum Likelihood for the same data-generating process;
- unbiased vs biased estimators;
- bias-variance-MSE trade-off;
- consistency under increasing sample size;
- relative efficiency;
- sufficient statistics and data reduction intuition;
- Fisher information accumulation with sample size;
- Cramér-Rao lower bound and attainability;
- small-sample vs asymptotic behavior;
- estimator sampling distributions;
- optimization geometry of MLE;
- score and Hessian/curvature intuition.

Where a theorem requires regularity conditions, the lab must expose those conditions and, where practical, a counterexample or failure case.

### Flagship MLE lab

Controls should allow:

- distribution choice;
- parameter choice;
- sample size;
- sample realization;
- likelihood/log-likelihood view;
- score function;
- curvature;
- information intuition;
- small vs large sample likelihood concentration;
- compare MLE with method of moments where possible.

---

# 12. Confidence Intervals and Confidence Sets

- confidence level;
- coverage probability;
- repeated-sampling interpretation;
- standard error;
- margin of error;
- z vs t intervals;
- mean/proportion/variance intervals;
- one-sided vs two-sided intervals;
- exact vs asymptotic intervals;
- profile-likelihood ideas where appropriate;
- bootstrap confidence intervals;
- simultaneous intervals;
- confidence regions in multiple dimensions;
- connection between confidence intervals and hypothesis tests;
- one-population mean with known variance;
- one-population mean with unknown variance;
- one-population proportion;
- one-population variance;
- difference between two means;
- difference between two proportions;
- ratio of two variances;
- paired-sample intervals;
- equal-variance vs unequal-variance two-sample intervals;
- optimal/required sample-size calculations for a target margin of error.

The GUI should expose the distinction between **confidence level**, **coverage probability in repeated sampling**, and the incorrect interpretation that a realized fixed interval has a frequentist probability of containing a fixed parameter.

Flagship animation: repeatedly generate samples and display which intervals cover the true parameter.

---

# 13. Hypothesis Testing and Statistical Inference

This must be one of the richest modules in the project.

## 13.1 Foundations

- null hypothesis;
- alternative hypothesis;
- simple vs composite hypotheses;
- test statistic;
- null distribution;
- critical value;
- rejection region;
- significance level `alpha`;
- Type I error;
- Type II error;
- `beta`;
- **size**;
- **power**;
- power function;
- effect size;
- p-value;
- one-sided vs two-sided tests;
- exact vs asymptotic tests.

## 13.2 Test Theory

- Neyman–Pearson lemma;
- most powerful tests;
- uniformly most powerful tests;
- likelihood-ratio tests;
- Wald tests;
- Score/Lagrange Multiplier tests;
- asymptotic equivalence of Wald/LR/Score under regularity conditions;
- nuisance parameters;
- test consistency;
- local alternatives where appropriate.

## 13.3 Common Tests

- z tests;
- one-sample t test;
- paired t test;
- independent two-sample t test;
- Welch test;
- proportion tests;
- chi-square goodness-of-fit;
- chi-square independence;
- variance tests;
- F tests;
- ANOVA;
- correlation tests;
- nonparametric tests;
- permutation tests;
- normality tests as diagnostic concepts.

### 13.3.1 Core hypothesis-testing families

The **Hypothesis Testing** module must make the following families explicit rather than hiding everything behind a generic “test” screen:

**One-sample parametric tests**

- mean, known and unknown population variance cases;
- proportion;
- variance.

**Two-sample parametric tests**

- difference in means;
- paired means;
- equal-variance vs Welch unequal-variance cases;
- difference in proportions;
- ratio/equality of variances.

**Goodness-of-fit and nonparametric tests**

- Kolmogorov-Smirnov;
- Cramér-von Mises;
- Wilcoxon signed-rank;
- Mann-Whitney / Wilcoxon rank-sum;
- Kruskal-Wallis;
- permutation alternatives where pedagogically helpful.

**Neyman-Pearson visual laboratory**

Provide a simple-vs-simple hypothesis lab that lets the learner:

- move the likelihood-ratio threshold;
- see the rejection region change;
- view Type I and Type II areas;
- compare tests with the same size;
- see why the likelihood-ratio ordering yields the most powerful rejection region under the lemma's assumptions.

Every test lab should include an **Assumption Inspector** and an **Alternative Test** panel so users see when the classical test is inappropriate.

## 13.4 Multiple Testing

- family-wise error rate;
- Bonferroni;
- Holm;
- false discovery rate;
- Benjamini–Hochberg;
- p-hacking intuition;
- multiple-comparison visualization.

## 13.5 Power Lab Requirements

User controls should include where applicable:

- `alpha`;
- `beta`;
- power;
- sample size;
- effect size;
- variance;
- standard deviation;
- one-sided/two-sided test;
- distribution;
- degrees of freedom;
- group-size ratio;
- allocation ratio;
- paired vs independent design;
- alternative direction;
- exact/asymptotic mode;
- multiple testing adjustment.

The visualization should display:

- H0 distribution;
- H1 distribution;
- rejection region;
- Type I area;
- Type II area;
- power area;
- critical value;
- observed statistic when present;
- p-value region;
- power curve;
- sample-size curve;
- sensitivity table.

---

# 14. Bayesian Statistics

- prior;
- likelihood;
- posterior;
- posterior predictive;
- conjugacy;
- Beta-Binomial;
- Normal-Normal;
- Gamma-Poisson;
- Bayesian updating;
- credible intervals;
- posterior probability;
- Bayes factors;
- prior sensitivity;
- weak vs informative priors;
- hierarchical modeling intuition;
- MCMC visual intuition;
- uncertainty propagation.

A flagship Bayesian lab should animate prior -> likelihood -> posterior while the user adds observations sequentially.

---

# 15. Regression and Linear Models

## 15.1 Simple Linear Regression

The GUI must expose a broad scenario space, including:

- positive slope;
- negative slope;
- zero slope;
- near-zero slope;
- steep vs shallow slope;
- intercept positive/negative/zero;
- strong vs weak relationship;
- low vs high noise;
- different sample sizes;
- normal vs non-normal errors;
- homoskedastic vs heteroskedastic errors;
- symmetric vs skewed errors;
- outliers;
- leverage points;
- influential points;
- restricted x range;
- extrapolation;
- missing-data patterns where pedagogically useful.

### Functional-form controls

- level-level;
- log-level;
- level-log;
- log-log;
- quadratic;
- cubic;
- exponential;
- inverse;
- threshold;
- piecewise linear;
- interaction effect;
- saturation pattern;
- U-shape;
- inverted U-shape.

### Visual outputs

- scatter;
- fitted line;
- residual segments;
- SSE visualization;
- confidence band;
- prediction band;
- coefficient changes;
- residual plot;
- geometry/projection view;
- loss/objective surface when relevant.

## 15.2 Multiple Regression

- partial effects;
- ceteris paribus intuition;
- 3D plane where useful;
- partial regression plots;
- interactions;
- polynomial terms;
- categorical variables;
- dummy-variable trap;
- standardization;
- scaling;
- partialling-out / Frisch–Waugh–Lovell theorem.

## 15.2.1 Econometrics I curriculum extensions

To align with Algerian and Arab-world Econometrics I courses, implement dedicated labs for:

### OLS vs Maximum Likelihood in the Gaussian linear model

- same fitted coefficients under the standard normal-error linear model;
- SSE minimization vs log-likelihood maximization;
- role of the error variance;
- what changes when distributional assumptions change;
- finite-sample inference vs likelihood-based inference.

### Restricted linear regression

Support restrictions of the form:

```text
R beta = r
```

Visualize:

- unrestricted coefficient space;
- restriction surface/line/plane;
- constrained optimum;
- loss in fit;
- restricted vs unrestricted estimator;
- F/Wald-style restriction testing where applicable.

### Dummy variables and structural changes

- binary dummy intercept shift;
- slope dummy / interaction;
- multiple-category reference groups;
- dummy-variable trap;
- pre/post structural-change visualization;
- Chow-style break intuition where appropriate.

### Partial correlation and partial regression

- raw correlation vs partial correlation;
- residualize X and Y on controls;
- Frisch-Waugh-Lovell connection;
- confounding intuition without falsely equating statistical control with causal identification.

### Explanatory-variable selection

Provide educational comparison of:

- forward selection;
- backward elimination;
- stepwise procedures;
- information criteria where relevant;
- regularization alternatives;
- train/test validation for prediction;
- post-selection inference caveat;
- data-snooping and researcher-degrees-of-freedom warnings.

Do not present mechanical stepwise selection as a universally recommended scientific method.

## 15.3 OLS Geometry

- vector space;
- column space of X;
- projection of y onto X;
- fitted values;
- residual vector;
- orthogonality;
- normal equations;
- P matrix;
- M matrix;
- geometric FWL explanation.

This area should include actual geometric proofs where mathematically valid.

---

# 16. Gauss–Markov and Classical Regression Assumptions

Each assumption must have its own lab and also participate in a unified “Assumption Control Center.”

Topics:

- linearity in parameters;
- random sampling / sampling assumptions;
- no perfect multicollinearity;
- zero conditional mean / exogeneity;
- homoskedasticity;
- no serial correlation where model context requires it;
- normality for exact finite-sample classical inference;
- BLUE;
- distinction between unbiasedness, consistency and efficiency.

For each assumption provide:

```text
Valid case
-> controlled violation
-> severe violation
-> effect on coefficients
-> effect on standard errors
-> effect on inference
-> remedies / robust alternatives
```

---

# 17. Regression Failure Modes and Diagnostics

- omitted-variable bias;
- irrelevant variables;
- measurement error;
- reverse causality;
- simultaneity;
- endogeneity;
- multicollinearity;
- heteroskedasticity;
- autocorrelation;
- misspecified functional form;
- nonlinearity;
- outliers;
- leverage;
- influence;
- non-normal residuals;
- data snooping;
- overfitting;
- extrapolation;
- selection bias;
- missing-data bias.

Diagnostic concepts:

- residual plots;
- leverage;
- Cook’s distance;
- VIF;
- heteroskedasticity tests;
- serial-correlation tests;
- robust standard errors;
- HAC;
- clustered standard errors;
- Breusch-Pagan / Cook-Weisberg intuition;
- White heteroskedasticity test;
- Park-test historical/teaching coverage where a syllabus requires it;
- Durbin-Watson statistic and its limitations;
- Breusch-Godfrey serial-correlation test;
- GLS and FGLS intuition;
- WLS as a heteroskedasticity remedy;
- heteroskedasticity-robust covariance estimators;
- specification tests and residual-pattern diagnostics;
- structural-break testing;
- measurement-error consequences;
- errors-in-variables models;
- data problems such as missing observations, coding errors, aggregation, mixed frequencies and influential revisions where pedagogically relevant.

### 17.1 Diagnostic-to-remedy visual workflow

For every major regression failure mode, VisualMetrics should expose a consistent sequence:

```text
What assumption is at risk?
-> Generate/choose a valid case
-> Introduce the violation gradually
-> Inspect graphical symptoms
-> Inspect formal diagnostic/test
-> Show estimator/inference consequence
-> Apply remedy/robust alternative
-> Compare before vs after
-> Explain what the remedy does NOT fix
```

This workflow is especially important for econometrics teaching in Algeria and the Arab world, where diagnostic-and-remedy sequences are explicit parts of many applied econometrics courses.

---

# 18. Endogeneity, Identification and Instrumental Variables

- endogeneity definition;
- correlation between regressor and error;
- omitted-variable mechanism;
- measurement-error mechanism;
- simultaneity mechanism;
- identification;
- under-identification;
- exact identification;
- over-identification;
- instrument relevance;
- exclusion restriction;
- instrument exogeneity;
- weak instruments;
- first stage;
- reduced form;
- 2SLS;
- LIML;
- IV-GMM;
- overidentification tests;
- weak-IV diagnostics;
- local average treatment effect intuition;
- control-function intuition where relevant.

Visual ideas:

- causal DAG view;
- correlation-with-error slider;
- first-stage projection;
- OLS vs IV path comparison;
- weak vs strong instrument animation;
- coefficient distribution under repeated sampling.

## 18.1 Simultaneous-Equation Systems, SUR and Multi-Equation Econometrics

This must be a first-class econometrics area, not merely a footnote inside endogeneity.

Topics:

- structural vs reduced-form equations;
- simultaneity bias;
- identification of structural equations;
- order-condition intuition;
- rank-condition intuition at an advanced level;
- endogenous vs predetermined vs exogenous variables;
- indirect least squares where pedagogically useful;
- 2SLS;
- 3SLS;
- systems GMM where appropriate;
- Seemingly Unrelated Regressions (SUR/SURE);
- cross-equation error correlation;
- efficiency gains from system estimation;
- equation-by-equation OLS vs system estimators;
- restrictions across equations.

### Required simultaneous-equations visual lab

A supply-demand style structural system may be used as one preset, but it must not be the only scenario. Controls should cover:

- strength of simultaneity;
- instrument strength;
- number of excluded exogenous variables;
- cross-equation error correlation;
- under/exact/over-identification;
- sample size;
- structural shock sizes;
- 2SLS vs 3SLS vs OLS comparison.

The visualization should distinguish **structural causal equations**, **reduced forms**, and **estimated regression relationships**.

### Required SUR lab

Show when SUR collapses to equation-by-equation OLS and when correlated equation errors plus differing regressor sets create efficiency gains.

---

# 19. Generalized Linear and Limited Dependent Variable Models

- binary outcomes;
- Linear Probability Model;
- Logit;
- Probit;
- marginal effects;
- odds and odds ratios;
- classification threshold;
- count data;
- Poisson;
- Negative Binomial;
- overdispersion;
- zero inflation;
- ordered response;
- multinomial choice;
- Tobit/censoring;
- truncation;
- selection models and Heckman intuition;
- duration/survival basics where useful.

---

# 20. Time-Series Foundations

- index and time ordering;
- lag operator;
- lead vs lag;
- trend;
- seasonality;
- cycles;
- white noise;
- autocovariance;
- autocorrelation;
- ACF;
- PACF;
- stationarity;
- weak/covariance stationarity;
- strict stationarity;
- ergodicity intuition;
- random walk;
- drift;
- deterministic vs stochastic trend;
- spurious regression.

## 20.1 ARIMA Family

- AR(p);
- MA(q);
- ARMA;
- ARIMA;
- SARIMA;
- invertibility;
- causality/stability;
- Box–Jenkins workflow;
- forecasts and intervals.

## 20.2 Unit Roots and Cointegration

- unit root;
- near-unit-root behavior;
- Dickey–Fuller intuition;
- ADF;
- PP;
- KPSS;
- structural-break complications;
- cointegration;
- Engle–Granger;
- Johansen;
- error-correction mechanism;
- speed of adjustment;
- short-run vs long-run effects;
- ARDL / bounds-testing intuition;
- distributed lags.

## 20.3 VAR and Structural Time Series

- VAR;
- lag selection;
- Granger causality;
- impulse-response functions;
- forecast-error variance decomposition;
- SVAR identification;
- sign/short-run/long-run restriction intuition;
- local projections;
- structural breaks;
- regime changes;
- threshold models.

## 20.4 Dynamic Econometrics and Model-Building Methodology

Add an explicit bridge between static regression and time-series econometrics:

- distributed-lag models;
- autoregressive distributed lag (ADL/ARDL) structures;
- lagged dependent variables;
- short-run vs long-run multipliers;
- dynamic stability;
- common-factor / error-correction representation intuition;
- general-to-specific modeling;
- Hendry/LSE-style model-building principles as historical and methodological context;
- encompassing and specification checking;
- lag-length choice;
- residual diagnostics;
- forecasting evaluation;
- structural breaks and parameter instability.

The GUI should support a **Dynamic Response Builder** where users change lag coefficients and immediately see the implied temporal response, long-run multiplier and stability behavior.

---

# 21. Volatility and Financial Econometrics

- conditional heteroskedasticity;
- ARCH;
- GARCH;
- persistence;
- leverage/asymmetry;
- EGARCH;
- GJR-GARCH;
- volatility clustering;
- Value at Risk intuition;
- Expected Shortfall intuition;
- heavy tails;
- volatility forecasting;
- realized volatility concepts;
- dynamic correlation concepts.

---

# 22. Panel Data Econometrics

- panel structure;
- balanced vs unbalanced panels;
- pooled OLS;
- individual effects;
- time effects;
- fixed effects;
- within transformation;
- between estimator;
- random effects;
- first differences;
- FE vs RE intuition;
- Hausman logic;
- two-way fixed effects;
- clustered inference;
- serial correlation in panels;
- cross-sectional dependence;
- dynamic panels;
- Nickell bias;
- Arellano–Bond intuition;
- Difference GMM;
- System GMM;
- instrument proliferation;
- panel unit roots;
- panel cointegration;
- panel quantile concepts where appropriate.

---

# 23. Causal Inference

## 23.1 Foundations

- association vs causation;
- prediction vs causation;
- counterfactual thinking;
- potential outcomes;
- treatment and control;
- individual treatment effects;
- ATE;
- ATT;
- ATC;
- SUTVA;
- exchangeability;
- positivity/overlap;
- consistency assumptions;
- selection bias.

## 23.2 DAGs

- nodes and edges;
- causal direction;
- chains;
- forks;
- colliders;
- confounders;
- mediators;
- descendants;
- backdoor paths;
- d-separation intuition;
- conditioning;
- collider bias;
- bad controls;
- front-door intuition;
- adjustment sets.

## 23.3 Designs and Estimators

- randomized experiments;
- matching;
- propensity scores;
- inverse probability weighting;
- doubly robust estimation;
- instrumental variables;
- regression discontinuity;
- Difference-in-Differences;
- event studies;
- staggered treatment;
- synthetic control;
- sensitivity analysis;
- partial identification concepts;
- heterogeneous treatment effects;
- causal forests;
- Double/Debiased Machine Learning.

---

# 24. Nonparametric and Semiparametric Statistics

- empirical distribution;
- kernel density estimation;
- bandwidth;
- kernel regression;
- local polynomial regression;
- splines;
- rank-based tests;
- sign tests;
- Wilcoxon/Mann–Whitney intuition;
- Kruskal–Wallis;
- permutation inference;
- bootstrap;
- semiparametric efficiency intuition;
- influence functions at an advanced level.

---

# 25. Multivariate Statistics

- covariance matrix;
- correlation geometry;
- multivariate normal;
- Mahalanobis distance;
- PCA;
- factor analysis;
- discriminant analysis;
- canonical correlation;
- MANOVA intuition;
- clustering;
- multidimensional scaling;
- dimensionality reduction.

---

# 26. Spatial Statistics and Spatial Econometrics

- spatial data structures;
- neighbors;
- weight matrices;
- spatial autocorrelation;
- Moran’s I intuition;
- local spatial association;
- spatial lag;
- spatial error;
- spillovers;
- direct/indirect effects;
- spatial heterogeneity;
- map-based diagnostics;
- spatial panel extensions as advanced modules.

---

# 27. Machine Learning Foundations

## 27.1 Core Ideas

- inference vs prediction;
- supervised vs unsupervised learning;
- features and targets;
- training/validation/test;
- loss functions;
- empirical risk;
- expected risk;
- bias–variance trade-off;
- underfitting;
- overfitting;
- data leakage;
- regularization;
- hyperparameters;
- cross-validation;
- learning curves;
- generalization;
- class imbalance;
- threshold selection;
- metrics.

## 27.2 Regression ML

- linear regression as ML;
- Ridge;
- Lasso;
- Elastic Net;
- polynomial regression;
- KNN regression;
- trees;
- random forests;
- boosting;
- gradient boosting;
- kernel regression.

## 27.3 Classification

- logistic classification;
- threshold;
- confusion matrix;
- accuracy;
- precision;
- recall;
- specificity;
- F1;
- ROC;
- AUC;
- precision-recall curves;
- calibration;
- cost-sensitive classification;
- KNN;
- Naive Bayes;
- LDA/QDA;
- decision trees;
- Random Forest;
- boosting;
- SVM;
- kernels.

## 27.4 Unsupervised Learning

- distance metrics;
- k-means;
- hierarchical clustering;
- DBSCAN intuition;
- Gaussian mixtures;
- EM algorithm intuition;
- PCA;
- manifold-learning intuition;
- anomaly detection.

---

# 28. Deep Learning

- perceptron;
- neuron;
- weights and biases;
- activation functions;
- layers;
- forward propagation;
- loss;
- computational graphs;
- gradients;
- chain rule;
- backpropagation;
- gradient descent;
- SGD;
- momentum;
- Adam intuition;
- learning rate;
- initialization;
- vanishing gradients;
- exploding gradients;
- normalization;
- dropout;
- regularization;
- CNN;
- convolutions;
- pooling;
- RNN;
- LSTM/GRU intuition;
- embeddings;
- attention;
- self-attention;
- transformers;
- encoder/decoder concepts.

---

# 29. Modern AI and Generative AI

- representation learning;
- tokenization;
- embeddings;
- positional encoding;
- attention weights;
- transformer blocks;
- autoregressive prediction;
- temperature;
- top-k/top-p intuition;
- context windows;
- hallucination as an uncertainty/grounding problem;
- retrieval-augmented generation concepts;
- vector search intuition;
- agents: model/tools/memory/planning loop;
- evaluation concepts;
- responsible AI;
- robustness;
- model uncertainty;
- distribution shift;
- calibration;
- conformal prediction;
- fairness metrics;
- explainability.

The package should focus on concepts that can be visualized rigorously and avoid superficial “AI animation” without educational substance.

---

# 30. Explainable AI

- global vs local explanation;
- feature importance;
- permutation importance;
- partial dependence;
- ICE;
- SHAP values;
- additive attribution;
- interaction effects;
- LIME intuition;
- counterfactual explanations;
- sensitivity;
- calibration;
- explanation instability;
- distinction between predictive explanation and causal interpretation.

---

# 31. Curriculum Benchmarking

The initial curriculum map should be informed by real university offerings while remaining broader than any single syllabus.

## 31.1 Algerian curriculum evidence — coverage reference only

Algerian university curricula and handouts are used here strictly as **coverage evidence**. They help determine which scientific concepts VisualMetrics must include, but their local course numbering or naming must **never become the canonical library taxonomy, GUI module names, Python API names, or concept IDs**. VisualMetrics must use standard scientific names such as **Probability**, **Inferential Statistics**, **Econometrics**, **Time Series Econometrics**, **Panel Data Econometrics**, **Causal Inference**, **Machine Learning**, and **Deep Learning**.

### 31.1.1 Probability distributions and bivariate random variables — Algerian coverage evidence

Current Algerian course pages and curriculum documents from universities such as Mila, Tlemcen, Jijel and Study in Algeria show a recurring Statistics 3 structure built around four axes:

1. important discrete probability distributions;
2. important continuous probability distributions;
3. convergence/approximation of selected distributions;
4. bivariate random variables.

Canonical VisualMetrics destination:

```text
Probability & Random Variables
```

with the following concept map:

```text
Probability & Random Variables
├── Discrete distributions
│   ├── Bernoulli
│   ├── Binomial
│   ├── Poisson
│   ├── Geometric
│   └── Hypergeometric
├── Continuous distributions
│   ├── Normal
│   ├── Uniform
│   ├── Exponential
│   ├── Gamma
│   ├── Beta
│   ├── Chi-square
│   ├── Student t
│   └── Fisher F
├── Distribution approximations/convergence
│   ├── Binomial -> Normal
│   ├── Binomial -> Poisson
│   └── Poisson -> Normal where justified
└── Bivariate random variables
    ├── Discrete joint distributions
    ├── Continuous joint distributions
    ├── Marginals
    ├── Conditionals
    ├── Independence
    ├── Covariance/correlation
    └── Functions of bivariate random variables
```

Important implementation implication: the canonical **Probability & Random Variables** module should include a first-class **Distribution & Joint-Random-Variable Explorer**.

Representative current/near-current references:

- University Abdelhafid Boussouf — Mila, Statistics 3 course page: https://elearning.univ-mila.dz/a2026/course/view.php?id=3428&lang=en
- Mila detailed Statistics 3 program: https://elearning.univ-mila.dz/a2024/mod/page/view.php?id=44844
- University of Tlemcen Statistics 3 course: https://elearn.univ-tlemcen.dz/course/view.php?id=7651
- University of Jijel Statistics 3 course: https://elearning.univ-jijel.dz/course/view.php?id=3238&lang=fr
- Study in Algeria curriculum example (Statistics 3, 2024-2025): https://studyinalgeria.dz/storage/Uploads/pages/20976/_full_curriculum_1743353447.pdf

### 31.1.2 Inferential Statistics — Algerian coverage evidence

Across current Algerian Statistics 4 course pages, the recurring core is:

```text
Sampling Theory
-> Estimation Theory
-> Confidence Intervals
-> Hypothesis Testing
```

Canonical VisualMetrics destination:

```text
Inferential Statistics
```

Required VisualMetrics coverage:

**Sampling**

- population/sample/sampling frame;
- sampling schemes;
- sampling distribution of the mean;
- sampling distribution of the proportion;
- sampling distribution of variance;
- difference of means;
- difference of proportions;
- ratio of variances;
- finite-population effects;
- standard errors;
- order statistics where present in the national/course syllabus;
- CLT and WLLN as inference foundations.

**Estimation**

- point estimation;
- Method of Moments;
- Maximum Likelihood;
- finite-sample estimator properties;
- large-sample properties;
- Fisher information;
- Cramér-Rao bound;
- confidence intervals for mean, proportion and variance;
- confidence intervals for differences in means/proportions;
- confidence interval for variance ratios;
- sample-size determination.

**Testing**

- null/alternative hypotheses;
- Type I and Type II errors;
- size and power;
- critical/rejection regions;
- p-values;
- Neyman-Pearson principle;
- one/two-sided tests;
- tests for mean, proportion and variance;
- two-sample comparisons;
- nonparametric tests when included in a course profile;
- Kolmogorov-Smirnov;
- Cramér-von Mises;
- Wilcoxon;
- Mann-Whitney;
- Kruskal-Wallis.

Representative references:

- University Abdelhafid Boussouf — Mila, Statistics 4 course: https://elearning.univ-mila.dz/a2024/course/view.php?id=3161
- University of Souk Ahras, Statistics 4: https://univ-soukahras.dz/moodle/course/view.php?id=5050
- University of Jijel, Statistics 4: https://elearning.univ-jijel.dz/course/view.php?id=4489
- Current national/program curriculum example exposed through Study in Algeria: https://studyinalgeria.dz/storage/Uploads/pages/30971/_full_curriculum_1772526724.pdf

### 31.1.3 Older Algerian inferential-statistics handouts — coverage evidence

Older Algerian handouts — and institutional repositories preserving previous programs — use different local numbering for inferential statistics. This reinforces the rule that VisualMetrics must ignore local numbering in its user-facing taxonomy and map all relevant content to the canonical **Inferential Statistics** module.

Canonical VisualMetrics destination:

```text
Inferential Statistics
```

Recurring content:

- probability-distribution review;
- sampling theory and sampling distributions;
- point estimation;
- interval estimation / confidence intervals;
- optimal sample size;
- hypothesis testing;
- solved applications/exercises.

Representative University of Algiers 3 repository materials:

- Amina Hand (2018), *Statistics 3: Inferential Statistics*: https://dspace.univ-alger3.dz/jspui/handle/123456789/6326
- Hamza Ben Sebaa (2020), *Lessons and Lectures in Inferential Statistics (Statistics 3)*: https://dspace.univ-alger3.dz/jspui/handle/123456789/6330
- Farida Hemal (2022), *Statistics 3*: https://dspace.univ-alger3.dz/jspui/handle/123456789/9056
- Asma Bouzenoura (2022), Statistics 3 handout: https://dspace.univ-alger3.dz/jspui/handle/123456789/7937
- Amal Belaidi (2019), Statistics 3 lectures including sampling, estimation, optimal sample size and testing: https://dspace.univ-alger3.dz/jspui/handle/123456789/6329

### 31.1.4 Core Econometrics — Algerian coverage evidence

The current Algerian Quantitative Economics curriculum provides a useful source for validating the **Econometrics** coverage. Its local course sequencing is not used as the VisualMetrics naming system.

**Core regression and econometric foundations to include**

- introduction to econometrics;
- mathematical/statistical prerequisites;
- simple linear regression by OLS;
- simple linear regression by MLE;
- multiple linear regression by OLS;
- multiple linear regression by MLE;
- extensions of multiple regression;
- structural changes/breaks;
- restricted models;
- dummy variables;
- partial correlation;
- multicollinearity;
- explanatory-variable selection.

**Diagnostics, violations, systems, and applied econometric foundations to include**

- autocorrelation of errors;
- heteroskedasticity;
- non-normality of errors;
- model misspecification;
- measurement error;
- data-related problems;
- simultaneous-equation models;
- SUR/SURE;
- modeling methodology;
- econometric-software applications.

Primary curriculum reference:

- Study in Algeria, Quantitative Economics curriculum: https://studyinalgeria.dz/pages/14888
- Full current curriculum document: https://studyinalgeria.dz/storage/Uploads/pages/30971/_full_curriculum_1772526724.pdf

### 31.1.5 Advanced Econometrics — Algerian coverage evidence

Advanced/master-level Algerian curricula broaden the expected coverage. VisualMetrics should therefore include canonical scientific modules/labs for:

- panel-data basics and advanced panel methods;
- instrumental variables;
- 2SLS;
- 3SLS;
- simultaneous equations;
- ADL/ARDL;
- ECM;
- VAR;
- serial correlation and heteroskedasticity;
- time-series diagnostics;
- unit roots;
- Box-Jenkins;
- financial econometrics and ARCH/GARCH;
- dynamic econometrics;
- forecasting.

A 2024-2025 Algerian Econometrics 2 master curriculum explicitly lists panel methods, IV, 2SLS, 3SLS, simultaneous equations and ADL/ECM/VAR among its content. VisualMetrics should treat this as validation for first-class labs in these areas rather than optional notes.

Reference example:

- Study in Algeria 2024-2025 master curriculum: https://studyinalgeria.dz/storage/Uploads/pages/28788/_full_curriculum_1749635171.pdf

### 31.1.6 Data Analysis and Statistical Computing — Algerian coverage evidence

The current Quantitative Economics curriculum also links econometrics to modules that should inform VisualMetrics navigation:

**Data Analysis 1**

- matrix operations;
- eigenvalues/eigenvectors and linear transformations;
- statistical inference;
- multiple linear regression;
- canonical correlation;
- factor analysis;
- PCA.

**Data Analysis 2**

- correspondence analysis;
- multiple correspondence analysis;
- cluster analysis;
- discriminant analysis;
- CART;
- logistic regression.

**Statistical Software**

- simple/multiple regression;
- autocorrelation diagnostics and remedies;
- multicollinearity diagnostics and remedies;
- heteroskedasticity diagnostics and remedies;
- randomness tests;
- seasonal-component detection and adjustment;
- ACF/PACF;
- stationarity tests;
- Box-Jenkins workflow.

These do not mean VisualMetrics should reproduce EViews or SPSS. They justify adding **software-independent visual concept labs** plus reproducible Python demonstrations.

### 31.1.7 Taxonomy rule derived from Algerian curriculum evidence

The GUI must **not** expose local course numbering such as “Statistics 3”, “Statistics 4”, “Econometrics 1”, or country-specific course trees as its primary navigation. The user-facing navigation must use universal scientific domains:

```text
Explore
├── Probability & Random Variables
├── Inferential Statistics
│   ├── Sampling Theory
│   ├── Estimation Theory
│   ├── Confidence Intervals
│   └── Hypothesis Testing
├── Econometrics
│   ├── Regression Foundations
│   ├── Diagnostics & Assumption Violations
│   ├── Endogeneity & Instrumental Variables
│   ├── Simultaneous Equations & Systems
│   └── Limited Dependent Variables
├── Time Series Econometrics
├── Panel Data Econometrics
├── Causal Inference
├── Statistical Learning
├── Machine Learning
├── Deep Learning
├── Explainable AI
└── Econometrics + AI
```

University curricula from Algeria and elsewhere remain an **internal coverage-validation layer**. They may be documented in a syllabus-mapping appendix or maintainer tools, but they must not dictate the main GUI/API taxonomy. One canonical concept implementation should satisfy all source curricula that contain that concept.

## 31.2 Arab World

VisualMetrics should benchmark Arab-world economics/econometrics programs so that the library remains useful beyond one national syllabus.

### 31.2.1 King Saud University — Saudi Arabia

Undergraduate and graduate econometrics descriptions validate a broad sequence including:

- linear regression assumptions, estimation, properties, testing and prediction;
- specification errors;
- multicollinearity;
- heteroskedasticity;
- autocorrelation;
- simultaneous equations;
- Least Squares;
- Maximum Likelihood;
- Instrumental Variables;
- Method of Moments / GMM;
- asymptotic inference;
- qualitative/limited dependent variables;
- panel data;
- unit roots and cointegration;
- SUR;
- VAR;
- applied Box-Jenkins, ECM, ARCH and panel methods.

Current/relevant references:

- ECON 416 Econometrics: https://cba.ksu.edu.sa/en/node/9401
- ECON 541 Econometrics: https://cba.ksu.edu.sa/en/ECON541
- ECON 542 Applied Econometrics: https://cba.ksu.edu.sa/en/ECON542
- ECON 423 Applied Econometrics: https://cba.ksu.edu.sa/ar/node/9407
- PhD Econometrics I/II sequence: https://cba.ksu.edu.sa/en/node/32167

### 31.2.2 American University of Beirut — Lebanon

AUB graduate econometrics provides a strong regional advanced benchmark.

Econometrics I includes:

- parameter estimation and hypothesis testing in the classical linear model;
- GLS;
- heteroskedasticity;
- autocorrelation;
- multivariate regression;
- GMM;
- simultaneous equations;
- panel data.

Econometrics II includes:

- dynamic models;
- structural VAR;
- impulse responses;
- variance decomposition;
- cointegration;
- error correction;
- ARCH;
- forecasting.

Reference:

- AUB Faculty of Arts and Sciences Graduate Catalogue 2025-26, ECON 305/306: https://www.aub.edu.lb/Registrar/catalogue2025-26/gr/Documents/gr-fas.pdf
- AUB catalogue portal: https://www.aub.edu.lb/Registrar/catalogue2026-27/gr/Pages/default.aspx

### 31.2.3 Cairo University — Egypt

Applied econometric teaching materials hosted by Cairo University provide useful diagnostic/remedy coverage for the GUI, including:

- multicollinearity and VIF;
- serial-correlation visualization;
- Durbin-Watson;
- AR(1)-based GLS;
- heteroskedasticity visualization;
- Park and White tests;
- WLS and robust standard errors;
- Box-Jenkins;
- ACF/PACF;
- unit roots;
- cointegration.

Reference:

- Cairo University, *Econometric Analysis using EViews*: https://scholar.cu.edu.eg/?q=mohamed_abonazel%2Fclasses%2Feconometric-analysis-using-eviews

VisualMetrics should teach the **underlying concepts**, not reproduce the EViews interface.

### 31.2.4 Other Arab-university profiles

The curriculum registry should be designed to add and version additional university profiles from Jordan, UAE, Qatar, Egypt, Morocco, Tunisia, Saudi Arabia and other Arab systems without changing the scientific engine.

When a syllabus introduces a concept not yet implemented, the curriculum mapper should mark it:

```text
covered
partial
planned
out_of_scope
needs_review
```

rather than silently omitting it.

### 31.2.5 Regional minimum econometrics coverage

Based on recurring Algerian and Arab curricula, the **regional minimum** for VisualMetrics should include:

```text
Statistical inference
-> Simple OLS
-> Multiple OLS
-> OLS inference
-> MLE link
-> Functional form / restrictions / dummies
-> Multicollinearity
-> Heteroskedasticity
-> Autocorrelation
-> Specification error
-> Measurement error
-> GLS/FGLS/WLS/robust covariance
-> Endogeneity / IV
-> Simultaneous equations / 2SLS / 3SLS
-> SUR
-> Limited dependent variables
-> Time-series stationarity / ARIMA
-> Unit roots / cointegration / ECM
-> VAR / IRF / FEVD
-> ARCH/GARCH
-> Panel FE/RE
-> Dynamic panel / GMM
-> Forecasting and model evaluation
```

This regional minimum should be fully represented in the concept registry even if some labs are initially marked `planned`.

## 31.3 International Benchmarks

### MIT

MIT Statistical Method in Economics provides a particularly important backbone for advanced statistical inference:

- samples and sample characteristics;
- convergence and limit theorems;
- LLN;
- CLT;
- Slutsky;
- sufficient statistics;
- estimator properties;
- Cramér–Rao;
- consistency/asymptotic normality/efficiency;
- Method of Moments;
- MLE;
- size and power;
- Neyman–Pearson;
- Wald testing;
- confidence sets;
- Bayesian inference.

References:

- https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/pages/syllabus/
- https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/pages/syllabus/
- https://catalog.mit.edu/schools/humanities-arts-social-sciences/economics/

### Stanford

Modern Statistical Learning curricula provide modern components including:

- decision theory;
- hold-out methods;
- linear models;
- ridge;
- bias–variance;
- cross-validation;
- learning/generalization;
- calibration;
- proper scoring rules;
- conformal prediction.

References:

- https://web.stanford.edu/class/stats315a/syllabus.html
- https://web.stanford.edu/class/stats202/intro.html

### Harvard

Relevant foundations and causal-inference offerings emphasize:

- probability and statistics;
- econometric foundations;
- regression and causal inference;
- potential outcomes;
- identification;
- randomized experiments;
- IV;
- matching;
- RDD;
- DiD;
- sensitivity analysis;
- partial identification.

References:

- https://my.harvard.edu/course/API114/2026-Fall/001
- https://imai.sites.fas.harvard.edu/teaching/cause.html

## 31.4 Curriculum Policy

VisualMetrics should not copy proprietary teaching text. It should use university curricula only to validate **topic coverage and sequencing**. All explanations, animations, proofs, diagrams and exercises must be original or based on properly licensed/public-domain material.

The architecture must allow the maintainer to add future handouts or university syllabi and map them to existing concepts without rewriting the package.


### 31.4.1 Curriculum registry and alias rules

Local curriculum names must never be treated as universal scientific definitions. A university may number or title the same scientific material differently across institutions or academic years.

If curriculum metadata is retained for maintainers, it should be explicitly marked as **source evidence**, for example:

```yaml
source_country: DZ
source_institution: univ_mila
source_program_family: SEGC
source_course_label: Statistics 3
academic_year: 2025-2026
source_profile_id: dz.segc.stat3.current
language: [ar, fr]
source_url: ...
maps_to_canonical_modules:
  - probability_random_variables
concept_ids: [...]
```

The GUI, documentation navigation, public API, concept IDs, package modules, and search taxonomy must display **canonical scientific names only**. Local course labels may appear only in an optional source/coverage appendix for traceability.

---

# 32. Concept Lab Standard

Every concept is represented by a structured `ConceptSpec` rather than scattered hard-coded GUI logic.

Recommended schema:

```yaml
id: inference.power.one_sample_t
version: 1
status: stable
domain: statistical_inference
subdomain: hypothesis_testing
level: [beginner, intermediate, advanced]

title_key: concepts.inference.power.title
summary_key: concepts.inference.power.summary

prerequisites:
  - probability.normal_distribution
  - inference.hypothesis_testing.basics

learning_objectives:
  - understand_power
  - distinguish_alpha_beta
  - see_effect_of_n

modes:
  - learn
  - visualize
  - animate
  - experiment
  - compare
  - derive
  - simulate
  - quiz

controls:
  - id: alpha
    type: slider
    min: 0.001
    max: 0.20
    default: 0.05
  - id: effect_size
    type: slider
    min: 0.0
    max: 2.0
    default: 0.5
  - id: n
    type: slider
    min: 5
    max: 1000
    default: 50
  - id: alternative
    type: select
    values: [two_sided, larger, smaller]

scenarios:
  - canonical
  - low_power
  - high_power
  - tiny_effect
  - large_effect
  - small_sample
  - large_sample

renderers:
  - plotly

backends:
  - statsmodels
  - scipy
```

---

# 33. Scenario Matrix Standard

Each important lab should consider these scenario categories.

| Category | Meaning |
|---|---|
| Canonical | textbook baseline |
| Positive | positive direction/effect where meaningful |
| Negative | negative direction/effect |
| Null | zero/no-effect/null case |
| Weak | weak relationship/signal/instrument/effect |
| Strong | strong relationship/signal/instrument/effect |
| Boundary | parameter near mathematical/statistical boundary |
| Violation | violated assumption |
| Counterexample | example showing why a common claim can fail |
| Sensitivity | one parameter varied systematically |
| Small sample | finite-sample issues |
| Large sample | asymptotic behavior |
| High noise | weak signal-to-noise |
| Low noise | clean signal |
| Robustness | compare robust vs non-robust method |
| Misspecification | wrong model/functional form |
| Real data | empirical dataset |
| Compare methods | competing estimators/algorithms |

A concept does not need every row, but every relevant row should be explicitly considered during design review.

---

# 34. Scientific Classification of Visual Explanations

Every visual must carry a `evidence_type` field:

```text
formal_proof
symbolic_derivation
geometric_proof
visual_derivation
visual_intuition
simulation
numerical_demonstration
counterexample
empirical_example
```

The GUI should display a small badge so learners know what they are looking at.

Example:

```text
[GEOMETRIC PROOF] OLS residuals are orthogonal to the column space of X
```

versus:

```text
[SIMULATION] Sampling distribution approaching normality as n grows
```

This prevents pedagogical overclaiming.

---

# 35. Flagship Visual Proof / Derivation Candidates

Prioritize mathematically meaningful visual proofs/derivations such as:

1. OLS as orthogonal projection.
2. Normal equations and residual orthogonality.
3. Frisch–Waugh–Lovell theorem geometry.
4. Omitted-variable-bias decomposition.
5. Variance inflation and near-collinearity geometry.
6. 2SLS as a two-projection procedure.
7. Confidence-interval repeated-sampling interpretation.
8. Neyman–Pearson rejection-region construction.
9. Likelihood-ratio geometry where practical.
10. Cramér–Rao information/curvature intuition with formal derivation panel.
11. Bayes updating as density multiplication/normalization.
12. PCA as variance-maximizing projection.
13. SVD geometry.
14. Ridge as constrained/penalized geometry.
15. Lasso constraint geometry and sparsity.
16. Gradient descent on convex/non-convex surfaces.
17. Chain rule and backpropagation computational graph.
18. Logistic link mapping real values to probabilities.
19. ROC threshold movement.
20. SHAP additive contribution decomposition.
21. DAG conditioning, blocking and collider activation.
22. Difference-in-Differences as counterfactual trend construction.
23. RDD local discontinuity logic.
24. Synthetic control weighted counterfactual construction.
25. Error-correction mechanism as disequilibrium adjustment.

For LLN and CLT, use the label **visual intuition/simulation** unless an actual proof is implemented in the derivation layer.

---

# 36. GUI Product Design

The GUI is a first-class component, not a demo.

## 36.1 Default Application Shell

```text
+--------------------------------------------------------------------------------+
| VisualMetrics | Search | Level | Language | Theme | Presentation | Help         |
+----------------------+---------------------------------------------------------+
| Domains / Concepts   |                                                         |
|                      |                  Visual Canvas                           |
| Statistics           |                                                         |
| Inference            |                                                         |
| Econometrics         |                                                         |
| Causal               |                                                         |
| ML / AI              |                                                         |
|                      +--------------------------------------+------------------+
| Favorites            | Explanation / Math / Proof / Code   | Controls         |
| Recent               |                                      | Sliders          |
| Learning Paths       |                                      | Toggles          |
|                      |                                      | Presets          |
+----------------------+--------------------------------------+------------------+
| Status | Evidence Type | Seed | Backend | Export | Reset | Reproduce Code      |
+--------------------------------------------------------------------------------+
```

## 36.2 Primary Navigation

- Home
- Explore Concepts
- Learning Paths
- Visual Proofs
- Statistical Labs
- Econometrics Labs
- Causal Labs
- AI Labs
- Compare Methods
- Datasets
- Instructor Mode
- Favorites
- Recent
- Glossary
- References
- Settings

## 36.3 Lab Workspace Panels

### Header

- concept title;
- translated title;
- difficulty;
- estimated prerequisite level;
- evidence badge;
- favorite button;
- share/export configuration.

### Center Visual Canvas

- primary visualization;
- animation controls;
- zoom/pan;
- pause/resume;
- step forward/backward;
- reset camera;
- compare overlay;
- snapshot;
- synchronized explanation panel;
- current-step caption;
- highlighted changing values/objects;
- visible interpretation of the current output;
- active-assumption and violated-assumption indicators where relevant.

### 36.3.1 Mandatory Animation Explanation Layer

Animation is a primary teaching mechanism in VisualMetrics, but **motion without explanation is not acceptable**. Every scientifically meaningful animation must explain the animation while it runs and after it ends. A learner or researcher should never have to infer the meaning of moving points, changing curves, highlighted areas or parameter transitions without guidance.

Each animation should provide, where applicable:

- **Purpose** — what question the animation is answering;
- **What you are seeing** — the objects, distributions, points, equations, parameters or model components currently shown;
- **What changed** — the exact quantity, assumption, parameter, sample, coefficient, boundary or model state that changed in the current step;
- **Why it changed** — the statistical, econometric, causal or algorithmic reason for the transition;
- **How to interpret it** — the scientific meaning of the visual output;
- **What to conclude** — the appropriate conclusion and any limits on that conclusion;
- **Assumptions in force** — which assumptions are active and which are intentionally violated;
- **Warnings** — common misinterpretations, invalid inferences, boundary cases and cases where the visualization is only intuition or simulation;
- **Synchronized mathematics** — formulas or derivation steps when they materially improve understanding;
- **Synchronized numeric values** — changing statistics, coefficients, probabilities, loss values, gradients, test statistics, p-values, power, errors or other relevant outputs;
- **Legend and semantic highlighting** — labels, shapes, arrows and colors that explain what each visual object represents;
- **End summary** — a concise interpretation panel after playback explaining what happened and why it matters.

Explanations should update with the animation state. For example, moving an alpha slider in a hypothesis-testing animation should update the critical region, Type I error area, power, explanatory text and any relevant conclusion at the same time.

The default beginner experience should favor plain-language explanation first, while Advanced/PhD modes may additionally expose mathematical derivations, formal assumptions, asymptotic arguments and implementation details. The same explanatory content must be available in English, Arabic and French.

### Control Panel

- parameter sliders;
- dropdowns;
- switches;
- distribution selector;
- scenario presets;
- sample size;
- seed;
- noise controls;
- data source;
- assumptions on/off;
- animation speed;
- “randomize scenario” button;
- reset to canonical case.

### Knowledge Tabs

- Overview
- Intuition
- Assumptions
- Math
- Proof/Derivation
- Simulation
- Diagnostics
- Misconceptions
- Code
- Data
- Quiz
- References

## 36.4 Compare Mode

User can pin 2–4 scenarios.

Examples:

- positive slope vs negative slope;
- homoskedastic vs heteroskedastic;
- OLS vs IV;
- FE vs RE;
- stationary vs unit-root process;
- one-sided vs two-sided test;
- low vs high power;
- Ridge vs Lasso;
- underfitting vs optimal vs overfitting;
- calibrated vs uncalibrated classifier.

Controls can be linked or independent.

## 36.5 Presentation Mode

For classroom projection:

- hide side navigation;
- increase font size;
- enlarge plot;
- simplify controls;
- enable keyboard shortcuts;
- animation step mode;
- annotation pointer;
- full-screen;
- no-code by default.

---

# 37. Color System

Colors must have meaning, remain customizable, and never be the only carrier of information.

## 37.1 Recommended Semantic Palette

| Semantic role | Default color | Meaning |
|---|---:|---|
| Primary action | `#0F766E` | main controls / active state |
| Secondary | `#7C3AED` | secondary concept/method |
| Positive / increase | `#16A34A` | positive effect / valid success |
| Negative / decrease | `#DC2626` | negative effect / failure |
| Warning | `#F59E0B` | caution / boundary condition |
| Information | `#0284C7` | neutral information |
| Null / baseline | `#64748B` | H0 / baseline / reference |
| Treatment | `#7C3AED` | treatment group |
| Control | `#0F766E` | control group |
| Type I error | `#E11D48` | alpha region |
| Type II error | `#D97706` | beta region |
| Power | `#16A34A` | 1-beta |
| Training | `#2563EB` | training data |
| Validation | `#9333EA` | validation data |
| Test | `#EA580C` | test data |

These are defaults, not hard-coded immutable values.

## 37.2 Accessibility Rules

- always combine color with label, shape, line style, texture or icon;
- support high-contrast mode;
- support common color-vision-deficiency palettes;
- contrast should target WCAG AA where applicable;
- user can disable semantic colors;
- user can select monochrome/publishing mode;
- legends are mandatory for multi-state charts;
- avoid red/green-only distinction.

## 37.3 Themes

At minimum:

- Light
- Dark
- High Contrast
- Classroom
- Publication
- Color-Blind Friendly
- Custom

A custom theme should be serializable to configuration.

---

# 38. Multilingual and i18n Architecture

Required first-class languages:

- English — `en`
- Arabic — `ar`
- French — `fr`

## 38.1 Translation Structure

```text
src/visualmetrics/i18n/
├── en/
│   ├── common.json
│   ├── concepts.json
│   ├── glossary.json
│   └── errors.json
├── ar/
│   ├── common.json
│   ├── concepts.json
│   ├── glossary.json
│   └── errors.json
└── fr/
    ├── common.json
    ├── concepts.json
    ├── glossary.json
    └── errors.json
```

## 38.2 RTL

When Arabic is active:

- main page flow switches to RTL;
- sidebars may move to the right;
- navigation aligns appropriately;
- tables remain readable;
- mathematical equations remain in correct mathematical direction;
- English variable names remain LTR inside Arabic text;
- mixed Arabic/Latin text is tested explicitly;
- chart labels should respect chosen language without mirroring numerical axes incorrectly.

## 38.3 Terminology Modes

User setting:

```text
Translated only
Bilingual terminology
English technical terminology
```

Arabic bilingual example:

```text
القوة الإحصائية (Statistical Power)
```

French bilingual example:

```text
Puissance statistique (Statistical Power)
```

## 38.4 Glossary

Every concept should have:

- English term;
- Arabic term;
- French term;
- aliases;
- acronym;
- concise definition;
- related concepts;
- common mistranslations or confusing terms when useful.

## 38.5 Search Across Languages

Typing any of these should reach the same concept:

```text
power
statistical power
القوة الإحصائية
القدرة الإحصائية
puissance
puissance statistique
```

Search indexing should include translations, aliases, acronyms and tags.

---

# 39. Accessibility and Inclusive Interaction

Required capabilities:

- keyboard navigation;
- visible focus states;
- screen-reader-friendly labels where framework permits;
- text scaling;
- reduced-motion mode;
- animation pause;
- step-through animation;
- high contrast;
- color-blind-friendly palettes;
- labels in addition to colors;
- descriptive figure captions;
- alternative textual explanation for important visuals;
- readable formulas;
- larger classroom font mode.

Users sensitive to motion should be able to disable continuous animation globally.

---

# 40. Technical Architecture

VisualMetrics should be organized as a **layered architecture** so scientific calculations, visual rendering, GUI and educational content remain separable.

```text
                         VisualMetrics
                              │
                 ┌────────────┴────────────┐
                 │                         │
            GUI / Notebook              Python API
                 │                         │
                 └────────────┬────────────┘
                              │
                        Concept Engine
                              │
        ┌───────────────┬─────┴─────┬────────────────┐
        │               │           │                │
   Scenario Engine   Proof Engine  Math Engine   Data/Simulation
        │               │           │                │
        └───────────────┴─────┬─────┴────────────────┘
                              │
                       Visual Renderers
                 Plotly / Manim / Graphs
                              │
                      Scientific Adapters
        NumPy / SciPy / statsmodels / linearmodels / PyFixest
        arch / scikit-learn / DoWhy / EconML / SHAP / InterpretML
```

## 40.1 Layer Responsibilities

### `core`

- data models;
- concept specifications;
- state management;
- validation;
- configuration;
- registries;
- common utilities.

### `backends`

Thin adapters around external scientific packages. Avoid leaking package-specific result objects throughout the application.

### `education`

- explanations;
- prerequisites;
- learning objectives;
- misconceptions;
- quizzes;
- curriculum metadata.

### `scenarios`

- synthetic data generators;
- parameter presets;
- violation generators;
- edge cases;
- Monte Carlo experiments.

### `visuals`

- figures;
- animations;
- layouts;
- visual encodings;
- annotations;
- comparison views.

### `proofs`

- formal/symbolic/geometric proof specs;
- proof steps;
- derivation state;
- rendering adapters.

### `gui`

- pages;
- navigation;
- controls;
- themes;
- event handling;
- no-code workflows.

### `notebook`

- ipywidgets integration;
- notebook-friendly views.

### `i18n`

- language resources;
- terminology modes;
- RTL helpers;
- locale formatting.

---

# 41. Recommended Repository Structure

```text
visualmetrics/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── concept_request.yml
│   │   ├── visual_proof_request.yml
│   │   └── translation_issue.yml
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── docs.yml
│   │   ├── release.yml
│   │   └── security.yml
│   ├── dependabot.yml
│   └── pull_request_template.md
├── docs/
│   ├── en/
│   ├── ar/
│   ├── fr/
│   ├── assets/
│   └── curriculum/
├── examples/
│   ├── statistics/
│   ├── inference/
│   ├── econometrics/
│   ├── causal/
│   └── ai/
├── notebooks/
│   ├── en/
│   ├── ar/
│   └── fr/
├── src/
│   └── visualmetrics/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── version.py
│       ├── core/
│       │   ├── concepts.py
│       │   ├── controls.py
│       │   ├── scenarios.py
│       │   ├── evidence.py
│       │   ├── registry.py
│       │   ├── state.py
│       │   └── exceptions.py
│       ├── catalog/
│       │   ├── search.py
│       │   ├── index.py
│       │   ├── graph.py
│       │   └── paths.py
│       ├── concepts/
│       │   ├── math/
│       │   ├── probability/
│       │   ├── descriptive/
│       │   ├── inference/
│       │   ├── regression/
│       │   ├── econometrics/
│       │   ├── timeseries/
│       │   ├── panel/
│       │   ├── causal/
│       │   ├── multivariate/
│       │   ├── spatial/
│       │   ├── ml/
│       │   ├── deep_learning/
│       │   └── ai/
│       ├── backends/
│       │   ├── scipy_backend.py
│       │   ├── sympy_backend.py
│       │   ├── statsmodels_backend.py
│       │   ├── linearmodels_backend.py
│       │   ├── pyfixest_backend.py
│       │   ├── arch_backend.py
│       │   ├── sklearn_backend.py
│       │   ├── dowhy_backend.py
│       │   ├── econml_backend.py
│       │   ├── shap_backend.py
│       │   └── interpret_backend.py
│       ├── data/
│       │   ├── generators/
│       │   ├── loaders/
│       │   ├── datasets/
│       │   ├── validation.py
│       │   └── schemas.py
│       ├── simulation/
│       │   ├── monte_carlo.py
│       │   ├── sampling.py
│       │   └── random.py
│       ├── math/
│       │   ├── symbolic.py
│       │   ├── geometry.py
│       │   └── formatting.py
│       ├── proofs/
│       │   ├── specs.py
│       │   ├── steps.py
│       │   ├── validation.py
│       │   └── library/
│       ├── visuals/
│       │   ├── plotly/
│       │   ├── manim/
│       │   ├── graphs/
│       │   ├── matplotlib/
│       │   ├── themes/
│       │   └── common/
│       ├── gui/
│       │   ├── app.py
│       │   ├── router.py
│       │   ├── pages/
│       │   ├── components/
│       │   ├── labs/
│       │   ├── state/
│       │   ├── themes/
│       │   └── accessibility/
│       ├── notebook/
│       │   ├── widgets.py
│       │   └── display.py
│       ├── education/
│       │   ├── objectives.py
│       │   ├── misconceptions.py
│       │   ├── quizzes.py
│       │   └── pathways.py
│       ├── i18n/
│       │   ├── en/
│       │   ├── ar/
│       │   ├── fr/
│       │   ├── translator.py
│       │   ├── rtl.py
│       │   └── glossary.py
│       ├── export/
│       │   ├── images.py
│       │   ├── html.py
│       │   ├── animation.py
│       │   ├── code.py
│       │   └── configs.py
│       └── plugins/
│           ├── discovery.py
│           └── protocol.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── gui/
│   ├── visual/
│   ├── translations/
│   ├── scientific/
│   └── regression/
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── SECURITY.md
├── pyproject.toml
├── mkdocs.yml
└── uv.lock
```

---

# 42. Dependency Strategy

The base installation must remain reasonably lightweight. Heavy capabilities should use optional extras.

## 42.1 Core Dependencies

Recommended core:

- `numpy` — arrays, numerical calculations, simulations;
- `scipy` — probability distributions, statistical tests, optimization;
- `pandas` — familiar tabular data interface;
- `sympy` — symbolic mathematics and derivations;
- `pydantic` — concept/config/schema validation;
- `platformdirs` — platform-specific config/cache locations;
- `babel` — locale-aware formatting if useful;
- `typing-extensions` where required.

## 42.2 GUI Dependencies

Primary recommendation for the initial GUI:

- `nicegui` — Python-first modern GUI/web interface;
- `plotly` — interactive visualizations;
- `pywebview` — optional native desktop window for NiceGUI native mode.

Keep GUI framework interactions behind VisualMetrics components so a future PySide6 frontend remains possible.

### Why NiceGUI first

- Python-native development;
- modern browser UI;
- easy controls and layouts;
- local application deployment;
- Plotly integration;
- can operate in a native-style desktop window with an appropriate webview backend;
- easier to make one GUI serve desktop and local browser use.

### Why keep PySide6 compatibility in mind

Qt/PySide6 offers mature desktop internationalization and bidirectional-layout support. If future requirements demand a pure Qt desktop product, the scientific core should not need to change.

## 42.3 Visualization Dependencies

- `plotly` — primary interactive charts, frames and animations;
- `matplotlib` — fallback and publication/static support;
- `manim` — optional high-quality mathematical animation / proof scenes;
- `networkx` — graph structures;
- optional Cytoscape/Dash integration if advanced draggable DAG interfaces are added.

## 42.4 Statistics / Econometrics Extras

- `statsmodels` — regression, statistical models, diagnostics, power analysis, time series;
- `linearmodels` — IV, GMM, panel and system models;
- `pyfixest` — fixed-effects regression, IV, modern DiD/event-study workflows and inference;
- `arch` — volatility, unit roots, cointegration and bootstrap-related tools.

## 42.5 Causal Inference Extras

- `dowhy` — causal modeling, identification, estimation and refutation workflows;
- `econml` — heterogeneous treatment effects, DML and causal ML estimators;
- `pywhy-graphs` — causal graph structures;
- `networkx` — generic graph manipulation.

## 42.6 ML / AI Extras

- `scikit-learn` — standard ML models, metrics, validation and inspection;
- `shap` — Shapley-based model explanations;
- `interpret` — glassbox/blackbox explanation ecosystem;
- optional `xgboost` and `lightgbm` adapters;
- optional `torch` extra for deep-learning demonstrations;
- do not require PyTorch for the base install.

## 42.7 Notebook Extras

- `ipywidgets`;
- Jupyter support packages as needed.

## 42.8 Export Extras

- `kaleido` for Plotly static-image export;
- Manim/FFmpeg tooling for optional video rendering;
- `imageio`/appropriate video helper only if needed.

## 42.9 Data I/O Extras

- `openpyxl` — Excel;
- `pyarrow` — Parquet/Arrow;
- `pyreadstat` — SPSS/Stata/SAS formats where needed;
- keep specialized I/O optional.

## 42.10 Documentation / Developer Extras

- `mkdocs-material`;
- `mkdocstrings[python]`;
- multilingual MkDocs plugin such as `mkdocs-static-i18n` if compatible;
- `pymdown-extensions`;
- `pytest`;
- `pytest-cov`;
- `hypothesis`;
- `ruff`;
- `mypy` or equivalent static typing check;
- `pre-commit`;
- `nox` or equivalent task automation.

---

# 43. Proposed Extras in `pyproject.toml`

Conceptual design:

```toml
[project.optional-dependencies]
gui = [
  "nicegui",
  "plotly",
  "pywebview",
]

econometrics = [
  "statsmodels",
  "linearmodels",
  "pyfixest",
  "arch",
]

causal = [
  "dowhy",
  "econml",
  "pywhy-graphs",
  "networkx",
]

ai = [
  "scikit-learn",
  "shap",
  "interpret",
]

proofs = [
  "manim",
]

notebook = [
  "ipywidgets",
]

data = [
  "openpyxl",
  "pyarrow",
  "pyreadstat",
]

dev = [
  "pytest",
  "pytest-cov",
  "hypothesis",
  "ruff",
  "mypy",
  "pre-commit",
  "nox",
]
```

A convenience `all` extra may aggregate user-facing capabilities, but CI must ensure the base package works without all extras installed.

---

# 44. Backend Adapter Contract

External packages should be wrapped behind stable VisualMetrics interfaces.

Example:

```python
class RegressionBackend(Protocol):
    def fit_ols(self, data, y, x, *, options=None) -> RegressionResult:
        ...
```

Normalize output to VisualMetrics data classes such as:

```python
RegressionResult(
    coefficients=...,
    standard_errors=...,
    residuals=...,
    fitted_values=...,
    covariance=...,
    diagnostics=...,
    backend="statsmodels",
    metadata=...,
)
```

Benefits:

- GUI is not tied to one library;
- alternative estimators can be compared;
- tests can use small deterministic fake backends;
- package-specific changes are isolated;
- educational renderers receive predictable structures.

---

# 45. Data Engine

Three primary data modes:

## 45.1 Generated Data

Default for theoretical labs.

A reproducible generator receives:

- sample size;
- seed;
- parameter values;
- distribution choices;
- noise;
- violations;
- treatment assignment;
- time-series process settings;
- panel structure;
- missingness/outliers.

Every generated dataset should expose a readable **Data Generating Process (DGP)** summary.

## 45.2 Built-In Educational Data

Use small, clearly licensed datasets. Store only data that has a compatible license. Prefer external package datasets where appropriate rather than duplicating large files.

## 45.3 User Data

Support progressively:

- CSV;
- Excel;
- Parquet;
- Stata;
- SPSS where optional dependencies permit;
- clipboard/table paste where practical.

User workflow:

```text
Load -> Inspect -> Select variables -> Validate -> Map to concept -> Run lab
```

The GUI must communicate when a lab requires synthetic data because a proof/controlled DGP cannot be guaranteed from arbitrary user data.

---

# 46. Simulation Engine

Monte Carlo is central to understanding statistical behavior.

Required controls:

- repetitions;
- sample size;
- seed;
- parameter grid;
- distribution;
- estimator;
- DGP;
- parallelism if later supported;
- stop/reset;
- reproducibility.

Outputs:

- estimator sampling distribution;
- bias;
- variance;
- MSE;
- coverage;
- rejection frequency;
- empirical size;
- empirical power;
- convergence paths;
- confidence bands around simulation metrics.

Performance rules:

- vectorize where possible;
- debounce GUI sliders;
- do not rerun 100k simulations on every minor slider event;
- offer `Fast`, `Teaching`, `High Precision` presets;
- show progress for expensive simulations;
- allow deterministic seeds.

---

# 47. Mathematical and Symbolic Engine

Use SymPy where it reduces hand-coded algebra.

Responsibilities:

- symbolic formulas;
- simplification;
- derivatives;
- gradients/Hessians;
- expectation/variance expressions when supported;
- matrix derivation helpers;
- LaTeX conversion;
- controlled substitutions;
- proof-step verification for supported identities.

Security rule:

> Never use Python `eval()` on arbitrary user-entered mathematical expressions.

If custom expressions are allowed, use a constrained parser/whitelist and validate allowed symbols/functions.

---

# 48. Proof Engine

The proof system should be declarative where possible.

## 48.1 `ProofSpec`

Example concept:

```yaml
id: regression.ols.residual_orthogonality
kind: geometric_proof
claim_key: proofs.ols.residual_orthogonality.claim
prerequisites:
  - math.linear_algebra.projection
  - regression.ols.geometry
steps:
  - id: define_projection
    equation: "y = X beta_hat + e"
  - id: least_squares_projection
    visual_action: project_y_onto_colspace_x
  - id: orthogonal_residual
    equation: "X' e = 0"
    visual_action: mark_right_angle
conclusion_key: proofs.ols.residual_orthogonality.conclusion
```

## 48.2 Renderer Options

- lightweight in-GUI step renderer;
- Plotly geometric renderer;
- SVG/HTML renderer;
- Manim renderer for high-quality exported animation.

The GUI should not require Manim merely to view basic proof steps.

## 48.3 Proof Navigation

- Previous step;
- Next step;
- Auto-play;
- Pause;
- Show/hide algebra;
- Show/hide geometric annotations;
- Explain this step;
- Reset proof;
- Export proof sequence.

---

# 49. Visualization Engine

## 49.1 Plot Types

Support reusable visual primitives:

- distributions;
- density overlap;
- sampling distributions;
- scatter/regression;
- residual geometry;
- vector projection;
- loss surfaces;
- contour plots;
- 3D planes;
- coefficient paths;
- regularization geometry;
- time series;
- ACF/PACF;
- impulse responses;
- event-study plots;
- treatment/control trajectories;
- DAGs;
- trees;
- decision boundaries;
- confusion matrices;
- ROC/PR curves;
- calibration curves;
- PCA projections;
- clustering;
- neural-network graphs;
- attention matrices;
- SHAP-style contribution diagrams.

## 49.2 Visual State

Every figure should be generated from a serializable state, not from hidden mutable globals.

```python
FigureState(
    concept_id="inference.power",
    scenario="low_power",
    parameters={...},
    theme="light",
    locale="ar",
)
```

This enables:

- reproducibility;
- sharing;
- screenshots;
- tests;
- saved lessons.

## 49.3 Animation State

Animations should have:

- deterministic timeline where possible;
- playback speed;
- step mode;
- replay;
- pause;
- previous/next step navigation;
- timeline/scrubber where scientifically meaningful;
- reduced-motion alternative;
- fixed axis ranges where movement could otherwise mislead;
- synchronized explanatory state;
- synchronized numeric and mathematical state;
- explicit current-step interpretation;
- explicit final interpretation/summary.

An animation state must not store only frame position. It should be possible to recover the **pedagogical meaning of the current frame**. A recommended conceptual state is:

```python
AnimationState(
    concept_id="inference.power",
    animation_id="sample_size_effect",
    frame=42,
    step_id="increase_n",
    parameters={"alpha": 0.05, "n": 80, "effect_size": 0.4},
    explanation_key="power.sample_size.increase",
    active_assumptions=["independent_sampling"],
    violated_assumptions=[],
    highlighted_objects=["h1_curve", "power_region"],
    outputs={"power": 0.81, "beta": 0.19},
)
```

The rendering layer should be able to use this state to update the picture, captions, equations, highlighted values and interpretation together.

---

# 50. Causal Graph Engine

Recommended stack:

- `pywhy-graphs` for causal structures where applicable;
- `networkx` for generic graph manipulation;
- optional interactive graph renderer for drag-and-drop nodes.

DAG lab capabilities:

- add/remove nodes;
- draw directed edges;
- mark treatment/outcome;
- mark observed/unobserved variables;
- classify chain/fork/collider;
- highlight open/closed paths;
- condition on a variable;
- show how conditioning changes path status;
- suggest valid adjustment sets where backend support permits;
- show invalid controls;
- animate collider bias;
- compare association and causal estimand.

The system should distinguish **graphical identification** from **statistical estimation**.

---

# 51. Econometrics Backend Map

## SciPy

Reuse for:

- distributions;
- numerical probability;
- optimization;
- many classical tests.

## statsmodels

Reuse for:

- OLS/GLM;
- diagnostics;
- statistical inference;
- power analysis;
- standard time-series components.

## linearmodels

Reuse for:

- IV2SLS;
- LIML;
- IV-GMM;
- panel models;
- systems where appropriate.

## PyFixest

Reuse for:

- fixed effects;
- IV workflows;
- robust inference;
- DiD/event-study capabilities;
- multiple-testing related features where useful.

## arch

Reuse for:

- ARCH/GARCH;
- unit-root tools;
- cointegration tools;
- bootstrap-related functionality.

Principle:

> The backend computes; VisualMetrics teaches, visualizes, compares and explains.

---

# 52. AI Backend Map

## scikit-learn

Use for canonical educational implementations of:

- linear/logistic models;
- regularization;
- KNN;
- trees;
- ensembles;
- SVM;
- clustering;
- PCA;
- metrics;
- calibration;
- cross-validation.

## SHAP

Use as a backend for Shapley-based explanations. VisualMetrics should provide its own pedagogical layer explaining what a contribution means and warn that predictive attribution is not automatically causal attribution.

## InterpretML

Use where glassbox and blackbox explanation APIs are useful.

## PyTorch — optional

Use only for deep-learning labs requiring actual trainable neural models. Basic neuron/backprop/attention educational labs should not depend on huge models.

---

# 53. No-Code / Code Parity

A central differentiator.

Example GUI state:

```text
Concept: Simple Linear Regression
Slope: -1.8
Intercept: 3
n: 120
Noise: 0.7
Functional form: level-level
Heteroskedasticity: off
Seed: 42
```

The `Show Python` panel should generate something conceptually like:

```python
import visualmetrics as vm

vm.lab(
    "regression.simple_linear",
    slope=-1.8,
    intercept=3,
    n=120,
    noise=0.7,
    functional_form="level_level",
    heteroskedasticity=False,
    seed=42,
    language="en",
)
```

The generated code must reproduce the same state as closely as possible.

---

# 54. Public Python API

Keep the API discoverable and consistent.

## 54.1 Core API

```python
import visualmetrics as vm

vm.launch()
vm.search("statistical power")
vm.list_concepts(domain="inference")
vm.lab("inference.power")
vm.explain("econometrics.endogeneity")
vm.compare("regression.ols", "regression.iv")
```

## 54.2 Configuration

```python
vm.configure(
    language="ar",
    terminology="bilingual",
    theme="light",
    level="intermediate",
    reduced_motion=False,
)
```

## 54.3 Concept-Specific Convenience Namespaces

Optional convenience:

```python
vm.stats.power(...)
vm.stats.clt(...)
vm.econometrics.ols(...)
vm.econometrics.iv(...)
vm.causal.did(...)
vm.ml.bias_variance(...)
vm.ai.attention(...)
```

Avoid generating hundreds of inconsistent hand-written top-level functions. Prefer registry-backed convenience wrappers.

---

# 55. CLI Design

Commands:

```bash
visualmetrics
visualmetrics gui
visualmetrics list
visualmetrics search "power"
visualmetrics show inference.power
visualmetrics doctor
visualmetrics info
visualmetrics examples
visualmetrics export ...
```

### `visualmetrics doctor`

Checks:

- Python version;
- core dependencies;
- optional extras;
- GUI availability;
- Plotly/Kaleido availability;
- Manim availability;
- FFmpeg/LaTeX requirements if proof-video rendering is requested;
- locale resources;
- write access to cache/config directories.

Do not make `doctor` fail the entire package merely because an optional backend is missing.

---

# 56. Search and Concept Knowledge Graph

The library should contain a searchable concept graph.

Example links:

```text
Probability
 -> Random Variables
 -> Sampling Distributions
 -> LLN / CLT
 -> Standard Error
 -> Confidence Intervals
 -> Hypothesis Testing
 -> Size / Power

Linear Algebra
 -> Projection
 -> OLS Geometry
 -> FWL
 -> IV Projection

Prediction
 -> Loss
 -> Bias-Variance
 -> Regularization
 -> Cross-Validation
 -> Generalization
```

Each concept may declare:

- prerequisites;
- related concepts;
- next concepts;
- frequently confused concepts;
- alternative names;
- curriculum tags.

GUI features:

- interactive prerequisite map;
- “What should I learn first?”;
- “What comes next?”;
- “Compare with...”;
- search by Arabic, French or English term.

---

# 57. Learning Levels

Use levels as views, not separate duplicated content.

## Beginner

- intuition;
- simple equations;
- guided presets;
- minimal jargon;
- strong visual explanation.

## Intermediate

- standard formulas;
- assumptions;
- typical diagnostics;
- code view.

## Advanced

- derivations;
- asymptotic properties;
- alternative estimators;
- robust inference;
- counterexamples.

## PhD

- formal assumptions;
- identification;
- theorem/proof steps;
- semiparametric/advanced inference when implemented;
- paper/reference links;
- advanced comparison.

Switching level should show/hide depth, not change scientific truth.

---

# 58. Detailed Flagship Lab Specifications

## 58.1 Simple Regression Lab

Controls:

- slope `beta1`;
- intercept `beta0`;
- sample size;
- x distribution;
- x range;
- error distribution;
- noise scale;
- seed;
- positive/negative/zero slope preset;
- linear/quadratic/cubic/log/exponential relation;
- level-level / log-level / level-log / log-log;
- heteroskedasticity type;
- outlier count/magnitude;
- leverage point;
- measurement error;
- omitted-variable strength;
- endogeneity correlation;
- confidence level;
- show/hide confidence interval;
- show/hide prediction interval;
- standardize variables;
- center variables.

Views:

- scatter + true DGP + estimated line;
- residuals;
- SSE;
- sampling distribution of slope;
- coefficient table;
- assumption dashboard;
- generated code.

Presets:

- clean positive relation;
- clean negative relation;
- no relationship;
- high noise;
- outlier-driven slope;
- leverage-driven fit;
- heteroskedastic fan;
- nonlinear truth fitted linearly;
- omitted-variable bias;
- endogeneity;
- log-log elasticity.

## 58.2 LLN Lab

Controls:

- distribution;
- population parameters;
- number of observations;
- number of paths;
- seed;
- statistic (mean, proportion where appropriate);
- heavy-tail option;
- compare finite/infinite variance examples where scientifically appropriate.

Visuals:

- running sample mean;
- true expectation line;
- multiple convergence paths;
- distance from target;
- distribution of sample averages.

Label default visual as **simulation/visual intuition**, not proof.

## 58.3 CLT Lab

Controls:

- parent distribution;
- `n`;
- number of repeated samples;
- standardization toggle;
- seed;
- show population distribution;
- compare multiple sample sizes.

Visuals:

- parent distribution;
- distribution of sums/means;
- standardized sampling distribution;
- normal reference overlay;
- animation across `n`;
- convergence metric only if pedagogically justified.

Include examples where convergence is slow and explain required conditions.

## 58.4 Hypothesis Testing Lab

Controls:

- H0 parameter;
- H1 parameter/effect size;
- alpha;
- n;
- variance;
- test direction;
- test family;
- observed statistic;
- confidence level.

Visuals:

- H0/H1 distributions;
- critical region;
- alpha/beta/power;
- p-value;
- power curve;
- repeated-sampling rejection animation;
- CI/test duality.

## 58.5 OLS Geometry Lab

Controls:

- dimensions limited to visualization-capable cases;
- predictor correlation;
- y vector;
- noise;
- centering;
- intercept inclusion.

Visuals:

- column space;
- y;
- fitted projection;
- residual;
- orthogonality markers;
- normal equations;
- projection matrix.

Evidence badge: **Geometric Proof / Visual Derivation** where exact.

## 58.6 Multicollinearity Lab

Controls:

- correlation among predictors;
- sample size;
- noise;
- true coefficients;
- standardization;
- repeated samples.

Visuals:

- predictor geometry;
- near-parallel directions;
- VIF;
- standard errors;
- coefficient instability;
- confidence regions;
- prediction stability vs coefficient instability.

## 58.7 Endogeneity / IV Lab

Controls:

- `corr(X,u)`;
- instrument strength;
- direct instrument-to-outcome violation toggle;
- confounder strength;
- sample size;
- noise;
- weak-instrument preset.

Visuals:

- DAG;
- OLS vs IV slope;
- first stage;
- reduced form;
- repeated-sampling bias;
- weak-IV distribution;
- assumption status.

## 58.8 Unit Root Lab

Controls:

- AR coefficient rho;
- drift;
- trend;
- sample length;
- innovation variance;
- seed.

Presets:

- rho=0;
- stationary rho=0.5;
- near-unit-root 0.95/0.99;
- unit root 1.0;
- explosive >1;
- random walk with drift.

Visuals:

- time path;
- ACF;
- variance through time;
- differenced series;
- test results if backend enabled;
- repeated-sample distributions.

## 58.9 Difference-in-Differences Lab

Controls:

- pre-trend slopes;
- treatment timing;
- treatment effect;
- anticipation;
- heterogeneous effects;
- staggered adoption;
- noise;
- group sizes.

Visuals:

- treated/control trajectories;
- counterfactual trajectory;
- DiD rectangle/gaps;
- event-study coefficients;
- parallel-trends violation;
- TWFE warning scenarios.

## 58.10 Bias–Variance Lab

Controls:

- model complexity;
- sample size;
- noise;
- training sample realization;
- repeated datasets;
- regularization.

Visuals:

- multiple fitted models;
- bias;
- variance;
- train/test error;
- expected error decomposition intuition.

## 58.11 Classification Threshold Lab

Controls:

- threshold;
- class balance;
- score separation;
- cost ratio;
- sample size.

Visuals:

- score distributions;
- TP/FP/TN/FN;
- confusion matrix;
- precision/recall;
- ROC position;
- cost curve where relevant.

## 58.12 Gradient Descent Lab

Controls:

- learning rate;
- starting point;
- iterations;
- momentum where supported;
- convex/non-convex surface;
- noise/stochastic mode.

Visuals:

- contour/surface;
- optimization path;
- gradient vector;
- loss by iteration;
- divergence for excessive learning rate.

## 58.13 Backpropagation Lab

Visualize:

- computational graph;
- forward values;
- local derivatives;
- chain-rule multiplication;
- backward gradients;
- parameter update.

Animation should support step-by-step propagation.

## 58.14 PCA Lab

Controls:

- correlation;
- variance ratio;
- rotation;
- noise;
- number of components.

Visuals:

- point cloud;
- principal axes;
- projection;
- explained variance;
- reconstruction error.

## 58.15 SHAP / Explanation Lab

Controls:

- model;
- observation;
- baseline;
- feature perturbation where supported.

Visuals:

- baseline prediction;
- additive contributions;
- final prediction;
- local vs global explanation;
- explicit warning: predictive attribution is not automatically causal effect.

## 58.16 Attention Lab

Controls:

- short token sequence;
- query/key similarity examples;
- temperature/scaling illustration;
- head selection in educational toy model.

Visuals:

- token embeddings conceptually;
- Q/K/V flow;
- attention score matrix;
- softmax weights;
- weighted value aggregation.

Keep it a transparent toy model rather than relying on a giant pretrained model.

---

# 59. Practical README Specification

`README.md` must function as a practical product guide, not as a one-paragraph project description.

## 59.1 README Required Sections

1. Project name and one-line mission.
2. Screenshots/GIF placeholders once available.
3. What VisualMetrics is.
4. What VisualMetrics is not.
5. Main domains.
6. Feature matrix.
7. Installation.
8. Quick start — GUI.
9. Quick start — Python API.
10. Language switching.
11. Terminology modes.
12. GUI map.
13. Learning modes.
14. Scenario presets.
15. How to use your own data.
16. How to export visuals.
17. How to reproduce GUI work in Python.
18. Optional extras.
19. `visualmetrics doctor` troubleshooting.
20. Notebook usage.
21. Visual proof policy.
22. Scientific disclaimer / proof-vs-simulation distinction.
23. Accessibility.
24. Supported Python versions.
25. Contributing.
26. Adding a new concept.
27. Adding a translation.
28. Citation.
29. License.
30. Contact and repository.

## 59.2 README Feature Matrix Example

```markdown
| Capability | GUI | Python API | Notebook | Arabic | English | French |
|---|---:|---:|---:|---:|---:|---:|
| Statistical inference | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Econometrics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Visual proofs | ✅ | ✅ | Limited | ✅ | ✅ | ✅ |
| Interactive scenarios | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Causal DAGs | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AI explanations | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
```

## 59.3 GUI Walkthrough in README

Explain, with screenshots added later:

```text
1. Launch VisualMetrics
2. Choose language
3. Choose level
4. Select a domain
5. Open a concept
6. Select a scenario preset
7. Move controls
8. Compare cases
9. Open Math / Proof / Code
10. Export or save configuration
```

## 59.4 README Examples Must Be Real

Do not leave fake API examples after implementation. README examples should be smoke-tested in CI wherever possible.

---

# 60. Full Documentation Site

Recommended stack:

- MkDocs Material;
- mkdocstrings;
- multilingual plugin;
- Mermaid for architecture diagrams;
- searchable glossary;
- rendered notebooks selectively.

Documentation hierarchy:

```text
Docs
├── Getting Started
├── GUI Guide
├── Learning Paths
├── Statistics
├── Statistical Inference
├── Econometrics
├── Time Series
├── Panel Data
├── Causal Inference
├── Machine Learning
├── Deep Learning
├── AI / Explainability
├── Visual Proofs
├── Datasets
├── Python API
├── Developer Guide
├── Add a Concept
├── Add a Language
└── Scientific References
```

Each language should have a navigable documentation tree. If some advanced pages are initially available only in English, show a transparent fallback badge rather than a broken page.

---

# 61. Instructor Features

VisualMetrics should grow into a teaching tool.

## 61.1 Instructor Mode

Features:

- lesson playlist;
- choose concepts in sequence;
- lock selected controls;
- hide answers;
- show/hide formulas;
- quiz mode;
- random scenario generator;
- full-screen presentation;
- large labels;
- timed animation;
- pause at teaching checkpoints;
- export lesson configuration.

## 61.2 Classroom Presets

Example presets:

```text
Regression: slope sign
Regression: outlier trap
Power: effect of n
CLT: different parent distributions
OLS: projection geometry
IV: weak instrument
Time series: rho approaches 1
DID: parallel trends violation
ML: overfitting
Classification: threshold trade-off
```

---

# 62. Quiz and Assessment System

Question types:

- multiple choice;
- true/false;
- predict-the-visual-result;
- select the violated assumption;
- choose the correct graph;
- manipulate a slider to reach a target;
- match concept and definition;
- identify Type I/II regions;
- identify confounder/collider/mediator;
- compare model behavior.

Quiz design rules:

- explanation after answer;
- localized text;
- no high-stakes grading claim;
- question bank tagged by level;
- randomization with reproducible seeds;
- store progress locally only if user enables it.

---

# 63. Misconception Engine

Each concept may define common misconceptions.

Examples:

### p-value

- “The p-value is the probability H0 is true.” -> incorrect.
- “A small p-value measures effect size.” -> incorrect.

### Confidence interval

- clarify repeated-sampling interpretation.

### R-squared

- high R² does not imply causality;
- high R² does not guarantee correct specification.

### Correlation

- correlation is not causation.

### SHAP

- feature attribution is not automatically causal attribution.

### DiD

- a visually parallel pre-period alone does not prove the identifying assumption.

The GUI can display a “Common mistake” callout after relevant experiments.

---

# 64. Data-Generating Process (DGP) Builder

Advanced GUI feature.

User can configure a DGP without code:

- choose variables;
- distributions;
- coefficients;
- correlations;
- nonlinear transformations;
- interaction terms;
- error structure;
- heteroskedasticity;
- serial correlation;
- latent confounder;
- instrument;
- panel unit/time effects;
- treatment assignment;
- time-series persistence.

The GUI should then generate both:

1. dataset;
2. equivalent Python configuration/code.

This feature supports research teaching and custom counterexamples.

---

# 65. Reproducibility

Every lab state should be exportable as a small JSON/YAML configuration.

Example:

```json
{
  "concept": "inference.power.one_sample_t",
  "visualmetrics_version": "0.1.0",
  "language": "ar",
  "theme": "classroom",
  "seed": 42,
  "parameters": {
    "alpha": 0.05,
    "effect_size": 0.5,
    "n": 50
  }
}
```

Rules:

- include VisualMetrics version;
- include backend versions when scientific results depend on them;
- include seed;
- include important configuration;
- allow reloading the state;
- warn if a future version cannot reproduce an old state exactly.

---

# 66. Export System

Supported targets should progressively include:

- PNG;
- SVG;
- HTML interactive figure;
- JSON/YAML lab state;
- Python reproduction script;
- GIF for animations where supported;
- MP4 for Manim/video output where configured;
- CSV of simulation summaries;
- Markdown summary;
- publication-ready static plot mode.

Exports should include metadata optionally:

- concept;
- parameter values;
- seed;
- package version;
- backend;
- date/time generated.

---

# 67. Performance Architecture

## 67.1 Lazy Imports

Heavy optional libraries must be imported only when needed.

Example:

```text
Opening a simple probability lab must not import:
PyTorch + SHAP + EconML + Manim + PyFixest
```

## 67.2 Lazy Concept Loading

Do not construct all concept objects at app startup.

Catalog metadata can be lightweight; the lab implementation is loaded when opened.

## 67.3 Caching

Cache appropriately:

- generated static metadata;
- expensive deterministic visuals;
- rendered proof scenes;
- simulation results keyed by parameters and seed when beneficial.

Use platform cache directories and expose a “Clear cache” option.

## 67.4 GUI Responsiveness

- debounce sliders for expensive calculations;
- distinguish live-update controls from “Apply” controls;
- cancel stale computations if feasible;
- show progress for long tasks;
- avoid freezing the UI.

---

# 68. Error Handling

Errors should be educational and localized.

Bad:

```text
ValueError: shapes (50,2) and (3,) not aligned
```

Better GUI message:

```text
The selected model needs three predictor coefficients, but only two predictors are active.
Open “Model setup” to correct the specification.
```

Developer details can appear in an expandable technical panel.

Error categories:

- invalid parameter;
- unsupported scenario;
- missing optional dependency;
- data format error;
- missing values;
- model identification issue;
- numerical failure;
- proof renderer unavailable;
- export failure;
- translation key missing.

---

# 69. Optional Dependency UX

If a user opens a feature whose dependency is missing:

```text
This visual requires the Econometrics extra.
Install:
    pip install "visualmetrics[econometrics]"
```

Do not crash at import time.

`visualmetrics doctor` should list capabilities:

```text
Core ................ OK
GUI ................. OK
Econometrics ........ OK
Causal .............. Missing EconML
Visual proofs ....... Manim available; FFmpeg missing
Notebook ............ OK
```

---

# 70. Testing Strategy

Testing must cover **scientific correctness**, not only software execution.

## 70.1 Unit Tests

- schemas;
- translations;
- config;
- registry;
- scenario generation;
- transformations;
- utility functions.

## 70.2 Scientific Regression Tests

Compare outputs to known analytical results or trusted backend results.

Examples:

- OLS coefficient on deterministic data;
- standard-error formulas;
- power calculation benchmark;
- known probability values;
- fixed seed simulation summaries within tolerance;
- IV simple benchmark;
- FE transformation identities.

## 70.3 Property-Based Tests

Use Hypothesis where useful.

Examples:

- CDF stays within [0,1];
- probabilities sum appropriately;
- variance nonnegative;
- changing seed changes random realization but preserves shape constraints;
- projection residual is orthogonal within tolerance;
- normalized weights sum to one.

## 70.4 Translation Tests

- no missing mandatory keys;
- no accidental untranslated UI labels in stable modules;
- Arabic RTL smoke test;
- mixed Arabic/English strings;
- French accents;
- fallback behavior.

## 70.5 GUI Tests

Smoke test:

- app launches;
- major pages open;
- controls update state;
- reset works;
- language switch works;
- concept search works;
- optional dependency warning works.

## 70.6 Visual Regression Tests

Use selectively for stable layouts/figures. Avoid brittle pixel-perfect testing everywhere.

## 70.7 Accessibility Checks

Automate what the framework permits and maintain manual checklist for:

- keyboard;
- contrast;
- reduced motion;
- label redundancy.

---

# 71. Quality Gates

A pull request should not merge if it breaks:

- test suite;
- formatting/lint;
- type checking for designated modules;
- translation integrity;
- package build;
- minimal import test;
- GUI smoke test in supported CI context;
- docs build.

Recommended quality commands:

```bash
ruff check .
ruff format --check .
pytest
python -m build
```

Add type/docs checks according to final tooling.

---

# 72. Scientific Review Checklist for a New Concept

Before a concept becomes `stable`, review:

- Is the definition correct?
- Are assumptions explicit?
- Is the visualization faithful?
- Can axes/scaling visually mislead?
- Are edge cases represented?
- Are counterexamples available when important?
- Does the lab distinguish finite vs asymptotic results?
- Is simulation labeled as simulation?
- Is a claimed proof actually a proof?
- Are formulas verified?
- Are default parameters pedagogically sensible?
- Are backend calculations tested?
- Are Arabic/French translations reviewed?
- Are references included?
- Does GUI-generated code reproduce state?

Statuses:

```text
draft -> experimental -> reviewed -> stable -> deprecated
```

---

# 73. Security and Privacy

## 73.1 Local Binding

Local GUI server should bind to localhost by default, not public interfaces.

## 73.2 File Handling

- sanitize uploads;
- limit file size;
- validate extensions and parsers;
- do not execute uploaded files;
- never use `eval` on user content;
- protect against path traversal.

## 73.3 User Data

- no telemetry by default;
- no data upload by default;
- do not persist datasets unless the user asks;
- explain where saved states are stored.

## 73.4 Dependencies

- Dependabot or equivalent;
- dependency pinning/lockfile for development;
- regular security updates;
- `SECURITY.md` with reporting contact.

---

# 74. Configuration

Recommended user config location through `platformdirs`.

Possible settings:

```yaml
language: ar
terminology: bilingual
level: intermediate
theme: light
reduced_motion: false
live_update: true
precision: 4
seed: 42
show_backend: false
show_code: false
colorblind_mode: none
classroom_font_scale: 1.25
```

Priority:

```text
explicit function/CLI args
> project config
> user config
> defaults
```

---

# 75. Theme System

A theme object should control:

- semantic colors;
- backgrounds;
- foregrounds;
- grid lines;
- font scale;
- line widths;
- marker size;
- animation emphasis;
- plot templates;
- GUI component styling.

Do not hard-code colors inside individual concept modules.

Concept modules request semantic roles:

```python
color("null")
color("alternative")
color("power")
color("warning")
```

not raw values everywhere.

---

# 76. Plugin Architecture

Future third parties should be able to add concepts without modifying core VisualMetrics.

Possible plugin interface:

```python
from visualmetrics.plugins import ConceptPlugin

class MyConcept(ConceptPlugin):
    ...
```

or Python entry points:

```toml
[project.entry-points."visualmetrics.concepts"]
my_lab = "my_package.plugin:plugin"
```

Plugin validation must check:

- unique concept IDs;
- compatible VisualMetrics version;
- translation keys;
- control schemas;
- safe metadata;
- declared optional dependencies.

---

# 77. Adding a New Concept — Developer Workflow

Desired workflow:

```text
1. Create ConceptSpec
2. Define learning objectives
3. Define prerequisites
4. Define scenarios
5. Define controls
6. Connect scientific backend
7. Build visualization(s)
8. Add proof/derivation if applicable
9. Add explanations
10. Add misconceptions
11. Add quiz
12. Add EN/AR/FR translations
13. Add tests
14. Add documentation
15. Register concept
```

Provide a scaffolding command later:

```bash
visualmetrics dev new-concept econometrics.my_new_concept
```

This can generate template files.

---

# 78. Versioning and Releases

Use Semantic Versioning.

Examples:

- `0.1.0` initial public alpha;
- `0.2.0` new domain or substantial feature;
- `0.2.1` bug fix;
- `1.0.0` stable public API and mature core curriculum.

Maintain:

- `CHANGELOG.md`;
- GitHub releases;
- PyPI releases;
- tagged versions;
- migration notes for breaking changes.

---

# 79. GitHub Repository Standards

Required files:

- README.md
- LICENSE
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- SECURITY.md
- CITATION.cff
- CHANGELOG.md
- pyproject.toml
- uv.lock
- .gitignore
- pre-commit config
- GitHub Actions workflows
- issue templates
- PR template

Recommended labels:

```text
area:statistics
area:econometrics
area:causal
area:ai
area:gui
area:i18n
area:proof
area:docs
bug
concept-request
visual-proof-request
translation
scientific-review
good-first-issue
```

---

# 80. PyPI Packaging

Package goal:

```bash
pip install visualmetrics
```

Optional:

```bash
pip install "visualmetrics[gui]"
pip install "visualmetrics[econometrics]"
pip install "visualmetrics[causal]"
pip install "visualmetrics[ai]"
pip install "visualmetrics[proofs]"
pip install "visualmetrics[notebook]"
pip install "visualmetrics[all]"
```

Build should follow PEP 517/518/621 conventions through `pyproject.toml`.

Recommended build backend: a modern backend such as Hatchling, unless dependency/tooling tests show a better choice.

---

# 81. CI/CD

## CI on Pull Request

- install core;
- lint;
- format check;
- type check;
- unit/scientific tests;
- build package;
- import smoke test;
- docs build;
- selected GUI smoke test;
- optional dependency matrix where practical.

## Release

On version tag:

- full test suite;
- build sdist/wheel;
- verify metadata;
- optionally publish to TestPyPI first;
- publish to PyPI through trusted publishing when configured;
- create GitHub release.

Do not put raw PyPI passwords into repository secrets if trusted publishing is available.

---

# 82. Documentation and Citation Policy

Each advanced concept should cite authoritative sources such as:

- textbooks;
- original papers;
- reputable course materials;
- official package documentation.

Do not copy large copyrighted passages. Summaries must be original.

`CITATION.cff` should let researchers cite VisualMetrics.

Possible later integration:

- Zenodo DOI for releases.

---

# 83. Reference Backend Sources

The following are important external technical references for implementation.

## Core statistics

- SciPy statistics: https://docs.scipy.org/doc/scipy/
- SymPy: https://docs.sympy.org/
- statsmodels: https://www.statsmodels.org/

## Econometrics

- linearmodels: https://bashtage.github.io/linearmodels/
- PyFixest: https://pyfixest.org/
- arch: https://bashtage.github.io/arch/

## Causal

- DoWhy: https://www.pywhy.org/dowhy/
- EconML: https://github.com/py-why/EconML
- PyWhy: https://www.pywhy.org/

## Visualization

- Plotly Python: https://plotly.com/python/
- Manim Community: https://docs.manim.community/
- Matplotlib: https://matplotlib.org/

## GUI / Interaction

- NiceGUI: https://nicegui.io/
- PySide6 / Qt for Python: https://doc.qt.io/qtforpython-6/
- ipywidgets: https://ipywidgets.readthedocs.io/

## AI explainability

- SHAP: https://shap.readthedocs.io/
- InterpretML: https://interpret.ml/

---

# 84. Initial Development Scope vs Long-Term Catalog

The **architecture should support the full catalog from day one**, but implementation should prioritize a coherent high-quality initial set rather than hundreds of shallow screens.

Important: this is an implementation-priority statement, **not** permission for the coding agent to stop after “Phase 1” and ask for approval. When given a complete implementation task, the agent should complete the requested scope end-to-end.

Recommended flagship initial concepts:

1. Distribution explorer.
2. Sampling distributions.
3. LLN.
4. CLT.
5. Confidence interval coverage.
6. Hypothesis testing basics.
7. Size/power.
8. MLE.
9. Simple regression scenarios.
10. OLS geometry.
11. Multicollinearity.
12. Heteroskedasticity.
13. Omitted-variable bias.
14. Endogeneity/IV.
15. Unit root / stationarity.
16. Cointegration / ECM intuition.
17. FE vs RE.
18. DiD.
19. DAG basics.
20. Bias–variance.
21. Regularization.
22. Classification threshold.
23. Gradient descent.
24. PCA.
25. Backpropagation.
26. SHAP explanation.
27. Attention.

If the implementation request explicitly asks for more, implement more without gating.

---

# 85. Extended Concept Backlog

This is a catalog checklist to prevent important curriculum areas from being forgotten.

## Probability

- set operations;
- conditional probability;
- independence;
- Bayes theorem;
- random variables;
- transformations;
- joint/marginal/conditional distributions;
- expectation;
- variance;
- covariance;
- correlation;
- conditional expectation;
- total expectation;
- total variance;
- discrete distributions;
- continuous distributions;
- multivariate normal;
- order statistics;
- Markov/Chebyshev/Jensen.

## Statistical inference

- sampling distributions;
- LLN;
- CLT;
- convergence modes;
- Slutsky;
- Continuous Mapping;
- Delta Method;
- sufficiency;
- completeness;
- ancillary statistics;
- unbiasedness;
- consistency;
- efficiency;
- MSE;
- Rao–Blackwell;
- Cramér–Rao;
- MoM;
- MLE;
- Fisher information;
- Bayesian estimation;
- confidence intervals;
- tests;
- p-values;
- Type I/II;
- size;
- power;
- Neyman–Pearson;
- UMP;
- LR/Wald/Score;
- multiple testing;
- bootstrap;
- permutation;
- nonparametric tests.

## Econometrics

- simple/multiple regression;
- OLS geometry;
- FWL;
- Gauss–Markov;
- robust SE;
- multicollinearity;
- heteroskedasticity;
- autocorrelation;
- misspecification;
- endogeneity;
- IV/2SLS;
- weak instruments;
- GMM;
- nonlinear regression;
- Logit/Probit;
- count models;
- censoring/truncation;
- selection models;
- quantile regression;
- panel FE/RE;
- dynamic panel;
- time series;
- unit roots;
- cointegration;
- ECM;
- ARDL;
- VAR/SVAR;
- Granger causality;
- IRF/FEVD;
- local projections;
- volatility models;
- structural breaks;
- spatial econometrics;
- treatment effects;
- DiD;
- RDD;
- synthetic control.

## Machine learning

- train/test split;
- loss;
- bias/variance;
- regularization;
- CV;
- metrics;
- linear/logistic;
- KNN;
- Naive Bayes;
- LDA/QDA;
- trees;
- Random Forest;
- boosting;
- SVM;
- kernels;
- PCA;
- clustering;
- mixtures/EM;
- anomaly detection;
- calibration;
- conformal prediction;
- explainability.

## Deep learning / AI

- neuron;
- activation;
- forward pass;
- loss;
- gradient descent;
- backprop;
- initialization;
- dropout;
- CNN;
- RNN/LSTM;
- embeddings;
- attention;
- transformers;
- tokenization;
- autoregression;
- generative sampling;
- RAG concepts;
- agent loop concepts;
- fairness;
- robustness;
- uncertainty;
- distribution shift.

---

# 86. Naming and Branding

Current technical name:

```text
VisualMetrics
```

Tagline candidates:

```text
See the Theory. Change the Assumptions. Understand the Model.
```

```text
Statistics, Econometrics and AI — Made Visual.
```

```text
From Abstract Theory to Interactive Understanding.
```

Brand principles:

- academic but modern;
- bright and clear;
- not childish;
- strong visual hierarchy;
- consistent semantic colors;
- equations should look professional;
- multilingual identity should be visible.

---

# 87. Definition of Done for Version 0.1.0

A first meaningful release is not “done” simply because the package imports.

Minimum definition:

- package builds successfully;
- CLI works;
- GUI launches;
- EN/AR/FR language switch works;
- RTL is functional for Arabic;
- themes work;
- concept registry works;
- search works;
- at least a coherent flagship catalog is implemented;
- each flagship lab has multiple scenarios, not one demo;
- at least several genuine proof/derivation experiences exist;
- simulation is correctly labeled;
- generated-code view works for flagship labs;
- core exports work;
- README is practical and complete;
- documentation builds;
- tests pass;
- optional dependency failures are graceful;
- `visualmetrics doctor` works;
- repository includes governance/security/citation files;
- CI is active;
- package can be prepared for PyPI.

---

# 88. Future Research-Level Extensions

Possible later additions:

- interactive asymptotic experiment builder;
- semiparametric influence-function visualizer;
- identification-region visualizer;
- weak-identification labs;
- high-dimensional econometrics;
- post-selection inference;
- debiased Lasso;
- Double Machine Learning;
- causal forests;
- heterogeneous treatment effects;
- synthetic DiD;
- matrix-completion causal methods;
- Bayesian econometrics;
- state-space models/Kalman filter;
- DSGE visual intuition;
- local projections;
- quantile time series;
- spatial spillover decomposition;
- network econometrics;
- conformal inference for time series;
- uncertainty quantification for AI;
- modern calibration;
- interpretability stability;
- foundation-model/LLM visual concepts;
- teacher-authored custom concept plugins.

---

# 89. Implementation Rules for AI Coding Agents

When an AI coding agent works on this repository, it must follow these rules:

1. Read this blueprint first.
2. Inspect current repository before editing.
3. Do not destroy working code to simplify a task.
4. Preserve modular boundaries.
5. Use existing scientific libraries rather than unnecessary reimplementation.
6. Keep heavy dependencies optional.
7. Avoid circular imports.
8. Avoid global mutable GUI/scientific state.
9. Keep lab state serializable.
10. Use deterministic seeds in tests.
11. Separate scientific computation from rendering.
12. Separate UI strings from source code.
13. Implement Arabic RTL intentionally.
14. Keep all three languages synchronized for stable concepts.
15. Test all public APIs.
16. Test scientific identities/results.
17. Never call simulation a proof.
18. Do not add fake buttons or placeholder menus that appear functional but do nothing.
19. Avoid placeholder `TODO` implementations in requested scope.
20. Do not stop after internal phases to ask the user to confirm.
21. Run tests and repair failures before final report.
22. Update README/docs when behavior changes.
23. Report exactly what was implemented and any genuine unresolved limitation.
24. Do not claim functionality that was not verified.

---

# 90. MASTER PROMPT FOR A CODING AGENT

The following prompt is designed to be copied to a capable repository coding agent.

```text
You are the principal software architect, scientific Python engineer, econometrics/statistics developer, AI/ML developer, GUI engineer, visualization engineer, and technical documentation author for the VisualMetrics project.

PROJECT
=======
Name: VisualMetrics
Python package: visualmetrics
Repository: https://github.com/merwanroudane/visualmetrics
Maintainer contact: merwanroudane920@gmail.com
Reference runtime: Python 3.11

MISSION
=======
Build VisualMetrics as a multilingual, GUI-first and Python-API-accessible interactive visual laboratory for:

- mathematical foundations relevant to statistics/econometrics/AI;
- probability;
- descriptive statistics;
- sampling theory;
- statistical inference;
- estimation;
- confidence intervals;
- hypothesis testing, size and power;
- Bayesian statistics;
- regression and OLS geometry;
- econometrics;
- endogeneity and identification;
- IV/GMM;
- time-series econometrics;
- panel data;
- financial/volatility econometrics;
- causal inference;
- nonparametric/multivariate/spatial statistics where implemented;
- machine learning;
- deep learning;
- modern AI concepts;
- explainable AI;
- visual proofs, visual derivations, simulations and counterexamples.

The product is NOT merely a plotting library. Its primary intellectual contribution is a pedagogical visualization layer that turns abstract theoretical concepts into interactive, animated, comparative, reproducible and multilingual learning laboratories.

AUTHORITATIVE BLUEPRINT
=======================
The repository should contain and follow the project master blueprint. Read the full blueprint before coding. Treat its non-negotiable principles as requirements.

CRITICAL EXECUTION BEHAVIOR
===========================
Work end-to-end. Do NOT use a gated workflow.

Do not stop after architecture, phase 1, GUI skeleton, first module, or any internal milestone to ask:
- “Should I continue?”
- “Please confirm phase 1.”
- “Do you want me to implement phase 2?”

Instead, perform the complete requested scope in one continuous execution cycle whenever technically possible:

1. inspect repository;
2. plan internally;
3. implement architecture;
4. implement functionality;
5. implement GUI;
6. implement translations;
7. implement tests;
8. run tests;
9. fix failures;
10. build documentation;
11. verify package/build;
12. provide one final consolidated report.

Only ask the user for information if there is a truly blocking external fact that cannot be resolved safely with a reasonable default. Prefer sensible defaults and keep moving.

SCIENTIFIC INTEGRITY
====================
This is a scientific/educational project. Scientific correctness is more important than impressive animation.

Never label simulation as proof.

Every explanatory visual should be classifiable as one of:
- formal_proof
- symbolic_derivation
- geometric_proof
- visual_derivation
- visual_intuition
- simulation
- numerical_demonstration
- counterexample
- empirical_example

Show this classification in the lab where useful.

For LLN and CLT, simulations should be called simulations/visual intuition unless a genuine proof is separately implemented.

Do not imply causation from predictive explanations such as SHAP. Explicitly distinguish predictive attribution from causal effects.

ANIMATION-FIRST EXPLANATION REQUIREMENT
=======================================
Animation is a first-class scientific teaching surface in VisualMetrics. Do not treat animation as decorative movement or as a silent chart that happens to move.

For every concept where animation is scientifically meaningful, implement one or more animations as appropriate. A single concept may and often should have multiple animations for different intuitions, assumptions, parameter effects, derivation steps, violations, counterexamples and comparisons.

Every animation must include an explanatory layer. During playback and step mode, the user should be able to understand:
- what is currently being shown;
- what changed in this frame/step;
- which parameter, assumption or model component caused the change;
- why the change occurs scientifically;
- how the current output should be interpreted;
- what conclusion is justified;
- which assumptions are active or violated;
- whether the animation is a formal proof, derivation, simulation, numerical demonstration or visual intuition.

Synchronize the explanation with the visual state. When a point moves, curve changes, rejection region expands, residual changes, coefficient updates, gradient step occurs, treatment effect appears, decision boundary moves or neural activation propagates, update the relevant text, values, formulas, labels and interpretation at the same time.

Provide controls such as Play, Pause, Restart, Previous Step, Next Step, Step-by-Step, speed control, repeat and timeline/scrubber where appropriate. Add a concise end-of-animation summary. Beginner mode should use clear plain-language explanations; Advanced/PhD modes should optionally show equations, formal assumptions and derivation detail. All explanatory animation content must support English, Arabic and French, with correct RTL behavior for Arabic.

Do not mark an animation complete merely because motion works. It is complete only when the user can understand the scientific meaning of the motion and the output.

CORE PRODUCT PRINCIPLES
=======================
1. GUI-first for beginners.
2. Reproducible Python API for advanced users.
3. Arabic + English + French from architecture level.
4. True RTL handling for Arabic.
5. Broad scenario coverage for every important concept.
6. Meaningful semantic colors and accessible alternatives.
7. Visualizations + animations + interactive controls + derivations + proofs where valid.
8. Animations are explanation-first: every important animation must explain what is visible, what changed, why it changed, how to interpret the output, which assumptions are active/violated, and what conclusion is justified. Motion without synchronized interpretation is incomplete.
9. No-code GUI must be able to generate equivalent Python code/configuration.
10. Heavy optional dependencies must be lazy and optional.
11. Reuse mature scientific libraries instead of rewriting standard estimators unnecessarily.
12. Local-first; no telemetry or data upload by default.
13. Documentation must be practical, not superficial.

SCENARIO COMPLETENESS
=====================
Do not build each lab as one canned textbook example.

For every relevant concept, consider and expose appropriate combinations of:
- canonical case;
- positive direction;
- negative direction;
- zero/null case;
- weak vs strong effect;
- low vs high noise;
- small vs large sample;
- boundary cases;
- alternative distributions;
- alternative functional forms;
- assumption violations;
- misspecification;
- counterexamples;
- sensitivity to parameters;
- method comparisons;
- robust vs non-robust alternatives;
- empirical vs synthetic data.

REGRESSION EXAMPLE REQUIREMENT
==============================
Simple regression must not be a single positive line.

Expose controls/presets for at least:
- positive slope;
- negative slope;
- zero/near-zero slope;
- intercept;
- sample size;
- noise;
- x distribution/range;
- linear relationship;
- quadratic/cubic relationship;
- log-level;
- level-log;
- log-log;
- exponential where suitable;
- heteroskedasticity;
- non-normal errors;
- outliers;
- leverage;
- influential observations;
- omitted-variable bias;
- measurement error;
- endogeneity;
- confidence/prediction bands;
- standardized/centered variables.

Use the same philosophy across statistical inference, econometrics and AI.

CURRICULUM-GROUNDED COVERAGE WITH UNIVERSAL SCIENTIFIC NAMING
=============================================================
Treat Algerian, Arab-world, and international university curricula as **coverage references**, not as the library naming system.

Mandatory rule:
- Do NOT create user-facing modules named `Statistics 3`, `Statistics 4`, `Econometrics 1`, `Econometrics 2`, or any country/university-specific course label.
- Do NOT use local course numbering in package paths, Python APIs, concept IDs, GUI menus, search categories, or primary documentation headings.
- Use canonical scientific names only, including `Probability & Random Variables`, `Inferential Statistics`, `Sampling Theory`, `Estimation Theory`, `Hypothesis Testing`, `Econometrics`, `Time Series Econometrics`, `Panel Data Econometrics`, `Causal Inference`, `Machine Learning`, `Deep Learning`, and related standard fields.

Use the Algerian and Arab curriculum sources in this blueprint to ensure that the canonical modules include all recurring content: probability distributions and approximations, bivariate random variables, sampling distributions, MoM/MLE, Fisher information, Cramer-Rao, confidence intervals, Neyman-Pearson, size/power, parametric and nonparametric tests, OLS vs MLE, restrictions, dummies, partial regression/FWL, diagnostics/remedies, simultaneous equations, SUR, IV/2SLS/3SLS, dynamic models, panel data, unit roots, cointegration, VAR/SVAR, ARCH/GARCH, and the other topics described in the blueprint.

Curriculum mappings may exist only as internal/source metadata for maintainers and traceability. They must map local source labels to stable canonical concept IDs without duplicating concept implementations.

LANGUAGES / I18N
================
Required languages:
- en
- ar
- fr

No stable GUI scientific string should be hard-coded in page/component code.

Implement translation resources and a translation service.

Arabic requirements:
- RTL application flow;
- mixed Arabic/English text;
- equations remain mathematically readable;
- variables/numbers are not incorrectly mirrored;
- sidebars/layout adapt sensibly.

Implement terminology modes:
- translated only;
- bilingual terminology;
- English technical terminology.

Example Arabic bilingual label:
القوة الإحصائية (Statistical Power)

Example French bilingual label:
Puissance statistique (Statistical Power)

Search must index all translations, aliases and acronyms.

GUI
===
Build a polished GUI as a first-class product.

Preferred first implementation: NiceGUI + Plotly, with local browser and native-window-ready architecture. Keep GUI-specific dependencies isolated so a future PySide6 frontend can be added without modifying the scientific core.

The application should launch through:
- visualmetrics
- visualmetrics gui
- python -m visualmetrics

Main GUI areas:
- top navigation;
- multilingual language control;
- user level control;
- theme control;
- concept search;
- domain explorer;
- prerequisite/related concept graph;
- visual lab canvas;
- parameter controls;
- scenario presets;
- learning tabs;
- proof/derivation tabs;
- code tab;
- export/reset;
- diagnostics/status.

Lab tabs should support as relevant:
- Overview
- Intuition
- Visualize
- Animate
- Experiment
- Compare
- Assumptions
- Math
- Proof/Derivation
- Simulation
- Diagnostics
- Misconceptions
- Code
- Data
- Quiz
- References

Add presentation/classroom mode and reduced-motion mode.

COLORS / THEMES
===============
Use semantic color roles, not scattered hard-coded colors.

Provide themes:
- Light
- Dark
- High Contrast
- Classroom
- Publication
- Color-Blind Friendly
- Custom

Possible semantic defaults:
- primary: #0F766E
- secondary: #7C3AED
- positive: #16A34A
- negative: #DC2626
- warning: #F59E0B
- info: #0284C7
- baseline/null: #64748B
- Type I error: #E11D48
- Type II error: #D97706
- power: #16A34A

Never use color as the only distinction. Add labels/line styles/shapes/icons.

ARCHITECTURE
============
Use a src layout and keep these concerns separated:

src/visualmetrics/
- core
- catalog
- concepts
- backends
- data
- simulation
- math
- proofs
- visuals
- gui
- notebook
- education
- i18n
- export
- plugins

Use Pydantic/dataclasses/protocols where appropriate for stable internal models such as:
- ConceptSpec
- ScenarioSpec
- ControlSpec
- ProofSpec
- FigureState
- LabState
- normalized backend result types.

Do not pass raw statsmodels/linearmodels/etc result objects throughout GUI code. Normalize external results through adapters.

SCIENTIFIC DEPENDENCIES
=======================
Reuse mature packages.

Core candidates:
- numpy
- scipy
- pandas
- sympy
- pydantic
- platformdirs
- babel where useful

GUI/visualization:
- nicegui
- plotly
- matplotlib fallback
- pywebview optional

Econometrics optional extra:
- statsmodels
- linearmodels
- pyfixest
- arch

Causal optional extra:
- dowhy
- econml
- pywhy-graphs
- networkx

AI optional extra:
- scikit-learn
- shap
- interpret

Proofs optional extra:
- manim

Notebook optional extra:
- ipywidgets

Data optional extra:
- openpyxl
- pyarrow
- pyreadstat

Do not make all of these mandatory on base import.

OPTIONAL EXTRAS
===============
Design pyproject extras conceptually as:
- visualmetrics[gui]
- visualmetrics[econometrics]
- visualmetrics[causal]
- visualmetrics[ai]
- visualmetrics[proofs]
- visualmetrics[notebook]
- visualmetrics[data]
- visualmetrics[dev]
- visualmetrics[all]

Missing optional dependencies should produce a friendly localized capability message, not an import crash.

CONCEPT REGISTRY
================
Concepts must be registry-driven.

Each concept declares:
- stable ID;
- domain/subdomain;
- title/summary translation keys;
- levels;
- prerequisites;
- related concepts;
- learning objectives;
- supported learning modes;
- controls;
- scenarios;
- renderers;
- scientific backends;
- evidence type;
- references;
- quiz/misconception metadata.

Do not hard-code the entire navigation tree manually.

PROOF ENGINE
============
Implement a proof/derivation abstraction that can render a sequence of steps in the GUI without requiring Manim.

Manim is an optional renderer for high-quality exported mathematical animations.

Provide controls:
- previous;
- next;
- play;
- pause;
- speed;
- show algebra;
- show geometry;
- explain step;
- reset.

Good early proof/derivation candidates:
- OLS projection;
- residual orthogonality;
- normal equations;
- FWL geometry;
- omitted-variable-bias decomposition;
- 2SLS projection;
- Neyman-Pearson rejection region;
- Bayes updating;
- PCA projection;
- Ridge/Lasso geometry;
- gradient descent;
- chain rule/backpropagation.

STATISTICAL INFERENCE
=====================
The inference curriculum must include at least:
- populations/samples;
- sampling distributions;
- convergence;
- LLN;
- CLT;
- Slutsky;
- continuous mapping;
- Delta Method;
- unbiasedness;
- consistency;
- efficiency;
- sufficiency;
- completeness;
- ancillary statistics;
- Rao-Blackwell;
- Cramer-Rao;
- Method of Moments;
- MLE;
- Fisher information;
- confidence intervals;
- H0/H1;
- Type I/II;
- alpha/beta;
- size;
- power;
- p-values;
- one/two-sided tests;
- Neyman-Pearson;
- UMP;
- LR/Wald/Score tests;
- multiple testing;
- bootstrap/permutation;
- Bayesian inference foundations.

POWER LAB
=========
Build a rich power lab. Use statsmodels/scipy calculations where appropriate rather than reimplementing them unnecessarily.

Controls should include, where relevant:
- alpha;
- effect size;
- n;
- variance/SD;
- alternative direction;
- one/two-sided;
- df;
- group allocation;
- test family.

Visuals:
- H0/H1 distributions;
- alpha;
- beta;
- power;
- rejection region;
- critical values;
- p-value region;
- power curve;
- sample-size relationship.

ECONOMETRICS
============
Cover architecture/content for:
- OLS;
- Gauss-Markov;
- robust inference;
- multicollinearity;
- heteroskedasticity;
- serial correlation;
- specification;
- endogeneity;
- IV/2SLS/LIML/GMM;
- Logit/Probit/count/censoring concepts;
- time series;
- unit roots;
- cointegration/ECM/ARDL;
- VAR/SVAR;
- Granger causality;
- IRF/FEVD;
- volatility;
- panel FE/RE;
- dynamic panel;
- DiD;
- event studies;
- RDD;
- synthetic control;
- causal inference links;
- spatial concepts as the catalog expands.

CAUSAL INFERENCE
================
Build visual support for:
- potential outcomes;
- treatment/control;
- ATE/ATT;
- confounders;
- mediators;
- colliders;
- DAGs;
- backdoor paths;
- conditioning;
- randomized experiments;
- matching;
- propensity scores;
- IPW;
- IV;
- RDD;
- DiD;
- event studies;
- synthetic control;
- heterogeneous effects;
- DML/causal forests where dependency support exists.

Use DoWhy/EconML/PyWhy adapters where valuable.

ML / AI
=======
Visual curriculum should cover:
- prediction vs inference;
- train/validation/test;
- loss;
- bias-variance;
- overfitting/underfitting;
- regularization;
- cross-validation;
- generalization;
- classification threshold;
- confusion matrix;
- precision/recall/ROC/PR;
- calibration;
- linear/logistic;
- KNN;
- trees/forests/boosting;
- SVM/kernels;
- PCA;
- clustering;
- neural networks;
- forward pass;
- backpropagation;
- gradient descent;
- CNN/RNN/LSTM concepts;
- embeddings;
- attention;
- transformers;
- explainability;
- SHAP;
- uncertainty/calibration/conformal concepts;
- responsible AI and distribution shift where visualizable.

DATA
====
Support three modes:
1. generated DGP data;
2. built-in/licensed educational data;
3. user data.

Generated data must be seed-reproducible and expose the DGP.

User data support should progressively include CSV, Excel, Parquet, Stata and SPSS through optional dependencies.

Never pretend an arbitrary user dataset verifies a theorem. Keep theoretical controlled labs distinct from empirical examples.

REPRODUCIBILITY
===============
Every lab must have serializable state.

Allow export/import of a JSON/YAML state with:
- concept ID;
- VisualMetrics version;
- language/theme;
- parameters;
- scenario;
- seed;
- relevant backend versions.

GUI state should generate equivalent Python code.

PERFORMANCE
===========
Use lazy imports and lazy concept loading.

A basic probability lab must not import Manim, PyTorch, SHAP, EconML or the full econometrics stack.

Debounce expensive controls.
Cache deterministic expensive artifacts appropriately.
Keep GUI responsive.

SECURITY / PRIVACY
==================
- bind local GUI to localhost by default;
- no telemetry by default;
- no cloud upload by default;
- sanitize data files;
- never execute uploaded content;
- never use eval() on arbitrary user input;
- use safe constrained mathematical parsing;
- use dependency security monitoring.

TESTING
=======
Testing must include scientific correctness.

Implement:
- unit tests;
- integration tests;
- scientific regression tests;
- property-based tests where useful;
- translation integrity tests;
- Arabic RTL smoke tests;
- GUI smoke tests;
- package build/import tests;
- selective visual regression tests.

Examples of scientific properties:
- projection residual orthogonality;
- CDF range;
- normalized weights;
- deterministic results with fixed seeds;
- known analytical power values within tolerance;
- known regression benchmarks.

QUALITY
=======
Use modern tooling such as:
- ruff
- pytest
- pytest-cov
- hypothesis
- mypy or equivalent
- pre-commit
- build

Do not finish with failing tests if failures can be repaired.

DOCUMENTATION
=============
README must be a practical guide, not a short marketing page.

It must include:
- what VisualMetrics is;
- features;
- domains;
- installation;
- GUI launch;
- GUI walkthrough;
- language switching;
- learning modes;
- scenarios;
- Python API;
- no-code-to-code workflow;
- data loading;
- exports;
- optional extras;
- troubleshooting/doctor;
- notebook mode;
- proof-vs-simulation policy;
- adding a concept;
- adding a translation;
- contribution;
- citation;
- license/contact.

Build a documentation site structure for EN/AR/FR.

GITHUB / PACKAGING
==================
Create/maintain:
- README.md
- LICENSE
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- SECURITY.md
- CHANGELOG.md
- CITATION.cff
- pyproject.toml
- uv.lock
- .gitignore
- pre-commit config
- GitHub Actions CI/docs/release workflows
- Dependabot
- issue templates
- pull request template

Prepare project for:
- pip install visualmetrics
- PyPI release
- Semantic Versioning
- GitHub releases.

Do not invent credentials. Do not attempt authenticated publication unless credentials/trusted publishing are actually configured.

VISUAL ACCESSIBILITY
====================
Implement:
- keyboard-friendly navigation where feasible;
- clear focus states;
- high contrast;
- reduced motion;
- pause/step animation;
- text scaling;
- color-blind-friendly theme;
- labels/shapes in addition to color;
- descriptive figure context.

INITIAL FLAGSHIP IMPLEMENTATION
===============================
Unless the current repository already contains a stronger implementation, prioritize a coherent set of working flagship labs including as many as feasible from:

- distribution explorer;
- LLN;
- CLT;
- confidence interval coverage;
- hypothesis testing;
- size/power;
- MLE;
- simple regression scenario explorer;
- OLS geometry;
- multicollinearity;
- heteroskedasticity;
- omitted-variable bias;
- endogeneity/IV;
- stationarity/unit root;
- FE vs RE;
- DiD;
- DAG fundamentals;
- bias-variance;
- classification threshold;
- gradient descent;
- PCA;
- backpropagation;
- SHAP explanation;
- attention.

The purpose of this list is not to encourage shallow placeholders. Prefer coherent fully working labs with rich scenario options. But if the user explicitly requested a larger scope in the current task, implement that larger scope without stopping for confirmation.

NO FAKE COMPLETENESS
====================
Do not create menu items or buttons that look complete but do nothing.

If an advanced concept is cataloged but not yet implemented, mark it transparently as planned/experimental and keep it out of the stable feature claims.

Do not fill the repository with empty modules simply to match the architecture tree.

IMPLEMENTATION ORDER — INTERNAL ONLY
====================================
You may organize your own work internally in stages, but do not use those stages as user confirmation gates.

A sensible internal order is:
1. inspect current repo;
2. establish package/build/testing skeleton;
3. implement core schemas/registry/state/i18n/theme;
4. implement scientific backend adapters;
5. implement reusable visual primitives;
6. implement flagship concept labs;
7. implement GUI/navigation/search;
8. implement generated code/state export;
9. implement docs/README;
10. implement tests/CI;
11. run full verification and fix issues.

FINAL VERIFICATION
==================
Before reporting completion:
- run lint/format checks;
- run tests;
- build package;
- import package from built/installed environment if practical;
- run GUI smoke test;
- verify EN/AR/FR switching;
- verify Arabic RTL;
- verify at least one export;
- verify missing optional dependency handling;
- verify README commands correspond to actual code.

FINAL REPORT
============
Return one consolidated report containing:
- architecture implemented;
- features implemented;
- flagship labs implemented;
- languages/status;
- dependencies/extras;
- tests executed and results;
- docs created;
- commands to run the GUI;
- commands to run tests/build;
- genuine limitations only;
- next technically logical enhancements.

Do not end by asking permission to continue work that was already part of the task.
```

---

# 91. Suggested First Repository Commit Structure

A clean initial commit should include real scaffolding rather than empty placeholders:

```text
feat: initialize VisualMetrics architecture and multilingual GUI core
```

Potential first commits afterward:

```text
feat: add concept registry and scenario engine
feat: add statistical inference visual labs
feat: add regression and OLS geometry labs
feat: add econometrics backend adapters
feat: add causal graph labs
feat: add ML visual labs
feat: add trilingual docs and glossary
feat: add proof engine and optional Manim renderer
ci: add scientific tests and release workflow
```

These are Git commit organization suggestions; they are not user-approval gates.

---

# 92. Maintainer Decisions That Can Remain Configurable

Do not block implementation on these unless technically necessary:

- final logo;
- final marketing tagline;
- exact default theme;
- whether docs content has a separate CC license;
- whether PySide6 becomes an official second GUI;
- exact number of flagship labs in the first public release;
- publication date;
- final PyPI classifiers.

Use sensible defaults and keep architecture flexible.

---

# 93. Final Product Standard

VisualMetrics should ultimately make it possible for a learner to ask:

> “What does this theorem, estimator, assumption or AI mechanism actually mean?”

and then **see it**, **change it**, **break it**, **compare it**, **derive it**, and when mathematically appropriate **prove it visually**.

A researcher should be able to move from the same GUI state to reproducible Python code.

An instructor should be able to use the same concept as a classroom demonstration.

An Arabic-, French- or English-speaking learner should access the same scientific content through an interface appropriate to their language.

That is the standard by which the project should be evaluated.

---

# 94. Research and Technical References Used to Shape This Blueprint

The following sources are useful starting points for implementation and curriculum validation. Their inclusion does not mean VisualMetrics copies their content.

## University / curriculum references

- MIT Statistical Method in Economics: https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/pages/syllabus/
- MIT Econometrics: https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/pages/syllabus/
- MIT Economics catalog: https://catalog.mit.edu/schools/humanities-arts-social-sciences/economics/
- Stanford Modern Statistical Learning: https://web.stanford.edu/class/stats315a/syllabus.html
- Stanford STATS 202: https://web.stanford.edu/class/stats202/intro.html
- Harvard causal inference course information: https://imai.sites.fas.harvard.edu/teaching/cause.html
- Harvard API-114: https://my.harvard.edu/course/API114/2026-Fall/001
- King Saud University MSc Economics: https://cba.ksu.edu.sa/ar/ECON/Master/AR
- King Saud University ECON 416 Econometrics: https://cba.ksu.edu.sa/en/node/9401
- King Saud University ECON 541 Econometrics: https://cba.ksu.edu.sa/en/ECON541
- King Saud University ECON 542 Applied Econometrics: https://cba.ksu.edu.sa/en/ECON542
- King Saud University ECON 423 Applied Econometrics: https://cba.ksu.edu.sa/ar/node/9407
- King Saud University PhD Econometrics I/II: https://cba.ksu.edu.sa/en/node/32167
- Study in Algeria Quantitative Economics: https://studyinalgeria.dz/pages/14888
- Study in Algeria Quantitative Economics full curriculum: https://studyinalgeria.dz/storage/Uploads/pages/30971/_full_curriculum_1772526724.pdf
- Study in Algeria advanced/master Econometrics 2 example: https://studyinalgeria.dz/storage/Uploads/pages/28788/_full_curriculum_1749635171.pdf
- University of Mila Statistics 3 (2025/2026 platform): https://elearning.univ-mila.dz/a2026/course/view.php?id=3428&lang=en
- University of Mila Statistics 3 detailed program: https://elearning.univ-mila.dz/a2024/mod/page/view.php?id=44844
- University of Mila Statistics 4: https://elearning.univ-mila.dz/a2024/course/view.php?id=3161
- University of Tlemcen Statistics 3: https://elearn.univ-tlemcen.dz/course/view.php?id=7651
- University of Jijel Statistics 3: https://elearning.univ-jijel.dz/course/view.php?id=3238&lang=fr
- University of Jijel Statistics 4: https://elearning.univ-jijel.dz/course/view.php?id=4489
- University of Souk Ahras Statistics 4: https://univ-soukahras.dz/moodle/course/view.php?id=5050
- University of Algiers 3, Statistics 3 Inferential Statistics (2018): https://dspace.univ-alger3.dz/jspui/handle/123456789/6326
- University of Algiers 3, Inferential Statistics / Statistics 3 (2020): https://dspace.univ-alger3.dz/jspui/handle/123456789/6330
- University of Algiers 3, Statistics 3 handout (2022): https://dspace.univ-alger3.dz/jspui/handle/123456789/9056
- University of Algiers 3, Statistics 3 handout (2022, Bouzenoura): https://dspace.univ-alger3.dz/jspui/handle/123456789/7937
- University of Algiers 3, Statistics 3 lectures (2019, Belaidi): https://dspace.univ-alger3.dz/jspui/handle/123456789/6329
- American University of Beirut Graduate FAS Catalogue 2025-26 (ECON 305/306): https://www.aub.edu.lb/Registrar/catalogue2025-26/gr/Documents/gr-fas.pdf
- Cairo University Econometric Analysis using EViews: https://scholar.cu.edu.eg/?q=mohamed_abonazel%2Fclasses%2Feconometric-analysis-using-eviews
- USTHB Big Data Analytics: https://finfo.usthb.dz/pages/Master-Big-Data-Analytics
- ENSSEA course listing: https://learning.enssea.edu.dz/course/index.php?categoryid=42&lang=ar
- ENSSEA programs: https://www.test.enssea.edu.dz/index.php/fr/formation/les-offres-de-formations
- American University of Beirut catalog: https://www.aub.edu.lb/Registrar/catalogue2026-27/ug/Pages/fas-dom.aspx
- Qatar University AI program: https://esc.qu.edu.qa/en-us/Colleges/engineering/academics/computer/ai/Pages/default.aspx
- Qatar University Data Science concentration: https://business.qu.edu.qa/en-us/Colleges/engineering/academics/computer/cs/Pages/datascience.aspx
- UAEU Master in Business Analytics: https://www.uaeu.ac.ae/en/catalog/graduate/programs/master-in-business-analytics.shtml
- KAUST ML course collection: https://ml.kaust.edu.sa/courses.html

## Scientific/backend references

- SciPy: https://docs.scipy.org/doc/scipy/
- SymPy: https://docs.sympy.org/
- statsmodels: https://www.statsmodels.org/
- statsmodels power APIs: https://www.statsmodels.org/dev/stats.html#power-and-sample-size-calculations
- linearmodels: https://bashtage.github.io/linearmodels/
- PyFixest: https://pyfixest.org/
- PyFixest Difference-in-Differences: https://pyfixest.org/difference-in-differences.html
- arch: https://bashtage.github.io/arch/
- DoWhy: https://www.pywhy.org/dowhy/
- EconML: https://github.com/py-why/EconML
- SHAP: https://shap.readthedocs.io/
- InterpretML: https://interpret.ml/
- Plotly animations: https://plotly.com/python/animations/
- Manim Community: https://docs.manim.community/
- NiceGUI: https://nicegui.io/
- Qt internationalization: https://doc.qt.io/qt-6/internationalization.html
- ipywidgets: https://ipywidgets.readthedocs.io/

---

# 95. Blueprint Maintenance

This document should evolve with the project.

When the maintainer later supplies:

- statistical-inference handouts;
- econometrics syllabi;
- AI course notes;
- new university curricula;
- references/books;
- desired visual proof ideas;

map them into the existing catalog through a **coverage matrix**:

```text
Source topic
-> VisualMetrics concept ID
-> coverage status
-> learning level
-> visual mode
-> proof/derivation status
-> missing scenarios
-> source institution
-> course label
-> academic year/version
-> source URL
-> curriculum alias/profile ID
```

Do not rewrite the architecture each time a new syllabus arrives.

---

## End of Master Blueprint

**Repository:** https://github.com/merwanroudane/visualmetrics  
**Contact:** merwanroudane920@gmail.com
