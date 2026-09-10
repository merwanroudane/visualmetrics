"""The built-in concept catalog.

This module holds *lightweight* metadata only. Opening a lab imports its
implementation module, which registers the complete specification (controls,
scenarios, references). Concepts marked ``planned`` are listed for transparency
and honestly refuse to open rather than showing an empty screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.concepts import ConceptSpec, Domain, LearningMode, Level, Status
from ..core.evidence import EvidenceType as E

if TYPE_CHECKING:  # pragma: no cover
    from ..core.registry import ConceptRegistry

__all__ = ["load_builtin_catalog", "CATALOG"]

_LV = {
    "b": Level.BEGINNER,
    "i": Level.INTERMEDIATE,
    "a": Level.ADVANCED,
    "p": Level.PHD,
}
_MODE = {
    "l": LearningMode.LEARN,
    "v": LearningMode.VISUALIZE,
    "n": LearningMode.ANIMATE,
    "x": LearningMode.EXPERIMENT,
    "c": LearningMode.COMPARE,
    "P": LearningMode.PROVE,
    "d": LearningMode.DERIVE,
    "s": LearningMode.SIMULATE,
    "g": LearningMode.DIAGNOSE,
    "k": LearningMode.COUNTEREXAMPLE,
    "q": LearningMode.QUIZ,
    "o": LearningMode.CODE,
    "t": LearningMode.DATA,
    "r": LearningMode.REFERENCES,
}


def _entry(
    id: str,
    domain: Domain,
    subdomain: str,
    module: str | None,
    levels: str = "i",
    modes: str = "lvxsoq",
    evidence: E = E.VISUAL_INTUITION,
    status: Status = Status.STABLE,
    *,
    prerequisites: tuple[str, ...] = (),
    related: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    curriculum: tuple[str, ...] = (),
) -> ConceptSpec:
    return ConceptSpec(
        id=id,
        domain=domain,
        subdomain=subdomain,
        title_key=f"concepts.{id}.title",
        summary_key=f"concepts.{id}.summary",
        levels=tuple(_LV[c] for c in levels),
        modes=tuple(_MODE[c] for c in modes),
        evidence=evidence,
        status=status,
        module=module,
        prerequisites=prerequisites,
        related=related,
        tags=tags,
        aliases=aliases,
        required_extras=extras,
        curriculum_tags=curriculum,
    )


D = Domain
P = "visualmetrics.concepts"

#: The catalog. ``module=None`` + ``status=PLANNED`` means "listed, not built".
CATALOG: tuple[ConceptSpec, ...] = (
    # ---------------- Probability and random variables --------------------
    _entry(
        "probability.distributions", D.PROBABILITY, "distributions",
        f"{P}.probability.distributions", "biap", "lvnxcsoqrt", E.NUMERICAL_DEMONSTRATION,
        related=("probability.approximations", "inference.clt"),
        tags=("distribution", "pdf", "cdf", "quantile", "moments"),
        aliases=("distribution explorer", "pdf", "cdf", "loi de probabilite", "التوزيعات"),
        curriculum=("dz.stat3", "ksu.econ416"),
    ),
    _entry(
        "probability.bivariate", D.PROBABILITY, "bivariate_random_variables",
        f"{P}.probability.bivariate", "iap", "lvxcsoqrt", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("probability.distributions",),
        tags=("joint", "marginal", "conditional", "covariance", "independence"),
        aliases=("joint distribution", "bivariate", "المتغيرات الثنائية", "loi conjointe"),
        curriculum=("dz.stat3",),
    ),
    _entry(
        "probability.approximations", D.PROBABILITY, "convergence_of_distributions",
        f"{P}.probability.approximations", "biap", "lvnxcsoqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("probability.distributions",),
        related=("inference.clt",),
        tags=("approximation", "continuity correction", "convergence"),
        aliases=("binomial normal approximation", "التقريب", "approximation"),
        curriculum=("dz.stat3",),
    ),
    _entry(
        "probability.inequalities", D.PROBABILITY, "inequalities_and_bounds", None,
        "ap", "lvx", E.VISUAL_DERIVATION, Status.PLANNED,
        tags=("markov", "chebyshev", "jensen"),
    ),
    _entry(
        "probability.order_statistics", D.PROBABILITY, "order_statistics", None,
        "ap", "lvxs", E.SIMULATION, Status.PLANNED, tags=("order statistics",),
    ),
    # ---------------- Descriptive statistics -------------------------------
    _entry(
        "descriptive.explorer", D.DESCRIPTIVE, "exploratory_data_analysis",
        f"{P}.descriptive.explorer", "biap", "lvxcgoqrt", E.EMPIRICAL_EXAMPLE,
        tags=("histogram", "boxplot", "ecdf", "outliers", "skewness", "kurtosis"),
        aliases=("eda", "descriptive statistics", "الإحصاء الوصفي", "statistique descriptive"),
    ),
    # ---------------- Inferential statistics -------------------------------
    _entry(
        "inference.lln", D.INFERENCE, "limit_theory",
        f"{P}.inference.lln", "biap", "lvnxcsokqr", E.SIMULATION,
        prerequisites=("probability.distributions",),
        related=("inference.clt",),
        tags=("law of large numbers", "convergence", "consistency"),
        aliases=("lln", "loi des grands nombres", "قانون الأعداد الكبيرة"),
        curriculum=("dz.stat4", "mit.14381"),
    ),
    _entry(
        "inference.clt", D.INFERENCE, "limit_theory",
        f"{P}.inference.clt", "biap", "lvnxcsokqr", E.SIMULATION,
        prerequisites=("inference.lln", "probability.distributions"),
        related=("inference.sampling_distributions",),
        tags=("central limit theorem", "asymptotic normality"),
        aliases=("clt", "tcl", "theoreme central limite", "نظرية النهاية المركزية"),
        curriculum=("dz.stat4", "mit.14381"),
    ),
    _entry(
        "inference.sampling_distributions", D.INFERENCE, "sampling_theory",
        f"{P}.inference.sampling_distributions", "biap", "lvnxcsoqrt", E.SIMULATION,
        prerequisites=("probability.distributions",),
        related=("inference.clt", "inference.confidence_intervals"),
        tags=("sampling distribution", "standard error", "finite population correction"),
        aliases=("sampling distribution builder", "معاينة", "distribution d'echantillonnage"),
        curriculum=("dz.stat4",),
    ),
    _entry(
        "inference.confidence_intervals", D.INFERENCE, "interval_estimation",
        f"{P}.inference.confidence_intervals", "biap", "lvnxcsokqr", E.SIMULATION,
        prerequisites=("inference.sampling_distributions",),
        related=("inference.hypothesis_testing",),
        tags=("confidence interval", "coverage", "margin of error"),
        aliases=("ci", "intervalle de confiance", "مجال الثقة"),
        curriculum=("dz.stat4", "ksu.econ416"),
    ),
    _entry(
        "inference.hypothesis_testing", D.INFERENCE, "hypothesis_testing",
        f"{P}.inference.hypothesis_testing", "biap", "lvnxcsgokqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("inference.sampling_distributions",),
        related=("inference.power", "inference.confidence_intervals"),
        tags=("p-value", "type i error", "type ii error", "critical region"),
        aliases=("test", "hypothesis test", "اختبار الفرضيات", "test d'hypothese"),
        curriculum=("dz.stat4",),
    ),
    _entry(
        "inference.power", D.INFERENCE, "hypothesis_testing",
        f"{P}.inference.power", "biap", "lvnxcsokqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("inference.hypothesis_testing",),
        tags=("power", "effect size", "sample size", "beta"),
        aliases=("statistical power", "puissance statistique", "القوة الإحصائية"),
        curriculum=("dz.stat4", "mit.14381"),
    ),
    _entry(
        "inference.neyman_pearson", D.INFERENCE, "test_theory",
        f"{P}.inference.neyman_pearson", "ap", "lvnxcdPoqr", E.SYMBOLIC_DERIVATION,
        prerequisites=("inference.hypothesis_testing", "inference.power"),
        tags=("neyman-pearson", "likelihood ratio", "most powerful test"),
        aliases=("np lemma", "lemme de neyman-pearson", "نيمان-بيرسون"),
        curriculum=("mit.14381",),
    ),
    _entry(
        "inference.mle", D.INFERENCE, "estimation_theory",
        f"{P}.inference.mle", "iap", "lvnxcdsoqr", E.VISUAL_DERIVATION,
        prerequisites=("probability.distributions",),
        related=("inference.cramer_rao",),
        tags=("maximum likelihood", "score", "fisher information", "method of moments"),
        aliases=("mle", "maximum de vraisemblance", "الإمكان الأعظم"),
        curriculum=("dz.stat4", "mit.14381"),
    ),
    _entry(
        "inference.cramer_rao", D.INFERENCE, "estimation_theory",
        f"{P}.inference.cramer_rao", "ap", "lvxcdsoqr", E.SYMBOLIC_DERIVATION,
        prerequisites=("inference.mle",),
        tags=("cramer-rao", "efficiency", "information bound"),
        aliases=("crlb", "borne de cramer-rao", "حد كرامر-راو"),
        curriculum=("mit.14381",),
    ),
    _entry(
        "inference.bayesian_updating", D.INFERENCE, "bayesian_statistics",
        f"{P}.inference.bayesian_updating", "biap", "lvnxcdsoqr", E.VISUAL_DERIVATION,
        prerequisites=("probability.distributions",),
        tags=("prior", "posterior", "conjugate", "credible interval"),
        aliases=("bayes", "bayesien", "بايز"),
    ),
    _entry(
        "inference.multiple_testing", D.INFERENCE, "multiple_testing",
        f"{P}.inference.multiple_testing", "iap", "lvxcsgoqr", E.SIMULATION,
        prerequisites=("inference.hypothesis_testing",),
        tags=("bonferroni", "holm", "fdr", "benjamini-hochberg", "p-hacking"),
        aliases=("multiple comparisons", "tests multiples", "الاختبارات المتعددة"),
    ),
    _entry(
        "inference.bootstrap", D.INFERENCE, "resampling",
        f"{P}.inference.bootstrap", "iap", "lvnxcsoqr", E.SIMULATION,
        prerequisites=("inference.sampling_distributions",),
        tags=("bootstrap", "resampling", "percentile interval"),
        aliases=("bootstrap", "reechantillonnage", "إعادة المعاينة"),
    ),
    _entry(
        "inference.convergence_modes", D.INFERENCE, "limit_theory", None,
        "ap", "lvncs", E.SIMULATION, Status.PLANNED,
        tags=("almost sure", "in probability", "in distribution", "slutsky"),
    ),
    # ---------------- Regression -------------------------------------------
    _entry(
        "regression.simple_linear", D.REGRESSION, "simple_regression",
        f"{P}.regression.simple_linear", "biap", "lvnxcsgokqrt", E.NUMERICAL_DEMONSTRATION,
        related=("regression.ols_geometry", "econometrics.heteroskedasticity"),
        tags=("ols", "slope", "residuals", "functional form", "outliers", "leverage"),
        aliases=("simple regression", "regression simple", "الانحدار البسيط"),
        curriculum=("dz.econometrics1", "ksu.econ416"),
    ),
    _entry(
        "regression.ols_geometry", D.REGRESSION, "ols_geometry",
        f"{P}.regression.ols_geometry", "iap", "lvnxcdPoqr", E.GEOMETRIC_PROOF,
        prerequisites=("regression.simple_linear",),
        related=("regression.fwl",),
        tags=("projection", "orthogonality", "column space", "normal equations"),
        aliases=("ols geometry", "projection", "geometrie des mco", "الإسقاط"),
    ),
    _entry(
        "regression.fwl", D.REGRESSION, "partialling_out",
        f"{P}.regression.fwl", "ap", "lvnxcdPoqr", E.GEOMETRIC_PROOF,
        prerequisites=("regression.ols_geometry",),
        tags=("frisch-waugh-lovell", "partial regression", "partial correlation"),
        aliases=("fwl", "frisch waugh", "الانحدار الجزئي"),
        curriculum=("dz.econometrics1",),
    ),
    _entry(
        "regression.multicollinearity", D.REGRESSION, "diagnostics",
        f"{P}.regression.multicollinearity", "iap", "lvnxcsgoqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("regression.simple_linear",),
        tags=("vif", "collinearity", "variance inflation", "condition number"),
        aliases=("multicollinearity", "multicolinearite", "الازدواج الخطي"),
        curriculum=("dz.econometrics1", "cairo.eviews"),
    ),
    _entry(
        "regression.restricted", D.REGRESSION, "restricted_regression",
        f"{P}.regression.restricted", "iap", "lvxcdoqr", E.SYMBOLIC_DERIVATION,
        prerequisites=("regression.simple_linear",),
        tags=("restrictions", "f test", "wald", "dummy variables", "chow"),
        aliases=("restricted least squares", "regression contrainte", "الانحدار المقيد"),
        curriculum=("dz.econometrics1",),
    ),
    _entry(
        "regression.gauss_markov", D.REGRESSION, "classical_assumptions", None,
        "iap", "lvxcg", E.SYMBOLIC_DERIVATION, Status.PLANNED,
        tags=("blue", "gauss-markov", "assumption control center"),
    ),
    # ---------------- Econometrics ------------------------------------------
    _entry(
        "econometrics.heteroskedasticity", D.ECONOMETRICS, "diagnostics_and_remedies",
        f"{P}.econometrics.heteroskedasticity", "biap", "lvnxcsgokqr", E.SIMULATION,
        prerequisites=("regression.simple_linear",),
        tags=("heteroskedasticity", "white", "breusch-pagan", "robust", "wls"),
        aliases=("heteroskedasticity", "heteroscedasticite", "عدم تجانس التباين"),
        curriculum=("dz.econometrics1", "cairo.eviews", "ksu.econ416"),
    ),
    _entry(
        "econometrics.autocorrelation", D.ECONOMETRICS, "diagnostics_and_remedies",
        f"{P}.econometrics.autocorrelation", "iap", "lvnxcsgoqr", E.SIMULATION,
        prerequisites=("regression.simple_linear",),
        tags=("serial correlation", "durbin-watson", "breusch-godfrey", "hac", "newey-west"),
        aliases=("autocorrelation", "الارتباط الذاتي", "correlation serielle"),
        curriculum=("dz.econometrics1", "cairo.eviews"),
    ),
    _entry(
        "econometrics.omitted_variable_bias", D.ECONOMETRICS, "specification",
        f"{P}.econometrics.omitted_variable_bias", "biap", "lvnxcdsokqr", E.SYMBOLIC_DERIVATION,
        prerequisites=("regression.simple_linear",),
        related=("causal.dag", "econometrics.endogeneity_iv"),
        tags=("omitted variable bias", "confounding", "specification error"),
        aliases=("ovb", "biais de variable omise", "تحيز المتغير المحذوف"),
        curriculum=("dz.econometrics1",),
    ),
    _entry(
        "econometrics.endogeneity_iv", D.ECONOMETRICS, "endogeneity_and_identification",
        f"{P}.econometrics.endogeneity_iv", "iap", "lvnxcdsgokqr", E.SIMULATION,
        prerequisites=("econometrics.omitted_variable_bias",),
        related=("causal.dag",),
        tags=("endogeneity", "instrumental variables", "2sls", "weak instruments", "sargan"),
        aliases=("iv", "2sls", "variables instrumentales", "المتغيرات الوسيطة"),
        curriculum=("dz.econometrics2", "ksu.econ541", "aub.econ305"),
    ),
    _entry(
        "econometrics.simultaneous_equations", D.ECONOMETRICS, "systems",
        f"{P}.econometrics.simultaneous_equations", "ap", "lvxcdsgoqr", E.SIMULATION,
        prerequisites=("econometrics.endogeneity_iv",),
        tags=("simultaneity", "2sls", "3sls", "sur", "identification", "order condition"),
        aliases=("simultaneous equations", "equations simultanees", "المعادلات الآنية"),
        curriculum=("dz.econometrics2", "ksu.econ541"),
    ),
    _entry(
        "econometrics.logit_probit", D.ECONOMETRICS, "limited_dependent_variables",
        f"{P}.econometrics.logit_probit", "iap", "lvnxcdsgoqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("regression.simple_linear", "inference.mle"),
        tags=("logit", "probit", "lpm", "marginal effects", "odds ratio"),
        aliases=("binary choice", "logit probit", "الاختيار الثنائي"),
        curriculum=("dz.econometrics2", "ksu.econ541"),
    ),
    _entry(
        "econometrics.measurement_error", D.ECONOMETRICS, "specification", None,
        "iap", "lvxcs", E.SIMULATION, Status.PLANNED,
        tags=("errors in variables", "attenuation bias"),
    ),
    _entry(
        "econometrics.quantile_regression", D.ECONOMETRICS, "robust_methods", None,
        "ap", "lvxc", E.NUMERICAL_DEMONSTRATION, Status.PLANNED,
        tags=("quantile regression", "check function"),
    ),
    _entry(
        "econometrics.gmm", D.ECONOMETRICS, "gmm", None,
        "p", "lvxcd", E.SYMBOLIC_DERIVATION, Status.PLANNED,
        tags=("gmm", "moment conditions", "j statistic"),
    ),
    # ---------------- Time series -------------------------------------------
    _entry(
        "timeseries.stationarity", D.TIMESERIES, "unit_roots",
        f"{P}.timeseries.stationarity", "biap", "lvnxcsgokqr", E.SIMULATION,
        tags=("stationarity", "unit root", "random walk", "adf", "spurious regression"),
        aliases=("unit root", "racine unitaire", "جذر الوحدة"),
        curriculum=("dz.econometrics2", "cairo.eviews", "aub.econ306"),
    ),
    _entry(
        "timeseries.arma", D.TIMESERIES, "arima_family",
        f"{P}.timeseries.arma", "iap", "lvnxcsgoqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("timeseries.stationarity",),
        tags=("ar", "ma", "arma", "acf", "pacf", "box-jenkins"),
        aliases=("arima", "box jenkins", "بوكس جنكينز"),
        curriculum=("dz.econometrics2", "cairo.eviews"),
    ),
    _entry(
        "timeseries.cointegration", D.TIMESERIES, "cointegration_and_ecm",
        f"{P}.timeseries.cointegration", "ap", "lvnxcsgoqr", E.SIMULATION,
        prerequisites=("timeseries.stationarity",),
        tags=("cointegration", "engle-granger", "error correction", "ardl", "long run"),
        aliases=("ecm", "cointegration", "التكامل المشترك"),
        curriculum=("dz.econometrics2", "aub.econ306", "ksu.econ542"),
    ),
    _entry(
        "timeseries.var", D.TIMESERIES, "var_and_structural",
        f"{P}.timeseries.var", "ap", "lvnxcsoqr", E.NUMERICAL_DEMONSTRATION,
        prerequisites=("timeseries.arma",),
        tags=("var", "impulse response", "fevd", "granger causality"),
        aliases=("vector autoregression", "irf", "الانحدار الذاتي المتجه"),
        curriculum=("dz.econometrics2", "aub.econ306"),
    ),
    _entry(
        "timeseries.garch", D.TIMESERIES, "volatility",
        f"{P}.timeseries.garch", "ap", "lvnxcsgoqr", E.SIMULATION,
        prerequisites=("timeseries.arma",),
        tags=("arch", "garch", "volatility clustering", "leverage", "value at risk"),
        aliases=("garch", "volatilite", "التقلب"),
        curriculum=("dz.econometrics2", "aub.econ306"),
    ),
    _entry(
        "timeseries.ardl", D.TIMESERIES, "dynamic_models", None,
        "ap", "lvxc", E.NUMERICAL_DEMONSTRATION, Status.PLANNED,
        tags=("ardl", "distributed lag", "long-run multiplier"),
    ),
    # ---------------- Panel --------------------------------------------------
    _entry(
        "panel.fixed_vs_random", D.PANEL, "static_panel",
        f"{P}.panel.fixed_vs_random", "iap", "lvnxcsgoqr", E.SIMULATION,
        prerequisites=("regression.simple_linear",),
        tags=("fixed effects", "random effects", "within", "hausman", "clustered"),
        aliases=("fe re", "panel", "المعطيات الطولية", "donnees de panel"),
        curriculum=("dz.econometrics2", "ksu.econ541", "aub.econ305"),
    ),
    _entry(
        "panel.dynamic_gmm", D.PANEL, "dynamic_panel", None,
        "p", "lvxcs", E.SIMULATION, Status.PLANNED,
        tags=("nickell bias", "arellano-bond", "system gmm"),
    ),
    # ---------------- Causal inference ---------------------------------------
    _entry(
        "causal.potential_outcomes", D.CAUSAL, "foundations",
        f"{P}.causal.potential_outcomes", "biap", "lvnxcsokqr", E.SIMULATION,
        tags=("potential outcomes", "ate", "att", "selection bias", "overlap"),
        aliases=("rubin causal model", "resultats potentiels", "النواتج المحتملة"),
        curriculum=("harvard.api114",),
    ),
    _entry(
        "causal.dag", D.CAUSAL, "graphical_models",
        f"{P}.causal.dag", "biap", "lvnxcgokqr", E.VISUAL_DERIVATION,
        related=("econometrics.omitted_variable_bias",),
        tags=("dag", "confounder", "collider", "mediator", "backdoor", "bad control"),
        aliases=("causal graph", "graphe causal", "الرسم السببي"),
        extras=("causal",),
        curriculum=("harvard.api114",),
    ),
    _entry(
        "causal.did", D.CAUSAL, "designs",
        f"{P}.causal.did", "biap", "lvnxcsgokqr", E.VISUAL_DERIVATION,
        prerequisites=("causal.potential_outcomes",),
        tags=("difference in differences", "parallel trends", "event study", "twfe"),
        aliases=("did", "differences en differences", "الفروق في الفروق"),
        curriculum=("harvard.api114", "ksu.econ542"),
    ),
    _entry(
        "causal.rdd", D.CAUSAL, "designs",
        f"{P}.causal.rdd", "iap", "lvnxcsgoqr", E.VISUAL_DERIVATION,
        prerequisites=("causal.potential_outcomes",),
        tags=("regression discontinuity", "cutoff", "bandwidth", "mccrary"),
        aliases=("rdd", "regression sur discontinuite", "الانقطاع الانحداري"),
        curriculum=("harvard.api114",),
    ),
    _entry(
        "causal.synthetic_control", D.CAUSAL, "designs", None,
        "ap", "lvxc", E.VISUAL_DERIVATION, Status.PLANNED,
        tags=("synthetic control", "donor pool"),
    ),
    _entry(
        "causal.dml", D.CAUSAL, "causal_machine_learning", None,
        "p", "lvxcs", E.SIMULATION, Status.PLANNED,
        tags=("double machine learning", "causal forest", "heterogeneous effects"),
    ),
    # ---------------- Multivariate --------------------------------------------
    _entry(
        "multivariate.pca", D.MULTIVARIATE, "dimension_reduction",
        f"{P}.multivariate.pca", "biap", "lvnxcdPoqr", E.GEOMETRIC_PROOF,
        tags=("pca", "eigenvector", "explained variance", "projection", "svd"),
        aliases=("principal component analysis", "acp", "المكونات الرئيسية"),
        curriculum=("dz.data_analysis1",),
    ),
    _entry(
        "multivariate.clustering", D.MULTIVARIATE, "clustering",
        f"{P}.multivariate.clustering", "biap", "lvnxcsoqr", E.NUMERICAL_DEMONSTRATION,
        tags=("k-means", "hierarchical", "silhouette", "elbow"),
        aliases=("clustering", "classification automatique", "التصنيف العنقودي"),
        extras=("ai",),
        curriculum=("dz.data_analysis2",),
    ),
    # ---------------- Machine learning ------------------------------------------
    _entry(
        "ml.bias_variance", D.ML, "generalization",
        f"{P}.ml.bias_variance", "biap", "lvnxcsokqr", E.SIMULATION,
        tags=("bias", "variance", "overfitting", "underfitting", "learning curve"),
        aliases=("bias variance tradeoff", "biais variance", "المفاضلة بين التحيز والتباين"),
        curriculum=("stanford.stats315a",),
    ),
    _entry(
        "ml.regularization", D.ML, "regularization",
        f"{P}.ml.regularization", "iap", "lvnxcdPoqr", E.GEOMETRIC_PROOF,
        prerequisites=("ml.bias_variance",),
        tags=("ridge", "lasso", "elastic net", "shrinkage", "sparsity"),
        aliases=("ridge lasso", "regularisation", "الانكماش"),
        curriculum=("stanford.stats315a",),
    ),
    _entry(
        "ml.classification_threshold", D.ML, "classification",
        f"{P}.ml.classification_threshold", "biap", "lvnxcsgoqr", E.NUMERICAL_DEMONSTRATION,
        tags=("roc", "auc", "precision", "recall", "confusion matrix", "calibration"),
        aliases=("roc curve", "seuil de classification", "عتبة التصنيف"),
        curriculum=("stanford.stats202",),
    ),
    _entry(
        "ml.gradient_descent", D.ML, "optimization",
        f"{P}.ml.gradient_descent", "biap", "lvnxcdoqr", E.VISUAL_DERIVATION,
        tags=("gradient descent", "learning rate", "momentum", "convergence", "sgd"),
        aliases=("gradient descent", "descente de gradient", "الانحدار التدريجي"),
    ),
    _entry(
        "ml.cross_validation", D.ML, "model_selection", None,
        "ib", "lvxs", E.SIMULATION, Status.PLANNED,
        tags=("cross validation", "data leakage", "nested cv"),
    ),
    # ---------------- Deep learning / AI ------------------------------------------
    _entry(
        "deep_learning.backpropagation", D.DEEP_LEARNING, "training",
        f"{P}.deep_learning.backpropagation", "biap", "lvnxcdPoqr", E.SYMBOLIC_DERIVATION,
        prerequisites=("ml.gradient_descent",),
        tags=("backpropagation", "chain rule", "computational graph", "gradients"),
        aliases=("backprop", "retropropagation", "الانتشار العكسي"),
    ),
    _entry(
        "ai.attention", D.AI, "transformers",
        f"{P}.ai.attention", "biap", "lvnxcdoqr", E.NUMERICAL_DEMONSTRATION,
        tags=("attention", "self-attention", "softmax", "query key value", "transformer"),
        aliases=("attention", "الانتباه", "mecanisme d'attention"),
    ),
    _entry(
        "xai.shap", D.XAI, "local_explanations",
        f"{P}.ai.shap_lab", "iap", "lvnxcgoqr", E.NUMERICAL_DEMONSTRATION,
        tags=("shap", "shapley", "feature attribution", "additive explanation"),
        aliases=("shap values", "valeurs de shapley", "قيم شابلي"),
        extras=("ai",),
    ),
    _entry(
        "ai.tokenization", D.AI, "language_models", None,
        "bi", "lvx", E.NUMERICAL_DEMONSTRATION, Status.PLANNED,
        tags=("tokenization", "embeddings", "context window"),
    ),
    _entry(
        "xai.partial_dependence", D.XAI, "global_explanations", None,
        "ia", "lvxc", E.NUMERICAL_DEMONSTRATION, Status.PLANNED,
        tags=("partial dependence", "ice", "permutation importance"),
    ),
    # ---------------- Spatial ---------------------------------------------------
    _entry(
        "spatial.autocorrelation", D.SPATIAL, "spatial_dependence", None,
        "ap", "lvxg", E.NUMERICAL_DEMONSTRATION, Status.PLANNED,
        tags=("moran i", "weight matrix", "spatial lag"),
    ),
    # ---------------- Mathematical foundations -----------------------------------
    _entry(
        "math.projection", D.MATH, "linear_algebra",
        f"{P}.math.projection", "biap", "lvnxdPoqr", E.GEOMETRIC_PROOF,
        related=("regression.ols_geometry",),
        tags=("vector", "projection", "orthogonality", "span", "basis"),
        aliases=("orthogonal projection", "projection orthogonale", "الإسقاط المتعامد"),
    ),
    _entry(
        "math.optimization", D.MATH, "calculus", None,
        "ia", "lvxd", E.VISUAL_DERIVATION, Status.PLANNED,
        tags=("gradient", "hessian", "lagrange", "convexity"),
    ),
)


def load_builtin_catalog(reg: ConceptRegistry) -> None:
    """Register every catalog entry into ``reg``."""
    for spec in CATALOG:
        reg.register(spec, replace=True)
