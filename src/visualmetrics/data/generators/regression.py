"""Regression data-generating processes.

One generator covers the whole scenario matrix of the regression labs:
functional form, error distribution, heteroskedasticity, autocorrelation,
outliers, leverage, measurement error, omitted variables and endogeneity.
Every dataset carries a readable DGP description (blueprint section 45.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...simulation.random import rng

__all__ = [
    "Dataset",
    "FUNCTIONAL_FORMS",
    "ERROR_DISTRIBUTIONS",
    "HETERO_TYPES",
    "X_DISTRIBUTIONS",
    "simple_regression",
    "multiple_regression",
    "collinear_design",
    "iv_design",
    "apply_functional_form",
    "draw_errors",
]

FUNCTIONAL_FORMS: tuple[str, ...] = (
    "linear",
    "quadratic",
    "cubic",
    "log_level",
    "level_log",
    "log_log",
    "exponential",
    "inverse",
    "threshold",
    "piecewise",
    "saturation",
    "u_shape",
    "inverted_u",
)

ERROR_DISTRIBUTIONS: tuple[str, ...] = (
    "normal",
    "student_t",
    "skewed",
    "uniform",
    "laplace",
    "mixture",
    "cauchy",
)

HETERO_TYPES: tuple[str, ...] = ("none", "increasing", "decreasing", "u_shaped", "grouped")

X_DISTRIBUTIONS: tuple[str, ...] = ("uniform", "normal", "lognormal", "bimodal", "discrete")


@dataclass
class Dataset:
    """A generated sample plus a human-readable account of how it was made."""

    columns: dict[str, np.ndarray]
    dgp: str
    truth: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def __contains__(self, key: object) -> bool:
        return key in self.columns

    @property
    def n(self) -> int:
        return len(next(iter(self.columns.values())))

    def matrix(self, *names: str, constant: bool = True) -> np.ndarray:
        cols = [self.columns[n] for n in names]
        if constant:
            cols.insert(0, np.ones(self.n))
        return np.column_stack(cols)

    def to_frame(self):
        """The generated columns as a pandas DataFrame.

        pandas is not a hard dependency - one convenience method does not
        justify putting it in every install - so this reports the extra to
        install rather than failing with an import error.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            from ...core.exceptions import MissingDependencyError

            raise MissingDependencyError(
                "pandas", "data", feature="converting a dataset to a DataFrame"
            ) from exc

        return pd.DataFrame(self.columns)


def draw_x(
    gen: np.random.Generator,
    n: int,
    distribution: str = "uniform",
    *,
    low: float = 0.0,
    high: float = 10.0,
) -> np.ndarray:
    """Draw the regressor with a chosen shape but a comparable location/scale."""
    if distribution == "normal":
        x = gen.normal((low + high) / 2, (high - low) / 6, n)
    elif distribution == "lognormal":
        x = gen.lognormal(0.0, 0.7, n)
        x = low + (high - low) * (x - x.min()) / max(x.max() - x.min(), 1e-12)
    elif distribution == "bimodal":
        pick = gen.random(n) < 0.5
        x = np.where(
            pick,
            gen.normal(low + 0.2 * (high - low), 0.08 * (high - low), n),
            gen.normal(low + 0.8 * (high - low), 0.08 * (high - low), n),
        )
    elif distribution == "discrete":
        x = gen.choice(np.linspace(low, high, 6), size=n)
    else:
        x = gen.uniform(low, high, n)
    return x


def apply_functional_form(
    x: np.ndarray, beta0: float, beta1: float, form: str = "linear", *, threshold: float = 5.0
) -> np.ndarray:
    """Systematic part E[y|x] for the requested functional form.

    ``log_level`` means log(y) is linear in x; the generator applies the
    inverse transform so the *observed* y follows the stated relationship.
    """
    xs = np.asarray(x, dtype=float)
    xpos = np.clip(xs, 1e-6, None)
    if form == "linear":
        return beta0 + beta1 * xs
    if form == "quadratic":
        return beta0 + beta1 * xs - 0.08 * beta1 * xs**2
    if form == "cubic":
        return beta0 + beta1 * xs - 0.15 * beta1 * xs**2 + 0.01 * beta1 * xs**3
    if form == "log_level":
        return np.exp(np.clip(beta0 * 0.1 + beta1 * xs * 0.1, -20, 20))
    if form == "level_log":
        return beta0 + beta1 * np.log(xpos)
    if form == "log_log":
        return np.exp(np.clip(beta0 * 0.1 + beta1 * np.log(xpos), -20, 20))
    if form == "exponential":
        return beta0 + np.exp(np.clip(beta1 * xs * 0.25, -20, 20))
    if form == "inverse":
        return beta0 + beta1 * 10.0 / xpos
    if form == "threshold":
        return beta0 + beta1 * xs * (xs > threshold)
    if form == "piecewise":
        return beta0 + beta1 * xs + (-1.8 * beta1) * np.clip(xs - threshold, 0, None)
    if form == "saturation":
        return beta0 + beta1 * 10.0 * xs / (2.0 + xs)
    if form == "u_shape":
        centre = xs.mean()
        return beta0 + abs(beta1) * 0.25 * (xs - centre) ** 2
    if form == "inverted_u":
        centre = xs.mean()
        return beta0 + abs(beta1) * 3.0 - abs(beta1) * 0.25 * (xs - centre) ** 2
    return beta0 + beta1 * xs


def draw_errors(
    gen: np.random.Generator, n: int, distribution: str = "normal", scale: float = 1.0
) -> np.ndarray:
    """Standardized error draws (unit variance where the variance exists)."""
    if distribution == "student_t":
        df = 4.0
        u = gen.standard_t(df, n) / np.sqrt(df / (df - 2.0))
    elif distribution == "skewed":
        u = gen.gumbel(0.0, 1.0, n)
        u = (u - np.euler_gamma) / (np.pi / np.sqrt(6.0))
    elif distribution == "uniform":
        u = gen.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    elif distribution == "laplace":
        u = gen.laplace(0.0, 1.0 / np.sqrt(2.0), n)
    elif distribution == "mixture":
        contaminated = gen.random(n) < 0.1
        u = np.where(contaminated, gen.normal(0, 4.0, n), gen.normal(0, 1.0, n))
        u = u / np.sqrt(0.9 + 0.1 * 16.0)
    elif distribution == "cauchy":
        u = gen.standard_cauchy(n)  # no finite variance - deliberately
    else:
        u = gen.standard_normal(n)
    return scale * u


def _hetero_weights(x: np.ndarray, kind: str, strength: float) -> np.ndarray:
    """Multiplicative scale factor for the error, mean-normalized to 1.

    An exponential form is used rather than a linear one: a linear variance
    ramp inflates the slope's true variance by only a few percent even at full
    strength, which is far too subtle to teach with. The exponential form gives
    a genuine order-of-magnitude spread in the error standard deviation.
    """
    if kind == "none" or strength <= 0:
        return np.ones_like(x)
    z = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    centred = 2.0 * z - 1.0
    if kind == "increasing":
        w = np.exp(strength * 2.0 * centred)
    elif kind == "decreasing":
        w = np.exp(-strength * 2.0 * centred)
    elif kind == "u_shaped":
        w = np.exp(strength * 2.5 * (2.0 * centred**2 - 0.6))
    elif kind == "grouped":
        w = np.where(z > 0.5, np.exp(strength * 1.6), np.exp(-strength * 1.6))
    else:
        w = np.ones_like(z)
    return w / w.mean()


def simple_regression(
    *,
    n: int = 120,
    beta0: float = 2.0,
    beta1: float = 1.5,
    noise: float = 1.0,
    seed: int = 42,
    functional_form: str = "linear",
    x_distribution: str = "uniform",
    x_low: float = 0.0,
    x_high: float = 10.0,
    error_distribution: str = "normal",
    heteroskedasticity: str = "none",
    hetero_strength: float = 0.0,
    autocorrelation: float = 0.0,
    n_outliers: int = 0,
    outlier_magnitude: float = 6.0,
    leverage_point: bool = False,
    leverage_x: float = 20.0,
    leverage_y_shift: float = -8.0,
    measurement_error: float = 0.0,
    omitted_strength: float = 0.0,
    omitted_x_correlation: float = 0.0,
    endogeneity: float = 0.0,
    threshold: float = 5.0,
    standardize: bool = False,
    center: bool = False,
) -> Dataset:
    """The workhorse DGP behind the simple-regression and diagnostics labs."""
    gen = rng(seed)
    n = int(max(n, 5))
    x_true = draw_x(gen, n, x_distribution, low=x_low, high=x_high)

    # omitted variable: correlated with x, and it moves y
    omitted = gen.standard_normal(n)
    if omitted_x_correlation:
        rho = float(np.clip(omitted_x_correlation, -0.99, 0.99))
        xs = (x_true - x_true.mean()) / max(x_true.std(), 1e-12)
        omitted = rho * xs + np.sqrt(max(1 - rho**2, 0.0)) * omitted

    u = draw_errors(gen, n, error_distribution, 1.0)
    if autocorrelation:
        rho = float(np.clip(autocorrelation, -0.99, 0.99))
        e = np.empty(n)
        e[0] = u[0] / np.sqrt(max(1 - rho**2, 1e-6))
        for t in range(1, n):
            e[t] = rho * e[t - 1] + u[t]
        u = e / max(e.std(), 1e-12)

    weights = _hetero_weights(x_true, heteroskedasticity, hetero_strength)
    error = noise * weights * u

    # endogeneity: correlation between the regressor and the structural error
    if endogeneity:
        rho = float(np.clip(endogeneity, -0.95, 0.95))
        x_true = x_true + rho * (x_true.std() / max(error.std(), 1e-12)) * error
        x_true = np.asarray(x_true, dtype=float)

    mean = apply_functional_form(x_true, beta0, beta1, functional_form, threshold=threshold)
    y = mean + error + omitted_strength * omitted

    if n_outliers:
        idx = gen.choice(n, size=min(int(n_outliers), n), replace=False)
        sign = gen.choice([-1.0, 1.0], size=idx.size)
        y[idx] = y[idx] + sign * outlier_magnitude * max(noise, 0.5)

    x_observed = x_true.copy()
    if leverage_point:
        x_observed = np.append(x_observed, leverage_x)
        x_true = np.append(x_true, leverage_x)
        y = np.append(y, apply_functional_form(
            np.array([leverage_x]), beta0, beta1, functional_form, threshold=threshold
        )[0] + leverage_y_shift)
        omitted = np.append(omitted, 0.0)
        error = np.append(error, leverage_y_shift)
        weights = np.append(weights, 1.0)
        mean = np.append(mean, 0.0)
        n += 1

    if measurement_error:
        x_observed = x_observed + measurement_error * gen.standard_normal(n)

    if center or standardize:
        x_observed = x_observed - x_observed.mean()
        if standardize:
            x_observed = x_observed / max(x_observed.std(), 1e-12)
            y = (y - y.mean()) / max(y.std(), 1e-12)

    notes: list[str] = []
    parts = [f"y = {beta0:g} + {beta1:g}*x + u" if functional_form == "linear"
             else f"y = f(x; {functional_form}) + u"]
    parts.append(f"n = {n}")
    parts.append(f"x ~ {x_distribution}[{x_low:g}, {x_high:g}]")
    parts.append(f"u ~ {error_distribution}, sd = {noise:g}")
    if heteroskedasticity != "none" and hetero_strength > 0:
        parts.append(f"heteroskedasticity: {heteroskedasticity} (strength {hetero_strength:g})")
        notes.append("assumption_homoskedasticity")
    if autocorrelation:
        parts.append(f"AR(1) errors with rho = {autocorrelation:g}")
        notes.append("assumption_no_autocorrelation")
    if omitted_strength and omitted_x_correlation:
        parts.append(
            f"omitted variable z: effect {omitted_strength:g}, corr(x, z) = {omitted_x_correlation:g}"
        )
        notes.append("assumption_exogeneity")
    if endogeneity:
        parts.append(f"corr(x, u) induced = {endogeneity:g}")
        notes.append("assumption_exogeneity")
    if measurement_error:
        parts.append(f"x measured with error sd = {measurement_error:g}")
        notes.append("assumption_exogeneity")
    if n_outliers:
        parts.append(f"{n_outliers} vertical outlier(s) of size {outlier_magnitude:g}")
    if leverage_point:
        parts.append(f"one high-leverage point at x = {leverage_x:g}")
    if error_distribution != "normal":
        notes.append("assumption_normal_errors")

    # asymptotic bias of OLS from the omitted variable (for the teaching panel)
    bias = 0.0
    if omitted_strength and omitted_x_correlation:
        sx = float(np.std(x_true))
        bias = float(omitted_strength * omitted_x_correlation / max(sx, 1e-12))

    return Dataset(
        columns={
            "x": x_observed,
            "x_true": x_true,
            "y": y,
            "z_omitted": omitted,
            "error": error,
            "hetero_weight": weights,
        },
        dgp="; ".join(parts),
        truth={
            "beta0": float(beta0),
            "beta1": float(beta1),
            "noise": float(noise),
            "omitted_bias": bias,
        },
        notes=tuple(dict.fromkeys(notes)),
        meta={"functional_form": functional_form, "error_distribution": error_distribution},
    )


def multiple_regression(
    *,
    n: int = 200,
    betas: Any = (1.0, 2.0, -1.0),
    correlation: float = 0.0,
    noise: float = 1.0,
    seed: int = 42,
    intercept: float = 1.0,
    hetero_strength: float = 0.0,
) -> Dataset:
    """k regressors with an exchangeable correlation structure."""
    gen = rng(seed)
    betas = np.asarray(betas, dtype=float)
    k = betas.size
    rho = float(np.clip(correlation, -0.99, 0.99))
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-10, None)))
    X = gen.standard_normal((n, k)) @ L.T
    weights = _hetero_weights(X[:, 0], "increasing" if hetero_strength else "none",
                              hetero_strength)
    u = noise * weights * gen.standard_normal(n)
    y = intercept + X @ betas + u
    columns = {"y": y}
    for j in range(k):
        columns[f"x{j + 1}"] = X[:, j]
    return Dataset(
        columns=columns,
        dgp=(
            f"y = {intercept:g} + "
            + " + ".join(f"{b:g}*x{j + 1}" for j, b in enumerate(betas))
            + f" + u;  n = {n};  corr(x_i, x_j) = {rho:g};  sd(u) = {noise:g}"
        ),
        truth={"intercept": float(intercept), **{f"beta{j + 1}": float(b)
                                                 for j, b in enumerate(betas)}},
        meta={"k": k, "correlation": rho},
    )


def collinear_design(
    *, n: int = 150, correlation: float = 0.9, beta1: float = 1.0, beta2: float = 1.0,
    noise: float = 1.0, seed: int = 42,
) -> Dataset:
    """Two regressors with a controlled correlation - the multicollinearity lab."""
    gen = rng(seed)
    rho = float(np.clip(correlation, -0.999, 0.999))
    x1 = gen.standard_normal(n)
    x2 = rho * x1 + np.sqrt(max(1 - rho**2, 1e-9)) * gen.standard_normal(n)
    u = noise * gen.standard_normal(n)
    y = 1.0 + beta1 * x1 + beta2 * x2 + u
    return Dataset(
        columns={"y": y, "x1": x1, "x2": x2},
        dgp=(
            f"y = 1 + {beta1:g}*x1 + {beta2:g}*x2 + u;  corr(x1, x2) = {rho:g};  "
            f"n = {n};  sd(u) = {noise:g}"
        ),
        truth={"beta1": float(beta1), "beta2": float(beta2), "correlation": rho},
    )


def iv_design(
    *,
    n: int = 300,
    beta: float = 1.0,
    endogeneity: float = 0.6,
    instrument_strength: float = 0.8,
    invalid_instrument: float = 0.0,
    n_instruments: int = 1,
    noise: float = 1.0,
    seed: int = 42,
) -> Dataset:
    """Structural equation y = a + beta*x + u with corr(x, u) != 0.

    ``instrument_strength`` is the first-stage coefficient; ``invalid_instrument``
    is a direct z -> y path that violates the exclusion restriction.
    """
    gen = rng(seed)
    n_inst = max(int(n_instruments), 1)
    Z = gen.standard_normal((n, n_inst))
    confounder = gen.standard_normal(n)
    v = gen.standard_normal(n)
    u = noise * (endogeneity * confounder + np.sqrt(max(1 - endogeneity**2, 0.0)) * v)
    x = (
        instrument_strength * Z.sum(axis=1) / np.sqrt(n_inst)
        + endogeneity * confounder
        + np.sqrt(max(1 - endogeneity**2, 0.0)) * gen.standard_normal(n)
    )
    y = 0.5 + beta * x + u + invalid_instrument * Z[:, 0]
    columns: dict[str, np.ndarray] = {"y": y, "x": x, "confounder": confounder, "u": u}
    for j in range(n_inst):
        columns[f"z{j + 1}"] = Z[:, j]
    # OLS asymptotic bias: cov(x, u) / var(x)
    bias = float(np.cov(x, u, ddof=1)[0, 1] / np.var(x, ddof=1))
    return Dataset(
        columns=columns,
        dgp=(
            f"structural: y = 0.5 + {beta:g}*x + u;  "
            f"first stage: x = {instrument_strength:g}*z + confounder;  "
            f"corr driver = {endogeneity:g};  n = {n}"
            + (f";  INVALID direct path z -> y of {invalid_instrument:g}"
               if invalid_instrument else "")
        ),
        truth={"beta": float(beta), "ols_bias": bias,
               "instrument_strength": float(instrument_strength)},
        notes=("assumption_exogeneity",) + (
            ("assumption_exclusion",) if invalid_instrument else ()),
        meta={"n_instruments": n_inst},
    )
