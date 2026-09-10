"""Normalized result types.

Blueprint section 44: raw statsmodels / linearmodels / sklearn objects never
travel through the GUI. Backends adapt them into these stable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["RegressionResult", "TestResult", "IVResult", "SeriesResult", "ModelComparison"]


@dataclass
class RegressionResult:
    """Everything a lab needs from a fitted linear model."""

    names: tuple[str, ...]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    nobs: int
    df_resid: int
    df_model: int
    sigma2: float
    r_squared: float
    adj_r_squared: float
    cov_type: str = "nonrobust"
    backend: str = "numpy"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- inference ---------------------------------------------------------
    @property
    def tvalues(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.standard_errors > 0,
                            self.coefficients / self.standard_errors, np.nan)

    @property
    def pvalues(self) -> np.ndarray:
        from scipy import stats

        return 2.0 * stats.t.sf(np.abs(self.tvalues), max(self.df_resid, 1))

    def conf_int(self, level: float = 0.95) -> np.ndarray:
        from scipy import stats

        crit = stats.t.ppf(0.5 + level / 2.0, max(self.df_resid, 1))
        return np.column_stack(
            [
                self.coefficients - crit * self.standard_errors,
                self.coefficients + crit * self.standard_errors,
            ]
        )

    @property
    def ssr(self) -> float:
        return float(self.residuals @ self.residuals)

    def coef(self, name: str) -> float:
        return float(self.coefficients[self.names.index(name)])

    def se(self, name: str) -> float:
        return float(self.standard_errors[self.names.index(name)])

    def to_table(self, level: float = 0.95) -> dict[str, list[Any]]:
        ci = self.conf_int(level)
        return {
            "term": list(self.names),
            "coefficient": [float(v) for v in self.coefficients],
            "std_error": [float(v) for v in self.standard_errors],
            "t": [float(v) for v in self.tvalues],
            "p_value": [float(v) for v in self.pvalues],
            "ci_low": [float(v) for v in ci[:, 0]],
            "ci_high": [float(v) for v in ci[:, 1]],
        }

    def summary_lines(self, precision: int = 4) -> list[str]:
        lines = [
            f"OLS ({self.cov_type} SE, backend={self.backend})  "
            f"n={self.nobs}  R2={self.r_squared:.{precision}f}  "
            f"adj-R2={self.adj_r_squared:.{precision}f}"
        ]
        width = max(len(n) for n in self.names)
        lines.append(f"{'term':<{width}}  {'coef':>10}  {'se':>10}  {'t':>8}  {'p':>8}")
        table = self.to_table()
        for i, name in enumerate(self.names):
            lines.append(
                f"{name:<{width}}  {table['coefficient'][i]:>10.{precision}f}  "
                f"{table['std_error'][i]:>10.{precision}f}  {table['t'][i]:>8.3f}  "
                f"{table['p_value'][i]:>8.4f}"
            )
        return lines


@dataclass
class TestResult:
    """A hypothesis test in a form the GUI can always render."""

    name: str
    statistic: float
    p_value: float
    df: Any = None
    distribution: str = ""
    critical_value: float | None = None
    alternative: str = "two_sided"
    null_hypothesis: str = ""
    alternative_hypothesis: str = ""
    reject: bool | None = None
    alpha: float = 0.05
    backend: str = "scipy"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reject is None and self.p_value == self.p_value:
            self.reject = bool(self.p_value < self.alpha)

    def verdict(self) -> str:
        return "reject H0" if self.reject else "fail to reject H0"


@dataclass
class IVResult:
    """Two-stage least squares plus the diagnostics that decide credibility."""

    second_stage: RegressionResult
    first_stage: RegressionResult
    reduced_form: RegressionResult | None = None
    first_stage_f: float = float("nan")
    partial_r2: float = float("nan")
    endogenous: tuple[str, ...] = ()
    instruments: tuple[str, ...] = ()
    overid: TestResult | None = None
    endogeneity_test: TestResult | None = None
    backend: str = "numpy"

    @property
    def weak_instruments(self) -> bool:
        """Staiger-Stock rule of thumb: first-stage F below 10."""
        return bool(self.first_stage_f < 10.0)


@dataclass
class SeriesResult:
    """Time-series estimation output (ARIMA/AR/VAR-style)."""

    name: str
    params: dict[str, float]
    fitted: np.ndarray | None = None
    residuals: np.ndarray | None = None
    aic: float | None = None
    bic: float | None = None
    loglik: float | None = None
    backend: str = "numpy"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelComparison:
    """Side-by-side comparison of competing estimators."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def add(self, method: str, **values: Any) -> None:
        self.rows.append({"method": method, **values})

    def columns(self) -> list[str]:
        cols: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
        return cols
