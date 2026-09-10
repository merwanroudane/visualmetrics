"""Dependency-light linear-model engine.

Why this exists even though statsmodels is excellent: Monte Carlo labs fit tens
of thousands of models per interaction, the base install must work without the
econometrics extra, and OLS geometry labs need the projection matrices
themselves rather than a summary table.

Every formula here is cross-checked against statsmodels in
``tests/scientific/test_linear_backend.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg, stats

from ..core.exceptions import NumericalError
from .results import IVResult, RegressionResult, TestResult

__all__ = [
    "add_constant",
    "ols",
    "wls",
    "restricted_ols",
    "iv_2sls",
    "projection_matrices",
    "vif",
    "f_test",
    "wald_test",
    "breusch_pagan",
    "white_test",
    "durbin_watson",
    "breusch_godfrey",
    "jarque_bera",
    "reset_test",
    "hac_cov",
    "cluster_cov",
    "partial_out",
]


def add_constant(X: np.ndarray, *, prepend: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X]) if prepend else np.hstack([X, ones])


def _pinv_solve(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares solve returning ``(beta, XtX_inv)`` with a safe fallback."""
    XtX = X.T @ X
    try:
        XtX_inv = linalg.inv(XtX)
        beta = XtX_inv @ (X.T @ y)
        if not np.all(np.isfinite(beta)):
            raise linalg.LinAlgError("non-finite solution")
    except linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (X.T @ y)
    return beta, XtX_inv


def _cov_matrix(
    X: np.ndarray,
    resid: np.ndarray,
    XtX_inv: np.ndarray,
    cov_type: str,
    *,
    groups: np.ndarray | None = None,
    maxlags: int | None = None,
) -> np.ndarray:
    n, k = X.shape
    u = resid
    ct = cov_type.lower()
    if ct in ("nonrobust", "ols", "classical"):
        sigma2 = float(u @ u) / max(n - k, 1)
        return sigma2 * XtX_inv
    if ct in ("hc0", "hc1", "hc2", "hc3", "robust", "white"):
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        if ct in ("hc0", "robust", "white"):
            w = u**2
        elif ct == "hc1":
            w = u**2 * (n / max(n - k, 1))
        elif ct == "hc2":
            w = u**2 / np.clip(1.0 - h, 1e-10, None)
        else:  # hc3
            w = u**2 / np.clip(1.0 - h, 1e-10, None) ** 2
        meat = (X * w[:, None]).T @ X
        return XtX_inv @ meat @ XtX_inv
    if ct == "hac":
        return hac_cov(X, u, XtX_inv, maxlags=maxlags)
    if ct == "cluster":
        if groups is None:
            raise NumericalError("cluster-robust covariance requires a group vector")
        return cluster_cov(X, u, XtX_inv, groups)
    raise NumericalError(f"unknown covariance type {cov_type!r}")


def hac_cov(
    X: np.ndarray, u: np.ndarray, XtX_inv: np.ndarray, *, maxlags: int | None = None
) -> np.ndarray:
    """Newey-West heteroskedasticity- and autocorrelation-consistent covariance."""
    n, k = X.shape
    if maxlags is None:
        maxlags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    maxlags = max(0, min(int(maxlags), n - 1))
    h = X * u[:, None]
    S = h.T @ h
    for lag in range(1, maxlags + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        gamma = h[lag:].T @ h[:-lag]
        S += w * (gamma + gamma.T)
    S *= n / max(n - k, 1)
    return XtX_inv @ S @ XtX_inv


def cluster_cov(
    X: np.ndarray, u: np.ndarray, XtX_inv: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    n, k = X.shape
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    g = len(uniq)
    meat = np.zeros((k, k))
    for gid in uniq:
        idx = groups == gid
        sg = X[idx].T @ u[idx]
        meat += np.outer(sg, sg)
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return correction * (XtX_inv @ meat @ XtX_inv)


def ols(
    y: Any,
    X: Any,
    *,
    names: Any = None,
    cov_type: str = "nonrobust",
    groups: Any = None,
    maxlags: int | None = None,
    has_constant: bool = True,
) -> RegressionResult:
    """Ordinary least squares with classical, HC, HAC or clustered covariance."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if n != y.size:
        raise NumericalError(f"y has {y.size} rows but X has {n}")
    if n <= k:
        raise NumericalError(
            f"cannot estimate {k} coefficients from {n} observations "
            "(the model is not identified)"
        )
    beta, XtX_inv = _pinv_solve(X, y)
    fitted = X @ beta
    resid = y - fitted
    df_resid = n - k
    sigma2 = float(resid @ resid) / df_resid
    cov = _cov_matrix(
        X, resid, XtX_inv, cov_type,
        groups=None if groups is None else np.asarray(groups), maxlags=maxlags,
    )
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_res = float(resid @ resid)
    if has_constant:
        ss_tot = float(((y - y.mean()) ** 2).sum())
        df_model = k - 1
    else:
        ss_tot = float((y**2).sum())
        df_model = k
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1.0 - (1.0 - r2) * (n - (1 if has_constant else 0)) / max(df_resid, 1)
    if names is None:
        names = tuple(
            ["const"] + [f"x{i}" for i in range(1, k)] if has_constant
            else [f"x{i}" for i in range(1, k + 1)]
        )
    return RegressionResult(
        names=tuple(names),
        coefficients=beta,
        standard_errors=se,
        covariance=cov,
        residuals=resid,
        fitted_values=fitted,
        nobs=n,
        df_resid=df_resid,
        df_model=max(df_model, 0),
        sigma2=sigma2,
        r_squared=r2,
        adj_r_squared=adj,
        cov_type=cov_type,
        backend="numpy",
        metadata={"has_constant": has_constant, "XtX_inv": XtX_inv},
    )


def wls(y: Any, X: Any, weights: Any, *, names: Any = None, **kw: Any) -> RegressionResult:
    """Weighted least squares - the classic heteroskedasticity remedy."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if np.any(w <= 0):
        raise NumericalError("WLS weights must be strictly positive")
    rw = np.sqrt(w)
    res = ols(y * rw, X * rw[:, None], names=names, **kw)
    res.metadata["weighted"] = True
    res.fitted_values = X @ res.coefficients
    res.residuals = y - res.fitted_values
    return res


def restricted_ols(
    y: Any, X: Any, R: Any, r: Any, *, names: Any = None
) -> tuple[RegressionResult, TestResult]:
    """Least squares subject to ``R beta = r``, plus the F test of the restriction.

    Closed form: b_R = b + (X'X)^-1 R' [R (X'X)^-1 R']^-1 (r - R b).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))
    unrestricted = ols(y, X, names=names)
    b = unrestricted.coefficients
    XtX_inv = unrestricted.metadata["XtX_inv"]
    middle = R @ XtX_inv @ R.T
    middle_inv = np.linalg.pinv(middle)
    adjustment = XtX_inv @ R.T @ middle_inv @ (r - R @ b)
    b_r = b + adjustment
    fitted = X @ b_r
    resid = y - fitted
    q, k = R.shape[0], X.shape[1]
    n = X.shape[0]
    df_resid = n - k + q
    sigma2 = float(resid @ resid) / max(df_resid, 1)
    cov = sigma2 * (XtX_inv - XtX_inv @ R.T @ middle_inv @ R @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    restricted = RegressionResult(
        names=unrestricted.names,
        coefficients=b_r,
        standard_errors=se,
        covariance=cov,
        residuals=resid,
        fitted_values=fitted,
        nobs=n,
        df_resid=df_resid,
        df_model=max(k - q - 1, 0),
        sigma2=sigma2,
        r_squared=r2,
        adj_r_squared=1.0 - (1.0 - r2) * (n - 1) / max(df_resid, 1),
        cov_type="nonrobust",
        backend="numpy",
        metadata={"restricted": True, "R": R, "r": r},
    )
    test = f_test(unrestricted, R, r)
    return restricted, test


def f_test(result: RegressionResult, R: Any, r: Any = None) -> TestResult:
    """F test of the linear restriction ``R beta = r``."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    q = R.shape[0]
    r = np.zeros(q) if r is None else np.atleast_1d(np.asarray(r, dtype=float))
    b = result.coefficients
    diff = R @ b - r
    middle = R @ result.covariance @ R.T
    stat = float(diff @ np.linalg.pinv(middle) @ diff / q)
    p = float(stats.f.sf(stat, q, result.df_resid))
    return TestResult(
        name="F test of linear restrictions",
        statistic=stat,
        p_value=p,
        df=(q, result.df_resid),
        distribution="F",
        null_hypothesis="R beta = r",
        alternative_hypothesis="R beta != r",
    )


def wald_test(result: RegressionResult, R: Any, r: Any = None) -> TestResult:
    """Chi-square Wald test (asymptotic counterpart of :func:`f_test`)."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    q = R.shape[0]
    r = np.zeros(q) if r is None else np.atleast_1d(np.asarray(r, dtype=float))
    diff = R @ result.coefficients - r
    middle = R @ result.covariance @ R.T
    stat = float(diff @ np.linalg.pinv(middle) @ diff)
    return TestResult(
        name="Wald test",
        statistic=stat,
        p_value=float(stats.chi2.sf(stat, q)),
        df=q,
        distribution="chi2",
        null_hypothesis="R beta = r",
    )


def projection_matrices(X: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(P, M)`` where ``P = X(X'X)^-1 X'`` and ``M = I - P``.

    These are the objects behind the OLS geometry lab: ``P`` projects onto the
    column space of X, ``M`` annihilates it, and ``M X = 0`` exactly.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    XtX_inv = np.linalg.pinv(X.T @ X)
    P = X @ XtX_inv @ X.T
    M = np.eye(X.shape[0]) - P
    return P, M


def partial_out(y: Any, X: Any, Z: Any) -> tuple[np.ndarray, np.ndarray]:
    """Residualize ``y`` and ``X`` on ``Z`` - the Frisch-Waugh-Lovell operation."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    _, M = projection_matrices(Z)
    return M @ np.asarray(y, dtype=float), M @ np.asarray(X, dtype=float)


def vif(X: Any, *, has_constant: bool = True) -> np.ndarray:
    """Variance inflation factors for each non-constant column."""
    X = np.asarray(X, dtype=float)
    start = 1 if has_constant else 0
    out = []
    for j in range(start, X.shape[1]):
        others = np.delete(X, j, axis=1)
        if others.shape[1] == 0:
            out.append(1.0)
            continue
        res = ols(X[:, j], others, has_constant=has_constant)
        r2 = res.r_squared
        out.append(float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------
def breusch_pagan(result: RegressionResult, X: Any | None = None) -> TestResult:
    """Breusch-Pagan / Cook-Weisberg test for heteroskedasticity."""
    X = result.metadata.get("X") if X is None else X
    if X is None:
        raise NumericalError("breusch_pagan needs the regressor matrix")
    X = np.asarray(X, dtype=float)
    u2 = result.residuals**2
    aux = ols(u2 / u2.mean(), X)
    n = result.nobs
    lm = 0.5 * float(((aux.fitted_values - 1.0) ** 2).sum())
    df = X.shape[1] - 1
    return TestResult(
        name="Breusch-Pagan",
        statistic=lm,
        p_value=float(stats.chi2.sf(lm, df)),
        df=df,
        distribution="chi2",
        null_hypothesis="homoskedasticity",
        alternative_hypothesis="variance depends on the regressors",
        extra={"nobs": n},
    )


def white_test(result: RegressionResult, X: Any | None = None) -> TestResult:
    """White's general heteroskedasticity test (squares and cross-products)."""
    X = result.metadata.get("X") if X is None else X
    X = np.asarray(X, dtype=float)
    core = X[:, 1:] if result.metadata.get("has_constant", True) else X
    cols = [np.ones(X.shape[0])]
    for j in range(core.shape[1]):
        cols.append(core[:, j])
    for j in range(core.shape[1]):
        for m in range(j, core.shape[1]):
            cols.append(core[:, j] * core[:, m])
    Z = np.column_stack(cols)
    # drop collinear columns
    keep = [0]
    for j in range(1, Z.shape[1]):
        trial = Z[:, keep + [j]]
        if np.linalg.matrix_rank(trial) > len(keep):
            keep.append(j)
    Z = Z[:, keep]
    aux = ols(result.residuals**2, Z)
    lm = result.nobs * aux.r_squared
    df = max(Z.shape[1] - 1, 1)
    return TestResult(
        name="White",
        statistic=float(lm),
        p_value=float(stats.chi2.sf(lm, df)),
        df=df,
        distribution="chi2",
        null_hypothesis="homoskedasticity",
        alternative_hypothesis="general heteroskedasticity",
    )


def durbin_watson(residuals: Any) -> float:
    """Durbin-Watson statistic; ~2 means no first-order serial correlation."""
    u = np.asarray(residuals, dtype=float).ravel()
    return float(np.sum(np.diff(u) ** 2) / np.sum(u**2))


def breusch_godfrey(result: RegressionResult, X: Any | None = None, lags: int = 1) -> TestResult:
    """Breusch-Godfrey LM test for serial correlation up to ``lags`` order.

    Unlike Durbin-Watson it stays valid with a lagged dependent variable.
    """
    X = result.metadata.get("X") if X is None else X
    X = np.asarray(X, dtype=float)
    u = result.residuals
    n = u.size
    lagged = np.column_stack(
        [np.concatenate([np.zeros(k), u[:-k]]) for k in range(1, lags + 1)]
    )
    Z = np.column_stack([X, lagged])
    aux = ols(u, Z)
    lm = (n - lags) * aux.r_squared
    return TestResult(
        name=f"Breusch-Godfrey (lags={lags})",
        statistic=float(lm),
        p_value=float(stats.chi2.sf(lm, lags)),
        df=lags,
        distribution="chi2",
        null_hypothesis="no serial correlation up to the chosen order",
    )


def jarque_bera(residuals: Any) -> TestResult:
    u = np.asarray(residuals, dtype=float).ravel()
    n = u.size
    m = u - u.mean()
    s2 = float((m**2).mean())
    skew = float((m**3).mean() / s2**1.5)
    kurt = float((m**4).mean() / s2**2)
    jb = n / 6.0 * (skew**2 + 0.25 * (kurt - 3.0) ** 2)
    return TestResult(
        name="Jarque-Bera",
        statistic=float(jb),
        p_value=float(stats.chi2.sf(jb, 2)),
        df=2,
        distribution="chi2",
        null_hypothesis="normally distributed errors",
        extra={"skewness": skew, "kurtosis": kurt},
    )


def reset_test(result: RegressionResult, X: Any | None = None, power: int = 3) -> TestResult:
    """Ramsey RESET test for functional-form misspecification."""
    X = result.metadata.get("X") if X is None else X
    X = np.asarray(X, dtype=float)
    y = result.fitted_values + result.residuals
    extra = np.column_stack([result.fitted_values**p for p in range(2, power + 1)])
    Z = np.column_stack([X, extra])
    aug = ols(y, Z)
    q = extra.shape[1]
    num = (result.ssr - aug.ssr) / q
    den = aug.ssr / aug.df_resid
    stat = float(num / den)
    return TestResult(
        name=f"Ramsey RESET (powers up to {power})",
        statistic=stat,
        p_value=float(stats.f.sf(stat, q, aug.df_resid)),
        df=(q, aug.df_resid),
        distribution="F",
        null_hypothesis="the linear functional form is adequate",
    )


# ---------------------------------------------------------------------------
# Instrumental variables
# ---------------------------------------------------------------------------
def iv_2sls(
    y: Any,
    X: Any,
    Z: Any,
    *,
    endog_index: int = 1,
    names: Any = None,
    cov_type: str = "nonrobust",
) -> IVResult:
    """Two-stage least squares as an explicit two-projection procedure.

    ``X`` holds every regressor of the structural equation (including the
    endogenous one at ``endog_index``); ``Z`` holds the full instrument matrix
    (exogenous regressors + excluded instruments).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n, k = X.shape
    if Z.shape[1] < k:
        raise NumericalError(
            f"under-identified: {Z.shape[1]} instruments for {k} parameters. "
            "You need at least as many instruments as regressors."
        )
    # first stage: regress the endogenous regressor on all instruments
    endog = X[:, endog_index]
    first = ols(endog, Z, names=tuple(f"z{i}" for i in range(Z.shape[1])))
    first.metadata["X"] = Z
    X_hat = X.copy()
    X_hat[:, endog_index] = first.fitted_values

    PZ_X = Z @ np.linalg.pinv(Z.T @ Z) @ (Z.T @ X)
    beta, _ = _pinv_solve(PZ_X, y)
    resid = y - X @ beta
    df_resid = n - k
    sigma2 = float(resid @ resid) / df_resid
    A_inv = np.linalg.pinv(PZ_X.T @ PZ_X)
    if cov_type.lower() in ("hc0", "hc1", "robust", "white"):
        meat = (PZ_X * (resid**2)[:, None]).T @ PZ_X
        cov = A_inv @ meat @ A_inv
        if cov_type.lower() == "hc1":
            cov *= n / max(n - k, 1)
    else:
        cov = sigma2 * A_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    if names is None:
        names = tuple(["const"] + [f"x{i}" for i in range(1, k)])
    second = RegressionResult(
        names=tuple(names),
        coefficients=beta,
        standard_errors=se,
        covariance=cov,
        residuals=resid,
        fitted_values=X @ beta,
        nobs=n,
        df_resid=df_resid,
        df_model=k - 1,
        sigma2=sigma2,
        r_squared=r2,
        adj_r_squared=1.0 - (1.0 - r2) * (n - 1) / max(df_resid, 1),
        cov_type=cov_type,
        backend="numpy",
        metadata={"method": "2SLS", "X": X, "X_hat": X_hat},
    )

    # weak-instrument diagnostics: F on the excluded instruments
    exog_cols = [j for j in range(X.shape[1]) if j != endog_index]
    exog = X[:, exog_cols] if exog_cols else np.ones((n, 1))
    restricted = ols(endog, exog)
    n_excluded = Z.shape[1] - exog.shape[1]
    if n_excluded > 0 and first.df_resid > 0:
        f_stat = ((restricted.ssr - first.ssr) / n_excluded) / (first.ssr / first.df_resid)
        partial_r2 = max((restricted.ssr - first.ssr) / restricted.ssr, 0.0)
    else:
        f_stat, partial_r2 = float("nan"), float("nan")

    reduced = ols(y, Z, names=tuple(f"z{i}" for i in range(Z.shape[1])))

    overid = None
    if Z.shape[1] > k:
        aux = ols(resid, Z)
        j_stat = n * aux.r_squared
        dof = Z.shape[1] - k
        overid = TestResult(
            name="Sargan over-identification test",
            statistic=float(j_stat),
            p_value=float(stats.chi2.sf(j_stat, dof)),
            df=dof,
            distribution="chi2",
            null_hypothesis="all instruments are exogenous (valid exclusion restrictions)",
        )

    # Durbin-Wu-Hausman: is OLS actually inconsistent here?
    aug = ols(y, np.column_stack([X, first.residuals]))
    t_v = aug.tvalues[-1]
    endo_test = TestResult(
        name="Durbin-Wu-Hausman (augmented regression)",
        statistic=float(t_v**2),
        p_value=float(stats.f.sf(float(t_v**2), 1, aug.df_resid)),
        df=(1, aug.df_resid),
        distribution="F",
        null_hypothesis="the regressor is exogenous (OLS is consistent)",
    )

    return IVResult(
        second_stage=second,
        first_stage=first,
        reduced_form=reduced,
        first_stage_f=float(f_stat),
        partial_r2=float(partial_r2),
        overid=overid,
        endogeneity_test=endo_test,
        backend="numpy",
    )
