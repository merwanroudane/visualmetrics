"""Time-series data-generating processes.

Covers the stationarity/unit-root, ARMA, cointegration/ECM, VAR and volatility
labs from one consistent, seed-reproducible family.
"""

from __future__ import annotations

import numpy as np

from ...simulation.random import rng
from .regression import Dataset

__all__ = [
    "ar_process",
    "arma_process",
    "random_walk",
    "trend_stationary",
    "cointegrated_pair",
    "var_process",
    "garch_process",
    "structural_break_series",
    "acf",
    "pacf",
]


def ar_process(
    *,
    n: int = 200,
    rho: float = 0.5,
    drift: float = 0.0,
    trend: float = 0.0,
    sigma: float = 1.0,
    seed: int = 42,
    burn_in: int = 100,
    y0: float = 0.0,
) -> Dataset:
    """AR(1): y_t = drift + trend*t + rho*y_{t-1} + e_t."""
    gen = rng(seed)
    total = n + (burn_in if abs(rho) < 1 else 0)
    e = sigma * gen.standard_normal(total)
    y = np.empty(total)
    y[0] = y0 if abs(rho) >= 1 else drift / max(1 - rho, 1e-9)
    for t in range(1, total):
        y[t] = drift + rho * y[t - 1] + e[t]
    y = y[-n:]
    t_index = np.arange(n, dtype=float)
    if trend:
        y = y + trend * t_index
    kind = (
        "unit root (random walk)" if abs(rho - 1.0) < 1e-9
        else "explosive" if abs(rho) > 1.0
        else "stationary"
    )
    return Dataset(
        columns={"t": t_index, "y": y, "e": e[-n:]},
        dgp=(
            f"y_t = {drift:g} + {rho:g}*y_(t-1) + e_t"
            + (f" + {trend:g}*t" if trend else "")
            + f";  e_t ~ N(0, {sigma:g}^2);  n = {n};  process is {kind}"
        ),
        truth={"rho": float(rho), "drift": float(drift), "trend": float(trend),
               "sigma": float(sigma)},
        meta={"kind": kind, "stationary": abs(rho) < 1.0 and trend == 0.0},
    )


def arma_process(
    *, n: int = 250, ar: tuple[float, ...] = (0.6,), ma: tuple[float, ...] = (),
    sigma: float = 1.0, seed: int = 42, burn_in: int = 200, const: float = 0.0,
) -> Dataset:
    """ARMA(p, q) with the sign convention y_t = c + sum(ar_i y_{t-i}) + e_t + sum(ma_j e_{t-j})."""
    gen = rng(seed)
    p, q = len(ar), len(ma)
    total = n + burn_in
    e = sigma * gen.standard_normal(total)
    y = np.zeros(total)
    for t in range(max(p, q), total):
        val = const + e[t]
        for i, phi in enumerate(ar, start=1):
            val += phi * y[t - i]
        for j, theta in enumerate(ma, start=1):
            val += theta * e[t - j]
        y[t] = val
    y = y[-n:]
    roots_ok = True
    if p:
        companion = np.r_[1.0, -np.asarray(ar, dtype=float)]
        roots = np.roots(companion[::-1])
        roots_ok = bool(np.all(np.abs(roots) > 1.0)) if roots.size else True
    return Dataset(
        columns={"t": np.arange(n, dtype=float), "y": y, "e": e[-n:]},
        dgp=(
            f"ARMA({p},{q}):  AR = {tuple(round(float(a), 3) for a in ar)}, "
            f"MA = {tuple(round(float(m), 3) for m in ma)};  sd(e) = {sigma:g};  n = {n}"
        ),
        truth={"sigma": float(sigma)},
        meta={"p": p, "q": q, "ar": tuple(ar), "ma": tuple(ma), "stationary": roots_ok},
    )


def random_walk(*, n: int = 200, drift: float = 0.0, sigma: float = 1.0,
                seed: int = 42) -> Dataset:
    return ar_process(n=n, rho=1.0, drift=drift, sigma=sigma, seed=seed, burn_in=0)


def trend_stationary(*, n: int = 200, trend: float = 0.05, rho: float = 0.4,
                     sigma: float = 1.0, seed: int = 42) -> Dataset:
    """Deterministic trend + stationary noise - visually similar to a random walk."""
    base = ar_process(n=n, rho=rho, sigma=sigma, seed=seed)
    t = base["t"]
    y = base["y"] + trend * t
    return Dataset(
        columns={"t": t, "y": y, "e": base["e"]},
        dgp=f"y_t = {trend:g}*t + v_t with v_t stationary AR(1), rho = {rho:g};  n = {n}",
        truth={"trend": float(trend), "rho": float(rho)},
        meta={"kind": "trend stationary", "stationary": False},
    )


def cointegrated_pair(
    *, n: int = 200, beta: float = 1.5, adjustment: float = -0.25, sigma_x: float = 1.0,
    sigma_e: float = 0.8, seed: int = 42, cointegrated: bool = True,
) -> Dataset:
    """Two I(1) series that share a stationary long-run relation y - beta*x.

    ``adjustment`` is the error-correction speed (must be negative and > -2).
    """
    gen = rng(seed)
    dx = sigma_x * gen.standard_normal(n)
    x = np.cumsum(dx)
    y = np.empty(n)
    y[0] = beta * x[0] + gen.normal(0, sigma_e)
    for t in range(1, n):
        if cointegrated:
            ecm = y[t - 1] - beta * x[t - 1]
            y[t] = y[t - 1] + adjustment * ecm + beta * dx[t] + sigma_e * gen.standard_normal()
        else:
            y[t] = y[t - 1] + sigma_e * gen.standard_normal()
    spread = y - beta * x
    return Dataset(
        columns={"t": np.arange(n, dtype=float), "y": y, "x": x, "spread": spread},
        dgp=(
            (
                f"x_t is a random walk;  y_t - {beta:g}*x_t is stationary;  "
                f"ECM speed alpha = {adjustment:g};  n = {n}"
            )
            if cointegrated
            else f"x_t and y_t are INDEPENDENT random walks (no cointegration);  n = {n}"
        ),
        truth={"beta": float(beta), "adjustment": float(adjustment)},
        meta={"cointegrated": bool(cointegrated)},
    )


def var_process(
    *, n: int = 250, a11: float = 0.5, a12: float = 0.3, a21: float = 0.1, a22: float = 0.4,
    sigma1: float = 1.0, sigma2: float = 1.0, corr: float = 0.0, seed: int = 42,
    burn_in: int = 100,
) -> Dataset:
    """Bivariate VAR(1) - the basis of the impulse-response lab."""
    gen = rng(seed)
    A = np.array([[a11, a12], [a21, a22]], dtype=float)
    cov = np.array([[sigma1**2, corr * sigma1 * sigma2],
                    [corr * sigma1 * sigma2, sigma2**2]])
    total = n + burn_in
    shocks = gen.multivariate_normal(np.zeros(2), cov, size=total)
    y = np.zeros((total, 2))
    for t in range(1, total):
        y[t] = A @ y[t - 1] + shocks[t]
    y = y[-n:]
    eig = np.abs(np.linalg.eigvals(A))
    return Dataset(
        columns={"t": np.arange(n, dtype=float), "y1": y[:, 0], "y2": y[:, 1],
                 "e1": shocks[-n:, 0], "e2": shocks[-n:, 1]},
        dgp=(
            f"VAR(1):  A = [[{a11:g}, {a12:g}], [{a21:g}, {a22:g}]];  "
            f"corr(e1, e2) = {corr:g};  n = {n};  max |eigenvalue| = {eig.max():.3f}"
        ),
        truth={"a11": a11, "a12": a12, "a21": a21, "a22": a22},
        meta={"A": A, "cov": cov, "stable": bool(eig.max() < 1.0)},
    )


def garch_process(
    *, n: int = 500, omega: float = 0.05, alpha: float = 0.1, beta: float = 0.85,
    mu: float = 0.0, seed: int = 42, leverage: float = 0.0, burn_in: int = 200,
) -> Dataset:
    """GARCH(1,1) returns, optionally with a GJR leverage term."""
    gen = rng(seed)
    total = n + burn_in
    persistence = alpha + beta + 0.5 * leverage
    uncond = omega / max(1.0 - persistence, 1e-6) if persistence < 1 else omega * 10
    sigma2 = np.full(total, uncond)
    z = gen.standard_normal(total)
    r = np.zeros(total)
    for t in range(1, total):
        shock = r[t - 1] - mu
        neg = 1.0 if shock < 0 else 0.0
        sigma2[t] = omega + (alpha + leverage * neg) * shock**2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)
        r[t] = mu + np.sqrt(sigma2[t]) * z[t]
    return Dataset(
        columns={
            "t": np.arange(n, dtype=float),
            "r": r[-n:],
            "sigma": np.sqrt(sigma2[-n:]),
            "sigma2": sigma2[-n:],
        },
        dgp=(
            f"GARCH(1,1):  sigma2_t = {omega:g} + {alpha:g}*e_(t-1)^2 + "
            f"{beta:g}*sigma2_(t-1)"
            + (f" + {leverage:g}*e_(t-1)^2*I(e_(t-1)<0)" if leverage else "")
            + f";  persistence = {persistence:.3f};  n = {n}"
        ),
        truth={"omega": omega, "alpha": alpha, "beta": beta, "persistence": persistence},
        meta={"stationary": persistence < 1.0},
    )


def structural_break_series(
    *, n: int = 200, break_fraction: float = 0.5, level_shift: float = 3.0,
    rho: float = 0.4, sigma: float = 1.0, seed: int = 42,
) -> Dataset:
    """Stationary around a shifting mean - the classic unit-root false positive."""
    base = ar_process(n=n, rho=rho, sigma=sigma, seed=seed)
    t = base["t"]
    cut = int(break_fraction * n)
    shift = np.where(t >= cut, level_shift, 0.0)
    return Dataset(
        columns={"t": t, "y": base["y"] + shift, "break": shift},
        dgp=(
            f"stationary AR(1) with rho = {rho:g} around a mean that jumps by "
            f"{level_shift:g} at t = {cut};  n = {n}"
        ),
        truth={"rho": float(rho), "break_point": float(cut), "level_shift": float(level_shift)},
        meta={"kind": "break", "stationary": True},
    )


def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Sample autocorrelation function (biased/standard estimator)."""
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    denom = float(x @ x)
    out = np.ones(nlags + 1)
    for lag in range(1, nlags + 1):
        out[lag] = float(x[lag:] @ x[:-lag]) / denom if denom > 0 else 0.0
    return out


def pacf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    """Partial autocorrelations via the Durbin-Levinson recursion."""
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = np.ones(nlags + 1)
    if nlags >= 1:
        phi[1, 1] = r[1]
        out[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out[k] = phi[k, k]
    return out
