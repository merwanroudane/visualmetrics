"""Machine-learning data-generating processes.

Classification scores, nonlinear regression targets for the bias-variance and
regularization labs, correlated feature clouds for PCA, and clusterable blobs.
"""

from __future__ import annotations

import numpy as np

from ...simulation.random import rng
from .regression import Dataset

__all__ = [
    "classification_scores",
    "nonlinear_target",
    "correlated_features",
    "blobs",
    "sparse_regression",
]


def classification_scores(
    *,
    n: int = 600,
    separation: float = 1.5,
    positive_rate: float = 0.5,
    noise_scale: float = 1.0,
    miscalibration: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """Continuous scores for two classes - the threshold / ROC lab.

    ``separation`` is the standardized distance between class score means, i.e.
    exactly the quantity that fixes the achievable ROC curve.
    """
    gen = rng(seed)
    y = (gen.random(n) < positive_rate).astype(float)
    score = np.where(y > 0, gen.normal(separation, noise_scale, n), gen.normal(0.0, noise_scale, n))
    prob = 1.0 / (1.0 + np.exp(-(score - separation / 2)))
    if miscalibration:
        prob = np.clip(prob ** (1.0 + miscalibration), 1e-6, 1 - 1e-6)
    return Dataset(
        columns={"y": y, "score": score, "prob": prob},
        dgp=(
            f"score | y=1 ~ N({separation:g}, {noise_scale:g}^2), "
            f"score | y=0 ~ N(0, {noise_scale:g}^2);  P(y=1) = {positive_rate:g};  n = {n}"
            + (f";  probabilities distorted by exponent 1+{miscalibration:g}"
               if miscalibration else "")
        ),
        truth={"separation": float(separation), "positive_rate": float(positive_rate)},
        meta={"calibrated": miscalibration == 0.0},
    )


def nonlinear_target(
    *, n: int = 60, noise: float = 0.3, seed: int = 42, kind: str = "sine",
    x_low: float = 0.0, x_high: float = 1.0,
) -> Dataset:
    """A smooth true function sampled with noise - bias/variance and splines."""
    gen = rng(seed)
    x = np.sort(gen.uniform(x_low, x_high, n))
    if kind == "sine":
        f = np.sin(2 * np.pi * x)
        label = "f(x) = sin(2*pi*x)"
    elif kind == "polynomial":
        f = 1.5 * x**3 - 2.0 * x**2 + 0.5 * x
        label = "f(x) = 1.5x^3 - 2x^2 + 0.5x"
    elif kind == "step":
        f = np.where(x > 0.5, 1.0, -1.0)
        label = "f(x) = step at x = 0.5"
    else:
        f = x
        label = "f(x) = x"
    y = f + noise * gen.standard_normal(n)
    return Dataset(
        columns={"x": x, "y": y, "f": f},
        dgp=f"{label} + e,  e ~ N(0, {noise:g}^2);  n = {n}",
        truth={"noise": float(noise)},
        meta={"kind": kind, "x_low": x_low, "x_high": x_high},
    )


def correlated_features(
    *, n: int = 300, variance_ratio: float = 4.0, rotation: float = 30.0, noise: float = 0.0,
    n_features: int = 2, seed: int = 42,
) -> Dataset:
    """An elongated Gaussian cloud with a known principal direction - PCA lab."""
    gen = rng(seed)
    k = max(int(n_features), 2)
    sds = np.array([np.sqrt(variance_ratio)] + [1.0] * (k - 1))
    z = gen.standard_normal((n, k)) * sds
    theta = np.deg2rad(rotation)
    R = np.eye(k)
    R[0, 0] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    R[1, 1] = np.cos(theta)
    X = z @ R.T
    if noise:
        X = X + noise * gen.standard_normal((n, k))
    columns = {f"x{j + 1}": X[:, j] for j in range(k)}
    return Dataset(
        columns=columns,
        dgp=(
            f"Gaussian cloud with variance ratio {variance_ratio:g}:1 rotated by "
            f"{rotation:g} degrees;  n = {n};  {k} features"
        ),
        truth={"rotation": float(rotation), "variance_ratio": float(variance_ratio)},
        meta={"rotation_matrix": R, "k": k},
    )


def blobs(
    *, n: int = 300, n_clusters: int = 3, spread: float = 1.0, seed: int = 42,
    separation: float = 5.0,
) -> Dataset:
    """Isotropic clusters with a controllable overlap."""
    gen = rng(seed)
    k = max(int(n_clusters), 1)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centres = np.column_stack([separation * np.cos(angles), separation * np.sin(angles)])
    labels = gen.integers(0, k, n)
    X = centres[labels] + spread * gen.standard_normal((n, 2))
    return Dataset(
        columns={"x1": X[:, 0], "x2": X[:, 1], "label": labels.astype(float)},
        dgp=(
            f"{k} isotropic clusters, centre radius {separation:g}, "
            f"within-cluster sd {spread:g};  n = {n}"
        ),
        truth={"n_clusters": float(k)},
        meta={"centres": centres},
    )


def sparse_regression(
    *, n: int = 120, p: int = 20, n_active: int = 3, noise: float = 1.0,
    correlation: float = 0.0, signal: float = 2.0, seed: int = 42,
) -> Dataset:
    """High-dimensional design where only a few coefficients are non-zero."""
    gen = rng(seed)
    p, n_active = int(p), int(n_active)
    base = gen.standard_normal((n, p))
    if correlation:
        common = gen.standard_normal((n, 1))
        base = np.sqrt(1 - correlation) * base + np.sqrt(correlation) * common
    beta = np.zeros(p)
    beta[:n_active] = signal * np.sign(gen.standard_normal(n_active) + 0.1)
    y = base @ beta + noise * gen.standard_normal(n)
    columns = {"y": y}
    for j in range(p):
        columns[f"x{j + 1}"] = base[:, j]
    return Dataset(
        columns=columns,
        dgp=(
            f"y = X beta + e with {n_active} non-zero coefficients out of {p};  "
            f"|beta| = {signal:g};  sd(e) = {noise:g};  n = {n}"
        ),
        truth={"n_active": float(n_active), **{f"beta{j + 1}": float(b)
                                               for j, b in enumerate(beta)}},
        meta={"beta": beta, "p": p},
    )
