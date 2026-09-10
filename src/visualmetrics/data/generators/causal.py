"""Causal and panel data-generating processes.

Potential outcomes, panels with unobserved effects, difference-in-differences
(including staggered adoption and heterogeneous effects), and regression
discontinuity.
"""

from __future__ import annotations

import numpy as np

from ...simulation.random import rng
from .regression import Dataset

__all__ = [
    "potential_outcomes",
    "panel_dataset",
    "did_panel",
    "rdd_dataset",
    "collider_dataset",
]


def potential_outcomes(
    *,
    n: int = 400,
    ate: float = 2.0,
    confounding: float = 1.5,
    selection: float = 1.0,
    heterogeneity: float = 0.0,
    noise: float = 1.0,
    randomized: bool = False,
    seed: int = 42,
) -> Dataset:
    """Y(0), Y(1), a confounder X and a treatment assignment rule.

    When ``randomized`` is set, treatment ignores X and the naive difference in
    means is unbiased for the ATE; otherwise selection on X creates the gap
    between association and causation.
    """
    gen = rng(seed)
    x = gen.standard_normal(n)
    tau = ate + heterogeneity * x
    y0 = 1.0 + confounding * x + noise * gen.standard_normal(n)
    y1 = y0 + tau
    if randomized:
        propensity = np.full(n, 0.5)
    else:
        propensity = 1.0 / (1.0 + np.exp(-selection * x))
    d = (gen.random(n) < propensity).astype(float)
    y = np.where(d > 0, y1, y0)
    att = float(np.mean(tau[d > 0])) if d.sum() else float("nan")
    return Dataset(
        columns={"y": y, "d": d, "x": x, "y0": y0, "y1": y1, "tau": tau,
                 "propensity": propensity},
        dgp=(
            f"Y(0) = 1 + {confounding:g}*X + e;  tau = {ate:g}"
            + (f" + {heterogeneity:g}*X" if heterogeneity else "")
            + (";  D randomized (P = 0.5)" if randomized
               else f";  P(D=1|X) = logistic({selection:g}*X)")
            + f";  n = {n}"
        ),
        truth={
            "ate": float(np.mean(tau)),
            "att": att,
            "naive_difference": float(np.mean(y[d > 0]) - np.mean(y[d == 0])) if d.sum() else 0.0,
        },
        notes=() if randomized else ("assumption_exchangeability",),
        meta={"randomized": bool(randomized)},
    )


def panel_dataset(
    *,
    n_units: int = 40,
    n_periods: int = 8,
    beta: float = 1.0,
    effect_x_correlation: float = 0.8,
    sigma_alpha: float = 1.0,
    sigma_e: float = 1.0,
    time_effects: float = 0.0,
    seed: int = 42,
) -> Dataset:
    """y_it = beta*x_it + alpha_i + lambda_t + e_it with corr(alpha_i, x_it) tunable.

    ``effect_x_correlation`` is exactly the knob that decides between fixed and
    random effects: at 0 the RE assumption holds, above 0 RE is inconsistent.
    """
    gen = rng(seed)
    n_units, n_periods = int(n_units), int(n_periods)
    alpha = sigma_alpha * gen.standard_normal(n_units)
    lam = time_effects * gen.standard_normal(n_periods)
    rho = float(np.clip(effect_x_correlation, -0.99, 0.99))
    unit = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    x_noise = gen.standard_normal(n_units * n_periods)
    x = rho * np.repeat(alpha, n_periods) + np.sqrt(max(1 - rho**2, 0.0)) * x_noise
    e = sigma_e * gen.standard_normal(n_units * n_periods)
    y = beta * x + np.repeat(alpha, n_periods) + lam[period] + e
    return Dataset(
        columns={"unit": unit.astype(float), "period": period.astype(float),
                 "y": y, "x": x, "alpha": np.repeat(alpha, n_periods)},
        dgp=(
            f"y_it = {beta:g}*x_it + alpha_i"
            + (" + lambda_t" if time_effects else "")
            + f" + e_it;  corr(alpha_i, x_it) = {rho:g};  "
            f"N = {n_units}, T = {n_periods};  sd(alpha) = {sigma_alpha:g}, sd(e) = {sigma_e:g}"
        ),
        truth={"beta": float(beta), "effect_x_correlation": rho},
        notes=("assumption_random_effects",) if abs(rho) > 0.05 else (),
        meta={"n_units": n_units, "n_periods": n_periods},
    )


def did_panel(
    *,
    n_units: int = 60,
    n_periods: int = 10,
    treat_period: int = 5,
    effect: float = 2.0,
    treated_share: float = 0.5,
    pretrend: float = 0.0,
    anticipation: float = 0.0,
    heterogeneity: float = 0.0,
    staggered: bool = False,
    dynamic_growth: float = 0.0,
    noise: float = 1.0,
    seed: int = 42,
) -> Dataset:
    """Difference-in-differences panel with the failure modes that matter.

    ``pretrend`` breaks parallel trends, ``anticipation`` moves the effect
    before adoption, ``staggered`` + ``heterogeneity`` produces the setting
    where two-way fixed effects is biased by negative weighting.
    """
    gen = rng(seed)
    n_units, n_periods = int(n_units), int(n_periods)
    n_treated = max(1, int(round(treated_share * n_units)))
    treated = np.zeros(n_units)
    treated[:n_treated] = 1.0
    if staggered:
        cohorts = np.linspace(max(treat_period - 2, 1), min(treat_period + 3, n_periods - 1),
                              num=max(n_treated, 1))
        adoption = np.full(n_units, np.inf)
        adoption[:n_treated] = np.round(cohorts)
    else:
        adoption = np.where(treated > 0, float(treat_period), np.inf)

    unit_fe = gen.normal(0, 1.0, n_units)
    unit_effect = effect + heterogeneity * (np.arange(n_units) / max(n_units - 1, 1) - 0.5) * 2

    rows_unit, rows_time, rows_y, rows_d, rows_treated, rows_rel = [], [], [], [], [], []
    for i in range(n_units):
        for t in range(n_periods):
            common_trend = 0.15 * t
            extra_trend = pretrend * t * treated[i]
            post = 1.0 if t >= adoption[i] else 0.0
            rel = t - adoption[i] if np.isfinite(adoption[i]) else np.nan
            te = 0.0
            if post:
                te = unit_effect[i] * (1.0 + dynamic_growth * (t - adoption[i]))
            elif anticipation and np.isfinite(adoption[i]) and adoption[i] - t == 1:
                te = anticipation * unit_effect[i]
            y = (
                5.0 + unit_fe[i] + common_trend + extra_trend + te
                + noise * gen.standard_normal()
            )
            rows_unit.append(i)
            rows_time.append(t)
            rows_y.append(y)
            rows_d.append(post)
            rows_treated.append(treated[i])
            rows_rel.append(rel)

    notes: list[str] = []
    if pretrend:
        notes.append("assumption_parallel_trends")
    if anticipation:
        notes.append("assumption_no_anticipation")
    if staggered and heterogeneity:
        notes.append("assumption_homogeneous_effects")

    return Dataset(
        columns={
            "unit": np.asarray(rows_unit, dtype=float),
            "period": np.asarray(rows_time, dtype=float),
            "y": np.asarray(rows_y, dtype=float),
            "d": np.asarray(rows_d, dtype=float),
            "treated": np.asarray(rows_treated, dtype=float),
            "rel_time": np.asarray(rows_rel, dtype=float),
        },
        dgp=(
            f"y_it = unit FE + 0.15*t + tau*D_it + e_it;  tau = {effect:g}"
            + (f" (+/- {heterogeneity:g} across units)" if heterogeneity else "")
            + (f";  differential pre-trend {pretrend:g} per period" if pretrend else "")
            + (f";  anticipation {anticipation:g} one period early" if anticipation else "")
            + (";  staggered adoption" if staggered else f";  common adoption at t = {treat_period}")
            + f";  N = {n_units}, T = {n_periods}"
        ),
        truth={
            "effect": float(effect),
            "att": float(np.mean(unit_effect[:n_treated])),
            "pretrend": float(pretrend),
        },
        notes=tuple(notes),
        meta={"treat_period": treat_period, "staggered": bool(staggered),
              "adoption": adoption, "n_treated": n_treated},
    )


def rdd_dataset(
    *,
    n: int = 500,
    cutoff: float = 0.0,
    effect: float = 2.0,
    slope_left: float = 1.0,
    slope_right: float = 1.0,
    curvature: float = 0.0,
    noise: float = 1.0,
    manipulation: float = 0.0,
    fuzzy: float = 1.0,
    seed: int = 42,
) -> Dataset:
    """Sharp or fuzzy regression discontinuity with optional running-variable sorting."""
    gen = rng(seed)
    x = gen.uniform(-3, 3, n)
    if manipulation:
        push = (np.abs(x - cutoff) < 0.5) & (x < cutoff)
        x = np.where(push & (gen.random(n) < manipulation), x + 0.6, x)
    above = (x >= cutoff).astype(float)
    if fuzzy < 1.0:
        prob = np.where(above > 0, fuzzy, 1.0 - fuzzy)
        d = (gen.random(n) < prob).astype(float)
    else:
        d = above
    base = np.where(
        x < cutoff, slope_left * (x - cutoff), slope_right * (x - cutoff)
    ) + curvature * (x - cutoff) ** 2
    y = 3.0 + base + effect * d + noise * gen.standard_normal(n)
    return Dataset(
        columns={"y": y, "x": x, "d": d, "above": above},
        dgp=(
            f"y = f(x) + {effect:g}*D + e;  D = 1(x >= {cutoff:g})"
            + (f" with compliance {fuzzy:g} (fuzzy)" if fuzzy < 1.0 else " (sharp)")
            + (f";  running variable manipulated with probability {manipulation:g}"
               if manipulation else "")
            + f";  n = {n}"
        ),
        truth={"effect": float(effect), "cutoff": float(cutoff)},
        notes=("assumption_no_manipulation",) if manipulation else (),
        meta={"fuzzy": fuzzy < 1.0},
    )


def collider_dataset(
    *, n: int = 400, a_to_c: float = 1.0, b_to_c: float = 1.0, a_to_b: float = 0.0,
    noise: float = 1.0, seed: int = 42,
) -> Dataset:
    """A -> C <- B: conditioning on the collider C induces A-B association."""
    gen = rng(seed)
    a = gen.standard_normal(n)
    b = a_to_b * a + np.sqrt(max(1 - a_to_b**2, 0.0)) * gen.standard_normal(n)
    c = a_to_c * a + b_to_c * b + noise * gen.standard_normal(n)
    return Dataset(
        columns={"a": a, "b": b, "c": c},
        dgp=(
            f"A -> C <- B;  C = {a_to_c:g}*A + {b_to_c:g}*B + e"
            + (f";  A -> B with coefficient {a_to_b:g}" if a_to_b else "")
            + f";  n = {n}"
        ),
        truth={"a_b_marginal": float(a_to_b)},
    )
