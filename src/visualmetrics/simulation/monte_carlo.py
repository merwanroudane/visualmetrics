"""Monte Carlo engine.

Outputs the quantities that actually teach something about an estimator:
bias, variance, MSE, coverage, empirical size and empirical power - each with a
simulation standard error so the learner can tell noise from signal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .random import spawn

__all__ = ["MonteCarloResult", "monte_carlo", "coverage_study", "rejection_study"]


@dataclass
class MonteCarloResult:
    """Replicated draws of one or more scalar statistics."""

    draws: dict[str, np.ndarray]
    truth: dict[str, float] = field(default_factory=dict)
    reps: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def values(self, name: str | None = None) -> np.ndarray:
        if name is None:
            name = next(iter(self.draws))
        return self.draws[name]

    def bias(self, name: str | None = None) -> float:
        name = name or next(iter(self.draws))
        if name not in self.truth:
            return float("nan")
        return float(np.nanmean(self.draws[name]) - self.truth[name])

    def variance(self, name: str | None = None) -> float:
        name = name or next(iter(self.draws))
        return float(np.nanvar(self.draws[name], ddof=1))

    def mse(self, name: str | None = None) -> float:
        name = name or next(iter(self.draws))
        if name not in self.truth:
            return float("nan")
        return float(np.nanmean((self.draws[name] - self.truth[name]) ** 2))

    def mc_se(self, name: str | None = None) -> float:
        """Standard error of the simulated mean - honesty about simulation noise."""
        name = name or next(iter(self.draws))
        v = self.draws[name]
        return float(np.nanstd(v, ddof=1) / np.sqrt(np.sum(np.isfinite(v))))

    def summary(self, name: str | None = None) -> dict[str, float]:
        name = name or next(iter(self.draws))
        v = self.draws[name]
        return {
            "mean": float(np.nanmean(v)),
            "sd": float(np.nanstd(v, ddof=1)),
            "bias": self.bias(name),
            "variance": self.variance(name),
            "mse": self.mse(name),
            "mc_se": self.mc_se(name),
            "reps": int(np.sum(np.isfinite(v))),
        }


def monte_carlo(
    experiment: Callable[[np.random.Generator], dict[str, float] | float],
    reps: int,
    seed: int,
    *,
    truth: dict[str, float] | None = None,
    names: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> MonteCarloResult:
    """Run ``experiment`` ``reps`` times with independent seeded generators."""
    reps = max(int(reps), 1)
    gens = spawn(seed, reps)
    collected: dict[str, list[float]] = {}
    for i, gen in enumerate(gens):
        out = experiment(gen)
        if not isinstance(out, dict):
            out = {(names[0] if names else "statistic"): float(out)}
        for key, value in out.items():
            collected.setdefault(key, []).append(float(value))
        if progress is not None and (i % max(reps // 20, 1) == 0):
            progress(i, reps)
    draws = {k: np.asarray(v, dtype=float) for k, v in collected.items()}
    return MonteCarloResult(draws=draws, truth=dict(truth or {}), reps=reps)


def coverage_study(
    experiment: Callable[[np.random.Generator], tuple[float, float]],
    reps: int,
    seed: int,
    true_value: float,
) -> dict[str, Any]:
    """Repeated-sampling coverage of an interval procedure."""
    lows, highs = [], []
    for gen in spawn(seed, max(int(reps), 1)):
        lo, hi = experiment(gen)
        lows.append(float(lo))
        highs.append(float(hi))
    lo_arr = np.asarray(lows)
    hi_arr = np.asarray(highs)
    covered = (lo_arr <= true_value) & (true_value <= hi_arr)
    p = float(covered.mean())
    n = covered.size
    return {
        "low": lo_arr,
        "high": hi_arr,
        "covered": covered,
        "coverage": p,
        "mc_se": float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n else float("nan"),
        "width": float(np.mean(hi_arr - lo_arr)),
        "reps": n,
    }


def rejection_study(
    experiment: Callable[[np.random.Generator], float],
    reps: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Empirical rejection frequency = size under H0, power under H1."""
    pvals = np.asarray([float(experiment(g)) for g in spawn(seed, max(int(reps), 1))])
    rejected = pvals < alpha
    p = float(rejected.mean())
    n = pvals.size
    return {
        "p_values": pvals,
        "rejected": rejected,
        "rejection_rate": p,
        "mc_se": float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n else float("nan"),
        "alpha": float(alpha),
        "reps": n,
    }
