"""Deterministic random-number management.

Every lab draws from a generator derived from the lab state, so the same state
always produces the same picture - on any machine, in any language.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

__all__ = ["rng", "derive_seed", "spawn", "SIM_PRESETS", "preset_reps"]

#: repetition budgets for the three simulation presets (blueprint section 46)
SIM_PRESETS: dict[str, float] = {"fast": 0.25, "teaching": 1.0, "high_precision": 4.0}


def derive_seed(seed: int, *tags: Any) -> int:
    """Derive a stable child seed from a parent seed plus string tags.

    Two different animations inside one lab must not share a stream, otherwise
    an unrelated slider silently reshuffles another panel.
    """
    payload = "|".join([str(seed), *(str(t) for t in tags)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng(seed: int | None = None, *tags: Any) -> np.random.Generator:
    """Return a PCG64 generator for ``seed`` (plus optional stream tags)."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(derive_seed(int(seed), *tags) if tags else int(seed))


def spawn(seed: int, count: int) -> list[np.random.Generator]:
    """Independent child generators - safe for parallel Monte Carlo."""
    parent = np.random.SeedSequence(int(seed))
    return [np.random.default_rng(child) for child in parent.spawn(count)]


def preset_reps(base: int, preset: str = "teaching", *, minimum: int = 20,
                maximum: int = 200_000) -> int:
    """Scale a repetition count by the active precision preset."""
    factor = SIM_PRESETS.get(preset, 1.0)
    return int(np.clip(round(base * factor), minimum, maximum))
