# xp/crafter/stats.py
from __future__ import annotations

import collections
from typing import Dict, Iterable

import numpy as np

# The 22 Crafter achievements, in crafter.constants.achievements order (checked by a test marked `crafter`).
ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling", "collect_stone", "collect_wood",
    "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe",
    "make_stone_sword", "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone", "place_table", "wake_up",
)


def nan_to_zero(x: float) -> float:
    return 0.0 if x != x else float(x)


class EpisodeStats:
    """Distinct achievements per finished episode over a sliding window of the last `window` episodes.

    `ach()` is NaN until the window is full (counted as 0 in every AUC and budget rule, as in MiniGrid's `success_100`).
    """

    def __init__(self, window: int = 100):
        self.window = int(window)
        self.eps = collections.deque(maxlen=self.window)
        self.total = 0

    def add(self, unlocked: Iterable[str]) -> None:
        self.eps.append(frozenset(unlocked))
        self.total += 1

    def full(self) -> bool:
        return len(self.eps) >= self.window

    def ach(self) -> float:
        return float(np.mean([len(e) for e in self.eps])) if self.full() else float("nan")

    def rates(self, names: Iterable[str] = ACHIEVEMENTS) -> Dict[str, float]:
        n = max(1, len(self.eps))
        return {a: sum(a in e for e in self.eps) / n for a in names}


def crafter_score(rates: Dict[str, float]) -> float:
    """Official Crafter score: geometric mean of (1 + success rate in percent) over the achievements, minus 1."""
    vals = np.array([100.0 * float(r) for r in rates.values()])
    return float(np.exp(np.mean(np.log(1.0 + vals))) - 1.0)
