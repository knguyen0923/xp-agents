"""Friendly-fire reward penalty (spec 2026-11-02): charge an action that would damage the agent's own partner.

Works on the chosen action before the step, from the battle state the agent saw, using the matchup estimator, so an
immune partner (estimated fraction 0) is free and the number does not depend on what the opponent does.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np
from poke_env.battle import MoveCategory

FIRST_MOVE_ACTION = 7      # layout from xp/showdown/masks.py: 7 + 5 * k + (target + 2); Mega variants add 20
LAST_MOVE_ACTION = 46
N_TARGETS = 5
MEGA_OFFSET = 20
FF_BATTLE_CAP = 1.0        # total penalty per battle never exceeds this, so a win (return >= 0) always beats a clean loss (-1);
#                            that guarantee is for the undiscounted sum: with gamma 0.99 the discounted return can differ slightly


class PenaltyTally:
    """Per-battle friendly-fire totals for `n` envs: sum each env's per-step penalty until its battle is done.

    `recent()` is (mean total over the last 100 finished battles, how many); `totals()` is the cumulative (sum, count),
    so a caller can take a difference to measure only the battles finished in between (the held-out evaluation).
    """

    def __init__(self, n: int):
        self.running = np.zeros(n, dtype=np.float64)
        self.finished: deque = deque(maxlen=100)
        self.sum, self.count = 0.0, 0

    def add(self, penalties, dones) -> None:
        self.running += np.asarray(penalties, dtype=np.float64)
        for i in np.flatnonzero(np.asarray(dones) > 0):
            self.finished.append(float(self.running[i]))
            self.sum += float(self.running[i])
            self.count += 1
            self.running[i] = 0.0

    def recent(self) -> Tuple[float, int]:
        return (float(np.mean(self.finished)) if self.finished else float("nan")), len(self.finished)

    def totals(self) -> Tuple[float, int]:
        return self.sum, self.count


def capped_penalty(penalty: float, spent: float) -> float:
    """The part of `penalty` (zero or negative) that fits under the per-battle cap, given the positive total `spent` so far."""
    return max(penalty, -(FF_BATTLE_CAP - spent))


def decode_move_action(action: int) -> Optional[Tuple[int, int]]:
    """(move index 0..3, showdown target -2..2) for a move action, or None for pass, a switch or anything else."""
    if not FIRST_MOVE_ACTION <= action <= LAST_MOVE_ACTION:
        return None
    a = action - FIRST_MOVE_ACTION
    if a >= MEGA_OFFSET:
        a -= MEGA_OFFSET
    return a // N_TARGETS, a % N_TARGETS - 2


def hits_ally(move, target: int) -> bool:
    """A damaging move hits the partner when it is aimed at it, or when its target class is allAdjacent (Earthquake, Surf)."""
    # Zero base-power moves (Low Kick, Grass Knot: damage depends on weight) are not charged, by the spec.
    if move.category == MoveCategory.STATUS or move.base_power <= 0:
        return False
    return target in (-1, -2) or move.entry.get("target") == "allAdjacent"


def friendly_fire_penalty(battle, pos: int, action: int, estimator, scale: float) -> float:
    """`-scale * estimated damage fraction on the partner` for slot `pos` taking `action`; 0.0 when it does not hurt the partner."""
    decoded = decode_move_action(int(action))
    if decoded is None:
        return 0.0
    active, ally = battle.active_pokemon[pos], battle.active_pokemon[1 - pos]
    if active is None or active.fainted or ally is None or ally.fainted:
        return 0.0
    available = list(battle.available_moves[pos])      # same rule as xp/showdown/masks.py action_masks
    moves = (available if len(available) == 1 and available[0].id in ("struggle", "recharge") else list(active.moves.values()))[:4]
    k, target = decoded
    if k >= len(moves) or not hits_ally(moves[k], target):
        return 0.0
    est = estimator(active, ally, moves[k])
    return 0.0 if est is None else -scale * est[1]
