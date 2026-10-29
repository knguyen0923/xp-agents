from __future__ import annotations

from typing import Optional, Tuple

from poke_env.battle import Move, MoveCategory, Pokemon
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

# Stated assumption (spec): the opponent's real spreads are hidden, so fill missing stats from base stats at level 50
# with no EVs, perfect IVs and a neutral nature. Probe (docs/spikes/showdown_matchup_probe.md): every foe stat is None,
# own stats are real; foe abilities and items are not assumed.
FALLBACK_LEVEL = 50
FALLBACK_IV = 31
FALLBACK_EV = 0
FALLBACK_NATURE = "serious"
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


def _fill_stats(mon: Pokemon) -> None:
    """Give `mon.stats` numbers if any are missing, so the calculator's assertions hold. Never overwrites known stats.
    Fills foe stats in place; the main encoder reads `base_stats` only, so this does not leak into the observation."""
    if mon.stats and all(isinstance(v, (int, float)) for v in mon.stats.values()):
        return
    raw = compute_raw_stats(mon.species, [FALLBACK_EV] * 6, [FALLBACK_IV] * 6, FALLBACK_LEVEL, FALLBACK_NATURE, GenData.from_gen(9))
    fallback = {k: int(v) for k, v in zip(_STAT_ORDER, raw)}
    known = mon.stats or {}
    mon.stats = {k: (known[k] if isinstance(known.get(k), (int, float)) else fallback[k]) for k in _STAT_ORDER}


class PokeEnvEstimator:
    """`(attacker, defender, move) -> (type multiplier, damage fraction of the defender's current HP)` for one battle."""

    def __init__(self, battle) -> None:
        self.battle = battle
        self.errors = 0

    def _identifier(self, mon: Pokemon) -> str:
        for team in (self.battle.team, self.battle.opponent_team):
            for key, value in team.items():
                if value is mon:
                    return key
        raise KeyError(mon.species)

    def __call__(self, attacker: Pokemon, defender: Pokemon, move: Move) -> Optional[Tuple[float, float]]:
        if move.category == MoveCategory.STATUS or move.base_power <= 0:
            return None
        multiplier = float(defender.damage_multiplier(move))
        try:
            _fill_stats(attacker)
            _fill_stats(defender)
            lo, hi = calculate_damage(self._identifier(attacker), self._identifier(defender), move, self.battle)
        except Exception:
            self.errors += 1
            return None
        if hi == 0 and multiplier > 0:
            multiplier = 0.0   # ability immunity or Protect: the calculator sees no damage
        hp_points = max(1.0, defender.stats["hp"] * defender.current_hp_fraction)
        return multiplier, min(1.0, ((lo + hi) / 2.0) / hp_points)
