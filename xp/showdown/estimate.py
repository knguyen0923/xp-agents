from __future__ import annotations

from typing import Optional, Tuple

from poke_env.battle import Move, MoveCategory, Pokemon
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.stats import _raw_hp, _raw_stat

# Stated assumption (spec): the opponent's real spreads are hidden, so fill missing stats from base stats at level 50
# with no EVs, perfect IVs and a neutral nature. Probe (docs/spikes/showdown_matchup_probe.md): every foe stat is None,
# own stats are real; foe abilities and items are not assumed.
FALLBACK_LEVEL = 50
FALLBACK_IV = 31
FALLBACK_EV = 0
FALLBACK_NATURE = "serious"   # neutral; the formula uses multiplier 1.0
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


def _fallback_stats(mon: Pokemon) -> dict:
    """Level-50 / 0 EV / 31 IV / neutral-nature stats from the mon's CURRENT `base_stats` (a Mega Evolved foe keeps its
    base `species` in poke-env and changes only `base_stats`, so computing from the species would go stale)."""
    base = mon.base_stats
    out = {"hp": 1 if mon.species == "shedinja" else _raw_hp(base["hp"], FALLBACK_EV, FALLBACK_IV, FALLBACK_LEVEL)}
    for k in _STAT_ORDER[1:]:
        out[k] = _raw_stat(base[k], FALLBACK_EV, FALLBACK_IV, FALLBACK_LEVEL, 1.0)   # "serious" is neutral
    return out


def _fill_stats(mon: Pokemon, foe: bool = False) -> None:
    """Give `mon.stats` numbers if any are missing, so the calculator's assertions hold.
    Own mons: real stats are never overwritten, only missing ones are filled. Foes (poke-env never knows their stats):
    recomputed from the current `base_stats` on every call, so a foe that Mega Evolves follows its new base stats.
    Fills in place; the main encoder reads `base_stats` only, so this does not leak into the observation."""
    fallback = _fallback_stats(mon)
    if foe:
        mon.stats = fallback
        return
    if mon.stats and all(isinstance(v, (int, float)) for v in mon.stats.values()):
        return
    known = mon.stats or {}
    mon.stats = {k: (known[k] if isinstance(known.get(k), (int, float)) else fallback[k]) for k in _STAT_ORDER}


class PokeEnvEstimator:
    """`(attacker, defender, move) -> (type multiplier, damage fraction of the defender's current HP)` for one battle."""

    def __init__(self, battle) -> None:
        self.battle = battle
        self.errors = 0

    def _is_foe(self, mon: Pokemon) -> bool:
        return any(value is mon for value in self.battle.opponent_team.values())

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
            _fill_stats(attacker, self._is_foe(attacker))
            _fill_stats(defender, self._is_foe(defender))
            lo, hi = calculate_damage(self._identifier(attacker), self._identifier(defender), move, self.battle)
        except Exception:
            self.errors += 1
            return None
        if hi == 0 and multiplier > 0:
            multiplier = 0.0   # ability immunity or Protect: the calculator sees no damage
        hp_points = max(1.0, defender.stats["hp"] * defender.current_hp_fraction)
        return multiplier, min(1.0, ((lo + hi) / 2.0) / hp_points)
