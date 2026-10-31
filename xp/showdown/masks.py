from __future__ import annotations


import numpy as np
from poke_env.environment import DoublesEnv

N_ACTIONS = 107
PASS = 0
MEGA_ACTIONS = frozenset(a for a in range(27, 47))  # "move k and mega evolve", k = 1..4, 5 targets each


def _legal(action: int, battle, pos: int) -> bool:
    """Exactly poke-env's own per-slot validity rule (fake=False raises AssertionError on an invalid action)."""
    try:
        DoublesEnv._action_to_order_individual(np.int64(action), battle, False, pos)
    except Exception:
        return False
    return True


def slot_needs_action(battle, pos: int) -> bool:
    """True when this slot must choose something other than pass."""
    mon = battle.active_pokemon[pos]
    if any(battle.force_switch):
        return bool(battle.force_switch[pos]) and bool(battle.available_switches[pos])
    return mon is not None and not mon.fainted


def reference_masks(battle) -> np.ndarray:
    """Slow reference: ask poke-env's decoder about every action. `action_masks` must equal this on every real state."""
    out = np.zeros((2, N_ACTIONS), dtype=bool)
    for pos in range(2):
        if not slot_needs_action(battle, pos):
            out[pos, PASS] = True
            continue
        for a in range(1, N_ACTIONS):
            out[pos, a] = _legal(a, battle, pos)
        if not out[pos].any():  # nothing legal (for example a struggle turn the decoder rejects): fall back to pass
            out[pos, PASS] = True
    return out


def action_masks(battle) -> np.ndarray:
    """(2, N_ACTIONS) bool: which actions are legal for each slot on its own. Pass (0) is legal only for a slot with nothing to do.

    Built from the battle's own move, target and switch lists (about 100x faster than asking the decoder 212 times).
    """
    out = np.zeros((2, N_ACTIONS), dtype=bool)
    for pos in range(2):
        if not slot_needs_action(battle, pos):
            out[pos, PASS] = True
            continue
        active = battle.active_pokemon[pos]
        if not battle.trapped[pos] and battle.available_switches[pos]:
            ok = {p.base_species for p in battle.available_switches[pos]}
            for i, mon in enumerate(list(battle.team.values())[:6]):
                out[pos, 1 + i] = mon.base_species in ok
        if not battle.force_switch[pos] and active is not None:
            available = {m.id: m for m in battle.available_moves[pos]}
            moves = list(battle.available_moves[pos]) if (
                len(available) == 1 and next(iter(available)) in ("struggle", "recharge")) else list(active.moves.values())
            for k, mv in enumerate(moves[:4]):
                if mv.id not in available:
                    continue
                for target in battle.get_possible_showdown_targets(available[mv.id], active):
                    base = 7 + 5 * k + (target + 2)
                    out[pos, base] = True
                    if battle.can_mega_evolve[pos]:
                        out[pos, base + 20] = True
        if not out[pos].any():
            out[pos, PASS] = True
    return out


def restrict_second(mask1: np.ndarray, first_action: int) -> np.ndarray:
    """Slot 1's mask once slot 0 has chosen: no second Mega, no switching in the same Pokemon (pass stays legal if it was)."""
    m = mask1.copy()
    if first_action in MEGA_ACTIONS:
        for a in MEGA_ACTIONS:
            m[a] = False
    if 1 <= first_action <= 6:
        m[first_action] = False
    if not m.any():
        m[PASS] = True
    return m
