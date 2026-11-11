"""Tempo observation block (spec 2026-11-02): speed order, remaining turns of field effects, move flags.

The agent is taken to know what every move does (matchup spec); the numbers below are computed in code from battle state.
Foe Speed stats are unknown to poke-env, so they use the matchup spec's fallback spread (level 50, 0 EVs, 31 IVs, neutral).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from poke_env.battle import Field, Move, MoveCategory, Pokemon, SideCondition, Status

from xp.showdown.encode import N_MOVES, TEMPO_DIM
from xp.showdown.estimate import _fallback_stats
from xp.showdown.matchup import _live, _moves

TRICK_ROOM_TURNS = 5
TAILWIND_TURNS = 4
SCREEN_TURNS = 5
LIGHT_CLAY_SCREEN_TURNS = 8
SCREENS = (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL)
SETS_TRICK_ROOM = frozenset({"trickroom"})
SETS_TAILWIND = frozenset({"tailwind"})
SETS_SCREEN = frozenset({"reflect", "lightscreen", "auroraveil"})
N_FLAGS = 4
SPEED_DIM = 4
TURNS_DIM = 9
FLAGS_DIM = 4 * N_MOVES * N_FLAGS


def _boost_multiplier(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def effective_speed(mon: Pokemon, tailwind: bool, foe: bool) -> float:
    """Speed after boosts, paralysis, a known Choice Scarf and Tailwind. Foe (and any mon with unknown stats) uses the fallback spread."""
    known = mon.stats.get("spe") if mon.stats else None
    base = _fallback_stats(mon)["spe"] if (foe or not isinstance(known, (int, float))) else known
    speed = base * _boost_multiplier(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        speed *= 0.5
    if mon.item == "choicescarf":      # own items are known; a foe's Scarf counts once it has been revealed
        speed *= 1.5
    if tailwind:
        speed *= 2.0
    return float(speed)


def speed_order(own_actives: Sequence[Optional[Pokemon]], opp_actives: Sequence[Optional[Pokemon]],
                own_tailwind: bool, opp_tailwind: bool, trick_room: bool) -> np.ndarray:
    """For (my 0, my 1) x (foe 0, foe 1): +1 if mine moves first, -1 if theirs does, 0 for a tie or an absent mon. Trick Room flips it."""
    out = np.zeros(SPEED_DIM, dtype=np.float32)
    for i, mine in enumerate(list(own_actives)[:2]):
        for j, foe in enumerate(list(opp_actives)[:2]):
            if not (_live(mine) and _live(foe)):
                continue
            a, b = effective_speed(mine, own_tailwind, False), effective_speed(foe, opp_tailwind, True)
            if a == b:
                continue
            first = 1.0 if a > b else -1.0
            out[i * 2 + j] = -first if trick_room else first
    return out


def _left(start: int, duration: int, turn: int) -> float:
    return float(np.clip(start + duration - turn, 0, duration)) / duration


def turns_block(battle) -> np.ndarray:
    """Remaining turns / full duration: Trick Room, then own Tailwind and screens, then the foe's. poke-env stores each start turn."""
    out = np.zeros(TURNS_DIM, dtype=np.float32)
    turn = battle.turn
    if Field.TRICK_ROOM in battle.fields:
        out[0] = _left(battle.fields[Field.TRICK_ROOM], TRICK_ROOM_TURNS, turn)
    # scans the whole team, so a Light Clay holder that was not brought still counts (accepted approximation)
    own_clay = any(m.item == "lightclay" for m in battle.team.values())
    for base, conditions, clay in ((1, battle.side_conditions, own_clay), (5, battle.opponent_side_conditions, False)):
        if SideCondition.TAILWIND in conditions:
            out[base] = _left(conditions[SideCondition.TAILWIND], TAILWIND_TURNS, turn)
        for i, screen in enumerate(SCREENS):
            if screen in conditions:
                out[base + 1 + i] = _left(conditions[screen], LIGHT_CLAY_SCREEN_TURNS if clay else SCREEN_TURNS, turn)
    return out


def move_flags(move: Optional[Move]) -> np.ndarray:
    """sets_trick_room, sets_tailwind, sets_screen, hits_ally (a damaging allAdjacent move hits the partner whatever the target)."""
    if move is None:
        return np.zeros(N_FLAGS, dtype=np.float32)
    damaging = move.category != MoveCategory.STATUS and move.base_power > 0
    return np.array([move.id in SETS_TRICK_ROOM, move.id in SETS_TAILWIND, move.id in SETS_SCREEN,
                     damaging and move.entry.get("target") == "allAdjacent"], dtype=np.float32)


def flags_block(own_actives: Sequence[Optional[Pokemon]], opp_actives: Sequence[Optional[Pokemon]]) -> np.ndarray:
    """Four flags for each of the four move slots of each of my 2 actives and the 2 opposing actives (foe slots stay 0 until revealed)."""
    mons: List[Optional[Pokemon]] = (list(own_actives) + [None] * 2)[:2] + (list(opp_actives) + [None] * 2)[:2]
    rows = []
    for mon in mons:
        rows.extend(move_flags(m) for m in _moves(mon if _live(mon) else None))
    return np.concatenate(rows).astype(np.float32)


def tempo_block(battle) -> np.ndarray:
    own, opp = list(battle.active_pokemon), list(battle.opponent_active_pokemon)
    out = np.concatenate([
        speed_order(own, opp, SideCondition.TAILWIND in battle.side_conditions,
                    SideCondition.TAILWIND in battle.opponent_side_conditions, Field.TRICK_ROOM in battle.fields),
        turns_block(battle),
        flags_block(own, opp),
    ]).astype(np.float32)
    assert out.shape == (TEMPO_DIM,)
    return out
