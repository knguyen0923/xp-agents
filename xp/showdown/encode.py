from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from poke_env.battle import Field, Move, MoveCategory, Pokemon, PokemonType, SideCondition, Status, Weather

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
STATUSES = [s for s in Status if s is not Status.FNT]
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")
SPREAD_TARGETS = {"allAdjacent", "allAdjacentFoes", "all"}
SLOTS = 6                      # per side: 2 actives, then up to 4 others
N_MOVES = 4
MOVE_DIM = 3 + len(TYPES) + 3 + 2          # power, accuracy, priority | type | category | is_protect, is_spread
MON_DIM = 3 + len(STATUSES) + len(BOOSTS) + len(TYPES) + len(STATS) + 1 + N_MOVES * MOVE_DIM
FIELD_DIM = len(Weather) + len(Field) + 2 * len(SideCondition) + 1
OBS_DIM = 2 * SLOTS * MON_DIM + FIELD_DIM
MATCHUP_DIM = 2 * (2 * N_MOVES * 2) * 2 + 2 * N_MOVES   # outgoing 32 + incoming 32 + 8 revealed flags = 72


def obs_dim(matchup: bool = False) -> int:
    """Length of the observation vector; the matchup block (spec 2026-10-29) is appended only when switched on."""
    return OBS_DIM + (MATCHUP_DIM if matchup else 0)


def _onehot(item, order: list) -> np.ndarray:
    out = np.zeros(len(order), dtype=np.float32)
    if item in order:
        out[order.index(item)] = 1.0
    return out


def encode_move(move: Optional[Move]) -> np.ndarray:
    if move is None:
        return np.zeros(MOVE_DIM, dtype=np.float32)
    cats = [MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS]
    return np.concatenate([
        np.array([move.base_power / 150.0, float(move.accuracy), move.priority / 5.0], dtype=np.float32),
        _onehot(move.type, TYPES),
        _onehot(move.category, cats),
        np.array([float(move.id in ("protect", "detect", "spikyshield", "banefulbunker", "kingsshield")),
                  float(move.entry.get("target") in SPREAD_TARGETS)], dtype=np.float32),
    ])


def encode_pokemon(mon: Optional[Pokemon], can_mega: bool = False) -> np.ndarray:
    """Fixed-size features from the public enums and numbers only: no species id, so unseen species still encode."""
    if mon is None:
        return np.zeros(MON_DIM, dtype=np.float32)
    moves = list(mon.moves.values())[:N_MOVES]
    moves += [None] * (N_MOVES - len(moves))
    base = np.array([mon.base_stats.get(k, 0) / 255.0 for k in STATS], dtype=np.float32)
    boosts = np.array([mon.boosts.get(k, 0) / 6.0 for k in BOOSTS], dtype=np.float32)
    types = np.clip(sum((_onehot(t, TYPES) for t in mon.types if t is not None), np.zeros(len(TYPES), dtype=np.float32)), 0, 1)
    return np.concatenate([
        np.array([float(mon.current_hp_fraction), float(mon.fainted), float(mon.active)], dtype=np.float32),
        _onehot(mon.status, STATUSES), boosts, types, base,
        np.array([float(can_mega)], dtype=np.float32),
        *[encode_move(m) for m in moves],
    ])


def _side(team: Iterable[Pokemon], actives: List[Optional[Pokemon]], mega_flags: List[bool]) -> np.ndarray:
    active_ids = {id(m) for m in actives if m is not None}
    others = [m for m in team if id(m) not in active_ids][: SLOTS - 2]
    rows = [encode_pokemon(m, f) for m, f in zip(actives, mega_flags)]
    rows += [encode_pokemon(None)] * (2 - len(rows))
    rows += [encode_pokemon(m) for m in others]
    rows += [encode_pokemon(None)] * (SLOTS - len(rows))
    return np.concatenate(rows)


def _conditions(table: dict, order: list) -> np.ndarray:
    return np.array([float(k in table) for k in order], dtype=np.float32)


def encode_battle(battle) -> np.ndarray:
    """Flat float32 vector of length OBS_DIM for a DoubleBattle (own side, opponent side, field)."""
    own = _side(battle.team.values(), list(battle.active_pokemon), list(battle.can_mega_evolve))
    opp = _side(battle.opponent_team.values(), list(battle.opponent_active_pokemon), [False, False])
    field = np.concatenate([
        _conditions(battle.weather, list(Weather)), _conditions(battle.fields, list(Field)),
        _conditions(battle.side_conditions, list(SideCondition)),
        _conditions(battle.opponent_side_conditions, list(SideCondition)),
        np.array([min(battle.turn, 40) / 40.0], dtype=np.float32),
    ])
    out = np.concatenate([own, opp, field]).astype(np.float32)
    assert out.shape == (OBS_DIM,)
    return out
