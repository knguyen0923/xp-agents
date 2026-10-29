from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from poke_env.battle import Move, Pokemon

from xp.showdown.encode import MATCHUP_DIM, N_MOVES

# (attacker, defender, move) -> (type multiplier, expected damage as a fraction of the defender's current HP), or None
# when the move does no damage (status, zero power). Supplied by `xp.showdown.estimate` in a real battle, by a stub in tests.
Estimator = Callable[[Pokemon, Pokemon, Move], Optional[Tuple[float, float]]]

_HALF = 2 * N_MOVES * 2 * 2   # one direction: 2 attackers x 4 moves x 2 targets x 2 numbers = 32


def _live(mon: Optional[Pokemon]) -> bool:
    return mon is not None and not mon.fainted


def _moves(mon: Optional[Pokemon]) -> List[Optional[Move]]:
    """The same move order and cap the main encoder uses; unrevealed foe slots stay None."""
    moves = list(mon.moves.values())[:N_MOVES] if mon is not None else []
    return moves + [None] * (N_MOVES - len(moves))


def _direction(attackers: Sequence[Optional[Pokemon]], targets: Sequence[Optional[Pokemon]], estimate: Estimator) -> np.ndarray:
    out = np.zeros(_HALF, dtype=np.float32)
    for a, attacker in enumerate(attackers[:2]):
        if not _live(attacker):
            continue
        for m, move in enumerate(_moves(attacker)):
            if move is None:
                continue
            for t, target in enumerate(targets[:2]):
                if not _live(target):
                    continue
                est = estimate(attacker, target, move)
                if est is None:
                    continue
                i = ((a * N_MOVES + m) * 2 + t) * 2
                out[i], out[i + 1] = min(est[0] / 4.0, 1.0), min(est[1], 1.0)
    return out


def matchup_block(own_actives: Sequence[Optional[Pokemon]], opp_actives: Sequence[Optional[Pokemon]], estimate: Estimator) -> np.ndarray:
    """Outgoing (my moves into foes), incoming (foes' REVEALED moves into me), then one revealed flag per foe move slot."""
    own = list(own_actives) + [None] * (2 - len(own_actives))
    opp = list(opp_actives) + [None] * (2 - len(opp_actives))
    flags = np.zeros(2 * N_MOVES, dtype=np.float32)
    for f, foe in enumerate(opp[:2]):
        if _live(foe):
            for m, move in enumerate(_moves(foe)):
                flags[f * N_MOVES + m] = float(move is not None)
    out = np.concatenate([_direction(own, opp, estimate), _direction(opp, own, estimate), flags]).astype(np.float32)
    assert out.shape == (MATCHUP_DIM,)
    return out
