from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Status  # noqa: E402

from xp.showdown import encode as E  # noqa: E402
from xp.showdown.matchup import matchup_block  # noqa: E402


def mon(species: str, *moves: str) -> Pokemon:
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def stub(mult=2.0, frac=0.5, damaging=("earthquake", "surf")):
    def estimate(attacker, defender, move):
        return (mult, frac) if move.id in damaging else None
    return estimate


def test_shape_and_dtype():
    out = matchup_block([mon("garchomp", "earthquake")], [mon("rotom")], stub())
    assert out.shape == (E.MATCHUP_DIM,) and out.dtype == np.float32


def test_outgoing_values_and_scaling():
    own = [mon("garchomp", "earthquake", "protect"), None]
    foes = [mon("rotom"), mon("heatran")]
    out = matchup_block(own, foes, stub(mult=2.0, frac=0.5))
    # own active 0, move slot 0 (earthquake): both foes, (mult / 4, fraction)
    assert out[0:4].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    # move slot 1 (protect) is a status move: zeros
    assert not out[4:8].any()
    # own active 1 is empty: zeros
    assert not out[16:32].any()


def test_damage_fraction_is_capped_at_one_and_immunity_reads_zero():
    out = matchup_block([mon("garchomp", "earthquake")], [mon("rotom"), None], stub(mult=0.0, frac=3.0))
    assert out[0] == 0.0 and out[1] == 1.0          # mult 0 (immune), fraction capped
    assert not out[2:4].any()                        # absent target


def test_fainted_target_is_zeros():
    dead = mon("rotom")
    dead._status = Status.FNT
    dead._current_hp = 0
    out = matchup_block([mon("garchomp", "earthquake")], [dead, None], stub())
    assert not out[0:4].any()


def test_incoming_uses_only_revealed_moves_and_sets_flags():
    foes = [mon("rotom", "surf"), mon("heatran")]          # foe 1 has revealed nothing
    mine = [mon("garchomp"), mon("gyarados")]
    out = matchup_block(mine, foes, stub())
    incoming, flags = out[32:64], out[64:72]
    # foe 0, move slot 0 (surf), my two actives
    assert incoming[0:4].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert flags.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert not incoming[16:32].any()                        # foe 1: nothing revealed


def test_a_revealed_immune_move_differs_from_an_unknown_one():
    foes = [mon("rotom", "surf"), None]
    out = matchup_block([mon("garchomp"), None], foes, stub(mult=0.0, frac=0.0))
    assert out[64] == 1.0 and not out[32:36].any()


def test_every_estimate_and_flag_lands_at_its_spec_index():
    """A stub that encodes (direction, attacker, target, move slot) in its value: all 64 entries and 8 flags at the spec indices."""
    own = [mon("garchomp", "earthquake", "surf", "flamethrower", "icebeam"), mon("heatran", "thunderbolt", "tackle", "recover", "protect")]
    foes = [mon("rotom", "voltswitch", "shadowball"), mon("gardevoir", "moonblast", "psychic")]
    slot = {id(m): {mv.id: i for i, mv in enumerate(m.moves.values())} for m in own + foes}
    who = {id(m): i for i, m in enumerate(own)} | {id(m): i for i, m in enumerate(foes)}
    is_own = {id(m) for m in own}

    def code(att, tgt, move):
        d = 0 if id(att) in is_own else 1
        return 1 + ((d * 2 + who[id(att)]) * 2 + who[id(tgt)]) * 4 + slot[id(att)][move.id]   # 1..64

    def estimate(att, tgt, move):
        c = code(att, tgt, move)
        return c / 16.0, c / 64.0

    out = matchup_block(own, foes, estimate)
    expect = np.zeros(E.MATCHUP_DIM, dtype=np.float32)
    for d, (atts, tgts) in enumerate(((own, foes), (foes, own))):
        for a, att in enumerate(atts):
            for m, mv in enumerate(list(att.moves.values())[:4]):
                for t, tgt in enumerate(tgts):
                    i = d * 32 + ((a * 4 + m) * 2 + t) * 2
                    c = code(att, tgt, mv)
                    expect[i], expect[i + 1] = c / 64.0, c / 64.0     # multiplier c/16 is divided by 4 in the block
    for f in range(2):
        for m in range(2):
            expect[64 + f * 4 + m] = 1.0
    assert np.allclose(out, expect)
    assert out[64:].tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert np.count_nonzero(out[:64]) == 32 + 16   # 16 outgoing cells + 8 incoming cells (foes reveal 2 of 4 moves), 2 numbers each
