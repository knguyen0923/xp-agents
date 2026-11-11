from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Field, Move, Pokemon, SideCondition, Status  # noqa: E402

from xp.showdown import encode as E  # noqa: E402
from xp.showdown import tempo as T  # noqa: E402
from xp.showdown.estimate import _fallback_stats  # noqa: E402


def mon(species, *moves, spe=None):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    if spe is not None:
        p._stats = {"hp": 150, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": spe}
    return p


def test_dimension_constants_agree():
    assert T.SPEED_DIM + T.TURNS_DIM + T.FLAGS_DIM == E.TEMPO_DIM == 77


def test_faster_own_mon_reads_plus_one_and_trick_room_flips_it():
    own, foe = mon("garchomp", spe=150), mon("heatran")           # the foe's Speed is the level-50 fallback, about 97
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == 1.0
    assert T.speed_order([own, None], [foe, None], False, False, True)[0] == -1.0


def test_tailwind_on_the_foe_side_can_make_it_faster():
    own, foe = mon("garchomp", spe=150), mon("heatran")
    assert T.speed_order([own, None], [foe, None], False, True, False)[0] == -1.0     # about 194 against 150


def test_paralysis_boosts_and_a_scarf_change_the_order():
    own, foe = mon("garchomp", spe=150), mon("heatran")
    own._status = Status.PAR
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == -1.0    # 75 against about 97
    own._status = None
    own._boosts["spe"] = -2
    assert T.effective_speed(own, False, False) == pytest.approx(75.0)                # a -2 stage halves 150
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == -1.0    # 75 against about 97
    own._boosts["spe"] = 0
    own._item = "choicescarf"
    assert T.effective_speed(own, False, False) == pytest.approx(225.0)


def test_equal_speed_and_absent_mons_read_zero():
    foe = mon("heatran")
    own = mon("garchomp", spe=_fallback_stats(foe)["spe"])
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == 0.0
    assert T.speed_order([None, None], [foe, None], False, False, False).tolist() == [0.0] * 4


def battle_with(fields=None, side=None, opp_side=None, turn=4, items=()):
    team = {}
    for i, item in enumerate(items):
        m = mon("garchomp")
        m._item = item
        team[f"p1: {i}"] = m
    return SimpleNamespace(fields=fields or {}, side_conditions=side or {}, opponent_side_conditions=opp_side or {}, turn=turn, team=team)


def test_turns_left_for_each_condition():
    b = battle_with(fields={Field.TRICK_ROOM: 3}, side={SideCondition.TAILWIND: 3, SideCondition.REFLECT: 3},
                    opp_side={SideCondition.LIGHT_SCREEN: 2})
    out = T.turns_block(b)
    assert out[0] == pytest.approx(4 / 5)            # set on turn 3, lasts through turn 7, 4 left at turn 4
    assert out[1] == pytest.approx(3 / 4)            # tailwind: 3 + 4 - 4
    assert out[2] == pytest.approx(4 / 5)            # reflect: 3 + 5 - 4
    assert out[7] == pytest.approx(3 / 5)            # opposing light screen: 2 + 5 - 4
    assert out[3] == out[4] == out[5] == out[6] == out[8] == 0.0


def test_a_condition_at_or_past_expiry_reads_zero():
    b = battle_with(fields={Field.TRICK_ROOM: 3}, turn=8)
    assert T.turns_block(b)[0] == 0.0


def test_own_light_clay_lengthens_own_screens_only():
    b = battle_with(side={SideCondition.REFLECT: 3}, opp_side={SideCondition.REFLECT: 3}, items=("lightclay",))
    out = T.turns_block(b)
    assert out[2] == pytest.approx(7 / 8)            # 3 + 8 - 4
    assert out[6] == pytest.approx(4 / 5)            # the foe's item is hidden: assumed 5 turns


def test_move_flags_for_known_moves():
    f = lambda name: T.move_flags(Move(name, gen=9)).tolist()   # noqa: E731
    assert f("trickroom") == [1, 0, 0, 0] and f("tailwind") == [0, 1, 0, 0]
    assert f("reflect") == [0, 0, 1, 0] and f("lightscreen") == [0, 0, 1, 0] and f("auroraveil") == [0, 0, 1, 0]
    assert f("earthquake") == [0, 0, 0, 1]                      # hits the partner whatever the target
    assert f("heatwave") == [0, 0, 0, 0] and f("protect") == [0, 0, 0, 0] and f("closecombat") == [0, 0, 0, 0]
    assert T.move_flags(None).tolist() == [0, 0, 0, 0]


def test_flags_block_layout_and_foe_slots():
    own0 = mon("garchomp", "closecombat", "trickroom")
    foe1 = mon("heatran", "flamethrower", "earthquake")
    out = T.flags_block([own0, None], [None, foe1])
    assert out.shape == (T.FLAGS_DIM,)
    assert out[0 * 16 + 1 * 4 + 0] == 1.0                      # my active 0, move slot 1, sets_trick_room
    assert out[3 * 16 + 1 * 4 + 3] == 1.0                      # opposing active 1, move slot 1, hits_ally
    assert out.sum() == 2.0


def test_tempo_block_shape_dtype_and_order():
    own0, foe0 = mon("garchomp", "trickroom", spe=150), mon("heatran")
    b = battle_with(fields={Field.TRICK_ROOM: 3}, turn=4)
    b.active_pokemon, b.opponent_active_pokemon = [own0, None], [foe0, None]
    out = T.tempo_block(b)
    assert out.shape == (E.TEMPO_DIM,) and out.dtype == np.float32
    assert out[0] == -1.0                                      # speed order, flipped by Trick Room
    assert out[T.SPEED_DIM] == pytest.approx(4 / 5)            # remaining turns of Trick Room
    assert out[T.SPEED_DIM + T.TURNS_DIM + 0] == 1.0           # my active 0's first move slot (trickroom) sets_trick_room
