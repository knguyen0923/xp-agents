from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Field, Move, Pokemon, SideCondition, Weather  # noqa: E402
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder  # noqa: E402

from xp.showdown import opponents as O  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"


def mon(species, *moves, spe=None):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    if spe is not None:
        p._stats = {"hp": 150, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": spe}
    return p


def make(available0=(), available1=(), mine=(80, 80), foes=("heatran", "heatran"), foe_moves=("flamethrower",), **kw):
    """The opponent's view: its own two actives are slower or faster than the foes' fallback Speed (about 97)."""
    a0, a1 = mon("garchomp", *available0, spe=mine[0]), mon("garchomp", *available1, spe=mine[1])
    f0, f1 = mon(foes[0], *foe_moves), mon(foes[1], *foe_moves)
    return SimpleNamespace(
        battle_tag="b1", active_pokemon=[a0, a1], opponent_active_pokemon=[f0, f1],
        available_moves=[[Move(m, gen=9) for m in available0], [Move(m, gen=9) for m in available1]],
        force_switch=[False, False], fields=kw.get("fields", {}), weather=kw.get("weather", {}),
        side_conditions=kw.get("side", {}), opponent_side_conditions=kw.get("opp_side", {}), team={})


@pytest.fixture
def player():
    return O.make_opponent("tempo", FORMAT)


def test_make_opponent_knows_tempo_and_the_ladder_is_unchanged():
    assert O.LADDER == ("random", "maxbp", "damage") and "tempo" in O.OPPONENTS
    assert isinstance(O.make_opponent("tempo", FORMAT), O.TempoPlayer)
    with pytest.raises(ValueError):
        O.make_opponent("nope", FORMAT)


def test_trick_room_when_slower_and_not_up(player):
    b = make(available0=("trickroom", "closecombat"), mine=(80, 80))
    assert player._setter(b, 0, set()).id == "trickroom"


def test_no_trick_room_when_faster_already_up_or_unavailable(player):
    assert player._setter(make(available0=("trickroom",), mine=(200, 200)), 0, set()) is None
    assert player._setter(make(available0=("trickroom",), mine=(80, 80), fields={Field.TRICK_ROOM: 2}), 0, set()) is None
    assert player._setter(make(available0=("closecombat",), mine=(80, 80)), 0, set()) is None


def test_tailwind_when_not_up_and_no_trick_room(player):
    b = make(available0=("tailwind",))
    assert player._setter(b, 0, set()).id == "tailwind"
    assert player._setter(make(available0=("tailwind",), side={SideCondition.TAILWIND: 2}), 0, set()) is None
    assert player._setter(make(available0=("tailwind",), fields={Field.TRICK_ROOM: 2}), 0, set()) is None


def test_trick_room_takes_precedence_over_tailwind(player):
    b = make(available0=("tailwind", "trickroom"), mine=(80, 80))
    assert player._setter(b, 0, set()).id == "trickroom"


def test_reflect_against_physical_and_light_screen_against_special(player):
    phys = make(available0=("reflect", "lightscreen"), foe_moves=("closecombat", "earthquake"))
    assert player._setter(phys, 0, set()).id == "reflect"
    spec = make(available0=("reflect", "lightscreen"), foe_moves=("flamethrower", "psychic"))
    assert player._setter(spec, 0, set()).id == "lightscreen"


def test_no_screen_with_nothing_revealed_a_tie_or_one_already_up(player):
    assert player._setter(make(available0=("reflect",), foe_moves=()), 0, set()) is None
    assert player._setter(make(available0=("reflect",), foe_moves=("closecombat", "flamethrower")), 0, set()) is None
    up = make(available0=("reflect",), foe_moves=("closecombat",), side={SideCondition.REFLECT: 2})
    assert player._setter(up, 0, set()) is None


def test_aurora_veil_only_in_snow_and_not_recast(player):
    snow = make(available0=("auroraveil",), weather={Weather.SNOWSCAPE: 1})
    assert player._setter(snow, 0, set()).id == "auroraveil"
    assert player._setter(make(available0=("auroraveil",), foe_moves=("closecombat",)), 0, set()) is None
    up = make(available0=("auroraveil",), weather={Weather.HAIL: 1}, side={SideCondition.AURORA_VEIL: 2})
    assert player._setter(up, 0, set()) is None


def test_both_slots_never_set_the_same_condition(player, monkeypatch):
    b = make(available0=("trickroom", "closecombat"), available1=("trickroom", "closecombat"), mine=(80, 80))
    parent = DamageOrders(b)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    order = player.choose_move(b)
    assert order.first_order.order.id == "trickroom"
    assert order.second_order is parent.second_order      # the second slot keeps the parent's order


def DamageOrders(b):
    cc0, cc1 = b.available_moves[0][1], b.available_moves[1][1]
    return DoubleBattleOrder(SingleBattleOrder(cc0, move_target=1), SingleBattleOrder(cc1, move_target=2))


def test_falls_back_to_the_parent_when_no_setter_applies(player, monkeypatch):
    b = make(available0=("closecombat",), available1=("closecombat",))
    parent = DoubleBattleOrder(SingleBattleOrder(b.available_moves[0][0], move_target=1),
                               SingleBattleOrder(b.available_moves[1][0], move_target=2))
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    order = player.choose_move(b)
    assert order.first_order is parent.first_order and order.second_order is parent.second_order


def test_a_setter_keeps_the_mega_flag_and_targets_nobody(player, monkeypatch):
    b = make(available0=("trickroom", "closecombat"), mine=(80, 80))
    parent = DoubleBattleOrder(SingleBattleOrder(b.available_moves[0][1], mega=True, move_target=1), None)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    order = player.choose_move(b)
    assert order.first_order.order.id == "trickroom" and order.first_order.mega and order.first_order.move_target == 0


def test_forced_switches_are_left_to_the_parent(player, monkeypatch):
    b = make(available0=("trickroom",), mine=(80, 80))
    b.force_switch = [True, False]
    parent = DoubleBattleOrder(None, None)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    assert player.choose_move(b) is parent


def test_counters_record_what_was_set(player, monkeypatch):
    b = make(available0=("trickroom", "closecombat"), mine=(80, 80))
    parent = DoubleBattleOrder(SingleBattleOrder(b.available_moves[0][1], move_target=1), None)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    player.choose_move(b)
    assert player.setter_counts["trickroom"] == 1 and player.battles_with_setter == {"b1"} and player.battles_seen == {"b1"}
