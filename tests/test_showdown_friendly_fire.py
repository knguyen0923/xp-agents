from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Status  # noqa: E402

from xp.showdown import friendly_fire as FF  # noqa: E402


def mon(species, *moves):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def act(k, target):
    return 7 + 5 * k + (target + 2)


def setup(frac=0.4, mult=1.0):
    a = mon("garchomp", "closecombat", "earthquake", "heatwave", "helpinghand")
    b = mon("heatran", "flamethrower")
    calls = []

    def estimator(attacker, defender, move):
        calls.append((attacker, defender, move.id))
        return (mult, frac)

    return SimpleNamespace(active_pokemon=[a, b], available_moves=[list(a.moves.values()), list(b.moves.values())]), estimator, calls


def test_decode_move_actions_and_mega_variants():
    assert FF.decode_move_action(7) == (0, -2)
    assert FF.decode_move_action(11) == (0, 2)
    assert FF.decode_move_action(12) == (1, -2)
    assert FF.decode_move_action(46) == (3, 2)
    assert FF.decode_move_action(27) == (0, -2)   # the Mega variant of action 7
    assert FF.decode_move_action(0) is None and FF.decode_move_action(3) is None and FF.decode_move_action(47) is None


def test_a_damaging_move_aimed_at_the_partner_is_charged_the_estimated_fraction():
    battle, est, calls = setup(frac=0.4)
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 1.0) == pytest.approx(-0.4)
    assert calls[0][1] is battle.active_pokemon[1]   # the defender is the partner


def test_slot_one_aims_at_target_minus_one():
    battle, est, _ = setup(frac=0.5)
    battle.active_pokemon[1] = mon("garchomp", "closecombat")
    battle.active_pokemon[0] = mon("heatran", "flamethrower")
    assert FF.friendly_fire_penalty(battle, 1, act(0, -1), est, 1.0) == pytest.approx(-0.5)


def test_a_move_aimed_at_a_foe_is_free():
    battle, est, calls = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(0, 1), est, 1.0) == 0.0 and not calls


def test_all_adjacent_moves_hit_the_partner_whatever_the_target():
    battle, est, _ = setup(frac=0.3)
    assert FF.friendly_fire_penalty(battle, 0, act(1, 0), est, 1.0) == pytest.approx(-0.3)   # earthquake


def test_all_adjacent_foes_moves_are_free():
    battle, est, calls = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(2, 0), est, 1.0) == 0.0 and not calls      # heat wave


def test_status_moves_aimed_at_the_partner_are_free():
    battle, est, calls = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(3, -2), est, 1.0) == 0.0 and not calls     # helping hand


def test_the_mega_variant_is_charged_like_the_plain_move():
    battle, est, _ = setup(frac=0.4)
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2) + 20, est, 1.0) == pytest.approx(-0.4)


def test_an_immune_partner_is_free():
    battle, est, _ = setup(frac=0.0, mult=0.0)
    assert FF.friendly_fire_penalty(battle, 0, act(1, 0), est, 1.0) == 0.0


def test_a_fainted_or_absent_partner_is_free():
    battle, est, calls = setup()
    battle.active_pokemon[1]._status = Status.FNT
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 1.0) == 0.0
    battle.active_pokemon[1] = None
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 1.0) == 0.0 and not calls


def test_the_scale_multiplies_and_non_move_actions_are_free():
    battle, est, _ = setup(frac=0.5)
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 2.0) == pytest.approx(-1.0)
    assert FF.friendly_fire_penalty(battle, 0, 3, est, 1.0) == 0.0     # a switch
    assert FF.friendly_fire_penalty(battle, 0, 0, est, 1.0) == 0.0     # pass


def test_an_estimator_that_returns_none_is_free():
    battle, _, _ = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), lambda a, d, m: None, 1.0) == 0.0


def test_the_battle_cap_stops_the_running_total_at_minus_one():
    assert FF.FF_BATTLE_CAP == 1.0
    assert FF.capped_penalty(-0.4, 0.0) == pytest.approx(-0.4)
    assert FF.capped_penalty(-0.4, 0.8) == pytest.approx(-0.2)     # only 0.2 of headroom left
    assert FF.capped_penalty(-0.4, 1.0) == 0.0                     # already at the cap
    assert FF.capped_penalty(-2.0, 0.0) == pytest.approx(-1.0)     # one huge decision still stops at the cap
    assert FF.capped_penalty(0.0, 0.5) == 0.0


def test_a_win_with_the_most_friendly_fire_still_returns_more_than_a_clean_loss():
    spent = 0.0
    for raw in (-2.0, -2.0, -2.0):                                 # three decisions of two full-HP hits on the partner
        spent += -FF.capped_penalty(raw, spent)
    assert spent == pytest.approx(1.0) and 1.0 - spent >= 0.0 > -1.0


def test_tally_sums_per_battle_penalty_until_done_and_keeps_a_window():
    import numpy as np
    t = FF.PenaltyTally(2)
    assert t.recent()[1] == 0 and t.recent()[0] != t.recent()[0]       # empty: NaN mean, zero count
    t.add(np.array([-0.2, 0.0]), np.array([0.0, 0.0]))
    t.add(np.array([-0.3, -0.1]), np.array([1.0, 0.0]))                # env 0 finishes with -0.5
    t.add(np.array([-0.4, 0.0]), np.array([0.0, 1.0]))                 # env 1 finishes with -0.1; env 0 restarts at -0.4
    assert t.recent() == (pytest.approx(-0.3), 2)
    assert t.totals() == (pytest.approx(-0.6), 2)
    t.add(np.array([0.0, 0.0]), np.array([1.0, 0.0]))                  # env 0: -0.4 carried over from the new battle
    assert t.totals() == (pytest.approx(-1.0), 3)


def test_tally_window_is_the_last_100_battles_but_totals_are_cumulative():
    import numpy as np
    t = FF.PenaltyTally(1)
    for i in range(150):
        t.add(np.array([-1.0 if i < 50 else 0.0]), np.array([1.0]))
    assert t.recent() == (0.0, 100) and t.totals() == (pytest.approx(-50.0), 150)


def test_struggle_or_recharge_turn_charges_the_available_move_not_the_known_one():
    battle, est, calls = setup(frac=0.4)
    battle.active_pokemon[0] = mon("garchomp", "earthquake", "closecombat")   # Earthquake (spreads to the partner) in slot 0
    battle.available_moves = [[Move("struggle", gen=9)], []]
    assert FF.friendly_fire_penalty(battle, 0, act(0, 0), est, 1.0) == 0.0 and not calls    # Struggle (untargeted) never hits the partner
    battle.available_moves = [[Move("closecombat", gen=9), Move("earthquake", gen=9)], []]
    assert FF.friendly_fire_penalty(battle, 0, act(0, 0), est, 1.0) == pytest.approx(-0.4)  # more than one move: the known list
