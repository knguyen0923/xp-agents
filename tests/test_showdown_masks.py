from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Status  # noqa: E402

from xp.showdown.masks import MEGA_ACTIONS, N_ACTIONS, PASS, action_masks, restrict_second  # noqa: E402


def mon(species, *moves):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def targets(move, user):
    return [0] if move.id == "protect" else [1, 2]  # Earthquake-like spread moves list both foes in this stub


def battle(actives, team, switches=((), ()), force=(False, False), mega=(False, False), trapped=(False, False)):
    return SimpleNamespace(
        active_pokemon=list(actives), team={f"p{i}": m for i, m in enumerate(team)}, force_switch=list(force),
        available_switches=[list(s) for s in switches], trapped=list(trapped), can_mega_evolve=list(mega),
        available_moves=[list(a.moves.values()) if a is not None else [] for a in actives],
        get_possible_showdown_targets=targets,
    )


def legal(mask_row):
    return set(np.flatnonzero(mask_row).tolist())


def test_slot_with_no_active_or_a_fainted_pokemon_can_only_pass():
    a = mon("garchomp", "protect")
    dead = mon("rillaboom", "protect")
    dead._status = Status.FNT
    m = action_masks(battle([a, dead], [a, dead]))
    assert legal(m[1]) == {PASS}
    m = action_masks(battle([a, None], [a]))
    assert legal(m[1]) == {PASS}


def test_moves_map_to_seven_plus_five_k_plus_target_and_mega_adds_twenty():
    a, b = mon("garchomp", "protect", "earthquake"), mon("rillaboom", "protect")
    m = action_masks(battle([a, b], [a, b]))
    assert legal(m[0]) == {9, 15, 16}                      # protect (target 0) and earthquake (targets 1, 2)
    m = action_masks(battle([a, b], [a, b], mega=(True, False)))
    assert legal(m[0]) == {9, 15, 16, 29, 35, 36}
    assert legal(m[1]) == {9}                              # slot 1 cannot Mega


def test_switch_actions_are_one_plus_the_team_position_of_each_available_switch():
    a, b, c, d = mon("garchomp", "protect"), mon("rillaboom", "protect"), mon("sneasler"), mon("excadrill")
    m = action_masks(battle([a, b], [a, b, c, d], switches=([c, d], [c, d])))
    assert {3, 4} <= legal(m[0]) and not ({1, 2} & legal(m[0]))


def test_trapped_slot_cannot_switch():
    a, b, c = mon("garchomp", "protect"), mon("rillaboom", "protect"), mon("sneasler")
    m = action_masks(battle([a, b], [a, b, c], switches=([c], [c]), trapped=(True, False)))
    assert not ({1, 2, 3} & legal(m[0])) and 3 in legal(m[1])


def test_forced_switch_on_one_slot_leaves_the_other_slot_passing():
    a, b, c = mon("garchomp", "protect"), mon("rillaboom", "protect"), mon("sneasler")
    m = action_masks(battle([a, b], [a, b, c], switches=([c], [c]), force=(True, False)))
    assert legal(m[0]) == {3} and legal(m[1]) == {PASS}


def test_restrict_second_removes_a_second_mega_and_a_duplicate_switch():
    row = np.zeros(N_ACTIONS, dtype=bool)
    row[[9, 29, 35, 3, 4]] = True
    after_mega = restrict_second(row, 29)
    assert legal(after_mega) == {9, 3, 4} and not (legal(after_mega) & MEGA_ACTIONS)
    after_switch = restrict_second(row, 3)
    assert legal(after_switch) == {9, 29, 35, 4}
    only_switch = np.zeros(N_ACTIONS, dtype=bool)
    only_switch[3] = True
    assert legal(restrict_second(only_switch, 3)) == {PASS}   # nothing left: pass rather than an empty mask
