from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Weather  # noqa: E402

from xp.showdown import encode as E  # noqa: E402


def mon(species: str, *moves: str) -> Pokemon:
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def battle(own, opp, own_active, opp_active, can_mega=(False, False), weather=None):
    return SimpleNamespace(
        team={m.species + str(i): m for i, m in enumerate(own)},
        opponent_team={m.species + str(i): m for i, m in enumerate(opp)},
        active_pokemon=list(own_active), opponent_active_pokemon=list(opp_active), can_mega_evolve=list(can_mega),
        weather=weather or {}, fields={}, side_conditions={}, opponent_side_conditions={}, turn=3,
    )


def test_dimension_constants_are_consistent():
    assert E.OBS_DIM == 2 * E.SLOTS * E.MON_DIM + E.FIELD_DIM
    assert E.MON_DIM == 3 + len(E.STATUSES) + len(E.BOOSTS) + len(E.TYPES) + len(E.STATS) + 1 + E.N_MOVES * E.MOVE_DIM


def test_missing_pokemon_and_move_are_all_zero():
    assert E.encode_pokemon(None).shape == (E.MON_DIM,) and not E.encode_pokemon(None).any()
    assert E.encode_move(None).shape == (E.MOVE_DIM,) and not E.encode_move(None).any()


def test_hp_fraction_and_boosts_are_encoded_and_scaled():
    p = mon("garchomp")
    p._current_hp, p._max_hp = 50, 100
    p._boosts["atk"] = 6
    v = E.encode_pokemon(p)
    assert v[0] == pytest.approx(0.5)
    boost_start = 3 + len(E.STATUSES)
    assert v[boost_start + E.BOOSTS.index("atk")] == pytest.approx(1.0)


def test_types_are_two_hot_for_dual_types_and_base_stats_are_scaled():
    v = E.encode_pokemon(mon("garchomp"))
    type_start = 3 + len(E.STATUSES) + len(E.BOOSTS)
    assert v[type_start:type_start + len(E.TYPES)].sum() == 2
    stat_start = type_start + len(E.TYPES)
    assert v[stat_start + E.STATS.index("atk")] == pytest.approx(130 / 255)


def test_move_features_flag_protect_and_scale_power():
    protect, quake = E.encode_move(Move("protect", gen=9)), E.encode_move(Move("earthquake", gen=9))
    flags = slice(E.MOVE_DIM - 2, E.MOVE_DIM)
    assert protect[flags].tolist() == [1.0, 0.0]
    assert quake[0] == pytest.approx(100 / 150)
    assert quake[flags][1] == 1.0  # Earthquake hits every adjacent Pokemon


def test_a_species_never_seen_in_training_still_encodes_to_the_same_shape():
    for species in ("rillaboom", "sneasler", "garchomp"):
        assert E.encode_pokemon(mon(species, "protect")).shape == (E.MON_DIM,)


def test_encode_battle_shape_dtype_and_determinism():
    own = [mon("garchomp", "protect", "earthquake"), mon("rillaboom", "fakeout"), mon("sneasler"), mon("excadrill"), mon("gardevoir"), mon("hydreigon")]
    opp = [mon("gholdengo"), mon("pelipper")]
    b = battle(own, opp, own[:2], opp, can_mega=(True, False))
    out = E.encode_battle(b)
    assert out.shape == (E.OBS_DIM,) and out.dtype == np.float32
    assert np.array_equal(out, E.encode_battle(b))
    assert out.min() >= -2.0 and out.max() <= 2.0


def test_first_own_slot_is_the_first_active_with_its_mega_flag():
    own = [mon("garchomp", "protect"), mon("rillaboom")]
    b = battle(own, [mon("gholdengo")], own[:2], [mon("gholdengo")], can_mega=(True, False))
    out = E.encode_battle(b)
    assert np.array_equal(out[:E.MON_DIM], E.encode_pokemon(own[0], True))
    assert np.array_equal(out[E.MON_DIM:2 * E.MON_DIM], E.encode_pokemon(own[1], False))


def test_unrevealed_opponent_slots_are_zero_padded():
    own = [mon("garchomp"), mon("rillaboom")]
    opp = [mon("gholdengo")]
    out = E.encode_battle(battle(own, opp, own, [opp[0], None]))
    opp_side = out[E.SLOTS * E.MON_DIM:2 * E.SLOTS * E.MON_DIM].reshape(E.SLOTS, E.MON_DIM)
    assert opp_side[0].any() and not opp_side[1:].any()


def test_weather_sets_exactly_one_field_bit_in_the_weather_block():
    own = [mon("garchomp"), mon("rillaboom")]
    out = E.encode_battle(battle(own, [mon("gholdengo")], own, [None, None], weather={Weather.SUNNYDAY: 1}))
    field = out[2 * E.SLOTS * E.MON_DIM:]
    assert field[:len(Weather)].sum() == 1 and field[list(Weather).index(Weather.SUNNYDAY)] == 1


def test_obs_dim_helper_is_unchanged_with_the_flag_off():
    assert E.obs_dim() == E.obs_dim(False) == E.OBS_DIM
    assert E.MATCHUP_DIM == 72 and E.obs_dim(True) == E.OBS_DIM + 72
