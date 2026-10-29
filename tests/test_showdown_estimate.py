from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon  # noqa: E402

from xp.showdown import estimate as ES  # noqa: E402


def mon(species, *moves):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def fake_battle(*mons):
    return SimpleNamespace(team={f"p1: {m.species}": m for m in mons[:1]}, opponent_team={f"p2: {m.species}": m for m in mons[1:]})


def test_status_and_zero_power_moves_give_none(monkeypatch):
    a, d = mon("garchomp", "protect"), mon("rotom")
    est = ES.PokeEnvEstimator(fake_battle(a, d))
    assert est(a, d, a.moves["protect"]) is None


def test_multiplier_comes_from_the_types_and_fraction_from_the_calculator(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")
    d._stats = {"hp": 150, "atk": 80, "def": 90, "spa": 120, "spd": 90, "spe": 70}
    a._stats = dict(d._stats)
    d._current_hp, d._max_hp = 100, 100              # a bare Pokemon may report 0 HP, so set the fraction explicitly
    monkeypatch.setattr(ES, "calculate_damage", lambda *args, **kw: (60, 90))
    mult, frac = ES.PokeEnvEstimator(fake_battle(a, d))(a, d, a.moves["earthquake"])
    assert mult == pytest.approx(d.damage_multiplier(a.moves["earthquake"]))
    assert frac == pytest.approx(75 / 150)


def test_zero_damage_from_a_damaging_move_reads_as_an_immunity(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")
    for p in (a, d):
        p._stats = {"hp": 150, "atk": 80, "def": 90, "spa": 120, "spd": 90, "spe": 70}
    monkeypatch.setattr(ES, "calculate_damage", lambda *args, **kw: (0, 0))
    assert ES.PokeEnvEstimator(fake_battle(a, d))(a, d, a.moves["earthquake"]) == (0.0, 0.0)


def test_a_calculator_error_is_counted_and_returns_none(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")

    def boom(*args, **kw):
        raise AssertionError("stats not defined")

    monkeypatch.setattr(ES, "calculate_damage", boom)
    est = ES.PokeEnvEstimator(fake_battle(a, d))
    assert est(a, d, a.moves["earthquake"]) is None and est.errors == 1


def test_missing_stats_are_filled_from_base_stats_at_the_fallback_spread(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")
    seen = {}

    def spy(att, dfn, move, battle, **kw):
        seen["d_stats"] = battle.opponent_team[dfn].stats
        return (10, 10)

    monkeypatch.setattr(ES, "calculate_damage", spy)
    ES.PokeEnvEstimator(fake_battle(a, d))(a, d, a.moves["earthquake"])
    assert all(isinstance(v, (int, float)) for v in seen["d_stats"].values())


def test_partial_stats_keep_known_values_and_fill_only_the_missing_ones():
    d = mon("heatran")
    d._stats = {"hp": 150, "atk": 80, "def": None, "spa": None, "spd": 90, "spe": None}
    ES._fill_stats(d)
    assert (d.stats["hp"], d.stats["atk"], d.stats["spd"]) == (150, 80, 90)
    assert all(isinstance(d.stats[k], int) for k in ("def", "spa", "spe"))


@pytest.mark.showdown
def test_real_battle_estimates_are_finite_and_cheap():
    """Needs the local server. After the first decision every estimate is None or in range, and a decision costs under 50 ms."""
    import time

    from tests.test_showdown_vec import real_first_battle

    battle = real_first_battle()
    est = ES.PokeEnvEstimator(battle)
    t0 = time.perf_counter()
    for a in battle.active_pokemon:
        for d in battle.opponent_active_pokemon:
            for move in a.moves.values():
                r = est(a, d, move)
                assert r is None or (0.0 <= r[0] <= 4.0 and 0.0 <= r[1] <= 1.0)
    assert (time.perf_counter() - t0) < 0.05
    own_to_foe = [est(a, d, m) for a in battle.active_pokemon for d in battle.opponent_active_pokemon for m in a.moves.values() if m.base_power > 0]
    assert any(r is not None for r in own_to_foe) and est.errors == 0
