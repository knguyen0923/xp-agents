from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("poke_env")

from xp.errors import ProtocolError  # noqa: E402
from xp.showdown.teams import PoolTeambuilder, assert_disjoint, load_pool, normalise, validate_team  # noqa: E402

A = "Garchomp @ Choice Scarf\nAbility: Rough Skin\nLevel: 50\nAdamant Nature\n- Earthquake\n- Rock Slide\n"
B = "Rillaboom @ Miracle Seed\nAbility: Grassy Surge\nLevel: 50\nJolly Nature\n- Fake Out\n- Wood Hammer\n"


def test_load_pool_reads_sorted_txt_files(tmp_path):
    (tmp_path / "b.txt").write_text(B)
    (tmp_path / "a.txt").write_text(A)
    (tmp_path / "notes.md").write_text("ignored")
    assert list(load_pool(tmp_path)) == ["a", "b"]


def test_load_pool_empty_directory_raises(tmp_path):
    with pytest.raises(ProtocolError, match="no team files"):
        load_pool(tmp_path)


def test_assert_disjoint_accepts_distinct_pools():
    assert_disjoint({"t1": A}, {"e1": B})


def test_assert_disjoint_rejects_same_text_even_with_whitespace_noise():
    with pytest.raises(ProtocolError, match="also appear"):
        assert_disjoint({"t1": A}, {"e1": A.replace("\n", "  \n") + "\n\n"})


def test_assert_disjoint_rejects_same_name():
    with pytest.raises(ProtocolError, match="share team names"):
        assert_disjoint({"x": A}, {"x": B})


def test_normalise_ignores_trailing_space_and_outer_blank_lines():
    assert normalise("\n" + A.replace("\n", "   \n") + "\n\n") == normalise(A)


def test_pool_teambuilder_draws_only_from_its_pool_and_logs_names():
    tb = PoolTeambuilder({"t1": A, "t2": B}, seed=3)
    for _ in range(30):
        packed = tb.yield_team()
        assert packed
    assert set(tb.drawn) <= {"t1", "t2"} and len(tb.drawn) == 30


def test_pool_teambuilder_is_deterministic_per_seed():
    one = PoolTeambuilder({"t1": A, "t2": B}, seed=5)
    two = PoolTeambuilder({"t1": A, "t2": B}, seed=5)
    for _ in range(10):
        one.yield_team(), two.yield_team()
    assert one.drawn == two.drawn


def test_pool_teambuilder_requires_a_team():
    with pytest.raises(ProtocolError):
        PoolTeambuilder({}, seed=0)


def _fake_run(returncode, stdout="", stderr=""):
    def run(cmd, **kwargs):
        run.cmd, run.kwargs = cmd, kwargs
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return run


def test_validate_team_passes_on_silent_success(monkeypatch, tmp_path):
    run = _fake_run(0)
    monkeypatch.setattr(subprocess, "run", run)
    validate_team(A, "gen9championsvgc2026regmc", tmp_path)
    assert run.cmd[-2:] == ["validate-team", "gen9championsvgc2026regmc"] and run.kwargs["input"] == A


def test_validate_team_raises_with_the_validator_message(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run(1, stdout="Pikachu's move Splash does not exist in Gen 9."))
    with pytest.raises(ProtocolError, match="Splash"):
        validate_team(A, "gen9championsvgc2026regmc", tmp_path)


def test_validate_team_raises_on_output_even_with_exit_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run(0, stdout="You must bring at least 6 Pokémon"))
    with pytest.raises(ProtocolError, match="at least 6"):
        validate_team(A, "gen9championsvgc2026regmc", tmp_path)
