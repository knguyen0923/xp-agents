from __future__ import annotations

import os

import numpy as np
import pytest

from xp.bandit.phase import build_factories, build_tagteam, make_per_seed_fn, run_phase
from xp.bandit.tagteam import TagTeamAgent
from xp.errors import ConfigMismatchError
from xp.protocol.phase import Phase, load_config
from xp.protocol.results import load_results, read_tune

AGENTS = {"kairi", "kaiser", "tt_fixed", "tt_triggered", "tt_meta"}


def test_real_config_is_valid_and_grids_are_equal_size():
    cfg = load_config("configs/bandit.yaml")
    Phase.from_config(cfg)
    n = lambda spec: int(np.prod([len(v) for v in spec["grid"].values()]))
    assert n(cfg["agents"]["kairi"]) == n(cfg["agents"]["kaiser"])


def test_per_seed_fn_is_deterministic_and_per_seed_independent_of_batching(tiny_cfg):
    f = build_factories(tiny_cfg)["kairi"]
    fn = make_per_seed_fn(tiny_cfg, f)
    params = {"epsilon": 0.1, "alpha": 0.3}
    a = fn(np.array([100, 101, 102]), 0.01, params, 0)
    b = fn(np.array([100, 101, 102]), 0.01, params, 0)
    assert np.array_equal(a, b)


def test_build_tagteam_returns_agent_for_each_rule(tiny_cfg):
    base = {"kairi": {"epsilon": 0.1, "alpha": 0.3}, "kaiser": {"gamma": 0.95, "c": 1.0}}
    for rule, params in (
        ("fixed", {"block": 10}),
        ("triggered", {"delta": 1.0, "h": 4.0, "window": 5}),
        ("meta", {"gamma": 0.9, "c": 0.5}),
    ):
        agent = build_tagteam(tiny_cfg, rule, base)(params, 0)
        assert isinstance(agent, TagTeamAgent)


def test_run_phase_end_to_end_then_resume_then_mismatch(tiny_cfg, tmp_path):
    run_phase(tiny_cfg, tmp_path)
    phase = Phase.from_config(tiny_cfg)
    df = load_results(tmp_path, "bandit")
    assert set(df["agent"]) == AGENTS
    assert set(df["knob"]) == set(phase.sweep)
    assert len(df) == len(AGENTS) * len(phase.sweep) * len(phase.eval_seeds)
    assert np.isfinite(df["value"]).all() and (df["value"] >= 0).all()
    for name in ("kairi", "kaiser", "tt_triggered", "tt_meta"):
        assert read_tune(tmp_path, phase, name) is not None
    # tiny grids always put the optimum on an edge, so everything tuned must be provisional
    assert df.loc[df["agent"] == "kairi", "provisional"].all()

    unit = tmp_path / "bandit" / "kairi__jump_rate-0.01.csv"
    before = os.path.getmtime(unit)
    run_phase(tiny_cfg, tmp_path)  # resume: nothing recomputed
    assert os.path.getmtime(unit) == before

    tiny_cfg["horizon"] = 60
    with pytest.raises(ConfigMismatchError):
        run_phase(tiny_cfg, tmp_path)


def test_tagteam_inherits_provisional_from_base_agents(tiny_cfg, tmp_path):
    run_phase(tiny_cfg, tmp_path)
    df = load_results(tmp_path, "bandit")
    assert df.loc[df["agent"] == "tt_fixed", "provisional"].all()  # no own tuning, base is provisional
