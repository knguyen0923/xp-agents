from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from tests.toy_showdown import PPO, ToyVec  # noqa: E402

from xp.showdown import gate  # noqa: E402
from xp.showdown.ppo import Learner  # noqa: E402


def test_measure_reports_throughput_curve_and_usable_range_flag():
    out = gate.measure(Learner(PPO, 0.01, seed=0), ToyVec(16), steps=16 * 16 * 100, target=0.9, log_every=10)
    assert out["decisions_per_s"] > 0 and out["wall_s"] > 0 and out["steps"] == 16 * 16 * 100
    assert out["curve"][-1]["win_rate_100"] == out["final_win_rate_100"]
    assert out["in_usable_range"] == (0.2 <= out["final_win_rate_100"] <= 0.8)
    assert out["steps_to_target"] is not None and json.dumps(out)  # JSON-serialisable


def test_default_ppo_has_every_key_the_learner_reads():
    assert {"n_envs", "n_steps", "epochs", "minibatches", "lr", "gamma", "lam", "clip", "vf_coef"} <= set(gate.DEFAULT_PPO)
    assert gate.DEFAULT_PPO["n_steps"] * gate.DEFAULT_PPO["n_envs"] % gate.DEFAULT_PPO["minibatches"] == 0
