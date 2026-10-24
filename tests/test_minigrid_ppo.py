from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

import torch  # noqa: E402

from xp.minigrid.ppo import Learner  # noqa: E402
from xp.minigrid.rnd import IntrinsicScaler  # noqa: E402

PPO = {"n_envs": 2, "n_steps": 16, "epochs": 2, "minibatches": 2, "lr": 2.5e-4, "gamma": 0.99, "lam": 0.95,
       "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 1e-4}


def test_training_is_deterministic_per_seed():
    a, b = Learner(PPO, 0.01, None, seed=3), Learner(PPO, 0.01, None, seed=3)
    a.train(5, 128)
    b.train(5, 128)
    assert a.fingerprint() == b.fingerprint()
    c = Learner(PPO, 0.01, None, seed=4)
    c.train(5, 128)
    assert c.fingerprint() != a.fingerprint()


def test_evaluation_never_updates_anything():
    for rnd_coef in (None, 0.01):
        agent = Learner(PPO, 0.01, rnd_coef, seed=1)
        agent.train(5, 64)
        before = agent.fingerprint()
        res = agent.evaluate(5, 6)
        assert 0.0 <= res["success"] <= 1.0
        assert agent.fingerprint() == before


def test_evaluation_is_deterministic_for_fixed_weights():
    agent = Learner(PPO, 0.01, None, seed=2)
    assert agent.evaluate(5, 6) == agent.evaluate(5, 6)


def test_kaiser_changes_weights_differently_from_kairi():
    kairi, kaiser = Learner(PPO, 0.01, None, seed=5), Learner(PPO, 0.01, 0.01, seed=5)
    assert kairi.net.v_int is None and kaiser.net.v_int is not None
    kairi.train(5, 128)
    kaiser.train(5, 128)
    assert kairi.fingerprint() != kaiser.fingerprint()


def test_deepcopy_continues_independently():
    import copy

    base = Learner(PPO, 0.01, 0.01, seed=6)
    base.train(5, 64)
    before = base.fingerprint()
    clone = copy.deepcopy(base)
    assert clone.fingerprint() == before
    clone.train(6, 64, first_episode=500_000, limit=1_000_000)
    assert base.fingerprint() == before and clone.fingerprint() != before


def _max_discounted_return(rewards, gamma=0.99):
    ret, peak = np.zeros(rewards.shape[1]), 0.0
    for t in range(rewards.shape[0]):
        ret = ret * gamma + rewards[t]
        peak = max(peak, float(np.abs(ret).max()))
    return peak


def test_intrinsic_returns_stay_small():
    """Regression: scaling by the per-step std left discounted intrinsic returns in the hundreds or thousands."""
    rng = np.random.default_rng(0)
    raw = 5.0 + 0.1 * rng.standard_normal((2000, 4))
    scaled = IntrinsicScaler(4, 0.99)(raw)
    naive = (raw / raw.std()).astype(np.float32)
    assert _max_discounted_return(scaled) < 20.0
    assert _max_discounted_return(naive) > 1000.0


def test_steps_to_target_needs_a_full_episode_window():
    learner = Learner(PPO, 0.01, None, seed=3)
    out = learner.train(5, 1024, target=0.0)  # some episodes finish (<100); target 0 trips on any non-empty window
    assert out["steps_to_target"] is None
