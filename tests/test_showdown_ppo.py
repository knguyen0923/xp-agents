from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.toy_showdown import PPO, ToyVec  # noqa: E402
from xp.showdown.masks import N_ACTIONS  # noqa: E402
from xp.showdown.ppo import Learner, masked  # noqa: E402


def test_masked_categorical_gives_illegal_actions_zero_probability():
    logits = torch.zeros(2, N_ACTIONS)
    mask = torch.zeros(2, N_ACTIONS, dtype=torch.bool)
    mask[:, [5, 9]] = True
    probs = masked(logits, mask).probs
    assert probs[:, [5, 9]].sum().item() == pytest.approx(2.0) and probs[~mask].sum().item() == 0.0


def test_act_only_samples_legal_actions_and_conditions_the_second_slot():
    learner = Learner(PPO, 0.01, seed=0)
    env = ToyVec(8)
    env.masks[:, 1, :] = False
    env.masks[:, 1, 4] = env.masks[:, 1, 7] = True
    env.masks[:, 0, :] = False
    env.masks[:, 0, 4] = True  # slot 0 can only switch in "4", so slot 1 must not also pick switch target 4
    for _ in range(20):
        actions, logp, value, cond = learner.act(env.obs, env.masks)
        assert (actions[:, 0] == 4).all() and (actions[:, 1] == 7).all()
        assert logp.shape == (8,) and value.shape == (8,) and cond.shape == (8, 2, N_ACTIONS)


def test_training_is_deterministic_per_seed():
    runs = []
    for _ in range(2):
        learner = Learner(PPO, 0.01, seed=7)
        learner.train(ToyVec(8, seed=1), steps=8 * 16 * 3, log_every=1)
        runs.append([p.detach().numpy().copy() for p in learner.net.parameters()])
    assert all(np.array_equal(a, b) for a, b in zip(*runs))


def test_ppo_learns_the_toy_battle_with_illegal_actions_never_played():
    learner = Learner(PPO, 0.01, seed=0)
    env = ToyVec(16)
    out = learner.train(env, steps=16 * 16 * 100, log_every=5, target=0.9)
    assert out["curve"][-1]["win_rate_100"] > 0.9
    assert out["steps_to_target"] is not None and out["steps_to_target"] <= out["steps"]
    assert out["curve"][0]["win_rate_100"] < out["curve"][-1]["win_rate_100"]
