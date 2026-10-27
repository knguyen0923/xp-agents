from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.toy_showdown import PPO, ToyVec  # noqa: E402

PPO_N = {**PPO, "n_envs": 16, "rnd_lr": 0.0001}
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


def test_plain_ppo_has_no_intrinsic_head_and_kaiser_has_one():
    assert Learner(PPO_N, 0.01, seed=0).net.v_int is None and Learner(PPO_N, 0.01, seed=0).rnd is None
    kaiser = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    assert kaiser.net.v_int is not None and kaiser.rnd is not None


def test_rnd_error_falls_on_seen_observations_and_stays_higher_on_novel_ones():
    import torch as t
    from xp.showdown.encode import OBS_DIM
    from xp.showdown.rnd import VectorRND

    rnd = VectorRND()
    opt = t.optim.Adam(rnd.pred.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    seen = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    novel = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    x_seen, x_novel = t.as_tensor(seen), t.as_tensor(novel)
    start = rnd.error(x_seen).mean().item()
    for _ in range(200):
        loss = rnd.error(x_seen).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert rnd.error(x_seen).mean().item() < 0.2 * start
    assert rnd.error(x_novel).mean().item() > 3 * rnd.error(x_seen).mean().item()
    assert not any(p.requires_grad for p in rnd.target.parameters())


def test_kaiser_still_learns_the_toy_battle_and_never_plays_illegal_actions():
    learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    out = learner.train(ToyVec(16), steps=16 * 16 * 100, log_every=5)
    assert out["curve"][-1]["win_rate_100"] > 0.9


def test_evaluate_reports_a_win_rate_and_changes_nothing():
    for rnd_coef in (None, 0.1):
        learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=rnd_coef)
        learner.train(ToyVec(16), steps=16 * 16 * 5, log_every=5)
        before = learner.fingerprint()
        res = learner.evaluate(ToyVec(16, seed=3), n_battles=40)
        assert 0.0 <= res["win_rate"] <= 1.0 and res["battles"] >= 40
        assert learner.fingerprint() == before


def test_fingerprint_changes_when_training_happens():
    learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    before = learner.fingerprint()
    learner.train(ToyVec(16), steps=16 * 16, log_every=1)
    assert learner.fingerprint() != before


def test_a_mismatched_env_size_is_a_clear_error_for_rnd():
    learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    with pytest.raises(ValueError, match="n_envs=16"):
        learner.train(ToyVec(8), steps=8 * 16, log_every=1)
