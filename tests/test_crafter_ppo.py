from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xp.crafter.ppo import ActorCritic, Learner, gae, resolve_device  # noqa: E402
from xp.crafter.rnd import RND, IntrinsicScaler  # noqa: E402

PPO = {"n_envs": 2, "n_steps": 8, "epochs": 1, "minibatches": 2, "lr": 0.00025, "gamma": 0.99, "lam": 0.95,
       "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 0.0001}


class FakeVec:
    """Random 64x64x3 frames; every env finishes an episode every `period` steps, having unlocked `collect_wood` (and `eat_cow` on odd episodes)."""

    def __init__(self, n=2, period=5, seed=0):
        self.n, self.period, self.t = n, period, 0
        self.rng = np.random.RandomState(seed)
        self.obs = self.rng.randint(0, 255, (n, 64, 64, 3), dtype=np.uint8)
        self.closed = False

    def step(self, actions):
        assert len(actions) == self.n
        self.t += 1
        self.obs = self.rng.randint(0, 255, (self.n, 64, 64, 3), dtype=np.uint8)
        done = np.full(self.n, self.t % self.period == 0)
        eps = [(["collect_wood"] + (["eat_cow"] if (self.t // self.period) % 2 else [])) if d else None for d in done]
        return self.obs, np.zeros(self.n, np.float32), done, eps

    def close(self):
        self.closed = True


def test_network_shapes_and_intrinsic_head_only_for_rnd():
    plain, rnd = ActorCritic(False), ActorCritic(True)
    logits, v, v_int = rnd.heads(torch.zeros(3, 3, 64, 64))
    assert logits.shape == (3, 17) and v.shape == (3,) and v_int.shape == (3,)
    assert plain.v_int is None and plain.heads(torch.zeros(1, 3, 64, 64))[2] is None


def test_both_heads_output_n_actions_and_it_is_17():
    from xp.crafter.ppo import N_ACTIONS

    assert N_ACTIONS == 17
    for intrinsic in (False, True):
        logits = ActorCritic(intrinsic).heads(torch.zeros(2, 3, 64, 64))[0]
        assert logits.shape == (2, N_ACTIONS)


def test_rnd_target_is_frozen_and_error_shape():
    rnd = RND()
    assert not any(p.requires_grad for p in rnd.target.parameters())
    assert any(p.requires_grad for p in rnd.pred.parameters())
    x = rnd.normalize(np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8), "cpu")
    assert x.shape == (5, 3, 64, 64) and rnd.error(x).shape == (5,)


def test_intrinsic_scaler_shape_and_finite():
    out = IntrinsicScaler(2, 0.99)(np.abs(np.random.randn(8, 2)))
    assert out.shape == (8, 2) and np.isfinite(out).all() and out.dtype == np.float32


def test_gae_matches_a_hand_computed_value():
    r, v = torch.tensor([[1.0]]), torch.tensor([[0.5]])
    adv = gae(r, v, torch.tensor([2.0]), torch.tensor([[1.0]]), 0.9, 0.95)
    assert adv.item() == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


def test_train_returns_curve_rates_and_first_unlocks():
    learner = Learner(PPO, 0.01, None, seed=0)
    env = FakeVec(period=5)
    run = learner.train(env, steps=16 * 4, log_every=2)  # batch = 16, 4 updates
    assert run["steps"] == 64 and learner.steps_trained == 64
    assert [row["steps"] for row in run["curve"]] == [32, 64]  # every 2 updates, last update included
    row = run["curve"][-1]
    assert row["ach_100"] != row["ach_100"]  # NaN: fewer than 100 episodes
    assert set(row["rates"]) >= {"collect_wood", "eat_cow"} and row["episodes"] == run["episodes"] > 0
    assert 0 < run["first_unlock"]["collect_wood"] <= 64 and "make_iron_sword" not in run["first_unlock"]
    assert run["final_rates"]["collect_wood"] == 1.0 and run["secs"] >= 0


def test_train_is_deterministic_per_seed():
    a, b = Learner(PPO, 0.01, None, seed=3), Learner(PPO, 0.01, None, seed=3)
    a.train(FakeVec(seed=1), 32, log_every=1)
    b.train(FakeVec(seed=1), 32, log_every=1)
    assert a.fingerprint() == b.fingerprint()


def test_kaiser_trains_rnd_and_kairi_has_none():
    kairi, kaiser = Learner(PPO, 0.01, None, seed=0), Learner(PPO, 0.01, 0.3, seed=0)
    assert kairi.rnd is None and kaiser.rnd is not None
    before = kaiser.fingerprint()
    run = kaiser.train(FakeVec(), 32, log_every=1)
    assert kaiser.fingerprint() != before and len(run["curve"]) == 2


def test_evaluate_never_learns():
    learner = Learner(PPO, 0.01, 0.3, seed=0)
    frozen = learner.fingerprint()
    res = learner.evaluate(FakeVec(period=3), n_episodes=4)
    assert learner.fingerprint() == frozen
    assert res["episodes"] >= 4 and res["ach"] >= 1.0 and res["rates"]["collect_wood"] == 1.0
