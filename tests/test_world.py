from __future__ import annotations

import numpy as np

from xp.bandit.world import make_world, run_agent


def test_world_shapes():
    w = make_world(np.arange(5), lam=0.01, horizon=40, k=3, sigma=1.0)
    assert w.means.shape == (5, 40, 3)
    assert w.noise.shape == (5, 40)


def test_stationary_when_lambda_zero():
    w = make_world(np.arange(4), lam=0.0, horizon=60, k=4, sigma=1.0)
    assert np.all(w.means == w.means[:, :1, :])


def test_jump_count_matches_rate():
    horizon, n = 2000, 200
    w = make_world(np.arange(n), lam=0.01, horizon=horizon, k=3, sigma=1.0)
    changes = (np.diff(w.means, axis=1) != 0).any(axis=2).sum(axis=1)
    assert abs(changes.mean() - 0.01 * horizon) < 3.0


def test_deterministic_per_seed_independent_of_batching():
    a = make_world(np.array([3, 4, 5]), 0.02, 50, 3, 1.0)
    b = make_world(np.array([4]), 0.02, 50, 3, 1.0)
    assert np.array_equal(a.means[1], b.means[0])
    assert np.array_equal(a.noise[1], b.noise[0])


def test_noise_and_jump_timing_coupled_across_lambda():
    lo = make_world(np.arange(3), 0.01, 200, 3, 1.0)
    hi = make_world(np.arange(3), 0.05, 200, 3, 1.0)
    assert np.array_equal(lo.noise, hi.noise)
    lo_jumps = (np.diff(lo.means, axis=1) != 0).any(axis=2)
    hi_jumps = (np.diff(hi.means, axis=1) != 0).any(axis=2)
    assert np.all(hi_jumps | ~lo_jumps)  # every low-rate jump also occurs at the high rate


class _Oracle:
    def __init__(self, world):
        self.world = world

    def reset(self, n_envs):
        pass

    def select(self, t):
        return self.world.means[:, t, :].argmax(axis=1)

    def update(self, t, actions, rewards):
        pass


def test_oracle_has_zero_regret():
    w = make_world(np.arange(6), 0.02, 80, 4, 1.0)
    regret = run_agent(_Oracle(w), w)
    assert regret.shape == (6,)
    assert np.allclose(regret, 0.0)


class _FixedArm:
    def reset(self, n_envs):
        self.n = n_envs

    def select(self, t):
        return np.zeros(self.n, dtype=int)

    def update(self, t, actions, rewards):
        pass


def test_regret_of_fixed_arm_is_nonnegative_and_positive_on_average():
    w = make_world(np.arange(30), 0.0, 100, 5, 1.0)
    regret = run_agent(_FixedArm(), w)
    assert np.all(regret >= 0)
    assert regret.mean() > 0.1
