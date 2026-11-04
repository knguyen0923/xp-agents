# tests/test_crafter_env.py
from __future__ import annotations

import numpy as np
import pytest

from xp.crafter.env import HELDOUT_BASE, CrafterVec, assert_world_pools_disjoint, heldout_env_seeds, train_env_seeds, world_seed
from xp.errors import ProtocolError


def test_world_seed_is_deterministic_and_in_range():
    assert world_seed(5, 3) == world_seed(5, 3)
    assert world_seed(5, 3) != world_seed(5, 4)
    assert 0 <= world_seed(123456, 99) < 2**31 - 1


def test_env_seed_pools():
    assert train_env_seeds(3, 4) == [3000, 3001, 3002, 3003]
    assert heldout_env_seeds(3, 2) == [HELDOUT_BASE + 3000, HELDOUT_BASE + 3001]
    with pytest.raises(ProtocolError):
        train_env_seeds(0, 1001)
    with pytest.raises(ProtocolError):
        train_env_seeds(-1, 4)


def test_real_pools_pass_the_guard():
    assert_world_pools_disjoint({"tune": [0, 1, 2], "eval": list(range(100, 110)), "heldout": list(range(100, 110))}, n_envs=8, episodes=500)


def test_guard_rejects_a_shared_seed():
    with pytest.raises(ProtocolError, match="share"):
        assert_world_pools_disjoint({"tune": [0, 1], "eval": [1, 2]}, n_envs=4, episodes=10)


def test_guard_rejects_systematic_world_overlap(monkeypatch):
    import xp.crafter.env as env

    monkeypatch.setattr(env, "world_seed", lambda s, e: e)  # every env sees the same worlds: pools overlap completely
    with pytest.raises(ProtocolError, match="overlap"):
        env.assert_world_pools_disjoint({"tune": [0], "eval": [1]}, n_envs=2, episodes=50)


@pytest.mark.crafter
def test_real_env_steps_and_finishes_an_episode():
    vec = CrafterVec(train_env_seeds(0, 2), area=32)
    try:
        assert vec.obs.shape == (2, 64, 64, 3) and vec.obs.dtype == np.uint8
        finished = []
        for _ in range(12000):  # an episode is at most 10,000 steps
            obs, rew, done, eps = vec.step([0, 0])  # noop: the player eventually dies or times out
            assert obs.shape == (2, 64, 64, 3) and rew.shape == (2,) and done.shape == (2,)
            finished += [e for e in eps if e is not None]
            if finished:
                break
        assert finished and all(isinstance(a, str) for a in finished[0])
    finally:
        vec.close()


@pytest.mark.crafter
def test_same_env_seed_gives_the_same_first_world():
    a, b = CrafterVec([7], area=32), CrafterVec([7], area=32)
    try:
        assert np.array_equal(a.obs, b.obs)
    finally:
        a.close()
        b.close()


@pytest.mark.crafter
def test_n_actions_matches_the_real_env_and_the_last_action_steps():
    import crafter

    from xp.crafter.ppo import N_ACTIONS

    assert N_ACTIONS == crafter.Env(area=(32, 32), seed=0).action_space.n
    vec = CrafterVec(train_env_seeds(0, 2), area=32)
    try:
        obs, rew, done, eps = vec.step([16, 16])
        assert obs.shape == (2, 64, 64, 3)
    finally:
        vec.close()
