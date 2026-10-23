from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from xp.errors import ProtocolError  # noqa: E402
from xp.minigrid.env import (  # noqa: E402
    FEWSHOT_EPISODE_OFFSET, TEST_LAYOUT_BASE, TRAIN_STRIDE, VecEnv, eval_layout_seeds, train_layout_seed,
)


def test_train_and_test_layout_pools_are_disjoint():
    top_train = train_layout_seed(999, TRAIN_STRIDE - 1)
    assert top_train < TEST_LAYOUT_BASE <= min(eval_layout_seeds(5))


def test_training_layout_seed_cannot_reach_the_test_pool():
    with pytest.raises(ProtocolError):
        train_layout_seed(1000, 0)
    with pytest.raises(ProtocolError):
        train_layout_seed(0, TRAIN_STRIDE)


def test_validation_pool_is_disjoint_from_training_and_test():
    from xp.minigrid.env import VALIDATION_LAYOUT_BASE

    assert set(eval_layout_seeds(200)).isdisjoint(eval_layout_seeds(200, VALIDATION_LAYOUT_BASE))
    assert max(eval_layout_seeds(200)) < VALIDATION_LAYOUT_BASE
    with pytest.raises(ProtocolError):
        eval_layout_seeds(10, base=12345)


def test_vecenv_is_deterministic_and_resets_with_fresh_layouts():
    a, b = VecEnv(5, 3, seed=4), VecEnv(5, 3, seed=4)
    assert (a.obs == b.obs).all()
    rng = np.random.default_rng(0)
    for _ in range(400):
        act = rng.integers(0, 7, 3)
        oa, ra, da = a.step(act)
        ob, rb, db = b.step(act)
        assert (oa == ob).all() and (da == db).all()
    assert a.recent  # at least one episode finished (truncation at 10 * size**2 = 250 steps)
    assert a.next_episode > 3  # auto-resets consumed new layout seeds


def test_vecenv_raises_when_its_layout_stream_runs_out():
    env = VecEnv(5, 2, seed=1, first_episode=FEWSHOT_EPISODE_OFFSET - 2, limit=FEWSHOT_EPISODE_OFFSET)
    with pytest.raises(ProtocolError):
        for _ in range(600):
            env.step(np.zeros(2, dtype=int))
