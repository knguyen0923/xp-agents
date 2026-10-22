from __future__ import annotations

import numpy as np
import pytest

from xp.errors import ProtocolError
from xp.protocol.phase import Phase, config_hash


def test_from_config_builds_seed_pools(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    assert phase.name == "bandit"
    assert phase.tune_seeds == tuple(range(0, 10))
    assert phase.eval_seeds == tuple(range(100, 120))
    assert phase.sweep == (0.0, 0.01, 0.05)
    assert phase.familiar == 0.01
    assert phase.chunk == 10 and phase.agent_seed == 7


def test_overlapping_pools_rejected(tiny_cfg):
    tiny_cfg["seeds"]["eval"] = [5, 20]
    with pytest.raises(ProtocolError):
        Phase.from_config(tiny_cfg)


def test_familiar_must_be_in_sweep(tiny_cfg):
    tiny_cfg["knob"]["familiar"] = 0.3
    with pytest.raises(ProtocolError):
        Phase.from_config(tiny_cfg)


def test_hash_changes_with_config(tiny_cfg):
    h1 = config_hash(tiny_cfg)
    tiny_cfg["horizon"] = 51
    assert config_hash(tiny_cfg) != h1
    assert len(h1) == 12


def test_hash_is_key_order_independent(tiny_cfg):
    reordered = dict(reversed(list(tiny_cfg.items())))
    assert config_hash(reordered) == config_hash(tiny_cfg)


def test_assert_eval_seeds_rejects_tune_seed(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    phase.assert_eval_seeds(np.array([100, 101]))
    with pytest.raises(ProtocolError):
        phase.assert_eval_seeds(np.array([3]))


def test_assert_tune_seeds_rejects_eval_seed(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    phase.assert_tune_seeds(np.array([0, 1]))
    with pytest.raises(ProtocolError):
        phase.assert_tune_seeds(np.array([105]))
