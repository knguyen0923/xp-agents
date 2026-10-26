from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "spikes", "crafter_spike.py")
_SPEC = importlib.util.spec_from_file_location("crafter_spike", _PATH)
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)


def test_episode_stats_window_and_rates():
    st = cs.EpisodeStats(window=2)
    assert st.ach() == 0.0
    st.add(["a"])
    st.add(["a", "b", "c"])
    st.add(["b"])  # the first episode falls out of the window
    assert st.ach() == 2.0
    assert st.total == 3
    r = st.rates(["a", "b", "c", "d"])
    assert r == {"a": 0.5, "b": 1.0, "c": 0.5, "d": 0.0}


def test_gates_pass_and_fail():
    rates_ok = {k: 0.0 for k in "abcde"}
    rates_ok["f"] = 0.9
    g = cs.gates([(10240, 3.0), (500000, 4.0), (1000000, 6.0)], rates_ok)
    assert g["G1"] and g["G2"] and g["rare_achievements"] == 5
    low = cs.gates([(10240, 3.0), (1000000, 3.2)], rates_ok)
    assert not low["G1"]  # final below 5
    flat = cs.gates([(10240, 6.0), (1000000, 6.0)], rates_ok)
    assert flat["G1"] and not flat["G2"]  # no growth
    no_headroom = cs.gates([(10240, 3.0), (1000000, 7.0)], {k: 0.5 for k in "abcdef"})
    assert not no_headroom["G1"]  # fewer than 5 rare achievements


def test_world_seed_pools_are_disjoint():
    def pool(bases, episodes):
        return {cs.world_seed(b * 1000 + i, e) for b in bases for i in range(8) for e in range(1, episodes + 1)}

    train, ev = pool(range(0, 3), 2000), pool(range(100, 110), 200)
    assert len(train) > 0.99 * 3 * 8 * 2000  # almost no collisions inside a pool
    assert not (train & ev)


def test_same_env_seed_gives_same_first_world():
    crafter = pytest.importorskip("crafter")
    a = crafter.Env(area=(32, 32), seed=7)
    b = crafter.Env(area=(32, 32), seed=7)
    c = crafter.Env(area=(32, 32), seed=8)
    assert np.array_equal(a.reset(), b.reset())
    assert not np.array_equal(a.reset(), c.reset())
    assert cs.world_seed(7, 1) == hash((7, 1)) % (2 ** 31 - 1)


def test_achievement_names_has_22():
    pytest.importorskip("crafter")
    assert len(cs.achievement_names()) == 22
