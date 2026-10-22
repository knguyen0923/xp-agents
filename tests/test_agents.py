from __future__ import annotations

import numpy as np

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.world import make_world, run_agent


def test_kairi_greedy_picks_argmax_q():
    agent = KairiAgent(k=3, epsilon=0.0, alpha=0.5, seed=0)
    agent.reset(2)
    agent.q = np.array([[0.0, 1.0, 0.5], [2.0, 0.0, 1.0]])
    assert agent.select(0).tolist() == [1, 0]


def test_kairi_alpha_one_tracks_last_reward():
    agent = KairiAgent(k=3, epsilon=0.0, alpha=1.0, seed=0)
    agent.reset(2)
    agent.update(0, np.array([2, 0]), np.array([0.7, -0.3]))
    assert agent.q[0, 2] == 0.7 and agent.q[1, 0] == -0.3


def test_kairi_constant_step_update():
    agent = KairiAgent(k=2, epsilon=0.0, alpha=0.5, seed=0)
    agent.reset(1)
    agent.update(0, np.array([0]), np.array([1.0]))
    agent.update(1, np.array([0]), np.array([1.0]))
    assert np.isclose(agent.q[0, 0], 0.75)


def test_kairi_epsilon_one_explores_all_arms():
    agent = KairiAgent(k=4, epsilon=1.0, alpha=0.1, seed=1)
    agent.reset(500)
    assert set(agent.select(0).tolist()) == {0, 1, 2, 3}


def test_kairi_reset_makes_runs_reproducible():
    w = make_world(np.arange(10), 0.01, 100, 5, 1.0)
    agent = KairiAgent(5, 0.1, 0.2, seed=3)
    assert np.array_equal(run_agent(agent, w), run_agent(agent, w))


def test_kaiser_pulls_each_arm_once_first():
    agent = KaiserAgent(k=3, gamma=1.0, c=1.0)
    agent.reset(1)
    pulled = []
    for t in range(3):
        a = agent.select(t)
        pulled.append(int(a[0]))
        agent.update(t, a, np.array([0.0]))
    assert sorted(pulled) == [0, 1, 2]


def test_kaiser_with_zero_bonus_exploits_best_arm():
    agent = KaiserAgent(k=3, gamma=1.0, c=0.0)
    agent.reset(1)
    means = np.array([0.0, 1.0, 0.5])
    chosen = []
    for t in range(10):
        a = agent.select(t)
        chosen.append(int(a[0]))
        agent.update(t, a, means[a])
    assert chosen[3:] == [1] * 7


def test_kaiser_discount_forgets_old_evidence():
    agent = KaiserAgent(k=2, gamma=0.5, c=0.0)
    agent.reset(1)
    agent.update(0, np.array([0]), np.array([1.0]))
    before = agent.cnt[0, 0]
    agent.update(1, np.array([1]), np.array([0.0]))
    assert agent.cnt[0, 0] == before * 0.5


def test_both_beat_random_on_stationary_world():
    w = make_world(np.arange(40), 0.0, 500, 5, 1.0)
    rand = run_agent(KairiAgent(5, 1.0, 0.1, seed=0), w).mean()
    kairi = run_agent(KairiAgent(5, 0.1, 0.1, seed=0), w).mean()
    kaiser = run_agent(KaiserAgent(5, 1.0, 1.0), w).mean()
    assert kairi < rand and kaiser < rand
