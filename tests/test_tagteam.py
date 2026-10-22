from __future__ import annotations

import numpy as np

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.tagteam import ChangeTriggered, FixedSchedule, MetaBandit, TagTeamAgent
from xp.bandit.world import make_world, run_agent


def test_fixed_schedule_alternates_in_blocks():
    c = FixedSchedule(block=3)
    c.reset(2)
    seq = [int(c.who(t)[0]) for t in range(12)]
    assert seq == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    assert c.who(0).shape == (2,)


def test_change_triggered_starts_with_kairi_and_fires_on_big_residuals():
    c = ChangeTriggered(delta=1.0, h=3.0, window=4)
    c.reset(2)
    assert c.who(0).tolist() == [0, 0]
    pred = np.zeros(2)
    # env 0 sees huge residuals, env 1 sees tiny ones
    for t in range(3):
        c.observe(t, c.who(t), np.array([5.0, 0.1]), pred)
    assert c.who(3).tolist() == [1, 0]


def test_change_triggered_returns_to_kairi_after_window():
    c = ChangeTriggered(delta=0.0, h=1.0, window=2)
    c.reset(1)
    c.observe(0, c.who(0), np.array([2.0]), np.zeros(1))  # fires immediately
    assert c.who(1).tolist() == [1]
    c.observe(1, c.who(1), np.array([0.0]), np.zeros(1))
    c.observe(2, c.who(2), np.array([0.0]), np.zeros(1))
    assert c.who(3).tolist() == [0]


def test_meta_bandit_tries_both_then_prefers_better_agent():
    c = MetaBandit(gamma=1.0, c=0.01, block=2)
    c.reset(1)
    chosen = []
    for t in range(20):
        w = c.who(t)
        chosen.append(int(w[0]))
        reward = 1.0 if w[0] == 1 else 0.0  # Kaiser is better
        c.observe(t, w, np.array([reward]), np.zeros(1))
    assert chosen[0] == 0 and chosen[2] == 1  # explores both first
    assert chosen[-4:] == [1, 1, 1, 1]


class _Spy(KairiAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.updates = 0

    def update(self, t, actions, rewards):
        self.updates += 1
        super().update(t, actions, rewards)


def test_tagteam_updates_both_agents_every_step_even_when_other_chooses():
    kairi = _Spy(k=3, epsilon=0.0, alpha=1.0, seed=0)
    kaiser = KaiserAgent(k=3, gamma=1.0, c=1.0)
    team = TagTeamAgent(kairi, kaiser, FixedSchedule(block=1000))  # Kairi always in charge
    w = make_world(np.arange(4), 0.0, 30, 3, 1.0)
    run_agent(team, w)
    assert kairi.updates == 30
    assert kaiser.cnt.sum() > 0  # Kaiser learned from actions it did not choose


def test_tagteam_passes_pre_update_kairi_prediction_to_controller():
    seen = []

    class Probe(FixedSchedule):
        def observe(self, t, who, rewards, pred):
            seen.append(pred.copy())

    kairi = KairiAgent(k=2, epsilon=0.0, alpha=1.0, seed=0)
    team = TagTeamAgent(kairi, KaiserAgent(2, 1.0, 1.0), Probe(block=1000))
    team.reset(1)
    a = team.select(0)
    team.update(0, a, np.array([0.9]))
    assert seen[0][0] == 0.0  # q was still 0 before the update


def test_tagteam_runs_reproducibly_with_each_rule():
    w = make_world(np.arange(8), 0.02, 120, 4, 1.0)
    for ctrl in (FixedSchedule(10), ChangeTriggered(1.0, 4.0, 10), MetaBandit(0.95, 0.5, 10)):
        team = TagTeamAgent(KairiAgent(4, 0.1, 0.3, seed=2), KaiserAgent(4, 0.95, 1.0), ctrl)
        r1 = run_agent(team, w)
        r2 = run_agent(team, w)
        assert np.array_equal(r1, r2)
        assert r1.shape == (8,)
