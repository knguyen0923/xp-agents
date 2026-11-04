# tests/test_crafter_stats.py
from __future__ import annotations

import math

import pytest

from xp.crafter.stats import ACHIEVEMENTS, EpisodeStats, crafter_score, nan_to_zero


def test_achievement_list_has_22_unique_names():
    assert len(ACHIEVEMENTS) == 22 and len(set(ACHIEVEMENTS)) == 22
    assert list(ACHIEVEMENTS) == sorted(ACHIEVEMENTS)


def test_episode_stats_window_rates_and_nan_before_full():
    st = EpisodeStats(window=2)
    assert math.isnan(st.ach()) and not st.full()
    st.add(["collect_wood"])
    assert math.isnan(st.ach())  # one episode: window not full yet
    st.add(["collect_wood", "eat_cow", "wake_up"])
    assert st.full() and st.ach() == 2.0
    st.add(["eat_cow"])  # the first episode falls out of the window
    assert st.ach() == 2.0 and st.total == 3
    r = st.rates(["collect_wood", "eat_cow", "wake_up", "place_table"])
    assert r == {"collect_wood": 0.5, "eat_cow": 1.0, "wake_up": 0.5, "place_table": 0.0}


def test_rates_before_any_episode_are_zero():
    assert set(EpisodeStats(5).rates().values()) == {0.0}


def test_nan_to_zero():
    assert nan_to_zero(float("nan")) == 0.0 and nan_to_zero(3.5) == 3.5


def test_crafter_score_is_zero_for_no_unlocks_and_100_for_all():
    assert crafter_score({a: 0.0 for a in ACHIEVEMENTS}) == 0.0
    assert crafter_score({a: 1.0 for a in ACHIEVEMENTS}) == pytest.approx(100.0)
    half = crafter_score({a: 0.5 for a in ACHIEVEMENTS})
    assert half == pytest.approx(50.0)  # geometric mean of 1 + 50 is 51, minus 1


@pytest.mark.crafter
def test_achievement_names_match_the_installed_crafter():
    from crafter import constants

    assert tuple(constants.achievements) == ACHIEVEMENTS
