"""Needs a running local Showdown server (see docs/spikes/showdown_spike.md). Run with: pytest -m showdown

XP_SHOWDOWN_DIR: the Showdown checkout (default ~/xp-showdown-spike/pokemon-showdown).
XP_NODE: a Node >= 22 binary for the team validator (default `node`; the system Node may be older).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.environment import SingleAgentWrapper  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402

from xp.showdown.encode import OBS_DIM  # noqa: E402
from xp.showdown.env import XPDoublesEnv  # noqa: E402
from xp.showdown.teams import PoolTeambuilder, load_pool, validate_team  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"
TEAMS = Path(__file__).resolve().parent.parent / "configs" / "showdown_teams"
SHOWDOWN_DIR = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))

NODE = os.environ.get("XP_NODE", "node")

pytestmark = pytest.mark.showdown


@pytest.fixture(scope="module")
def tune_pool():
    if not SHOWDOWN_DIR.exists():
        pytest.skip(f"no Showdown checkout at {SHOWDOWN_DIR} (set XP_SHOWDOWN_DIR)")
    pool = load_pool(TEAMS / "tune")
    for text in pool.values():  # a rejected team makes poke-env hang with no error, so validate first
        validate_team(text, FORMAT, SHOWDOWN_DIR, node=NODE)
    return pool


def test_random_play_runs_to_the_end_with_the_xp_observation_and_sparse_reward(tune_pool):
    suffix = str(random.randrange(10**5))
    team = PoolTeambuilder(tune_pool, seed=0)
    env = XPDoublesEnv(
        battle_format=FORMAT, team=team, strict=False, log_level=40,
        account_configuration1=AccountConfiguration("xa" + suffix, None),
        account_configuration2=AccountConfiguration("xb" + suffix, None),
    )
    wrapped = SingleAgentWrapper(env, RandomPlayer(battle_format=FORMAT, start_listening=False))
    try:
        assert wrapped.observation_space.shape == (OBS_DIM,)
        finals = []
        for _ in range(3):
            obs, _ = wrapped.reset()
            assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
            done, reward = False, 0.0
            while not done:
                obs, reward, term, trunc, _ = wrapped.step(wrapped.action_space.sample())
                assert wrapped.observation_space.contains(obs)
                done = term or trunc
            finals.append(reward)
        assert all(r in (1.0, -1.0) for r in finals)
        assert set(team.drawn) <= set(tune_pool)
    finally:
        wrapped.close()
