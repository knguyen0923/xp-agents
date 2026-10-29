"""Needs a running local Showdown server (see test_showdown_env.py for XP_SHOWDOWN_DIR and XP_NODE). Run with: pytest -m showdown"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.environment import DoublesEnv, SingleAgentWrapper  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402

from xp.showdown.encode import OBS_DIM  # noqa: E402
from xp.showdown.env import XPDoublesEnv  # noqa: E402
from xp.showdown.masks import N_ACTIONS, action_masks, reference_masks, restrict_second  # noqa: E402
from xp.showdown.teams import PoolTeambuilder, load_pool, validate_team  # noqa: E402
from xp.showdown.vec_env import ShowdownVecEnv  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"
TEAMS = Path(__file__).resolve().parent.parent / "configs" / "showdown_teams"
SHOWDOWN_DIR = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))
NODE = os.environ.get("XP_NODE", "node")

pytestmark = pytest.mark.showdown


@pytest.fixture(scope="module")
def pool():
    if not SHOWDOWN_DIR.exists():
        pytest.skip(f"no Showdown checkout at {SHOWDOWN_DIR} (set XP_SHOWDOWN_DIR)")
    teams = load_pool(TEAMS / "tune")
    for text in teams.values():
        validate_team(text, FORMAT, SHOWDOWN_DIR, node=NODE)
    return teams


def real_first_battle():
    """A live battle right after reset(): for other server tests (needs the server and a Showdown checkout)."""
    suffix = str(random.randrange(10**5))
    env = XPDoublesEnv(
        battle_format=FORMAT, team=PoolTeambuilder(load_pool(TEAMS / "tune"), 3), strict=True, log_level=40,
        account_configuration1=AccountConfiguration("ra" + suffix, None), account_configuration2=AccountConfiguration("rb" + suffix, None),
    )
    wrapped = SingleAgentWrapper(env, RandomPlayer(battle_format=FORMAT, start_listening=False))
    try:
        wrapped.reset()
        return env.battle1
    finally:
        wrapped.close()


def test_fast_masks_equal_the_decoder_reference_and_every_masked_action_is_accepted(pool):
    suffix = str(random.randrange(10**5))
    env = XPDoublesEnv(
        battle_format=FORMAT, team=PoolTeambuilder(pool, 3), strict=True, log_level=40,
        account_configuration1=AccountConfiguration("ta" + suffix, None), account_configuration2=AccountConfiguration("tb" + suffix, None),
    )
    wrapped = SingleAgentWrapper(env, RandomPlayer(battle_format=FORMAT, start_listening=False))
    rng, states = np.random.default_rng(0), 0
    try:
        for _ in range(25):
            wrapped.reset()
            done = False
            while not done:
                m = action_masks(env.battle1)
                assert np.array_equal(m, reference_masks(env.battle1))
                a0 = int(rng.choice(np.flatnonzero(m[0])))
                a1 = int(rng.choice(np.flatnonzero(restrict_second(m[1], a0))))
                DoublesEnv.action_to_order(np.array([a0, a1]), env.battle1, strict=True)  # raises if the pair is illegal
                _, _, term, trunc, _ = wrapped.step(np.array([a0, a1]))
                done, states = term or trunc, states + 1
        assert states > 100
    finally:
        wrapped.close()


def test_vec_env_steps_in_parallel_and_resets_finished_battles(pool):
    venv = ShowdownVecEnv(2, FORMAT, pool, "random", seed=0)
    rng = np.random.default_rng(1)
    try:
        assert venv.obs.shape == (2, OBS_DIM) and venv.masks.shape == (2, 2, N_ACTIONS) and venv.masks.dtype == bool
        finished = 0
        for _ in range(120):
            acts = np.zeros((2, 2), dtype=np.int64)
            for i in range(2):
                a0 = int(rng.choice(np.flatnonzero(venv.masks[i, 0])))
                acts[i] = (a0, int(rng.choice(np.flatnonzero(restrict_second(venv.masks[i, 1], a0)))))
            rewards, dones = venv.step(acts)
            assert set(rewards[dones == 1].tolist()) <= {1.0, -1.0} and (rewards[dones == 0] == 0).all()
            assert venv.obs.shape == (2, OBS_DIM)
            finished += int(dones.sum())
        assert finished >= 4 and venv.recent_stats()[1] == min(finished, 100)
    finally:
        venv.close()


def test_a_worker_reports_an_illegal_action_instead_of_hanging(pool):
    venv = ShowdownVecEnv(1, FORMAT, pool, "random", seed=0)
    try:
        bad = np.array([[N_ACTIONS - 1, N_ACTIONS - 1]])  # a terastallize action: never legal in this format
        with pytest.raises(RuntimeError, match="env worker failed"):
            venv.step(bad)
    finally:
        venv.close()
