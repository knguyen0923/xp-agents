from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import DoublesEnv

from xp.showdown.encode import encode_battle, obs_dim


class XPDoublesEnv(DoublesEnv):
    """Doubles env with the fixed XP feature vector and a sparse reward: +1 win, -1 loss, 0 otherwise.

    Both sides' teams come from the `team` argument (pass a `PoolTeambuilder`), and team preview uses poke-env's
    default random pick of 4 for both sides. Pair with `SingleAgentWrapper(env, opponent)` for a gymnasium env.
    """

    def __init__(self, matchup: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.matchup = bool(matchup)
        self.estimator_calls = 0     # matchup estimates attempted / failed over this env's life; the worker reports them on close
        self.estimator_errors = 0
        space = Box(-2.0, 2.0, shape=(obs_dim(self.matchup),), dtype=np.float32)  # move priority reaches -7 / 5 = -1.4
        self.observation_spaces = {agent: space for agent in self.possible_agents}

    def embed_battle(self, battle) -> np.ndarray:
        if not self.matchup:
            return encode_battle(battle)
        from xp.showdown.estimate import PokeEnvEstimator
        est = PokeEnvEstimator(battle)
        try:
            return encode_battle(battle, matchup=True, estimator=est)
        finally:
            self.estimator_calls += est.calls
            self.estimator_errors += est.errors

    def calc_reward(self, battle) -> float:
        if battle.won:
            return 1.0
        if battle.lost:
            return -1.0
        return 0.0
