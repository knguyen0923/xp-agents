"""Shared toy pieces for the Showdown learner tests: PPO settings and a one-decision battle env."""
from __future__ import annotations

from collections import deque

import numpy as np

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS

PPO = {"n_steps": 16, "epochs": 4, "minibatches": 4, "lr": 0.003, "gamma": 0.99, "lam": 0.95, "clip": 0.2, "vf_coef": 0.5}
GOOD = 3  # the action that wins in the toy env


class ToyVec:
    """One-decision battles: win (+1) when slot 0 plays action GOOD, else lose (-1). Actions 50..106 are never legal."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.results: deque = deque(maxlen=100)
        self._draw()

    def _draw(self):
        self.obs = self.rng.normal(size=(self.n, OBS_DIM)).astype(np.float32)
        self.masks = np.zeros((self.n, 2, N_ACTIONS), dtype=bool)
        self.masks[:, :, :50] = True

    def step(self, actions):
        assert self.masks[np.arange(self.n), 0, actions[:, 0]].all(), "slot 0 played an illegal action"
        rewards = np.where(actions[:, 0] == GOOD, 1.0, -1.0).astype(np.float32)
        self.results.extend((rewards > 0).astype(float).tolist())
        self._draw()
        return rewards, np.ones(self.n, dtype=np.float32)

    def recent_stats(self):
        return (float(np.mean(self.results)) if self.results else float("nan")), len(self.results)

    def recent_window_full(self):
        return len(self.results) == self.results.maxlen

    def close(self):
        pass
