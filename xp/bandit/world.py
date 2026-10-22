from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class World:
    means: np.ndarray  # (n, T, k) true arm means at each step
    noise: np.ndarray  # (n, T) reward noise (already scaled by sigma)


def make_world(seeds, lam: float, horizon: int, k: int, sigma: float) -> World:
    """Pre-generate one realization per seed. Draw order is fixed and independent
    of `lam`, so the same seed gives the same noise and coupled jump times."""
    seeds = np.asarray(seeds)
    n = len(seeds)
    means = np.empty((n, horizon, k))
    noise = np.empty((n, horizon))
    steps = np.arange(1, horizon + 1)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        m0 = rng.normal(size=k)
        u = rng.random(horizon)
        new_means = rng.normal(size=(horizon, k))
        noise[i] = sigma * rng.normal(size=horizon)
        jump = u < lam
        idx = np.maximum.accumulate(np.where(jump, steps, 0))
        source = np.vstack([m0[None, :], new_means])
        means[i] = source[idx]
    return World(means=means, noise=noise)


def run_agent(agent, world: World) -> np.ndarray:
    """Run `agent` on `world`; return mean per-step dynamic regret per seed."""
    n, horizon, _ = world.means.shape
    rows = np.arange(n)
    best = world.means.max(axis=2)
    total = np.zeros(n)
    agent.reset(n)
    for t in range(horizon):
        actions = agent.select(t)
        mu = world.means[rows, t, actions]
        rewards = mu + world.noise[:, t]
        total += best[:, t] - mu
        agent.update(t, actions, rewards)
    return total / horizon
