from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class World:
    means: np.ndarray  # (n, T, k) true arm means at each step
    noise: np.ndarray  # (n, T) reward noise (already scaled by sigma)


def make_world(seeds, lam: float, horizon: int, k: int, sigma: float, drift: float = 0.0) -> World:
    """Pre-generate one realization per seed. Draw order is fixed and independent
    of `lam`, so the same seed gives the same noise and coupled jump times.
    `drift` adds a per-arm Gaussian random walk (std `drift` per step, starting after step 0) to the means; its draws come
    last and only when drift > 0, so worlds without drift are unchanged."""
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
        if drift > 0:
            steps_walk = drift * rng.normal(size=(horizon, k))
            steps_walk[0] = 0.0
            means[i] += np.cumsum(steps_walk, axis=0)
    return World(means=means, noise=noise)


def run_agent_trace(agent, world: World) -> np.ndarray:
    """Run `agent` on `world`; return the per-step dynamic regret, shape (n, horizon)."""
    n, horizon, _ = world.means.shape
    rows = np.arange(n)
    best = world.means.max(axis=2)
    trace = np.zeros((n, horizon))
    agent.reset(n)
    for t in range(horizon):
        actions = agent.select(t)
        mu = world.means[rows, t, actions]
        rewards = mu + world.noise[:, t]
        trace[:, t] = best[:, t] - mu
        agent.update(t, actions, rewards)
    return trace


def run_agent(agent, world: World) -> np.ndarray:
    """Run `agent` on `world`; return mean per-step dynamic regret per seed."""
    return run_agent_trace(agent, world).mean(axis=1)
