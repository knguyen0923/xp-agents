from __future__ import annotations

import multiprocessing as mp
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np

from xp.errors import ProtocolError

HELDOUT_BASE = 1_000_000  # held-out evaluation env seeds start here; every training env seed (seed * 1000 + i) is far below it
_MAX_ENVS = 1000


def world_seed(env_seed: int, episode: int) -> int:
    """The world seed `crafter.Env` uses for an episode (crafter 1.8.3 `env.py` line 74)."""
    return hash((env_seed, episode)) % (2**31 - 1)


def _check(seed: int, n: int) -> None:
    if seed < 0 or not 0 <= n <= _MAX_ENVS:
        raise ProtocolError(f"seed {seed} or env count {n} is outside the allowed range (seed >= 0, n <= {_MAX_ENVS})")


def train_env_seeds(seed: int, n: int) -> List[int]:
    _check(seed, n)
    return [int(seed) * 1000 + i for i in range(n)]


def heldout_env_seeds(seed: int, n: int) -> List[int]:
    _check(seed, n)
    return [HELDOUT_BASE + int(seed) * 1000 + i for i in range(n)]


def _pool_env_seeds(name: str, seeds: Iterable[int], n_envs: int) -> List[int]:
    fn = heldout_env_seeds if name == "heldout" else train_env_seeds
    return [e for s in seeds for e in fn(int(s), n_envs)]


def assert_world_pools_disjoint(pools: Dict[str, Iterable[int]], n_envs: int, episodes: int = 2000, max_overlap: float = 0.001) -> None:
    """No two pools may share an env seed, and their world seeds may overlap only by hash chance (<= `max_overlap` of a pool)."""
    env_seeds = {name: _pool_env_seeds(name, seeds, n_envs) for name, seeds in pools.items()}
    for a, b in combinations(env_seeds, 2):
        shared = set(env_seeds[a]) & set(env_seeds[b])
        if shared:
            raise ProtocolError(f"world pools {a!r} and {b!r} share env seeds, e.g. {sorted(shared)[:3]}")
    worlds = {name: {world_seed(e, ep) for e in es for ep in range(1, episodes + 1)} for name, es in env_seeds.items()}
    for a, b in combinations(worlds, 2):
        frac = len(worlds[a] & worlds[b]) / max(1, min(len(worlds[a]), len(worlds[b])))
        if frac > max_overlap:
            raise ProtocolError(f"world pools {a!r} and {b!r} overlap: {frac:.4%} of worlds are shared (limit {max_overlap:.4%})")


def _worker(conn, area, env_seed):
    import crafter

    env = crafter.Env(area=(area, area), seed=env_seed)
    conn.send(env.reset())
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            obs, reward, done, info = env.step(data)
            ep = None
            if done:
                ep = [k for k, v in info["achievements"].items() if v > 0]
                obs = env.reset()
            conn.send((obs, float(reward), bool(done), ep))
    finally:
        conn.close()


class CrafterVec:
    """One crafter env per entry of `env_seeds`, each in its own process; finished episodes auto-reset.

    `step` returns (obs, rewards, dones, episodes) where `episodes[i]` is None, or the achievements unlocked in the
    episode env i just finished.
    """

    def __init__(self, env_seeds: List[int], area: int = 32):
        ctx = mp.get_context("spawn")  # fork is unsafe with torch
        self.n = len(env_seeds)
        self.conns, self.procs = [], []
        for seed in env_seeds:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, int(area), int(seed)), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.obs = np.stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        out = [c.recv() for c in self.conns]
        self.obs = np.stack([o[0] for o in out])
        return self.obs, np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out]), [o[3] for o in out]

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
