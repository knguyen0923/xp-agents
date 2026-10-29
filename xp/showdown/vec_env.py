from __future__ import annotations

import multiprocessing
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


def _worker(conn, cfg: dict) -> None:
    """One process = one env (two accounts on the local server) against one scripted opponent, auto-resetting."""
    from poke_env import AccountConfiguration
    from poke_env.environment import DoublesEnv, SingleAgentWrapper

    from xp.showdown.env import XPDoublesEnv
    from xp.showdown.masks import action_masks
    from xp.showdown.opponents import make_opponent
    from xp.showdown.teams import PoolTeambuilder

    env = XPDoublesEnv(
        battle_format=cfg["battle_format"], team=PoolTeambuilder(cfg["teams"], cfg["team_seed"]), strict=False, log_level=40, save_replays=cfg.get("save_replays") or False,
        account_configuration1=AccountConfiguration(cfg["name"] + "a", None),
        account_configuration2=AccountConfiguration(cfg["name"] + "b", None),
    )
    wrapped = SingleAgentWrapper(env, make_opponent(cfg["opponent"], cfg["battle_format"]))
    try:
        obs, _ = wrapped.reset()
        conn.send((obs, action_masks(env.battle1)))
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            DoublesEnv.action_to_order(np.asarray(data, dtype=np.int64), env.battle1, strict=True)  # fail loudly on a mask bug
            obs, reward, term, trunc, _ = wrapped.step(data)
            done = bool(term or trunc)
            if done:
                obs, _ = wrapped.reset()
            conn.send((obs, float(reward), done, action_masks(env.battle1)))
    except BaseException as e:  # report instead of letting the parent wait forever
        conn.send(("error", repr(e)))
    finally:
        wrapped.close()


class ShowdownVecEnv:
    """`n` env processes stepped in parallel. After `step`, `obs` / `masks` belong to the next decision (auto-reset on done).

    Interface used by the learner: `n`, `obs` (n, D) float32, `masks` (n, 2, 107) bool, `step(actions (n, 2)) -> (rewards, dones)`,
    `recent_stats() -> (win_rate over the last 100 finished battles, count)`, `recent_window_full()`, `close()`.
    """

    def __init__(self, n: int, battle_format: str, teams: Dict[str, str], opponent: str, seed: int, name: str = "v",
                 save_replays: Optional[str] = None):
        ctx = multiprocessing.get_context("spawn")
        self.n = n
        suffix = "%05d" % random.SystemRandom().randrange(10**5)  # fresh per run, NOT from the seed: stale accounts/battles from an old run block reused names
        self.conns, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            cfg = {"battle_format": battle_format, "teams": teams, "opponent": opponent, "team_seed": seed * 1000 + i,
                   "name": f"{name}{suffix}{i}"[:16], "save_replays": save_replays}
            proc = ctx.Process(target=_worker, args=(child, cfg), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        self.results: deque = deque(maxlen=100)
        first = [self._recv(c) for c in self.conns]
        self.obs = np.stack([f[0] for f in first]).astype(np.float32)
        self.masks = np.stack([f[1] for f in first])

    @staticmethod
    def _recv(conn):
        msg = conn.recv()
        if isinstance(msg, tuple) and isinstance(msg[0], str) and msg[0] == "error":
            raise RuntimeError(f"env worker failed: {msg[1]}")
        return msg

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for conn, a in zip(self.conns, actions):
            conn.send(("step", np.asarray(a, dtype=np.int64)))
        rewards, dones, obs, masks = np.zeros(self.n, np.float32), np.zeros(self.n, np.float32), [], []
        for i, conn in enumerate(self.conns):
            o, r, d, m = self._recv(conn)
            rewards[i], dones[i] = r, float(d)
            obs.append(o)
            masks.append(m)
            if d:
                self.results.append(1.0 if r > 0 else 0.0)
        self.obs, self.masks = np.stack(obs).astype(np.float32), np.stack(masks)
        return rewards, dones

    def recent_stats(self) -> Tuple[float, int]:
        return (float(np.mean(self.results)) if self.results else float("nan")), len(self.results)

    def recent_window_full(self) -> bool:
        return len(self.results) == self.results.maxlen

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
