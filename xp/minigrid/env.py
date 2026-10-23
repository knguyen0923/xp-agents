from __future__ import annotations

import numpy as np
import torch
from minigrid.envs import DoorKeyEnv
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical

from xp.errors import ProtocolError

N_ACTIONS = 7
TRAIN_STRIDE = 1_000_000          # training seed s owns layout seeds [s * STRIDE, (s + 1) * STRIDE)
FEWSHOT_EPISODE_OFFSET = 500_000  # episodes below this index are for familiar-size training, above for few-shot
TEST_LAYOUT_BASE = 1_000_000_000        # final-evaluation layouts: seeds >= this; every training layout seed is below it
VALIDATION_LAYOUT_BASE = 2_000_000_000  # tuning-time layouts, disjoint from training and from the test pool


def make_env(size: int):
    return ImgObsWrapper(DoorKeyEnv(size=int(size)))


def prep(obs_np: np.ndarray, device) -> torch.Tensor:
    """(N, 7, 7, 3) uint8 -> (N, 3, 7, 7) float in [0, 1]."""
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device) / 10.0
    return x.permute(0, 3, 1, 2)


def train_layout_seed(seed: int, episode: int) -> int:
    """Layout seed for the `episode`-th training episode of training seed `seed`; never in the test pool."""
    if not 0 <= episode < TRAIN_STRIDE:
        raise ProtocolError(f"training episode index {episode} is outside [0, {TRAIN_STRIDE})")
    value = int(seed) * TRAIN_STRIDE + int(episode)
    if value >= TEST_LAYOUT_BASE:
        raise ProtocolError(f"training layout seed {value} reaches the test pool (>= {TEST_LAYOUT_BASE})")
    return value


def eval_layout_seeds(n: int, base: int = TEST_LAYOUT_BASE) -> list:
    if base not in (TEST_LAYOUT_BASE, VALIDATION_LAYOUT_BASE) or not 0 <= n < VALIDATION_LAYOUT_BASE - TEST_LAYOUT_BASE:
        raise ProtocolError(f"unknown layout pool base {base} or size {n}")
    return [base + i for i in range(n)]


class VecEnv:
    """Synchronous vector env. Every episode (including after auto-reset) gets an explicit training layout seed."""

    def __init__(self, size: int, n: int, seed: int, first_episode: int = 0, limit: int = FEWSHOT_EPISODE_OFFSET):
        self.n, self.seed, self.limit = n, int(seed), limit
        self.next_episode = first_episode
        self.envs = [make_env(size) for _ in range(n)]
        self.obs = np.stack([self._reset(e) for e in self.envs])
        self.ep_ret = np.zeros(n)
        self.recent = []  # (success, return) of finished episodes

    def _reset(self, env):
        if self.next_episode >= self.limit:
            raise ProtocolError(f"training seed {self.seed} used up its layout stream (limit {self.limit})")
        layout = train_layout_seed(self.seed, self.next_episode)
        self.next_episode += 1
        return env.reset(seed=layout)[0]

    def step(self, actions):
        rews = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = env.step(int(a))
            self.ep_ret[i] += r
            rews[i] = r
            if term or trunc:
                dones[i] = 1.0
                self.recent.append((float(term and r > 0), float(self.ep_ret[i])))
                self.ep_ret[i] = 0.0
                o = self._reset(env)
            self.obs[i] = o
        return self.obs.copy(), rews, dones

    def recent_stats(self, k: int = 100):
        window = self.recent[-k:]
        if not window:
            return float("nan"), float("nan")
        arr = np.array(window)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())


@torch.no_grad()
def evaluate_policy(net, size: int, n_layouts: int, device, base: int = TEST_LAYOUT_BASE) -> dict:
    """Success rate and mean return of a frozen policy on a fixed layout pool. Never updates anything."""
    env = make_env(size)
    wins, rets = 0.0, []
    for layout in eval_layout_seeds(n_layouts, base):
        gen = torch.Generator().manual_seed(layout)  # sampling is deterministic per layout and leaves global RNG alone
        obs, _ = env.reset(seed=layout)
        done, ret, term, r = False, 0.0, False, 0.0
        while not done:
            logits, _ = net(prep(obs[None], device))
            action = int(torch.multinomial(Categorical(logits=logits).probs, 1, generator=gen).item())
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        wins += float(term and r > 0)
        rets.append(ret)
    return {"success": wins / n_layouts, "mean_return": float(np.mean(rets))}
