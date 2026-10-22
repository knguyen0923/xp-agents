from __future__ import annotations

import numpy as np


class KairiAgent:
    """Model-free, reward-driven: epsilon-greedy with a constant step-size value update."""

    def __init__(self, k: int, epsilon: float, alpha: float, seed: int = 0):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.seed = seed

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.rng = np.random.default_rng(self.seed)
        self.q = np.zeros((n_envs, self.k))

    def select(self, t: int) -> np.ndarray:
        greedy = np.argmax(self.q + 1e-9 * self.rng.random(self.q.shape), axis=1)
        explore = self.rng.random(self.n) < self.epsilon
        random_arm = self.rng.integers(0, self.k, self.n)
        return np.where(explore, random_arm, greedy)

    def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n)
        self.q[rows, actions] += self.alpha * (rewards - self.q[rows, actions])


class KaiserAgent:
    """Optimism under uncertainty: discounted UCB (Garivier & Moulines)."""

    def __init__(self, k: int, gamma: float, c: float, seed: int = 0):
        self.k = k
        self.gamma = gamma
        self.c = c
        self.seed = seed  # unused; kept so factories share one signature

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.cnt = np.zeros((n_envs, self.k))
        self.sum = np.zeros((n_envs, self.k))

    def select(self, t: int) -> np.ndarray:
        total = np.maximum(self.cnt.sum(axis=1, keepdims=True), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            index = self.sum / self.cnt + self.c * np.sqrt(np.log(total) / self.cnt)
        index = np.where(self.cnt <= 0, np.inf, index)
        return np.argmax(index, axis=1)

    def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None:
        self.cnt *= self.gamma
        self.sum *= self.gamma
        rows = np.arange(self.n)
        self.cnt[rows, actions] += 1.0
        self.sum[rows, actions] += rewards
