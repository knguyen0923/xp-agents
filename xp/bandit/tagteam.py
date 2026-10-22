from __future__ import annotations

import numpy as np

KAIRI, KAISER = 0, 1


class FixedSchedule:
    """Baseline: alternate control between Kairi and Kaiser in fixed blocks."""

    def __init__(self, block: int):
        self.block = block

    def reset(self, n_envs: int) -> None:
        self.n = n_envs

    def who(self, t: int) -> np.ndarray:
        return np.full(self.n, (t // self.block) % 2, dtype=int)

    def observe(self, t, who, rewards, pred) -> None:
        pass


class ChangeTriggered:
    """Kairi by default; a Page-Hinkley change detector on x_t = |reward - pred| hands control
    to Kaiser for `window` steps.

    Per environment, the reference is the running mean m_t of x since reset, so the detector
    is insensitive to the baseline residual level. g = max(0, g + x_t - m_t - delta); it fires
    when g > h, then sets remaining = window and resets g (the running mean keeps going).
    `delta` is a margin above the baseline residual level."""

    def __init__(self, delta: float, h: float, window: int):
        self.delta = delta
        self.h = h
        self.window = int(window)

    def reset(self, n_envs: int) -> None:
        self.g = np.zeros(n_envs)
        self.remaining = np.zeros(n_envs, dtype=int)
        self.count = 0
        self.mean = np.zeros(n_envs)

    def who(self, t: int) -> np.ndarray:
        return (self.remaining > 0).astype(int)

    def observe(self, t, who, rewards, pred) -> None:
        x = np.abs(rewards - pred)
        self.count += 1
        self.mean = self.mean + (x - self.mean) / self.count
        self.g = np.maximum(0.0, self.g + x - self.mean - self.delta)
        fire = self.g > self.h
        self.remaining = np.where(fire, self.window, np.maximum(self.remaining - 1, 0))
        self.g = np.where(fire, 0.0, self.g)


class MetaBandit:
    """Learned meta-controller: discounted UCB over {Kairi, Kaiser}, one choice per block,
    rewarded by the mean reward earned during the block."""

    def __init__(self, gamma: float, c: float, block: int):
        self.gamma = gamma
        self.c = c
        self.block = int(block)

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.cnt = np.zeros((n_envs, 2))
        self.sum = np.zeros((n_envs, 2))
        self.cur = np.zeros(n_envs, dtype=int)
        self.acc = np.zeros(n_envs)

    def who(self, t: int) -> np.ndarray:
        if t % self.block == 0:
            total = np.maximum(self.cnt.sum(axis=1, keepdims=True), 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                index = self.sum / self.cnt + self.c * np.sqrt(np.log(total) / self.cnt)
            index = np.where(self.cnt <= 0, np.inf, index)
            self.cur = np.argmax(index, axis=1)
        return self.cur

    def observe(self, t, who, rewards, pred) -> None:
        self.acc += rewards
        if (t + 1) % self.block == 0:
            rows = np.arange(self.n)
            self.cnt *= self.gamma
            self.sum *= self.gamma
            self.cnt[rows, self.cur] += 1.0
            self.sum[rows, self.cur] += self.acc / self.block
            self.acc = np.zeros(self.n)


class TagTeamAgent:
    """One shared arm choice per step. Both agents propose every step and learn from every
    (arm, reward) pair, whoever chose the arm; the controller decides whose proposal is used."""

    def __init__(self, kairi, kaiser, controller):
        self.kairi = kairi
        self.kaiser = kaiser
        self.controller = controller

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.kairi.reset(n_envs)
        self.kaiser.reset(n_envs)
        self.controller.reset(n_envs)
        self._who = np.zeros(n_envs, dtype=int)

    def select(self, t: int) -> np.ndarray:
        proposal_kairi = self.kairi.select(t)
        proposal_kaiser = self.kaiser.select(t)
        self._who = self.controller.who(t)
        return np.where(self._who == KAISER, proposal_kaiser, proposal_kairi)

    def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None:
        pred = self.kairi.q[np.arange(self.n), actions].copy()
        self.kairi.update(t, actions, rewards)
        self.kaiser.update(t, actions, rewards)
        self.controller.observe(t, self._who, rewards, pred)
