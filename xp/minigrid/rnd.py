from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


def _cnn(out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, 2), nn.ReLU(), nn.Conv2d(16, 32, 2), nn.ReLU(),
        nn.Conv2d(32, 32, 2), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 4 * 4, out),
    )


class RND(nn.Module):
    """Random-network-distillation novelty bonus (the deep analog of UCB's confidence bonus)."""

    def __init__(self):
        super().__init__()
        self.target = _cnn(64)
        self.pred = nn.Sequential(_cnn(64), nn.ReLU(), nn.Linear(64, 64))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms = RunningMeanStd((3,))

    def normalize(self, obs_np: np.ndarray, device) -> torch.Tensor:
        flat = obs_np.reshape(-1, 3).astype(np.float64)
        self.obs_rms.update(flat)
        x = ((obs_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)).clip(-5, 5)
        return torch.as_tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)


class IntrinsicScaler:
    """Scales raw RND errors by the std of their discounted return (RND paper's forward filter).

    Dividing by the std of the per-step error instead leaves the intrinsic value targets in the hundreds; through
    the shared trunk and the global grad clip that crushed the policy gradient (found in the Phase 2 spike).
    """

    def __init__(self, n_envs: int, gamma: float):
        self.gamma = gamma
        self.ret = np.zeros(n_envs)
        self.rms = RunningMeanStd(())

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """raw: (t_len, n_envs) -> scaled float32 of the same shape."""
        for t in range(raw.shape[0]):
            self.ret = self.ret * self.gamma + raw[t]
            self.rms.update(self.ret)
        return (raw / np.sqrt(self.rms.var + 1e-8)).astype(np.float32)
