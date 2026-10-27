from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from xp.minigrid.rnd import IntrinsicScaler, RunningMeanStd  # numpy/torch only; shared with the MiniGrid phase
from xp.showdown.encode import OBS_DIM

__all__ = ["VectorRND", "IntrinsicScaler"]


class VectorRND(nn.Module):
    """Random-network-distillation novelty bonus on the fixed feature vector (the deep analog of UCB's confidence bonus)."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, out: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.target = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.pred = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms = RunningMeanStd((obs_dim,))

    def normalize(self, obs_np: np.ndarray, device) -> torch.Tensor:
        """Update the running statistics with this batch, then standardise and clip to [-5, 5]."""
        flat = obs_np.reshape(-1, self.obs_dim).astype(np.float64)
        self.obs_rms.update(flat)
        x = ((flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)).clip(-5, 5)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)
