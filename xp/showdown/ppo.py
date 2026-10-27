from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS, restrict_second

NEG = -1e9


class ActorCritic(nn.Module):
    """MLP on the fixed feature vector with two 107-way heads (one per active slot) and a value head."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.ModuleList([nn.Linear(hidden, N_ACTIONS), nn.Linear(hidden, N_ACTIONS)])
        self.v = nn.Linear(hidden, 1)

    def heads(self, x: torch.Tensor):
        h = self.body(x)
        return self.pi[0](h), self.pi[1](h), self.v(h).squeeze(-1)


def masked(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Categorical over the legal actions only (illegal logits pushed to -1e9, so their probability is exactly 0)."""
    return Categorical(logits=logits.masked_fill(~mask, NEG))


def gae(rewards, values, next_value, nonterm, gamma: float, lam: float):
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(rewards.shape[0])):
        nv = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm[t] - values[t]
        last = delta + gamma * lam * nonterm[t] * last
        adv[t] = last
    return adv


class Learner:
    """Plain PPO with action masks (Kairi). Slot 1's mask is conditioned on slot 0's sampled action (no duplicate switch, one Mega)."""

    def __init__(self, ppo: dict, ent_coef: float, seed: int, device: str = "cpu"):
        self.ppo, self.ent_coef, self.seed = ppo, ent_coef, int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo["lr"], eps=1e-5)
        self.steps_trained = 0

    def act(self, obs: np.ndarray, masks: np.ndarray):
        """Sample one joint action per env. Returns (actions (n, 2), logp (n,), value (n,), conditional masks (n, 2, 107))."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m0 = torch.as_tensor(masks[:, 0], device=self.device)
        with torch.no_grad():
            l0, l1, value = self.net.heads(x)
            d0 = masked(l0, m0)
            a0 = d0.sample()
            m1 = np.stack([restrict_second(masks[i, 1], int(a0[i])) for i in range(len(obs))])
            m1_t = torch.as_tensor(m1, device=self.device)
            d1 = masked(l1, m1_t)
            a1 = d1.sample()
            logp = d0.log_prob(a0) + d1.log_prob(a1)
        cond = np.stack([masks[:, 0], m1], axis=1)
        return torch.stack([a0, a1], dim=1).cpu().numpy(), logp, value, cond

    def train(self, env, steps: int, log_every: int = 10, target: Optional[float] = None) -> dict:
        """Train for about `steps` agent decisions. Returns the win-rate curve and the steps to reach `target` win rate."""
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, device = self.ppo, self.device
        n, t_len = env.n, c["n_steps"]
        batch = n * t_len
        updates = max(1, steps // batch)
        obs_buf = np.zeros((t_len, n, OBS_DIM), dtype=np.float32)
        mask_buf = np.zeros((t_len, n, 2, N_ACTIONS), dtype=bool)
        act_buf = torch.zeros((t_len, n, 2), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        val_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        curve, steps_to_target = [], None
        for u in range(updates):
            for t in range(t_len):
                obs_buf[t] = env.obs
                actions, logp, value, cond = self.act(env.obs, env.masks)
                mask_buf[t], act_buf[t], logp_buf[t], val_buf[t] = cond, torch.as_tensor(actions, device=device), logp, value
                rew_buf[t], done_buf[t] = env.step(actions)
            with torch.no_grad():
                _, _, next_value = self.net.heads(torch.as_tensor(env.obs, dtype=torch.float32, device=device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            ret = adv + val_buf
            b_obs = torch.as_tensor(obs_buf.reshape(-1, OBS_DIM), device=device)
            b_mask = torch.as_tensor(mask_buf.reshape(-1, 2, N_ACTIONS), device=device)
            b_act, b_logp = act_buf.reshape(-1, 2), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            for _ in range(c["epochs"]):
                perm = torch.randperm(batch, device=device)
                for s in range(0, batch, batch // c["minibatches"]):
                    idx = perm[s:s + batch // c["minibatches"]]
                    l0, l1, value = self.net.heads(b_obs[idx])
                    d0, d1 = masked(l0, b_mask[idx, 0]), masked(l1, b_mask[idx, 1])
                    new_logp = d0.log_prob(b_act[idx, 0]) + d1.log_prob(b_act[idx, 1])
                    ratio = (new_logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                    vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    ent = (d0.entropy() + d1.entropy()).mean()
                    loss = pg + c["vf_coef"] * vf - self.ent_coef * ent
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()
            self.steps_trained += batch
            if (u + 1) % log_every == 0 or u == updates - 1:
                win, _ = env.recent_stats()
                done_steps = (u + 1) * batch
                curve.append({"steps": done_steps, "win_rate_100": win})
                if target is not None and steps_to_target is None and env.recent_window_full() and win >= target:
                    steps_to_target = done_steps
        return {"curve": curve, "steps_to_target": steps_to_target, "steps": updates * batch}
