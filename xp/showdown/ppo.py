from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS, restrict_second
from xp.showdown.rnd import IntrinsicScaler, VectorRND

NEG = -1e9


class ActorCritic(nn.Module):
    """MLP on the fixed feature vector with two 107-way heads (one per active slot) and a value head."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, intrinsic: bool = False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.ModuleList([nn.Linear(hidden, N_ACTIONS), nn.Linear(hidden, N_ACTIONS)])
        self.v = nn.Linear(hidden, 1)
        # Separate head for the non-episodic intrinsic return; built only for RND agents, so plain PPO has exactly
        # the same parameters and random streams with or without this code path.
        self.v_int = nn.Linear(hidden, 1) if intrinsic else None

    def heads(self, x: torch.Tensor):
        h = self.body(x)
        return self.pi[0](h), self.pi[1](h), self.v(h).squeeze(-1)

    def heads_int(self, x: torch.Tensor):
        """Like `heads`, plus the intrinsic value (None when the net has no intrinsic head)."""
        h = self.body(x)
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi[0](h), self.pi[1](h), self.v(h).squeeze(-1), v_int


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
    """PPO with action masks. Kairi = `rnd_coef` None; Kaiser = the same PPO plus an RND bonus (`rnd_coef` a float).

    Slot 1's mask is conditioned on slot 0's sampled action (no duplicate switch, one Mega).
    """

    def __init__(self, ppo: dict, ent_coef: float, seed: int, device: str = "cpu", rnd_coef: Optional[float] = None):
        self.ppo, self.ent_coef, self.seed, self.rnd_coef = ppo, ent_coef, int(seed), rnd_coef
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = ActorCritic(intrinsic=rnd_coef is not None).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo["lr"], eps=1e-5)
        self.rnd = self.rnd_opt = self.scaler = None
        if rnd_coef is not None:
            self.rnd = VectorRND().to(self.device)
            self.rnd_opt = torch.optim.Adam(self.rnd.pred.parameters(), lr=ppo.get("rnd_lr", 0.0001))
            self.scaler = IntrinsicScaler(ppo["n_envs"], ppo["gamma"])
        self.steps_trained = 0

    def fingerprint(self) -> str:
        """Hash of everything an update could change; used to prove evaluation never learns."""
        h = hashlib.sha256()
        tensors = list(self.net.parameters()) + (list(self.rnd.parameters()) if self.rnd else [])
        for p in tensors:
            h.update(p.detach().cpu().numpy().tobytes())
        h.update(str(self.steps_trained).encode())
        if self.rnd is not None:
            h.update(str(self.rnd.obs_rms.count).encode())
        return h.hexdigest()

    def evaluate(self, env, n_battles: int) -> dict:
        """Win rate of the current (frozen) policy over `n_battles` finished battles on `env`. Never updates anything."""
        wins = finished = 0
        while finished < n_battles:
            actions, _, _, _ = self.act(env.obs, env.masks)
            rewards, dones = env.step(actions)
            finished += int(dones.sum())
            wins += int(((rewards > 0) & (dones == 1)).sum())
        return {"win_rate": wins / finished, "battles": finished}

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
        # Reseed from (seed, steps so far) so training is deterministic however Learners are interleaved.
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, device, rnd = self.ppo, self.device, self.rnd
        n, t_len = env.n, c["n_steps"]
        if rnd is not None and n != c["n_envs"]:
            raise ValueError(f"the RND scaler was built for n_envs={c['n_envs']} but the env has {n} processes")
        batch = n * t_len
        updates = max(1, steps // batch)
        obs_buf = np.zeros((t_len, n, OBS_DIM), dtype=np.float32)
        mask_buf = np.zeros((t_len, n, 2, N_ACTIONS), dtype=bool)
        act_buf = torch.zeros((t_len, n, 2), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        val_buf = torch.zeros((t_len, n), device=device)
        vint_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        curve, steps_to_target = [], None
        for u in range(updates):
            for t in range(t_len):
                obs_buf[t] = env.obs
                actions, logp, value, cond = self.act(env.obs, env.masks)
                mask_buf[t], act_buf[t], logp_buf[t], val_buf[t] = cond, torch.as_tensor(actions, device=device), logp, value
                if rnd is not None:
                    with torch.no_grad():
                        vint_buf[t] = self.net.heads_int(torch.as_tensor(env.obs, dtype=torch.float32, device=device))[3]
                rew_buf[t], done_buf[t] = env.step(actions)

            if rnd is not None:
                next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape(-1, OBS_DIM)
                x = rnd.normalize(next_seq, device)
                with torch.no_grad():
                    raw = rnd.error(x).cpu().numpy().reshape(t_len, n)
                int_rewards = self.scaler(raw)
                perm = np.random.permutation(len(next_seq))
                for s in range(0, len(perm), 256):
                    idx = perm[s:s + 256]
                    loss = rnd.error(x[idx]).mean()
                    self.rnd_opt.zero_grad()
                    loss.backward()
                    self.rnd_opt.step()

            with torch.no_grad():
                _, _, next_value, next_vint = self.net.heads_int(torch.as_tensor(env.obs, dtype=torch.float32, device=device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv_ext = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            ret, adv = adv_ext + val_buf, adv_ext
            if rnd is not None:
                # The intrinsic return is not cut at episode ends (RND paper): reaching a new state costs nothing at the end.
                adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint, torch.ones_like(nonterm), c["gamma"], c["lam"])
                ret_int = adv_int + vint_buf
                adv = adv_ext + self.rnd_coef * adv_int
            b_obs = torch.as_tensor(obs_buf.reshape(-1, OBS_DIM), device=device)
            b_mask = torch.as_tensor(mask_buf.reshape(-1, 2, N_ACTIONS), device=device)
            b_act, b_logp = act_buf.reshape(-1, 2), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            b_ret_int = ret_int.reshape(-1) if rnd is not None else None
            for _ in range(c["epochs"]):
                perm_t = torch.randperm(batch, device=device)
                for s in range(0, batch, batch // c["minibatches"]):
                    idx = perm_t[s:s + batch // c["minibatches"]]
                    l0, l1, value, value_int = self.net.heads_int(b_obs[idx])
                    d0, d1 = masked(l0, b_mask[idx, 0]), masked(l1, b_mask[idx, 1])
                    new_logp = d0.log_prob(b_act[idx, 0]) + d1.log_prob(b_act[idx, 1])
                    ratio = (new_logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                    vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    if rnd is not None:
                        vf = vf + 0.5 * (value_int - b_ret_int[idx]).pow(2).mean()
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
