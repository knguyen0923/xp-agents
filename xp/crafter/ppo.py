from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.crafter.rnd import RND, IntrinsicScaler
from xp.crafter.stats import ACHIEVEMENTS, EpisodeStats, nan_to_zero  # noqa: F401 (nan_to_zero re-exported for callers)

N_ACTIONS = 17
FRAME = (64, 64, 3)


def resolve_device(name: str) -> torch.device:
    """`auto` picks cuda, then mps, then cpu; anything else is passed to torch."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def prep(obs_np: np.ndarray, device) -> torch.Tensor:
    """uint8 (N, 64, 64, 3) -> float (N, 3, 64, 64) in [0, 1]."""
    return torch.as_tensor(obs_np, device=device).permute(0, 3, 1, 2).float() / 255.0


class ActorCritic(nn.Module):
    """The spike's small CNN encoder (identical for both agents), an actor head, and a critic head; RND agents add an intrinsic critic."""

    def __init__(self, intrinsic: bool = False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 4 * 4, 512), nn.ReLU(),
        )
        self.pi = nn.Linear(512, N_ACTIONS)
        self.v = nn.Linear(512, 1)
        # Built only for RND agents, so the plain-PPO parameter and random streams are identical with or without this path.
        self.v_int = nn.Linear(512, 1) if intrinsic else None

    def heads(self, x):
        h = self.enc(x)
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi(h), self.v(h).squeeze(-1), v_int


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
    """PPO, plus an RND bonus when `rnd_coef` is not None. Kairi = rnd_coef None; Kaiser = rnd_coef a float."""

    def __init__(self, ppo: dict, ent_coef: float, rnd_coef: Optional[float], seed: int, device="cpu"):
        self.ppo, self.ent_coef, self.rnd_coef, self.seed = ppo, ent_coef, rnd_coef, int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = ActorCritic(intrinsic=rnd_coef is not None).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo["lr"], eps=1e-5)
        self.rnd = self.rnd_opt = self.scaler = None
        if rnd_coef is not None:
            self.rnd = RND().to(self.device)
            self.rnd_opt = torch.optim.Adam(self.rnd.pred.parameters(), lr=ppo["rnd_lr"])
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

    def evaluate(self, env, n_episodes: int) -> dict:
        """Roll the frozen policy (sampled actions) until `n_episodes` episodes have finished; mean distinct achievements and unlock rates."""
        stats = EpisodeStats(window=max(1, n_episodes))
        done_eps = 0
        obs = env.obs
        while done_eps < n_episodes:
            with torch.no_grad():
                logits, _, _ = self.net.heads(prep(obs, self.device))
                action = Categorical(logits=logits).sample()
            obs, _, _, eps = env.step(action.cpu().numpy())
            for e in eps:
                if e is not None:
                    stats.add(e)
                    done_eps += 1
        return {"ach": float(np.mean([len(e) for e in stats.eps])), "rates": stats.rates(), "episodes": done_eps}

    def train(self, env, steps: int, log_every: int = 10) -> dict:
        """Train on `env` for about `steps` env steps (whole updates). Logs a curve row every `log_every` updates."""
        t0 = time.time()
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, rnd, device = self.ppo, self.rnd, self.device
        n, t_len = c["n_envs"], c["n_steps"]
        batch = n * t_len
        updates = max(1, steps // batch)
        net, opt = self.net, self.opt
        stats, first_unlock = EpisodeStats(100), {}
        obs_buf = np.zeros((t_len, n) + FRAME, dtype=np.uint8)
        act_buf = torch.zeros((t_len, n), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        val_buf = torch.zeros((t_len, n), device=device)
        vint_buf = torch.zeros((t_len, n), device=device)

        curve = []
        for u in range(updates):
            for t in range(t_len):
                obs_buf[t] = env.obs
                with torch.no_grad():
                    logits, value, value_int = net.heads(prep(env.obs, device))
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                act_buf[t], logp_buf[t], val_buf[t] = action, dist.log_prob(action), value
                if rnd is not None:
                    vint_buf[t] = value_int
                _, rew_buf[t], done_buf[t], eps = env.step(action.cpu().numpy())
                for e in eps:
                    if e is not None:
                        stats.add(e)
                        for name in e:
                            first_unlock.setdefault(name, (u * t_len + t + 1) * n)  # env steps so far in this call

            if rnd is not None:
                next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape((-1,) + FRAME)
                x = rnd.normalize(next_seq, device)
                with torch.no_grad():
                    raw = rnd.error(x).cpu().numpy().reshape(t_len, n)
                int_rewards = self.scaler(raw)
                perm = np.random.permutation(len(next_seq))
                for s in range(0, len(perm), 256):
                    idx = torch.as_tensor(perm[s:s + 256], device=device)
                    loss = rnd.error(x[idx]).mean()
                    self.rnd_opt.zero_grad()
                    loss.backward()
                    self.rnd_opt.step()

            with torch.no_grad():
                _, next_value, next_vint = net.heads(prep(env.obs, device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv_ext = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            returns, adv = adv_ext + val_buf, adv_ext
            if rnd is not None:
                # The intrinsic return is not cut at episode ends (RND paper), so dying costs no bonus.
                adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint, torch.ones_like(nonterm), c["gamma"], c["lam"])
                returns_int = adv_int + vint_buf
                adv = adv_ext + self.rnd_coef * adv_int

            b_obs = prep(obs_buf.reshape((-1,) + FRAME), device)
            b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
            b_ret_int = returns_int.reshape(-1) if rnd is not None else None
            for _ in range(c["epochs"]):
                perm_t = torch.randperm(batch, device=device)
                for s in range(0, batch, batch // c["minibatches"]):
                    idx = perm_t[s:s + batch // c["minibatches"]]
                    logits, value, value_int = net.heads(b_obs[idx])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                    vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    if rnd is not None:
                        vf = vf + 0.5 * (value_int - b_ret_int[idx]).pow(2).mean()
                    loss = pg + c["vf_coef"] * vf - self.ent_coef * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()

            self.steps_trained += batch
            if (u + 1) % log_every == 0 or u == updates - 1:
                curve.append({"steps": (u + 1) * batch, "ach_100": stats.ach(), "episodes": stats.total, "rates": stats.rates()})
        return {"curve": curve, "first_unlock": first_unlock, "episodes": stats.total, "final_rates": stats.rates(),
                "steps": updates * batch, "secs": time.time() - t0}
