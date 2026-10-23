from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.minigrid.env import FEWSHOT_EPISODE_OFFSET, N_ACTIONS, TEST_LAYOUT_BASE, VecEnv, evaluate_policy, prep
from xp.minigrid.rnd import RND, IntrinsicScaler


class ActorCritic(nn.Module):
    def __init__(self, intrinsic: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 2), nn.ReLU(), nn.Conv2d(16, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(), nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 64), nn.Tanh())
        self.pi = nn.Linear(64, N_ACTIONS)
        self.v = nn.Linear(64, 1)
        # Separate head for the non-episodic intrinsic return; only built for RND agents, so the plain-PPO
        # parameter and random streams are identical with or without this code path.
        self.v_int = nn.Linear(64, 1) if intrinsic else None

    def heads(self, x):
        h = self.fc(self.conv(x))
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi(h), self.v(h).squeeze(-1), v_int

    def forward(self, x):
        logits, v, _ = self.heads(x)
        return logits, v


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
    """PPO, plus an RND bonus when `rnd_coef` is not None. Kairi = rnd_coef None; Kaiser = rnd_coef a float.

    One Learner holds its weights and optimizer state across `train` calls, so the few-shot phase can continue
    training a deep copy on a new size. `evaluate` never updates anything.
    """

    def __init__(self, ppo: dict, ent_coef: float, rnd_coef: Optional[float], seed: int, device: str = "cpu"):
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

    def evaluate(self, size: int, n_layouts: int, base: int = TEST_LAYOUT_BASE) -> dict:
        return evaluate_policy(self.net, size, n_layouts, self.device, base)

    def train(self, size: int, steps: int, first_episode: int = 0, limit: int = FEWSHOT_EPISODE_OFFSET,
              log_every: int = 10, target: float = 0.8) -> dict:
        """Train on fresh layouts of `size` for about `steps` env steps. Returns the curve and steps-to-target."""
        # Reseed from (seed, steps so far) so training is deterministic however Learners are interleaved.
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, rnd, device = self.ppo, self.rnd, self.device
        n, t_len = c["n_envs"], c["n_steps"]
        batch = n * t_len
        updates = max(1, steps // batch)
        env = VecEnv(size, n, self.seed, first_episode, limit)
        net, opt = self.net, self.opt
        obs_buf = np.zeros((t_len, n, 7, 7, 3), dtype=np.uint8)
        act_buf = torch.zeros((t_len, n), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        val_buf = torch.zeros((t_len, n), device=device)
        vint_buf = torch.zeros((t_len, n), device=device)

        curve, steps_to_target = [], None
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
                _, rew_buf[t], done_buf[t] = env.step(action.cpu().numpy())

            if rnd is not None:
                next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape(-1, 7, 7, 3)
                x = rnd.normalize(next_seq.astype(np.float32), device)
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
                _, next_value, next_vint = net.heads(prep(env.obs, device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv_ext = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            returns, adv = adv_ext + val_buf, adv_ext
            if rnd is not None:
                # The intrinsic return is not cut at episode ends (RND paper), so reaching the goal costs no bonus.
                adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint,
                              torch.ones_like(nonterm), c["gamma"], c["lam"])
                returns_int = adv_int + vint_buf
                adv = adv_ext + self.rnd_coef * adv_int

            b_obs = prep(obs_buf.reshape(-1, 7, 7, 3), device)
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
                succ, ret = env.recent_stats()
                done_steps = (u + 1) * batch
                curve.append({"steps": done_steps, "success_100": succ, "return_100": ret})
                if steps_to_target is None and succ == succ and succ >= target:
                    steps_to_target = done_steps
        return {"curve": curve, "steps_to_target": steps_to_target, "steps": updates * batch}
