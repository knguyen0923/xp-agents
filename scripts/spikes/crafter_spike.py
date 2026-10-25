"""Crafter spike (spec 5.5): plain PPO only, to find out whether Crafter has a usable difficulty range.

Throwaway by design: the full phase would generalise xp.minigrid.ppo instead. No RND here.

python scripts/spikes/crafter_spike.py --area 32 --steps 1000000 --seed 0 --device cpu --out results/crafter-spike
python scripts/spikes/crafter_spike.py --analyze results/crafter-spike/area32-seed0
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import time

import numpy as np

LOG_EVERY = 10240  # env steps between curve rows (matches the MiniGrid logging grid)


def world_seed(env_seed, episode):
    """The world seed crafter.Env uses for an episode (checked against crafter 1.8.3 source)."""
    return hash((env_seed, episode)) % (2 ** 31 - 1)


def achievement_names():
    from crafter import constants

    return tuple(constants.achievements)


class EpisodeStats:
    """Distinct achievements per finished episode over a sliding window of the last `window` episodes."""

    def __init__(self, window=100):
        self.window = window
        self.eps = collections.deque(maxlen=window)
        self.total = 0

    def add(self, unlocked):
        self.eps.append(frozenset(unlocked))
        self.total += 1

    def ach(self):
        """Mean number of distinct achievements per episode in the window (0.0 before any episode)."""
        return float(np.mean([len(e) for e in self.eps])) if self.eps else 0.0

    def rates(self, names):
        n = max(1, len(self.eps))
        return {a: sum(a in e for e in self.eps) / n for a in names}


def gates(curve, rates):
    """Spec 5.5 gates G1 and G2 from a curve [(steps, ach_100)] and final per-achievement rates.

    G1: final ach_100 >= 5 (a near-random policy already averages about 3) and at least 5 of the achievements
    unlocked in under 5% of episodes.
    G2: final ach_100 exceeds the mean ach_100 over the first 10% of training.
    """
    final = curve[-1][1]
    cut = curve[-1][0] * 0.10
    early = [v for s, v in curve if s <= cut] or [curve[0][1]]
    rare = sum(1 for r in rates.values() if r < 0.05)
    return {
        "final_ach_100": final, "early_ach_100": float(np.mean(early)), "rare_achievements": rare,
        "G1": bool(final >= 5 and rare >= 5), "G2": bool(final > float(np.mean(early))),
    }


def _worker(conn, area, env_seed):
    import crafter

    env = crafter.Env(area=(area, area), seed=env_seed)
    conn.send(env.reset())
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            obs, reward, done, info = env.step(data)
            ep = None
            if done:
                ep = [k for k, v in info["achievements"].items() if v > 0]
                obs = env.reset()
            conn.send((obs, float(reward), bool(done), ep))
    finally:
        conn.close()


class SubprocVec:
    """n crafter envs in n processes; env i uses seed base_seed * 1000 + i."""

    def __init__(self, n, area, base_seed):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, area, base_seed * 1000 + i), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.obs = np.stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        out = [c.recv() for c in self.conns]
        self.obs = np.stack([o[0] for o in out])
        return self.obs, np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out]), [o[3] for o in out]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)


def make_net(n_actions=8):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 4 * 4, 512), nn.ReLU())
            self.pi = nn.Linear(512, n_actions)
            self.v = nn.Linear(512, 1)

        def forward(self, x):
            h = self.enc(x)
            return self.pi(h), self.v(h).squeeze(-1)

    return Net()


def train(area, steps, seed, device, out_dir, n_envs=8, n_steps=128, epochs=4, minibatches=4,
          lr=2.5e-4, gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    dev = torch.device(device)
    net = make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    names = achievement_names()
    stats = EpisodeStats(100)
    vec = SubprocVec(n_envs, area, seed)
    to_t = lambda o: torch.as_tensor(o, device=dev).permute(0, 3, 1, 2).float() / 255.0  # noqa: E731
    curve, t0, next_log, done_steps = [], time.time(), LOG_EVERY, 0
    obs = to_t(vec.obs)
    try:
        while done_steps < steps:
            ob_b = torch.zeros((n_steps, n_envs) + tuple(obs.shape[1:]), device=dev)
            act_b = torch.zeros((n_steps, n_envs), dtype=torch.long, device=dev)
            lp_b = torch.zeros((n_steps, n_envs), device=dev)
            val_b = torch.zeros((n_steps + 1, n_envs), device=dev)
            rew_b = torch.zeros((n_steps, n_envs), device=dev)
            done_b = torch.zeros((n_steps, n_envs), device=dev)
            for t in range(n_steps):
                with torch.no_grad():
                    logits, val = net(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                ob_b[t], act_b[t], lp_b[t], val_b[t] = obs, act, dist.log_prob(act), val
                nobs, rew, done, eps = vec.step(act.cpu().numpy())
                rew_b[t] = torch.as_tensor(rew, device=dev)
                done_b[t] = torch.as_tensor(done.astype(np.float32), device=dev)
                for e in eps:
                    if e is not None:
                        stats.add(e)
                obs = to_t(nobs)
            with torch.no_grad():
                val_b[n_steps] = net(obs)[1]
            adv = torch.zeros((n_steps, n_envs), device=dev)
            last = torch.zeros(n_envs, device=dev)
            for t in reversed(range(n_steps)):
                nonterm = 1.0 - done_b[t]
                delta = rew_b[t] + gamma * val_b[t + 1] * nonterm - val_b[t]
                last = delta + gamma * lam * nonterm * last
                adv[t] = last
            ret = adv + val_b[:n_steps]
            flat = lambda x: x.reshape((-1,) + tuple(x.shape[2:]))  # noqa: E731
            f_ob, f_act, f_lp, f_adv, f_ret = flat(ob_b), flat(act_b), flat(lp_b), flat(adv), flat(ret)
            n = f_ob.shape[0]
            for _ in range(epochs):
                perm = torch.randperm(n, device=dev)
                for idx in perm.chunk(minibatches):
                    logits, v = net(f_ob[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    ratio = torch.exp(dist.log_prob(f_act[idx]) - f_lp[idx])
                    a = f_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                    vl = 0.5 * ((v - f_ret[idx]) ** 2).mean()
                    loss = pl + vf_coef * vl - ent_coef * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
            done_steps += n_steps * n_envs
            if done_steps >= next_log:
                next_log += LOG_EVERY
                sps = done_steps / (time.time() - t0)
                curve.append((done_steps, stats.ach(), stats.total, sps))
                if len(curve) % 10 == 0:
                    print(f"[crafter-spike] area {area} steps {done_steps} ach_100 {stats.ach():.2f} "
                          f"episodes {stats.total} {sps:.0f} steps/s", flush=True)
    finally:
        vec.close()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "curve.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["steps", "ach_100", "episodes", "steps_per_s"])
        w.writerows(curve)
    summary = {"area": area, "steps": done_steps, "seed": seed, "device": device,
               "secs": time.time() - t0, "rates": stats.rates(names)}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return curve, summary


def analyze(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "curve.csv"))))
    curve = [(float(r["steps"]), float(r["ach_100"])) for r in rows]
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    g = gates(curve, summ["rates"])
    g.update({"area": summ["area"], "steps": summ["steps"], "secs": round(summ["secs"]),
              "steps_per_s": round(summ["steps"] / summ["secs"])})
    print(json.dumps(g, indent=1))
    print("unlock rate, last 100 episodes:")
    for a, r in sorted(summ["rates"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:20s} {r:5.2f}")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/crafter-spike")
    ap.add_argument("--analyze", default=None, help="run directory to analyze instead of training")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze)
        return
    out = os.path.join(a.out, f"area{a.area}-seed{a.seed}")
    train(a.area, a.steps, a.seed, a.device, out)
    analyze(out)


if __name__ == "__main__":
    main()
