"""THROWAWAY SPIKE (not part of the `xp` package; do not import from it).

Answers two questions before the Phase 2 (MiniGrid) plan is written:

1. How long does PPO (and PPO + RND) take on DoorKey? -> steps/sec, steps to 80% success.
2. Are the larger DoorKey sizes at a 0% floor for a policy trained on the familiar size?
   -> zero-shot success rate at sizes 5/6/8/12 vs a random-policy baseline.

Single file on purpose: a CleanRL-style PPO with an optional RND bonus. Results are written to one
JSON file per run. Delete this directory once the Phase 2 plan has absorbed the numbers.

Example:
    python spikes/spike_minigrid.py --train-size 6 --steps 500000 --out kairi.json
    python spikes/spike_minigrid.py --train-size 6 --steps 500000 --rnd --out kaiser.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch
import torch.nn as nn
from minigrid.envs import DoorKeyEnv
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical

N_ACTIONS = 7


def make_env(size: int):
    return ImgObsWrapper(DoorKeyEnv(size=size))


def prep(obs_np: np.ndarray, device) -> torch.Tensor:
    """(N, 7, 7, 3) uint8 -> (N, 3, 7, 7) float in [0, 1]."""
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device) / 10.0
    return x.permute(0, 3, 1, 2)


class VecEnv:
    """Minimal synchronous vector env with auto-reset (version-independent of gymnasium.vector)."""

    def __init__(self, size: int, n: int, seed: int):
        self.n = n
        self.envs = [make_env(size) for _ in range(n)]
        self.obs = np.stack([e.reset(seed=seed + i)[0] for i, e in enumerate(self.envs)])
        self.ep_ret = np.zeros(n)
        self.recent = []  # (success, return) of finished episodes

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
                o, _ = env.reset()
            self.obs[i] = o
        return self.obs.copy(), rews, dones

    def recent_stats(self, k: int = 100):
        window = self.recent[-k:]
        if not window:
            return float("nan"), float("nan")
        arr = np.array(window)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())


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
        # Separate head for the non-episodic intrinsic return; only built for RND runs so the plain-PPO
        # random stream (and results) are unchanged.
        self.v_int = nn.Linear(64, 1) if intrinsic else None

    def heads(self, x):
        h = self.fc(self.conv(x))
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi(h), self.v(h).squeeze(-1), v_int

    def forward(self, x):
        logits, v, _ = self.heads(x)
        return logits, v


def gae(rewards, values, next_value, nonterm, gamma, lam):
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(rewards.shape[0])):
        nv = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm[t] - values[t]
        last = delta + gamma * lam * nonterm[t] * last
        adv[t] = last
    return adv


def _cnn(out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, 2), nn.ReLU(), nn.Conv2d(16, 32, 2), nn.ReLU(),
        nn.Conv2d(32, 32, 2), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 4 * 4, out),
    )


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


class RND(nn.Module):
    """Random-network-distillation novelty bonus (the deep analog of UCB's confidence bonus)."""

    def __init__(self):
        super().__init__()
        self.target = _cnn(64)
        self.pred = nn.Sequential(_cnn(64), nn.ReLU(), nn.Linear(64, 64))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms = RunningMeanStd((3,))
        self.ret_rms = RunningMeanStd(())  # std of the discounted intrinsic return (forward filter)
        self.ret = None

    def norm_obs(self, obs_np: np.ndarray, device) -> torch.Tensor:
        flat = obs_np.reshape(-1, 3).astype(np.float64)
        self.obs_rms.update(flat)
        x = ((obs_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)).clip(-5, 5)
        return torch.as_tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)


def train(args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = VecEnv(args.train_size, args.n_envs, seed=args.seed * 1000)
    net = ActorCritic(intrinsic=args.rnd).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    rnd = RND().to(device) if args.rnd else None
    rnd_opt = torch.optim.Adam(rnd.pred.parameters(), lr=1e-4) if rnd else None

    n, t_len = args.n_envs, args.n_steps
    batch = n * t_len
    updates = max(1, args.steps // batch)
    obs_buf = np.zeros((t_len, n, 7, 7, 3), dtype=np.uint8)
    act_buf = torch.zeros((t_len, n), dtype=torch.long, device=device)
    logp_buf = torch.zeros((t_len, n), device=device)
    rew_buf = np.zeros((t_len, n), dtype=np.float32)
    done_buf = np.zeros((t_len, n), dtype=np.float32)
    val_buf = torch.zeros((t_len, n), device=device)
    vint_buf = torch.zeros((t_len, n), device=device)

    curve = []
    steps_to_80 = None
    bonus_mean, ret_int_abs, prev_eps, prev_steps = 0.0, 0.0, 0, 0
    t0 = time.perf_counter()
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

        rewards = rew_buf.copy()
        if rnd is not None:
            next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape(-1, 7, 7, 3)
            x = rnd.norm_obs(next_seq.astype(np.float32), device)
            with torch.no_grad():
                intrinsic = rnd.error(x).cpu().numpy()
            raw = intrinsic.reshape(t_len, n)
            if rnd.ret is None:
                rnd.ret = np.zeros(n)
            for t in range(t_len):  # scale by the std of the discounted intrinsic return (as in the RND paper), so
                rnd.ret = rnd.ret * args.gamma + raw[t]  # the intrinsic value targets stay O(1)
                rnd.ret_rms.update(rnd.ret)
            int_rewards = (raw / np.sqrt(rnd.ret_rms.var + 1e-8)).astype(np.float32)  # non-episodic stream, kept apart
            bonus_mean = float(args.rnd_coef * int_rewards.mean())
            perm = np.random.permutation(len(next_seq))
            for s in range(0, len(perm), 256):
                idx = perm[s:s + 256]
                loss = rnd.error(x[idx]).mean()
                rnd_opt.zero_grad()
                loss.backward()
                rnd_opt.step()

        with torch.no_grad():
            _, next_value, next_vint = net.heads(prep(env.obs, device))
        rewards_t = torch.as_tensor(rewards, device=device)
        nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
        adv_ext = gae(rewards_t, val_buf, next_value, nonterm, args.gamma, args.lam)
        returns = adv_ext + val_buf
        adv = adv_ext
        if rnd is not None:
            # RND paper: intrinsic return is not cut at episode ends, so reaching the goal costs no future bonus.
            adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint,
                          torch.ones_like(nonterm), args.gamma, args.lam)
            returns_int = adv_int + vint_buf
            ret_int_abs = float(returns_int.abs().mean())
            adv = adv_ext + args.rnd_coef * adv_int

        b_obs = prep(obs_buf.reshape(-1, 7, 7, 3), device)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
        b_ret_int = returns_int.reshape(-1) if rnd is not None else None
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for s in range(0, batch, batch // args.minibatches):
                idx = perm[s:s + batch // args.minibatches]
                logits, value, value_int = net.heads(b_obs[idx])
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                if rnd is not None:
                    vf = vf + 0.5 * (value_int - b_ret_int[idx]).pow(2).mean()
                loss = pg + args.vf_coef * vf - args.ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if (u + 1) % args.log_every == 0 or u == updates - 1:
            succ, ret = env.recent_stats()
            steps = (u + 1) * batch
            elapsed = time.perf_counter() - t0
            ep_len = (steps - prev_steps) / max(1, len(env.recent) - prev_eps)  # env steps per finished episode
            prev_eps, prev_steps = len(env.recent), steps
            curve.append({"steps": steps, "success_100": succ, "return_100": ret, "seconds": elapsed,
                          "ep_len": ep_len, "bonus_per_step": bonus_mean, "int_return_abs": ret_int_abs})
            if steps_to_80 is None and succ == succ and succ >= 0.8:
                steps_to_80 = steps
            print(f"[train] steps={steps} success100={succ:.2f} return100={ret:.2f} "
                  f"ep_len={ep_len:.0f} bonus/step={bonus_mean:.3f} |R_int|={ret_int_abs:.1f} elapsed={elapsed:.0f}s sps={steps / elapsed:.0f}", flush=True)

    elapsed = time.perf_counter() - t0
    total_steps = updates * batch
    return net, {
        "steps": total_steps, "seconds": elapsed, "sps": total_steps / elapsed,
        "steps_to_80pct_success": steps_to_80, "curve": curve,
    }


@torch.no_grad()
def evaluate(net, size: int, episodes: int, seed: int, device, random_policy: bool = False):
    env = make_env(size)
    env.action_space.seed(seed)
    wins, rets = 0.0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, term, r = False, 0.0, False, 0.0
        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                logits, _ = net(prep(obs[None], device))
                action = Categorical(logits=logits).sample().item()
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        wins += float(term and r > 0)
        rets.append(ret)
    return {"success": wins / episodes, "mean_return": float(np.mean(rets)), "episodes": episodes}


def env_only_sps(size: int, n_envs: int, steps: int = 2000) -> float:
    env = VecEnv(size, n_envs, seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(steps // n_envs):
        env.step(rng.integers(0, N_ACTIONS, n_envs))
    return (steps // n_envs) * n_envs / (time.perf_counter() - t0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-size", type=int, default=6)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--eval-sizes", type=int, nargs="+", default=[5, 6, 8, 12])
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--rnd", action="store_true", help="add the RND intrinsic bonus (Kaiser-style)")
    p.add_argument("--rnd-coef", type=float, default=0.1)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    import gymnasium
    import minigrid

    result = {
        "args": vars(args),
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "minigrid": getattr(minigrid, "__version__", "unknown"),
        },
        "env_only_sps": env_only_sps(args.train_size, args.n_envs),
    }
    print(f"[spike] env-only steps/sec (random actions, size {args.train_size}): {result['env_only_sps']:.0f}", flush=True)

    net, result["train"] = train(args, device)
    eval_seed = 10_000_000
    result["eval_zero_shot"] = {
        str(s): evaluate(net, s, args.eval_episodes, eval_seed, device) for s in args.eval_sizes
    }
    result["random_baseline"] = {
        str(s): evaluate(net, s, args.eval_episodes, eval_seed, device, random_policy=True) for s in args.eval_sizes
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n[spike] size | trained-policy success | random-policy success")
    for s in args.eval_sizes:
        print(f"[spike] {s:>4} | {result['eval_zero_shot'][str(s)]['success']:.2f} | "
              f"{result['random_baseline'][str(s)]['success']:.2f}")
    print(f"[spike] train sps={result['train']['sps']:.0f}, steps to 80% = {result['train']['steps_to_80pct_success']}")


if __name__ == "__main__":
    main()
