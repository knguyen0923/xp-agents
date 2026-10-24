"""Spike 1 (kept for reproducibility; not part of the protocol): plain PPO vs PPO+RND (coef 0.003) on MultiRoom, 2 to 6 rooms, max room size 6, 2 seeds, 2M steps.

Run from the repo root: python scripts/spikes/multiroom_spike1.py <out_dir>. Results are summarised in docs/spikes/multiroom_spike.csv."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

OUT = Path(sys.argv[1])
BUDGET = 2_000_000


def job(args):
    rooms, agent, seed = args
    path = OUT / f"{agent}_N{rooms}_s{seed}.json"
    if path.exists():
        return str(path)
    import torch
    import yaml
    import xp.minigrid.env as env
    from minigrid.envs import MultiRoomEnv
    from minigrid.wrappers import ImgObsWrapper
    from xp.minigrid.ppo import Learner

    torch.set_num_threads(1)
    env.make_env = lambda size: ImgObsWrapper(MultiRoomEnv(minNumRooms=int(size), maxNumRooms=int(size), maxRoomSize=6))
    cfg = yaml.safe_load(open("configs/minigrid.yaml"))
    ent, rnd = (0.01, None) if agent == "ppo" else (0.01, 0.003)
    t0 = time.time()
    learner = Learner(cfg["ppo"], ent, rnd, seed)
    run = learner.train(rooms, BUDGET, target=0.95)
    res = learner.evaluate(rooms, 100)
    rec = {"rooms": rooms, "agent": agent, "seed": seed, "final_success": res["success"], "steps_to_target": run["steps_to_target"],
           "curve": [(r.get("steps"), r["success_100"]) for r in run["curve"]][::5], "secs": time.time() - t0}
    path.write_text(json.dumps(rec))
    print(f"N{rooms} {agent} s{seed}: test success {res['success']:.2f}, steps_to_95 {run['steps_to_target']}, {rec['secs']:.0f}s", flush=True)
    return str(path)


if __name__ == "__main__":
    jobs = [(n, a, s) for n in (2, 3, 4, 5, 6) for a in ("ppo", "rnd") for s in (0, 1)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        list(pool.map(job, jobs))
    print("spike done", flush=True)
