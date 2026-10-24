"""Probe (kept for reproducibility; not part of the protocol): 4 rooms, max room size 4, 5M steps, plain PPO vs PPO+RND at coef 0.01, 0.03, 0.1, seeds 0-4.

Is the lack of an RND benefit at 2M steps a budget / coefficient artifact? Run from the repo root: python scripts/spikes/multiroom_probe.py <out_dir>."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path

OUT = Path(sys.argv[1])
BUDGET = 5_000_000


def job(args):
    rooms, room_size, agent, coef, seed = args
    path = OUT / f"{agent}{coef if coef else ''}_N{rooms}_S{room_size}_s{seed}.json"
    if path.exists():
        return str(path)
    import torch
    import yaml
    import xp.minigrid.env as env
    from minigrid.envs import MultiRoomEnv
    from minigrid.wrappers import ImgObsWrapper
    from xp.minigrid.ppo import Learner

    torch.set_num_threads(1)
    env.make_env = lambda size: ImgObsWrapper(MultiRoomEnv(minNumRooms=int(size), maxNumRooms=int(size), maxRoomSize=room_size))
    cfg = yaml.safe_load(open("configs/minigrid.yaml"))
    ent, rnd = (0.01, None) if agent == "ppo" else (0.01, coef)
    t0 = time.time()
    learner = Learner(cfg["ppo"], ent, rnd, seed)
    run = learner.train(rooms, BUDGET, target=0.95)
    res = learner.evaluate(rooms, 100)
    rec = {"rooms": rooms, "room_size": room_size, "agent": agent, "rnd_coef": coef, "seed": seed, "final_success": res["success"], "steps_to_target": run["steps_to_target"],
           "curve": [(r.get("steps"), r["success_100"]) for r in run["curve"]][::5], "secs": time.time() - t0}
    path.write_text(json.dumps(rec))
    print(f"N{rooms} S{room_size} {agent} {coef} s{seed}: test success {res['success']:.2f}, steps_to_95 {run['steps_to_target']}, {rec['secs']:.0f}s", flush=True)
    return str(path)


if __name__ == "__main__":
    jobs = [(4, 4, "ppo", None, sd) for sd in range(5)] + [(4, 4, "rnd", c, sd) for c in (0.01, 0.03, 0.1) for sd in range(5)]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        list(pool.map(job, jobs))
    print("spike done", flush=True)
