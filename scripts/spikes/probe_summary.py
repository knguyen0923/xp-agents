"""Summarise the 5M-step MultiRoom RND probe (scripts/spikes/multiroom_probe.py): per-run CSV plus paired plain-PPO - RND differences.

Steps to 95% training success; a run that never reaches it is censored at the 5M budget. Paired by seed, 95% bootstrap CI over the 5 seeds.
Positive difference = RND needed fewer steps (RND better). Post-hoc, outside the protocol.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from xp.stats.ci import paired_diff_ci

BUDGET = 5_000_000


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="results/spike-multiroom/probe")
    parser.add_argument("--csv", default="docs/spikes/multiroom_probe.csv")
    args = parser.parse_args(argv)
    rows = []
    for f in sorted(glob.glob(f"{args.dir}/*.json")):
        r = json.load(open(f))
        reached = r["steps_to_target"] is not None
        rows.append({"agent": "ppo" if r["agent"] == "ppo" else "ppo+rnd", "rnd_coef": r["rnd_coef"], "seed": r["seed"],
                     "test_success": r["final_success"], "reached_95": reached,
                     "steps_to_95": r["steps_to_target"] if reached else BUDGET, "secs": round(r["secs"])})
    df = pd.DataFrame(rows).sort_values(["agent", "rnd_coef", "seed"], na_position="first")
    df.to_csv(args.csv, index=False)
    ppo = df[df.agent == "ppo"].set_index("seed")["steps_to_95"]
    print(f"plain PPO: reached {int(df[df.agent == 'ppo'].reached_95.sum())}/5, median {np.median(ppo) / 1e6:.2f}M, mean {ppo.mean() / 1e6:.2f}M")
    for coef, g in df[df.agent == "ppo+rnd"].groupby("rnd_coef"):
        g = g.set_index("seed")["steps_to_95"].reindex(ppo.index)
        reached = int((g < BUDGET).sum())
        d, lo, hi = paired_diff_ci(ppo.to_numpy(float), g.to_numpy(float))
        print(f"rnd {coef:g}: reached {reached}/5, median {np.median(g) / 1e6:.2f}M, mean {g.mean() / 1e6:.2f}M; "
              f"PPO - RND mean {d / 1e6:+.2f}M [{lo / 1e6:+.2f}, {hi / 1e6:+.2f}]; RND faster on {int((g < ppo).sum())}/5 seeds")


if __name__ == "__main__":
    main()
