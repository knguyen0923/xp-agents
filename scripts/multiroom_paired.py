"""Paired Kairi-vs-Kaiser comparison for the MultiRoom confirmation run.

python scripts/multiroom_paired.py --run ../xp-agents-run/results/multiroom --csv docs/spikes/multiroom_confirmation.csv
Writes the per-seed values and prints paired bootstrap 95% CIs of (Kaiser - Kairi), resampling seeds.
"""
from __future__ import annotations

import argparse
import csv
import glob

import numpy as np

METRICS = ("auc", "steps_to_95", "reached_95", "success")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/multiroom")
    ap.add_argument("--csv", default="docs/spikes/multiroom_confirmation.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = {}
    for agent in ("kairi", "kaiser"):
        for f in sorted(glob.glob(f"{a.run}/minigrid/{agent}__seed-*.csv")):
            for r in csv.DictReader(open(f)):
                D[(agent, r["metric"], int(r["seed"]))] = float(r["value"])
    seeds = sorted({k[2] for k in D})
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "seed"] + list(METRICS))
        for agent in ("kairi", "kaiser"):
            for s in seeds:
                w.writerow([agent, s] + [D[(agent, m, s)] for m in METRICS])
    rng = np.random.default_rng(a.seed)
    print("metric,kairi_mean,kaiser_mean,diff,lo,hi,kaiser_better_seeds")
    for m in METRICS:
        x = np.array([D[("kairi", m, s)] for s in seeds])
        y = np.array([D[("kaiser", m, s)] for s in seeds])
        d = y - x
        boot = [rng.choice(d, len(d)).mean() for _ in range(20000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        better = int((d < 0).sum() if m == "steps_to_95" else (d > 0).sum())
        print(f"{m},{x.mean():.4g},{y.mean():.4g},{d.mean():.4g},{lo:.4g},{hi:.4g},{better}/{len(d)}")


if __name__ == "__main__":
    main()
