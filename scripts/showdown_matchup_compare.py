"""Paired cross-run comparison for Showdown: matchup run minus original run, per agent, paired by eval seed.

python scripts/showdown_matchup_compare.py [--orig results/showdown] [--new results/showdown-matchup2]
                                           [--out docs/spikes/showdown_matchup_results/cross_run_paired.csv]
Reads the per-seed result CSVs of both runs; the 95% interval is a bootstrap over the 10 shared eval seeds.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd

METRICS = ("auc", "win_eval", "steps_to_30")
LOWER_IS_BETTER = {"steps_to_30"}


def load(run: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(run, "showdown", "*__seed-*.csv")))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_seed(df: pd.DataFrame, agent: str, metric: str) -> pd.Series:
    return df[(df.agent == agent) & (df.metric == metric)].set_index("seed").value.sort_index()


def interval(rng, x: np.ndarray) -> tuple:
    means = [rng.choice(x, len(x)).mean() for _ in range(20000)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="results/showdown")
    ap.add_argument("--new", default="results/showdown-matchup2")
    ap.add_argument("--out", default="docs/spikes/showdown_matchup_results/cross_run_paired.csv")
    a = ap.parse_args()
    old, new = load(a.orig), load(a.new)
    rng = np.random.default_rng(0)
    rows, diffs = [], {}
    for agent in ("kairi", "kaiser"):
        for m in METRICS:
            x, y = per_seed(old, agent, m), per_seed(new, agent, m)
            d = y - x
            diffs[(agent, m)] = d
            lo, hi = interval(rng, d.values)
            better = int((d < 0).sum() if m in LOWER_IS_BETTER else (d > 0).sum())
            rows.append([agent, m, f"{x.mean():.6g}", f"{y.mean():.6g}", f"{d.mean():.6g}", f"{lo:.6g}", f"{hi:.6g}",
                         f"{better}/{len(d)}", int((d == 0).sum())])
    for m in METRICS:
        g = diffs[("kaiser", m)] - diffs[("kairi", m)]
        lo, hi = interval(rng, g.values)
        rows.append(["kaiser_minus_kairi", m, "", "", f"{g.mean():.6g}", f"{lo:.6g}", f"{hi:.6g}", "", ""])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "metric", "orig", "matchup", "diff", "lo", "hi", "seeds_improved", "ties"])
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
