"""Post-hoc check of Kaiser's tuned optimum in the bandit drift condition, on the eval seeds.

python scripts/bandit_drift_check.py [--config configs/bandit-drift.yaml] [--existing results/bandit-drift/bandit]
                                     [--out docs/bandit_drift_check/kaiser_neighbours.csv]
Scores Kaiser at the tuned point (gamma=1.0, c=1.0, a natural limit) and three neighbours at the familiar drift on the
eval seeds, with paired 95% bootstrap CIs of each neighbour minus the tuned point. Diagnostic only: it describes the
tuning surface, never chooses parameters, and writes no results folder. The tuned row is checked against the stored
eval results, so a mismatch means the setup differs from the main run.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import yaml

from xp.bandit.phase import build_factories, make_per_seed_fn
from xp.protocol.evaluate import evaluate_unit
from xp.protocol.phase import Phase
from xp.stats.ci import bootstrap_ci, paired_diff_ci

TUNED = {"gamma": 1.0, "c": 1.0}
NEIGHBOURS = [{"gamma": 0.999, "c": 0.5}, {"gamma": 0.999, "c": 1.0}, {"gamma": 0.995, "c": 0.2}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bandit-drift.yaml")
    ap.add_argument("--existing", default="results/bandit-drift/bandit")
    ap.add_argument("--out", default="docs/bandit_drift_check/kaiser_neighbours.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    phase = Phase.from_config(cfg)
    fn = make_per_seed_fn(cfg, build_factories(cfg)["kaiser"])
    familiar = phase.familiar

    def run(params):
        return evaluate_unit(phase, "kaiser", params, familiar, False, fn).set_index("seed")["value"]

    tuned = run(TUNED)
    stored = pd.read_csv(os.path.join(args.existing, f"kaiser__drift-{familiar:g}.csv")).set_index("seed")["value"]
    if not np.allclose(tuned.sort_index().values, stored.sort_index().values):
        raise SystemExit("tuned row does not reproduce the stored eval results; setup differs from the main run")
    print(f"tuned row reproduces the stored eval results ({len(tuned)} seeds)")

    mean, lo, hi = bootstrap_ci(tuned.values)
    rows = [{"gamma": TUNED["gamma"], "c": TUNED["c"], "mean_regret": mean, "ci_lo": lo, "ci_hi": hi,
             "diff_vs_tuned": 0.0, "diff_lo": np.nan, "diff_hi": np.nan}]
    for params in NEIGHBOURS:
        v = run(params)
        mean, lo, hi = bootstrap_ci(v.values)
        d, dlo, dhi = paired_diff_ci(v.loc[tuned.index].values, tuned.values)
        rows.append({"gamma": params["gamma"], "c": params["c"], "mean_regret": mean, "ci_lo": lo, "ci_hi": hi,
                     "diff_vs_tuned": d, "diff_lo": dlo, "diff_hi": dhi})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
