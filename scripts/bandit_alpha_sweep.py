"""Finer Kairi `alpha` sweep on the familiar bandit condition, tune seeds only (no eval seeds touched).

python scripts/bandit_alpha_sweep.py [--config configs/bandit.yaml] [--existing results/bandit-local/bandit/tuning]
                                     [--out docs/bandit_alpha_sweep/kairi_fine_alpha.csv]
Rescores Kairi on the original epsilon grid crossed with a denser alpha grid, and compares the best score with
the tuned scores already on disk for Kairi and Kaiser. Diagnostic only: it writes no results folder.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

from xp.bandit.phase import build_factories, make_per_seed_fn
from xp.protocol.phase import Phase
from xp.protocol.tune import score_params

FINE_ALPHA = [0.02, 0.03, 0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.55, 0.7, 0.85, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bandit.yaml")
    ap.add_argument("--existing", default="results/bandit-local/bandit/tuning")
    ap.add_argument("--out", default="docs/bandit_alpha_sweep/kairi_fine_alpha.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    phase = Phase.from_config(cfg)
    fn = make_per_seed_fn(cfg, build_factories(cfg)["kairi"])
    rows = []
    for eps in cfg["agents"]["kairi"]["grid"]["epsilon"]:
        for alpha in FINE_ALPHA:
            score = score_params(phase, fn, {"epsilon": eps, "alpha": alpha})
            rows.append({"epsilon": eps, "alpha": alpha, "score": score})
        print(f"epsilon={eps:g} done", flush=True)
    fine = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fine.to_csv(args.out, index=False)

    old_kairi = pd.read_csv(os.path.join(args.existing, "kairi.csv"))
    kaiser = pd.read_csv(os.path.join(args.existing, "kaiser.csv"))
    b_fine, b_old, b_kaiser = fine.loc[fine["score"].idxmin()], old_kairi.loc[old_kairi["score"].idxmin()], kaiser.loc[kaiser["score"].idxmin()]
    print(f"Kairi original grid ({len(old_kairi)} pts): best {b_old['score']:.4f} at epsilon={b_old['epsilon']:g}, alpha={b_old['alpha']:g}")
    print(f"Kairi fine alpha grid ({len(fine)} pts):    best {b_fine['score']:.4f} at epsilon={b_fine['epsilon']:g}, alpha={b_fine['alpha']:g}")
    print(f"Kaiser tuned ({len(kaiser)} pts):           best {b_kaiser['score']:.4f}")
    print("fine-grid best by alpha (min over epsilon):")
    print(fine.groupby("alpha")["score"].min().round(4).to_string())


if __name__ == "__main__":
    main()
