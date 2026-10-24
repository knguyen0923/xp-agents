"""Few-shot (frozen hyperparameters, 200k steps of adaptation) Kairi vs Kaiser, paired by eval seed, over every variant we ran.

Kairi variants: tuned (results/minigrid), ent_coef 0.1 (results/robustness/ent-0.1). Kaiser variants: tuned (rnd_coef 0.001, ent 0.01), rnd_coef
0.003 and 0.01 (results/robustness), and matched entropy (results/ent-fair: ent 0.03, rnd_coef 0.01). Post-hoc, outside the protocol: no new training.
Prints each variant's few-shot success (95% CI) and every Kairi - Kaiser paired difference (positive = Kairi better) at the few-shot sizes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xp.stats.ci import bootstrap_ci, paired_diff_ci

METRIC = "success_fewshot"


def _matrix(directory: Path, agent: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(directory.glob(f"{agent}__seed-*.csv"))])
    return df[df.metric == METRIC].pivot_table(index="seed", columns="knob", values="value").sort_index()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--csv", help="optional path to write the pairwise table")
    args = parser.parse_args(argv)
    r = Path(args.results)
    kairi = {"tuned (ent 0.03)": _matrix(r / "minigrid", "kairi"), "ent 0.1": _matrix(r / "robustness/ent-0.1/minigrid", "kairi")}
    kaiser = {"tuned (rnd 0.001, ent 0.01)": _matrix(r / "minigrid", "kaiser"),
              "rnd 0.003 (ent 0.01)": _matrix(r / "robustness/rnd-0.003/minigrid", "kaiser"),
              "rnd 0.01 (ent 0.01)": _matrix(r / "robustness/rnd-0.01/minigrid", "kaiser"),
              "matched entropy (rnd 0.01, ent 0.03)": _matrix(r / "ent-fair/minigrid", "kaiser")}
    sizes = list(next(iter(kairi.values())).columns)
    rows = []
    for agent, variants in (("kairi", kairi), ("kaiser", kaiser)):
        for name, m in variants.items():
            for s in sizes:
                mean, lo, hi = bootstrap_ci(m[s])
                rows.append({"agent": agent, "variant": name, "size": s, "n": len(m), "mean": mean, "lo": lo, "hi": hi})
    print("few-shot success by variant (95% CI):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    pairs = []
    for kn, km in kairi.items():
        for sn, sm in kaiser.items():
            for s in sizes:
                d, lo, hi = paired_diff_ci(km[s], sm[s])
                pairs.append({"kairi": kn, "kaiser": sn, "size": s, "kairi_minus_kaiser": d, "lo": lo, "hi": hi,
                              "excludes_0": "yes" if lo > 0 or hi < 0 else "no"})
    pairs = pd.DataFrame(pairs)
    print("\nKairi minus Kaiser, paired by eval seed (positive = Kairi better):")
    print(pairs.round(3).to_string(index=False))
    if args.csv:
        pairs.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
