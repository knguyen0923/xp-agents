"""Compare each agent's tuned setting with its tuning-tied alternatives, and every Kairi variant against every Kaiser variant.

Reads per-seed CSVs of the main run (`<main>/minigrid/`) and of scripts/tie_robustness.py
(`<robust>/{ent,rnd}-<value>/minigrid/`). Pairs are matched by eval seed, as the protocol requires.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xp.stats.ci import bootstrap_ci, paired_diff_ci

FAIL_BELOW = 0.5  # a seed that scores under this at the familiar size "failed outright"


def _load(directory: Path, agent: str) -> pd.DataFrame:
    files = sorted(directory.glob(f"{agent}__seed-*.csv"))
    return pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame()


def _matrix(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """seeds x sizes matrix of `metric`."""
    return df[df.metric == metric].pivot_table(index="seed", columns="knob", values="value").sort_index()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="results/minigrid")
    parser.add_argument("--robust", default="results/robustness")
    parser.add_argument("--familiar", type=float, default=6.0)
    parser.add_argument("--metric", default="success")
    parser.add_argument("--csv", help="optional path to write the pairwise table")
    args = parser.parse_args(argv)

    main_dir, root = Path(args.main), Path(args.robust)
    kairi = {"tuned": _load(main_dir, "kairi")}
    kaiser = {"tuned": _load(main_dir, "kaiser")}
    for d in sorted(root.glob("ent-*")):
        kairi[d.name[4:]] = _load(d / "minigrid", "kairi")
    for d in sorted(root.glob("rnd-*")):
        kaiser[d.name[4:]] = _load(d / "minigrid", "kaiser")
    mats = {"kairi": {k: _matrix(v, args.metric) for k, v in kairi.items() if not v.empty},
            "kaiser": {k: _matrix(v, args.metric) for k, v in kaiser.items() if not v.empty}}
    sizes = mats["kairi"]["tuned"].columns

    pd.set_option("display.width", 200)
    rows = []
    for agent, variants in mats.items():
        for name, m in variants.items():
            for size in sizes:
                mean, lo, hi = bootstrap_ci(m[size])
                rows.append({"agent": agent, "variant": name, "n": len(m), "size": size, "mean": mean, "lo": lo, "hi": hi})
    print("success by variant (95% CI):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    pairs = []
    for kn, km in mats["kairi"].items():
        for sn, sm in mats["kaiser"].items():
            seeds = km.index.intersection(sm.index)
            if len(seeds) < max(len(km), len(sm)):
                print(f"note: kairi[{kn}] vs kaiser[{sn}] compared on {len(seeds)} shared seeds")
            for size in sizes:
                d, lo, hi = paired_diff_ci(km.loc[seeds, size], sm.loc[seeds, size])
                pairs.append({"kairi": kn, "kaiser": sn, "n": len(seeds), "size": size, "kairi_minus_kaiser": d, "lo": lo, "hi": hi,
                              "excludes_0": "yes" if lo > 0 or hi < 0 else "no"})
    pairs = pd.DataFrame(pairs)
    print("\nKairi minus Kaiser, paired by eval seed (positive = Kairi better):")
    print(pairs.round(3).to_string(index=False))

    print(f"\nseeds scoring < {FAIL_BELOW} at size {args.familiar:g} (failed outright):")
    for agent, variants in mats.items():
        for name, m in variants.items():
            bad = [int(s) for s in m.index[m[args.familiar] < FAIL_BELOW]]
            print(f"  {agent}[{name}]: {bad if bad else 'none'}")
    if args.csv:
        pairs.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
