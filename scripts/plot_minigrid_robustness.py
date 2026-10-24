"""Figure for the MiniGrid conclusion: tuned Kairi and Kaiser plus Kaiser at Kairi's entropy (results/ent-fair), 95% bootstrap CIs per size."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xp.plots.curves import plot_degradation
from xp.stats.ci import bootstrap_ci


def _curves(directory: Path, agent: str, label: str, metric: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(directory.glob(f"{agent}__seed-*.csv"))])
    wide = df[df.metric == metric].pivot_table(index="seed", columns="knob", values="value").sort_index()
    rows = [{"agent": label, "knob": size, **dict(zip(("mean", "lo", "hi"), bootstrap_ci(wide[size])))} for size in wide.columns]
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="results/minigrid")
    parser.add_argument("--entfair", default="results/ent-fair/minigrid")
    parser.add_argument("--out", default="figs/minigrid_robustness.png")
    args = parser.parse_args(argv)
    curves = pd.concat([
        _curves(Path(args.main), "kairi", "kairi", "success"),
        _curves(Path(args.main), "kaiser", "kaiser", "success"),
        _curves(Path(args.entfair), "kaiser", "kaiser (matched entropy)", "success"),
    ])
    plot_degradation(curves, 6.0, args.out, title="DoorKey success vs grid size (10 seeds)",
                     xlabel="grid size (dashed line = familiar)", ylabel="success rate on held-out layout seeds (95% CI)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
