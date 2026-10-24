"""Kairi vs Kaiser only, paired by eval seed, from a bandit results dir (ignores the tag-team agents, so it works when only they are provisional).

Prints mean regret per agent and the paired Kairi - Kaiser difference with a 95% bootstrap CI at every knob value (positive = Kaiser better),
and refuses if either single agent's tuning is provisional.
"""
from __future__ import annotations

import argparse

import pandas as pd

from xp.protocol.phase import Phase, load_config
from xp.protocol.results import load_results, read_tune
from xp.stats.ci import bootstrap_ci, paired_diff_ci


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="results directory of the run")
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    for agent in ("kairi", "kaiser"):
        tune = read_tune(args.out, phase, agent)
        if tune.provisional:
            raise SystemExit(f"{agent} tuning is provisional ({tune.boundary_dims}); widen its grid and rerun")
        print(f"{agent} tuned at {phase.knob_name}={phase.familiar:g}: {tune.best_params}")
    df = load_results(args.out, phase.name)
    wide = df[(df.metric == "regret") & df.agent.isin(["kairi", "kaiser"])].pivot_table(index=["knob", "seed"], columns="agent", values="value")
    rows = []
    for knob, g in wide.groupby(level="knob"):
        d, lo, hi = paired_diff_ci(g["kairi"].to_numpy(), g["kaiser"].to_numpy())
        rows.append({"knob": knob, "kairi": bootstrap_ci(g["kairi"].to_numpy())[0], "kaiser": bootstrap_ci(g["kaiser"].to_numpy())[0],
                     "kairi_minus_kaiser": d, "lo": lo, "hi": hi,
                     "winner": "kaiser" if lo > 0 else "kairi" if hi < 0 else "tie"})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
