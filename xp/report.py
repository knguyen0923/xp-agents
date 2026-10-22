from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xp.plots.curves import plot_degradation
from xp.protocol.phase import Phase, load_config
from xp.protocol.results import load_results, require_final
from xp.stats.summary import crossings, curve_table, gap_table, pair_table, pair_vs_best_single


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build final tables and figures from results.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="results directory written by xp.run")
    parser.add_argument("--figs", required=True, help="directory for tables and figures")
    parser.add_argument("--allow-provisional", action="store_true", help="draft only: include provisional results")
    args = parser.parse_args(argv)

    phase = Phase.from_config(load_config(args.config))
    df = load_results(args.out, args.phase)
    if df.empty:
        raise SystemExit(f"no results found under {args.out}/{args.phase}")
    require_final(df, allow_provisional=args.allow_provisional)

    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    curves = curve_table(df)
    curves.to_csv(figs / "curves.csv", index=False)
    gap_table(df, phase.familiar).to_csv(figs / "gaps.csv", index=False)

    pair = pair_table(df, "kairi", "kaiser")
    pair["crossover_intervals"] = str(crossings(pair))
    pair.to_csv(figs / "crossovers.csv", index=False)

    team_rows = []
    for team in sorted(a for a in df["agent"].unique() if a.startswith("tt_")):
        t = pair_vs_best_single(df, team)
        t.insert(0, "team", team)
        team_rows.append(t)
    if team_rows:
        pd.concat(team_rows).to_csv(figs / "pair_vs_best_single.csv", index=False)

    plot_degradation(curves, phase.familiar, figs / "degradation.png")
    print(f"wrote tables and figure to {figs}")
    print(f"kairi-vs-kaiser crossover intervals: {crossings(pair) or 'none (one agent dominates, or all ties)'}")


if __name__ == "__main__":
    main()
