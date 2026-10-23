from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xp.errors import ConfigMismatchError, ProtocolError
from xp.plots.curves import plot_degradation
from xp.protocol.phase import Phase, load_config
from xp.protocol.results import load_results, require_final
from xp.stats.summary import crossings, curve_table, gap_table, pair_table, pair_vs_best_single


EXPECTED_AGENTS = ("kairi", "kaiser", "tt_fixed", "tt_triggered", "tt_meta")


def report_settings(cfg: dict) -> dict:
    """Per-phase report options; the bandit config has no `report` block, so these defaults reproduce Phase 1."""
    r = cfg.get("report", {})
    return {
        "metric": r.get("metric", "regret"),
        "lower_is_better": r.get("lower_is_better", True),
        "agents": tuple(r.get("agents", EXPECTED_AGENTS)),
        "extra_metrics": dict(r.get("extra_metrics", {})),
        "xlabel": r.get("xlabel", "jump rate λ (dashed line = familiar)"),
        "ylabel": r.get("ylabel", "mean per-step regret (95% CI)"),
        "title": r.get("title", "Regret vs jump rate"),
    }


def validate_results(df: pd.DataFrame, phase: Phase, settings: dict = None) -> None:
    """Results must come from this config and cover every agent x knob cell completely."""
    settings = settings or report_settings(phase.cfg)
    hashes = set(df["config_hash"])
    if hashes != {phase.config_hash}:
        raise ConfigMismatchError(f"results were written under configs {sorted(hashes)}, current config is {phase.config_hash}")
    counts = df[df["metric"] == settings["metric"]].groupby(["agent", "knob"]).size()
    expected = len(phase.eval_seeds)
    problems = []
    for agent in settings["agents"]:
        for knob in phase.sweep:
            n = int(counts.get((agent, knob), 0))
            if n != expected:
                problems.append(f"{agent} {phase.knob_name}={knob:g}: {n}/{expected} rows")
    if problems:
        raise ProtocolError("incomplete results, " + "; ".join(problems))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build final tables and figures from results.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="results directory written by xp.run")
    parser.add_argument("--figs", required=True, help="directory for tables and figures")
    parser.add_argument("--allow-provisional", action="store_true", help="draft only: include provisional results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    if args.phase != cfg["phase"]:
        raise ProtocolError(f"--phase {args.phase!r} does not match the config's phase {cfg['phase']!r}")
    df = load_results(args.out, args.phase)
    if df.empty:
        raise SystemExit(f"no results found under {args.out}/{args.phase}")
    settings = report_settings(cfg)
    validate_results(df, phase, settings)
    require_final(df, allow_provisional=args.allow_provisional)

    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    metric, lower = settings["metric"], settings["lower_is_better"]
    curves = curve_table(df, metric=metric)
    curves.to_csv(figs / "curves.csv", index=False)
    gap_table(df, phase.familiar, metric=metric).to_csv(figs / "gaps.csv", index=False)

    first, second = settings["agents"][0], settings["agents"][1]
    pair = pair_table(df, first, second, metric=metric, lower_is_better=lower)
    pair["crossover_intervals"] = str(crossings(pair))
    pair.to_csv(figs / "crossovers.csv", index=False)

    team_rows = []
    for team in sorted(a for a in df["agent"].unique() if a.startswith("tt_")):
        t = pair_vs_best_single(df, team, metric=metric, lower_is_better=lower)
        t.insert(0, "team", team)
        team_rows.append(t)
    if team_rows:
        pd.concat(team_rows).to_csv(figs / "pair_vs_best_single.csv", index=False)

    for extra, extra_lower in settings["extra_metrics"].items():
        curve_table(df, metric=extra).to_csv(figs / f"curves__{extra}.csv", index=False)
        extra_pair = pair_table(df, first, second, metric=extra, lower_is_better=extra_lower)
        extra_pair.to_csv(figs / f"crossovers__{extra}.csv", index=False)

    plot_degradation(
        curves, phase.familiar, figs / "degradation.png",
        title=settings["title"], xlabel=settings["xlabel"], ylabel=settings["ylabel"],
    )
    print(f"wrote tables and figure to {figs}")
    print(f"{first}-vs-{second} crossover intervals: {crossings(pair) or 'none (one agent dominates, or all ties)'}")


if __name__ == "__main__":
    main()
