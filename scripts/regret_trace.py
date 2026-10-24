"""Post-hoc ablation (outside the protocol): when in the horizon does Kaiser's regret differ from Kairi's?

Runs the frozen tuned agents (parameters from `<tuned>/bandit/tuning/<agent>.json`) on the eval seeds at one knob value and splits
per-step regret into time windows, reporting each agent's mean and the paired Kairi - Kaiser difference with a 95% bootstrap CI
(positive = Kaiser better). Uses the same chunking and agent seeds as the protocol evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xp.bandit.phase import build_factories, world_params
from xp.bandit.world import make_world, run_agent_trace
from xp.protocol.phase import Phase, load_config
from xp.stats.ci import paired_diff_ci

WINDOWS = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000)]


def traces(cfg: dict, phase: Phase, tuned_dir: Path, knob: float) -> dict:
    out = {}
    for agent, build in build_factories(cfg).items():
        params = json.loads((tuned_dir / "bandit" / "tuning" / f"{agent}.json").read_text())["best_params"]
        w = world_params(cfg, knob)
        parts = []
        seeds = np.array(phase.eval_seeds)
        for chunk_index, start in enumerate(range(0, len(seeds), phase.chunk)):
            chunk = seeds[start:start + phase.chunk]
            world = make_world(chunk, w["lam"], cfg["horizon"], w["k"], w["sigma"], w["drift"])
            parts.append(run_agent_trace(build(params, phase.agent_seed + chunk_index, w["k"]), world))
        out[agent] = np.vstack(parts)
    return out


def window_table(tr: dict, windows=WINDOWS) -> pd.DataFrame:
    rows = []
    for lo, hi in windows:
        a, b = tr["kairi"][:, lo:hi].mean(axis=1), tr["kaiser"][:, lo:hi].mean(axis=1)
        d, dlo, dhi = paired_diff_ci(a, b)
        rows.append({"window": f"{lo}-{hi}", "kairi": a.mean(), "kaiser": b.mean(), "kairi_minus_kaiser": d, "lo": dlo, "hi": dhi,
                     "share_of_total_gap": np.nan})
    table = pd.DataFrame(rows)
    widths = np.array([hi - lo for lo, hi in windows])
    contrib = table["kairi_minus_kaiser"].to_numpy() * widths
    table["share_of_total_gap"] = contrib / contrib.sum() if contrib.sum() != 0 else np.nan
    return table


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tuned", required=True, help="results dir whose tuning JSONs give the frozen parameters")
    parser.add_argument("--knob", type=float, required=True)
    parser.add_argument("--csv")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    table = window_table(traces(cfg, phase, Path(args.tuned), args.knob))
    print(table.round(4).to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
