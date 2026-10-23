from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import torch

from xp.minigrid.phase import _run_jobs, build_learner
from xp.protocol.phase import Phase, load_config


def suggest_budget(curves: List[List[dict]], target: float, cap: int) -> dict:
    """Smallest logged step count at which mean training success over the seeds reaches `target` (else `cap`).

    Each curve is a list of {"steps", "success_100"} dicts logged at the same steps for every seed.
    """
    steps = [row["steps"] for row in curves[0]]
    for curve in curves:
        if [row["steps"] for row in curve] != steps:
            raise ValueError("all curves must be logged at the same steps")
    mean_success = np.mean([[0.0 if row["success_100"] != row["success_100"] else row["success_100"] for row in c]
                            for c in curves], axis=0)
    hits = [s for s, m in zip(steps, mean_success) if m >= target and s <= cap]
    return {"steps": int(hits[0]) if hits else int(cap), "reached": bool(hits), "mean_success": mean_success.tolist(),
            "log_steps": steps}


def calibration_run(cfg: dict, seed: int) -> list:
    """Default-entropy Kairi on one tune seed for the whole cap; returns its training-success curve."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    default_ent = cfg["agents"]["kaiser"]["fixed"]["ent_coef"]
    learner = build_learner(cfg, "kairi", {"ent_coef": default_ent}, seed)
    return learner.train(int(phase.familiar), cfg["budget"]["cap"], target=2.0)["curve"]  # target 2.0: never stops early


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Suggest the Phase 2 training budget S from tune-seed Kairi runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="JSON file for the curves and the suggestion")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    curves = _run_jobs(calibration_run, [(cfg, s) for s in phase.tune_seeds], args.workers)
    suggestion = suggest_budget(curves, cfg["budget"]["target_success"], cfg["budget"]["cap"])
    with open(args.out, "w") as f:
        json.dump({"suggestion": suggestion, "curves": curves}, f, indent=2)
    state = "reached" if suggestion["reached"] else "NOT reached; using the cap"
    print(f"suggested budget S = {suggestion['steps']} steps (80% target {state}); set budget.steps in the config")


if __name__ == "__main__":
    main()
