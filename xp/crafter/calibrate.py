from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from xp.crafter import phase as cp
from xp.crafter.stats import ACHIEVEMENTS, nan_to_zero
from xp.errors import ProtocolError
from xp.protocol.jobs import grid_size, run_jobs
from xp.protocol.phase import Phase, config_hash, load_config

RARE_BELOW = 0.05  # an achievement is "rare" (headroom for exploration) when plain PPO unlocks it in under 5% of episodes
DEEP_RANGE = (0.05, 0.95)


def suggest_budget(curves: List[List[dict]], target: float, cap: int, sustain: int) -> dict:
    """S = first logged step from which the seed-mean `ach_100` (NaN counted as 0) is >= `target` for `sustain` consecutive log points.

    A crossing within the last `sustain - 1` log points of the run does not qualify. If none qualifies, S = `cap` and `reached` is False.
    """
    steps = [row["steps"] for row in curves[0]]
    for c in curves:
        if [row["steps"] for row in c] != steps:
            raise ValueError("all curves must be logged at the same steps")
    mean_ach = np.mean([[nan_to_zero(row["ach_100"]) for row in c] for c in curves], axis=0)
    hit = None
    for i in range(len(steps) - sustain + 1):
        if steps[i + sustain - 1] <= cap and all(mean_ach[i + j] >= target for j in range(sustain)):
            hit = i
            break
    index = hit if hit is not None else len(steps) - 1
    return {"steps": int(steps[index]) if hit is not None else int(cap), "reached": hit is not None, "index": int(index),
            "mean_ach": mean_ach.tolist(), "log_steps": steps}


def gate_checks(suggestion: dict, curves: List[List[dict]], min_rare: int = 5, min_deep: int = 3) -> dict:
    """G1 clause 2 (rare achievements), G2 (learning) and the deep set at S, averaged over the tune seeds; `go` is all of them plus G1 clause 1."""
    i = suggestion["index"]
    mean_ach = suggestion["mean_ach"]
    rates = {a: float(np.mean([c[i]["rates"][a] for c in curves])) for a in ACHIEVEMENTS}
    cut = suggestion["log_steps"][i] * 0.10
    early_pts = [m for s, m in zip(suggestion["log_steps"], mean_ach) if s <= cut] or [mean_ach[0]]
    early, final = float(np.mean(early_pts)), float(mean_ach[i])
    rare = sum(1 for r in rates.values() if r < RARE_BELOW)
    deep = sorted(a for a, r in rates.items() if DEEP_RANGE[0] <= r <= DEEP_RANGE[1])
    out = {"g1_target": bool(suggestion["reached"]), "rare_count": rare, "rare_ok": bool(rare >= min_rare), "g2": bool(final > early),
           "early_ach": early, "final_ach": final, "rates_at_s": rates, "deep_set": deep, "deep_ok": bool(len(deep) >= min_deep)}
    reasons = []
    if not out["g1_target"]:
        reasons.append("G1: the target ach_100 was not sustained within the cap")
    if not out["rare_ok"]:
        reasons.append(f"G1: only {rare} achievements are rare at S (need {min_rare})")
    if not out["g2"]:
        reasons.append(f"G2: ach_100 at S ({final:.2f}) does not exceed the early mean ({early:.2f})")
    if not out["deep_ok"]:
        reasons.append(f"deep set has {len(deep)} achievements (need {min_deep})")
    out["reasons"], out["go"] = reasons, not reasons
    return out


def phase_runs(cfg: dict) -> int:
    """Tuning runs (2 agents x grid x tune seeds) plus evaluation runs (2 agents x eval seeds): 56 in the real config."""
    n_tune, n_eval = cfg["seeds"]["tune"][1] - cfg["seeds"]["tune"][0], cfg["seeds"]["eval"][1] - cfg["seeds"]["eval"][0]
    return 2 * grid_size(cfg["agents"]["kairi"]) * n_tune + 2 * n_eval


def implied_hours(S: int, steps_per_s: float, cfg: dict) -> dict:
    n_tune = cfg["seeds"]["tune"][1] - cfg["seeds"]["tune"][0]
    return {"full_phase_hours": phase_runs(cfg) * S / steps_per_s / 3600.0,
            "calibration_hours": n_tune * cfg["budget"]["cap"] / steps_per_s / 3600.0}


def calibration_run(cfg: dict, seed: int) -> dict:
    """Default-entropy Kairi on one tune seed for the whole cap, no early stopping."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    learner = cp.build_learner(cfg, "kairi", {"ent_coef": cfg["agents"]["kaiser"]["fixed"]["ent_coef"]}, seed)
    env = cp.make_env(cfg, "tune", seed)
    try:
        run = learner.train(env, cfg["budget"]["cap"], log_every=cfg["ppo"].get("log_every", 10))
    finally:
        env.close()
    return {"curve": run["curve"], "secs": run["secs"], "steps": run["steps"]}


def _json_safe(obj):
    """Recursively replace NaN floats with None so the file is strict JSON."""
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _nan_back(obj):
    """Inverse of `_json_safe` for the cached curve rows: None floats come back as NaN."""
    if obj is None:
        return float("nan")
    if isinstance(obj, dict):
        return {k: _nan_back(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_back(v) for v in obj]
    return obj


def calibration_hash(cfg: dict) -> str:
    """Config hash without `budget.steps` and `budget.deep_set` (the calibration's own outputs, set after it runs)."""
    c = copy.deepcopy(cfg)
    for k in ("steps", "deep_set"):
        c.get("budget", {}).pop(k, None)
    return config_hash(c)


def seed_cache_path(out: str, seed: int) -> Path:
    o = Path(out)
    return o.with_name(f"{o.stem}.seed-{seed}.json")


def _load_cached(path: Path, h: str):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("config_hash") != h:
        raise ProtocolError(f"{path} was written for a different config (hash {data.get('config_hash')}, now {h}); delete the calibration cache files (*.seed-*.json) or use a fresh --out")
    return _nan_back(data["result"])


def cached_calibration_run(cfg: dict, seed: int, path: str) -> dict:
    """Run one seed and write its result atomically (tmp + os.replace) so a finished seed survives a disconnect."""
    result = calibration_run(cfg, seed)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"config_hash": calibration_hash(cfg), "result": _json_safe(result)}, allow_nan=False))
    os.replace(tmp, path)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate the Crafter budget S from tune-seed Kairi runs (spec 2026-11-02, Change 1).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="crafter_calibration.json")
    parser.add_argument("--workers", type=int, default=1, help="keep 1 for a clean throughput measurement")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    h = calibration_hash(cfg)
    cached = {s: _load_cached(seed_cache_path(args.out, s), h) for s in phase.tune_seeds}
    todo = [s for s in phase.tune_seeds if cached[s] is None]
    print(f"calibration: {len(cached) - len(todo)} seed(s) cached, {len(todo)} to run", flush=True)
    fresh = run_jobs(cached_calibration_run, [(cfg, s, str(seed_cache_path(args.out, s))) for s in todo], args.workers)
    cached.update(dict(zip(todo, fresh)))
    runs = [cached[s] for s in phase.tune_seeds]
    curves = [r["curve"] for r in runs]
    b = cfg["budget"]
    suggestion = suggest_budget(curves, b["target_ach"], b["cap"], b["sustain"])
    gates = gate_checks(suggestion, curves, min_deep=b["min_deep"])
    sps = float(np.mean([r["steps"] / r["secs"] for r in runs]))
    hours = implied_hours(suggestion["steps"], sps, cfg)
    result = {"suggestion": suggestion, "gates": gates, "throughput": {"steps_per_s": sps, "workers": args.workers},
              "hours": hours, "go": gates["go"], "curves": curves}
    with open(args.out, "w") as f:
        json.dump(_json_safe(result), f, indent=1, allow_nan=False)
    print(f"S = {suggestion['steps']} steps (target {'sustained' if suggestion['reached'] else 'NOT reached; the cap'})")
    print(f"G1 rare achievements {gates['rare_count']}/{len(ACHIEVEMENTS)} (need >= 5), G2 {gates['early_ach']:.2f} -> {gates['final_ach']:.2f}, deep set {gates['deep_set']}")
    print(f"throughput {sps:.0f} steps/s; full phase about {hours['full_phase_hours']:.1f} h (a floor: RND runs are slower than plain PPO), calibration itself {hours['calibration_hours']:.1f} h")
    if gates["go"]:
        print(f'GO: with the user\'s approval set in configs/crafter.yaml budget.steps: {suggestion["steps"]} and budget.deep_set: {json.dumps(gates["deep_set"])}')
    else:
        print("PHASE STOPS: " + "; ".join(gates["reasons"]) + ". Write the calibration up as a result; no tuning or RND run starts.")


if __name__ == "__main__":
    main()
