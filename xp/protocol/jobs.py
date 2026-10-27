"""Phase-agnostic job helpers shared by the trained-agent phases (MiniGrid, Showdown): process pool, tuning orchestration,
tie-breaking, matched entropy and result rows. Lifted unchanged from `xp.minigrid.phase`; the phase passes its own `tune_job`."""
from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from xp.errors import ProtocolError
from xp.protocol.phase import Phase
from xp.protocol.results import COLUMNS, TuneResult, read_tune, read_tune_job, read_tune_job_extra, write_tune
from xp.protocol.tune import boundary_dims, grid_points, tune


def run_jobs(fn: Callable, arg_tuples: List[tuple], workers: int) -> list:
    if workers <= 1 or len(arg_tuples) <= 1:
        return [fn(*args) for args in arg_tuples]
    ctx = multiprocessing.get_context("spawn")  # fork is unsafe with torch
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(fn, *args) for args in arg_tuples]
        try:
            return [f.result() for f in futures]
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)  # fail fast: do not keep training after an error
            raise


def grid_size(spec: dict) -> int:
    return int(np.prod([len(v) for v in spec["grid"].values()]))


def matched_entropy(cfg: dict, kairi: TuneResult) -> dict:
    """Extra params for Kaiser's tuning: Kairi's tuned entropy when the config asks for matched entropy."""
    source = cfg["agents"]["kaiser"].get("fixed", {}).get("ent_coef_from")
    if source is None:
        return {}
    if source != "kairi":
        raise ProtocolError(f"ent_coef_from must be 'kairi' or absent, got {source!r}")
    if kairi.provisional:
        raise ProtocolError(
            f"Kairi's tuned entropy sits on a grid edge ({kairi.boundary_dims}); widen its grid and rerun "
            "before tuning Kaiser, whose entropy is matched to it"
        )
    return {"ent_coef": kairi.best_params["ent_coef"]}


def break_ties(result: TuneResult, grid: dict, natural_limits, auc_by_point, steps_by_point=None) -> TuneResult:
    """Among grid points tied (within 1e-12) for the best tuning score, pick the highest mean training AUC, then grid order.

    `auc_by_point` holds one mean AUC per row of `result.table` (grid order). When `steps_by_point` is given
    (mean steps to target, censored at the budget), the fewest steps wins instead, then grid order.
    """
    table = result.table
    best = float(table["score"].min())
    tied = [i for i, sc in enumerate(table["score"]) if abs(float(sc) - best) <= 1e-12]
    if len(tied) <= 1:
        return result
    if steps_by_point is not None:
        pick = min(tied, key=lambda i: (float(steps_by_point[i]), i))
    else:
        pick = max(tied, key=lambda i: (float(auc_by_point[i]), -i))
    params = {k: table.loc[pick, k] for k in grid}
    params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    return TuneResult(params, result.best_score, boundary_dims(params, grid, natural_limits), table)


def rows(phase: Phase, agent: str, seed: int, provisional: bool, entries: list) -> pd.DataFrame:
    df = pd.DataFrame(entries, columns=["knob", "metric", "value"])
    df.insert(0, "seed", seed)
    df.insert(0, "agent", agent)
    df.insert(0, "phase", phase.name)
    df["provisional"] = bool(provisional)
    df["config_hash"] = phase.config_hash
    return df[COLUMNS]


def tune_agent(cfg: dict, phase: Phase, out_dir, agent: str, workers: int, tune_job: Callable, extra: Optional[dict] = None) -> TuneResult:
    extra = extra or {}
    cached = read_tune(out_dir, phase, agent)
    if cached is not None:
        for key, value in extra.items():
            if cached.best_params.get(key) != value:
                raise ProtocolError(
                    f"cached {agent} tuning has {key}={cached.best_params.get(key)!r} but Kairi's tuned value is {value!r}; "
                    "use a fresh --out"
                )
        return cached
    spec = cfg["agents"][agent]

    def full(p: dict) -> dict:
        return {**p, **extra}

    jobs = [(cfg, out_dir, agent, full(p), s) for p in grid_points(spec["grid"]) for s in phase.tune_seeds]
    run_jobs(tune_job, jobs, workers)  # each job caches its score; this call only forces completion

    def score(params: dict) -> float:
        return float(np.mean([read_tune_job(out_dir, phase, agent, full(params), s) for s in phase.tune_seeds]))

    result = tune(score, spec["grid"], spec.get("natural_limits"))

    def mean_auc(params: dict) -> float:
        vals = [(read_tune_job_extra(out_dir, phase, agent, full(params), s) or {}).get("auc", -1.0) for s in phase.tune_seeds]
        return float(np.mean(vals))

    steps_by_point = None
    if cfg.get("tuning", {}).get("objective") == "auc":

        def mean_steps(params: dict) -> float:
            budget = cfg["budget"]["steps"]
            vals = []
            for s in phase.tune_seeds:
                st = (read_tune_job_extra(out_dir, phase, agent, full(params), s) or {}).get("steps_to_target")
                vals.append(budget if st is None else st)
            return float(np.mean(vals))

        steps_by_point = [mean_steps(p) for p in grid_points(spec["grid"])]

    broken = break_ties(result, spec["grid"], spec.get("natural_limits"), [mean_auc(p) for p in grid_points(spec["grid"])],
                        steps_by_point=steps_by_point)
    if broken is not result:
        n_tied = int((abs(result.table["score"] - result.best_score) <= 1e-12).sum())
        rule = "steps-to-target" if steps_by_point is not None else "AUC"
        print(f"[{phase.name}] {agent}: tie of {n_tied} points broken by {rule} -> {broken.best_params}", flush=True)
    result = broken
    result.best_params.update(extra)
    write_tune(out_dir, phase, agent, result)
    print(f"[{phase.name}] tuned {agent}: {result.best_params} score {result.best_score:.3f} "
          f"boundary={result.boundary_dims or 'none'}", flush=True)
    return result


def training_auc(curve: list, key: str = "success_100") -> float:
    """Mean logged training success over the whole curve (NaN, before the first full window, counts as 0)."""
    succ = [0.0 if row[key] != row[key] else row[key] for row in curve]
    return float(np.mean(succ)) if succ else 0.0


def curve_points(curve: list, key: str = "success_100") -> list:
    """[[steps, success_100], ...] with NaN (before the first full window) as None, so the JSON stays strict."""
    return [[row["steps"], None if row[key] != row[key] else row[key]] for row in curve]
