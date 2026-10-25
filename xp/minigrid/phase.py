from __future__ import annotations

import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch

from xp.errors import ProtocolError
from xp.minigrid.env import FEWSHOT_EPISODE_OFFSET, TRAIN_STRIDE, VALIDATION_LAYOUT_BASE
from xp.minigrid.ppo import Learner
from xp.protocol.manifest import write_manifest
from xp.protocol.phase import Phase
from xp.protocol.results import (
    COLUMNS, TuneResult, read_tune, read_tune_job, read_tune_job_extra, seed_job_done, write_seed_job, write_tune, write_tune_job,
)
from xp.protocol.tune import boundary_dims, grid_points, tune

AGENTS = ("kairi", "kaiser")


def build_learner(cfg: dict, agent: str, params: dict, seed: int) -> Learner:
    """Kairi = PPO (tunes its entropy coefficient); Kaiser = the same PPO plus RND (tunes the RND coefficient)."""
    learner_seed = cfg["seeds"]["agent_seed"] + int(seed)
    if agent == "kairi":
        return Learner(cfg["ppo"], params["ent_coef"], None, learner_seed, env_spec=cfg.get("env"))
    if agent == "kaiser":
        fixed_ent = params.get("ent_coef", cfg["agents"]["kaiser"]["fixed"]["ent_coef"])
        if params["rnd_coef"] == 0:  # no bonus: exactly plain PPO at the default entropy (no RND, no intrinsic head)
            return Learner(cfg["ppo"], fixed_ent, None, learner_seed, env_spec=cfg.get("env"))
        return Learner(cfg["ppo"], fixed_ent, params["rnd_coef"], learner_seed, env_spec=cfg.get("env"))
    raise ValueError(f"unknown agent {agent!r}")


def training_auc(curve: list) -> float:
    """Mean logged training success over the whole curve (NaN, before the first full window, counts as 0)."""
    succ = [0.0 if row["success_100"] != row["success_100"] else row["success_100"] for row in curve]
    return float(np.mean(succ)) if succ else 0.0


def tune_job(cfg: dict, out_dir, agent: str, params: dict, seed: int) -> float:
    """Train at the familiar size with one tune seed (lower score is better).

    Objective `validation_error` (default): 1 - success on validation layouts. Objective `auc`: 1 - training AUC.
    """
    objective = cfg.get("tuning", {}).get("objective", "validation_error")
    if objective not in ("validation_error", "auc"):
        raise ProtocolError(f"unknown tuning objective {objective!r}")
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    cached = read_tune_job(out_dir, phase, agent, params, seed)
    if cached is not None:
        return cached
    learner = build_learner(cfg, agent, params, seed)
    run = learner.train(int(phase.familiar), cfg["budget"]["steps"], target=cfg["budget"]["target_success"])
    auc = training_auc(run["curve"])
    if objective == "auc":
        score = 1.0 - auc
    else:
        score = 1.0 - learner.evaluate(int(phase.familiar), cfg["eval"]["layouts"], VALIDATION_LAYOUT_BASE)["success"]
    extra = {"auc": auc, "steps_to_target": run["steps_to_target"]}
    write_tune_job(out_dir, phase, agent, params, seed, score, extra=extra)
    print(f"[{phase.name}] tune {agent} {params} seed {seed} score {score:.3f}", flush=True)
    return score


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


def _rows(phase: Phase, agent: str, seed: int, provisional: bool, entries: list) -> pd.DataFrame:
    df = pd.DataFrame(entries, columns=["knob", "metric", "value"])
    df.insert(0, "seed", seed)
    df.insert(0, "agent", agent)
    df.insert(0, "phase", phase.name)
    df["provisional"] = bool(provisional)
    df["config_hash"] = phase.config_hash
    return df[COLUMNS]


def seed_job(cfg: dict, out_dir, agent: str, params: dict, provisional: bool, seed: int) -> None:
    """Train one frozen-hyperparameter agent at the familiar size, then zero-shot and few-shot evaluate it."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_eval_seeds([seed])
    if seed_job_done(out_dir, phase, agent, seed):
        return
    steps, n_layouts = cfg["budget"]["steps"], cfg["eval"]["layouts"]
    learner = build_learner(cfg, agent, params, seed)
    run = learner.train(int(phase.familiar), steps, target=cfg["budget"]["target_success"])
    reached = run["steps_to_target"]
    pct = int(round(100 * cfg["budget"]["target_success"]))
    entries = [
        (phase.familiar, f"steps_to_{pct}", float(steps if reached is None else reached)),  # censored at the budget
        (phase.familiar, f"reached_{pct}", float(reached is not None)),
    ]
    if cfg.get("tuning", {}).get("objective") == "auc":
        entries.append((phase.familiar, "auc", training_auc(run["curve"])))
    frozen = learner.fingerprint()
    for size in phase.sweep:  # zero-shot: weights frozen
        res = learner.evaluate(int(size), n_layouts)
        entries += [(size, "success", res["success"]), (size, "return", res["mean_return"])]
    if learner.fingerprint() != frozen:
        raise ProtocolError("a frozen agent changed during zero-shot evaluation")
    fs = cfg.get("fewshot", {"sizes": [], "steps": 0})
    for size in fs["sizes"]:  # few-shot: hyperparameters frozen, adapt online on the unfamiliar size
        clone = copy.deepcopy(learner)
        clone.train(int(size), fs["steps"], first_episode=FEWSHOT_EPISODE_OFFSET, limit=TRAIN_STRIDE)
        res = clone.evaluate(int(size), n_layouts)
        entries += [(float(size), "success_fewshot", res["success"]), (float(size), "return_fewshot", res["mean_return"])]
    write_seed_job(out_dir, phase, agent, seed, _rows(phase, agent, seed, provisional, entries))
    print(f"[{phase.name}] seed job {agent} seed {seed} done (sweep {list(phase.sweep)}, few-shot sizes {list(fs['sizes'])})", flush=True)


def _run_jobs(fn: Callable, arg_tuples: List[tuple], workers: int) -> list:
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


def _grid_size(spec: dict) -> int:
    return int(np.prod([len(v) for v in spec["grid"].values()]))


def matched_entropy(cfg: dict, kairi: TuneResult) -> dict:
    """Extra params for Kaiser's tuning: Kairi's tuned entropy when the config asks for matched entropy."""
    if cfg["agents"]["kaiser"].get("fixed", {}).get("ent_coef_from") != "kairi":
        return {}
    if kairi.provisional:
        raise ProtocolError(
            f"Kairi's tuned entropy sits on a grid edge ({kairi.boundary_dims}); widen its grid and rerun "
            "before tuning Kaiser, whose entropy is matched to it"
        )
    return {"ent_coef": kairi.best_params["ent_coef"]}


def _tune_agent(cfg: dict, phase: Phase, out_dir, agent: str, workers: int, extra: Optional[dict] = None) -> TuneResult:
    cached = read_tune(out_dir, phase, agent)
    if cached is not None:
        return cached
    spec = cfg["agents"][agent]
    extra = extra or {}

    def full(p: dict) -> dict:
        return {**p, **extra}

    jobs = [(cfg, out_dir, agent, full(p), s) for p in grid_points(spec["grid"]) for s in phase.tune_seeds]
    _run_jobs(tune_job, jobs, workers)  # each job caches its score; this call only forces completion

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


def run_phase(cfg: dict, out_dir, workers: int = 1, allow_code_change: bool = False) -> None:
    sizes = {a: _grid_size(cfg["agents"][a]) for a in AGENTS}
    if sizes["kairi"] != sizes["kaiser"]:
        raise ProtocolError(f"kairi and kaiser tuning grids must have equal size, got {sizes}")
    phase = Phase.from_config(cfg)
    import gymnasium
    import minigrid

    versions = {"torch": torch.__version__, "gymnasium": gymnasium.__version__, "minigrid": getattr(minigrid, "__version__", "unknown")}
    write_manifest(cfg, phase, out_dir, versions, strict_code=True, allow_code_change=allow_code_change)
    tuned = {"kairi": _tune_agent(cfg, phase, out_dir, "kairi", workers)}
    extra = matched_entropy(cfg, tuned["kairi"])
    tuned["kaiser"] = _tune_agent(cfg, phase, out_dir, "kaiser", workers, extra)
    jobs = [(cfg, out_dir, a, tuned[a].best_params, tuned[a].provisional, s) for a in AGENTS for s in phase.eval_seeds]
    _run_jobs(seed_job, jobs, workers)
    print(f"[{phase.name}] finished {len(jobs)} seed jobs", flush=True)
