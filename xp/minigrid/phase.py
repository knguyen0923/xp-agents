from __future__ import annotations

import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List

import numpy as np
import pandas as pd
import torch

from xp.errors import ProtocolError
from xp.minigrid.env import FEWSHOT_EPISODE_OFFSET, TRAIN_STRIDE, VALIDATION_LAYOUT_BASE
from xp.minigrid.ppo import Learner
from xp.protocol.manifest import write_manifest
from xp.protocol.phase import Phase
from xp.protocol.results import (
    COLUMNS, TuneResult, read_tune, read_tune_job, seed_job_done, write_seed_job, write_tune, write_tune_job,
)
from xp.protocol.tune import grid_points, tune

AGENTS = ("kairi", "kaiser")


def build_learner(cfg: dict, agent: str, params: dict, seed: int) -> Learner:
    """Kairi = PPO (tunes its entropy coefficient); Kaiser = the same PPO plus RND (tunes the RND coefficient)."""
    learner_seed = cfg["seeds"]["agent_seed"] + int(seed)
    if agent == "kairi":
        return Learner(cfg["ppo"], params["ent_coef"], None, learner_seed)
    if agent == "kaiser":
        return Learner(cfg["ppo"], cfg["agents"]["kaiser"]["fixed"]["ent_coef"], params["rnd_coef"], learner_seed)
    raise ValueError(f"unknown agent {agent!r}")


def tune_job(cfg: dict, out_dir, agent: str, params: dict, seed: int) -> float:
    """Train at the familiar size with one tune seed; score = 1 - success on validation layouts (lower is better)."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    cached = read_tune_job(out_dir, phase, agent, params, seed)
    if cached is not None:
        return cached
    learner = build_learner(cfg, agent, params, seed)
    learner.train(int(phase.familiar), cfg["budget"]["steps"])
    score = 1.0 - learner.evaluate(int(phase.familiar), cfg["eval"]["layouts"], VALIDATION_LAYOUT_BASE)["success"]
    write_tune_job(out_dir, phase, agent, params, seed, score)
    return score


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
    run = learner.train(int(phase.familiar), steps)
    reached = run["steps_to_target"]
    entries = [
        (phase.familiar, "steps_to_80", float(steps if reached is None else reached)),  # censored at the budget
        (phase.familiar, "reached_80", float(reached is not None)),
    ]
    frozen = learner.fingerprint()
    for size in phase.sweep:  # zero-shot: weights frozen
        res = learner.evaluate(int(size), n_layouts)
        entries += [(size, "success", res["success"]), (size, "return", res["mean_return"])]
    if learner.fingerprint() != frozen:
        raise ProtocolError("a frozen agent changed during zero-shot evaluation")
    fs = cfg["fewshot"]
    for size in fs["sizes"]:  # few-shot: hyperparameters frozen, adapt online on the unfamiliar size
        clone = copy.deepcopy(learner)
        clone.train(int(size), fs["steps"], first_episode=FEWSHOT_EPISODE_OFFSET, limit=TRAIN_STRIDE)
        res = clone.evaluate(int(size), n_layouts)
        entries += [(float(size), "success_fewshot", res["success"]), (float(size), "return_fewshot", res["mean_return"])]
    write_seed_job(out_dir, phase, agent, seed, _rows(phase, agent, seed, provisional, entries))


def _run_jobs(fn: Callable, arg_tuples: List[tuple], workers: int) -> list:
    if workers <= 1 or len(arg_tuples) <= 1:
        return [fn(*args) for args in arg_tuples]
    ctx = multiprocessing.get_context("spawn")  # fork is unsafe with torch
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(fn, *args) for args in arg_tuples]
        return [f.result() for f in futures]


def _grid_size(spec: dict) -> int:
    return int(np.prod([len(v) for v in spec["grid"].values()]))


def _tune_agent(cfg: dict, phase: Phase, out_dir, agent: str, workers: int) -> TuneResult:
    cached = read_tune(out_dir, phase, agent)
    if cached is not None:
        return cached
    spec = cfg["agents"][agent]
    jobs = [(cfg, out_dir, agent, p, s) for p in grid_points(spec["grid"]) for s in phase.tune_seeds]
    _run_jobs(tune_job, jobs, workers)  # each job caches its score; this call only forces completion

    def score(params: dict) -> float:
        return float(np.mean([read_tune_job(out_dir, phase, agent, params, s) for s in phase.tune_seeds]))

    result = tune(score, spec["grid"], spec.get("natural_limits"))
    write_tune(out_dir, phase, agent, result)
    print(f"[{phase.name}] tuned {agent}: {result.best_params} score {result.best_score:.3f} "
          f"boundary={result.boundary_dims or 'none'}", flush=True)
    return result


def run_phase(cfg: dict, out_dir, workers: int = 1) -> None:
    sizes = {a: _grid_size(cfg["agents"][a]) for a in AGENTS}
    if sizes["kairi"] != sizes["kaiser"]:
        raise ProtocolError(f"kairi and kaiser tuning grids must have equal size, got {sizes}")
    phase = Phase.from_config(cfg)
    import gymnasium
    import minigrid

    versions = {"torch": torch.__version__, "gymnasium": gymnasium.__version__, "minigrid": getattr(minigrid, "__version__", "unknown")}
    write_manifest(cfg, phase, out_dir, versions)
    tuned = {a: _tune_agent(cfg, phase, out_dir, a, workers) for a in AGENTS}
    jobs = [(cfg, out_dir, a, tuned[a].best_params, tuned[a].provisional, s) for a in AGENTS for s in phase.eval_seeds]
    _run_jobs(seed_job, jobs, workers)
    print(f"[{phase.name}] finished {len(jobs)} seed jobs", flush=True)
