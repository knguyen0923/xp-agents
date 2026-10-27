from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import torch

from xp.errors import ProtocolError
from xp.minigrid.env import FEWSHOT_EPISODE_OFFSET, TRAIN_STRIDE, VALIDATION_LAYOUT_BASE
from xp.minigrid.ppo import Learner
from xp.protocol.jobs import break_ties, curve_points, grid_size as _grid_size, matched_entropy, rows as _rows, run_jobs as _run_jobs, training_auc
from xp.protocol.jobs import tune_agent as _generic_tune_agent
from xp.protocol.manifest import write_manifest
from xp.protocol.phase import Phase
from xp.protocol.results import TuneResult, atomic_write, read_tune_job, seed_job_done, write_seed_job, write_tune_job

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
    extra = {"auc": auc, "steps_to_target": run["steps_to_target"], "curve": curve_points(run["curve"])}
    write_tune_job(out_dir, phase, agent, params, seed, score, extra=extra)
    print(f"[{phase.name}] tune {agent} {params} seed {seed} score {score:.3f}", flush=True)
    return score


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
    curve_path = Path(out_dir) / phase.name / "curves" / f"{agent}__seed-{seed}.json"  # not *.csv: load_results ignores it
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(curve_path, lambda tmp: tmp.write_text(json.dumps(curve_points(run["curve"]))))  # before the done-marker
    write_seed_job(out_dir, phase, agent, seed, _rows(phase, agent, seed, provisional, entries))
    print(f"[{phase.name}] seed job {agent} seed {seed} done (sweep {list(phase.sweep)}, few-shot sizes {list(fs['sizes'])})", flush=True)


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


def _tune_agent(cfg: dict, phase: Phase, out_dir, agent: str, workers: int, extra: Optional[dict] = None) -> TuneResult:
    return _generic_tune_agent(cfg, phase, out_dir, agent, workers, tune_job, extra)
