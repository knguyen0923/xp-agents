from __future__ import annotations

import json
from pathlib import Path

import torch

from xp.crafter.env import CrafterVec, assert_world_pools_disjoint, heldout_env_seeds, train_env_seeds
from xp.crafter.ppo import Learner, resolve_device
from xp.crafter.stats import ACHIEVEMENTS, crafter_score, nan_to_zero
from xp.errors import ProtocolError
from xp.protocol.jobs import curve_points, grid_size, matched_entropy, rows, run_jobs, training_auc, tune_agent
from xp.protocol.manifest import write_manifest
from xp.protocol.phase import Phase
from xp.protocol.results import atomic_write, read_tune_job, seed_job_done, write_seed_job, write_tune_job

AGENTS = ("kairi", "kaiser")


def require_budget(cfg: dict) -> int:
    """The calibrated budget S (`budget.steps`) and the deep set must be set (by the user, from the calibration) before any training job."""
    budget = cfg["budget"]
    if budget.get("steps") is None:
        raise ProtocolError("budget.steps is unset: run `python -m xp.crafter.calibrate`, get the user's approval of S, then set it in the config")
    deep = budget.get("deep_set")
    if not deep:
        raise ProtocolError("budget.deep_set is unset: copy the deep achievement set printed by the calibration into the config")
    unknown = [a for a in deep if a not in ACHIEVEMENTS]
    if unknown:
        raise ProtocolError(f"budget.deep_set has unknown achievements {unknown}, e.g. {unknown[0]}")
    return int(budget["steps"])


def build_learner(cfg: dict, agent: str, params: dict, seed: int) -> Learner:
    """Kairi = PPO (tunes its entropy coefficient); Kaiser = the same PPO plus RND (tunes the RND coefficient)."""
    learner_seed = cfg["seeds"]["agent_seed"] + int(seed)
    device = resolve_device(cfg["ppo"].get("device", "auto"))
    if agent == "kairi":
        return Learner(cfg["ppo"], params["ent_coef"], None, learner_seed, device)
    if agent == "kaiser":
        fixed_ent = params.get("ent_coef", cfg["agents"]["kaiser"]["fixed"]["ent_coef"])
        if params["rnd_coef"] == 0:  # no bonus: exactly plain PPO at the default entropy (no RND, no intrinsic head)
            return Learner(cfg["ppo"], fixed_ent, None, learner_seed, device)
        return Learner(cfg["ppo"], fixed_ent, params["rnd_coef"], learner_seed, device)
    raise ValueError(f"unknown agent {agent!r}")


def make_env(cfg: dict, stage: str, seed: int) -> CrafterVec:
    """`n_envs` crafter processes for `seed`; `stage` is `tune` or `eval` (training worlds) or `heldout` (held-out evaluation worlds)."""
    n = cfg["ppo"]["n_envs"]
    seeds = heldout_env_seeds(seed, n) if stage == "heldout" else train_env_seeds(seed, n)
    return CrafterVec(seeds, area=cfg["env"]["area"])


def _train(cfg: dict, stage: str, learner: Learner, seed: int) -> dict:
    env = make_env(cfg, stage, seed)
    try:
        return learner.train(env, require_budget(cfg), log_every=cfg["ppo"].get("log_every", 10))
    finally:
        env.close()


def tune_job(cfg: dict, out_dir, agent: str, params: dict, seed: int) -> float:
    """Train on the tune worlds with one tune seed (lower score is better): score = 1 - training AUC of ach_100 / 22."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    require_budget(cfg)
    cached = read_tune_job(out_dir, phase, agent, params, seed)
    if cached is not None:
        return cached
    run = _train(cfg, "tune", build_learner(cfg, agent, params, seed), seed)
    auc = training_auc(run["curve"], "ach_100") / len(ACHIEVEMENTS)
    final = nan_to_zero(run["curve"][-1]["ach_100"])
    extra = {"auc": auc, "final": final, "curve": curve_points(run["curve"], "ach_100")}
    write_tune_job(out_dir, phase, agent, params, seed, 1.0 - auc, extra=extra)
    print(f"[{phase.name}] tune {agent} {params} seed {seed} score {1.0 - auc:.3f} final ach_100 {final:.2f}", flush=True)
    return 1.0 - auc


def seed_job(cfg: dict, out_dir, agent: str, params: dict, provisional: bool, seed: int) -> None:
    """Train one frozen-hyperparameter agent on the eval worlds, then evaluate the frozen policy on the held-out worlds."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_eval_seeds([seed])
    if seed_job_done(out_dir, phase, agent, seed):
        return
    steps = require_budget(cfg)
    learner = build_learner(cfg, agent, params, seed)
    run = _train(cfg, "eval", learner, seed)
    frozen = learner.fingerprint()
    env = make_env(cfg, "heldout", seed)
    try:
        held = learner.evaluate(env, cfg["eval"]["episodes"])
    finally:
        env.close()
    if learner.fingerprint() != frozen:
        raise ProtocolError("a frozen agent changed during held-out evaluation")
    first = {a: float(run["first_unlock"].get(a, steps)) for a in ACHIEVEMENTS}  # censored at the budget
    deep = cfg["budget"]["deep_set"]
    k = phase.familiar
    entries = [
        (k, "auc", training_auc(run["curve"], "ach_100") / len(ACHIEVEMENTS)),
        (k, "ach_final", nan_to_zero(run["curve"][-1]["ach_100"])),
        (k, "crafter_score", crafter_score(run["final_rates"])),
        (k, "deep_first_unlock", float(sum(first[a] for a in deep) / len(deep))),
        (k, "heldout_ach", float(held["ach"])),
        (k, "heldout_crafter_score", crafter_score(held["rates"])),
    ]
    entries += [(k, f"rate_{a}", float(run["final_rates"][a])) for a in ACHIEVEMENTS]
    entries += [(k, f"first_{a}", first[a]) for a in ACHIEVEMENTS]
    curve_path = Path(out_dir) / phase.name / "curves" / f"{agent}__seed-{seed}.json"  # not *.csv: load_results ignores it
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(curve_path, lambda tmp: tmp.write_text(json.dumps(curve_points(run["curve"], "ach_100"))))  # before the done-marker
    write_seed_job(out_dir, phase, agent, seed, rows(phase, agent, seed, provisional, entries))
    print(f"[{phase.name}] seed job {agent} seed {seed} done (ach_final {nan_to_zero(run['curve'][-1]['ach_100']):.2f}, heldout {held['ach']:.2f})", flush=True)


def run_phase(cfg: dict, out_dir, workers: int = 1, allow_code_change: bool = False) -> None:
    sizes = {a: grid_size(cfg["agents"][a]) for a in AGENTS}
    if sizes["kairi"] != sizes["kaiser"]:
        raise ProtocolError(f"kairi and kaiser tuning grids must have equal size, got {sizes}")
    require_budget(cfg)
    phase = Phase.from_config(cfg)
    assert_world_pools_disjoint({"tune": phase.tune_seeds, "eval": phase.eval_seeds, "heldout": phase.eval_seeds}, cfg["ppo"]["n_envs"])
    import importlib.metadata as md

    try:
        crafter_version = md.version("crafter")
    except md.PackageNotFoundError:
        crafter_version = "unknown"
    versions = {"torch": torch.__version__, "crafter": crafter_version}
    write_manifest(cfg, phase, out_dir, versions, strict_code=True, allow_code_change=allow_code_change)
    tuned = {"kairi": tune_agent(cfg, phase, out_dir, "kairi", workers, tune_job, tie_key="final")}
    extra = matched_entropy(cfg, tuned["kairi"])
    tuned["kaiser"] = tune_agent(cfg, phase, out_dir, "kaiser", workers, tune_job, extra, tie_key="final")
    jobs = [(cfg, out_dir, a, tuned[a].best_params, tuned[a].provisional, s) for a in AGENTS for s in phase.eval_seeds]
    run_jobs(seed_job, jobs, workers)
    print(f"[{phase.name}] finished {len(jobs)} seed jobs", flush=True)
