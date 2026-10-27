from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import torch

from xp.errors import ProtocolError
from xp.protocol.jobs import curve_points, grid_size, matched_entropy, rows, run_jobs, training_auc, tune_agent
from xp.protocol.manifest import write_manifest
from xp.protocol.phase import Phase
from xp.protocol.results import atomic_write, read_tune_job, seed_job_done, write_seed_job, write_tune_job
from xp.showdown import patch
from xp.showdown.ppo import Learner
from xp.showdown.teams import assert_disjoint, load_pool, pool_digest, validate_team

AGENTS = ("kairi", "kaiser")
REPO = Path(__file__).resolve().parent.parent.parent


def team_dir(cfg: dict, stage: str) -> Path:
    path = Path(cfg["env"][f"{stage}_teams"])
    return path if path.is_absolute() else REPO / path


def make_env(cfg: dict, stage: str, seed: int):
    """`n_envs` env processes against the configured opponent, drawing teams only from the stage's pool (`tune` or `eval`)."""
    from xp.showdown.vec_env import ShowdownVecEnv

    return ShowdownVecEnv(cfg["ppo"]["n_envs"], cfg["env"]["format"], load_pool(team_dir(cfg, stage)), cfg["env"]["opponent"], seed=seed)


def build_learner(cfg: dict, agent: str, params: dict, seed: int) -> Learner:
    """Kairi = PPO (tunes its entropy coefficient); Kaiser = the same PPO plus RND (tunes the RND coefficient)."""
    learner_seed = cfg["seeds"]["agent_seed"] + int(seed)
    if agent == "kairi":
        return Learner(cfg["ppo"], params["ent_coef"], learner_seed)
    if agent == "kaiser":
        fixed_ent = params.get("ent_coef", cfg["agents"]["kaiser"]["fixed"]["ent_coef"])
        if params["rnd_coef"] == 0:  # no bonus: exactly plain PPO at the default entropy (no RND, no intrinsic head)
            return Learner(cfg["ppo"], fixed_ent, learner_seed)
        return Learner(cfg["ppo"], fixed_ent, learner_seed, rnd_coef=params["rnd_coef"])
    raise ValueError(f"unknown agent {agent!r}")


def preflight(cfg: dict, check_server: bool = True) -> None:
    """Refuse to start unless the patch, the team pools, the validator and the server are all in order (each failure otherwise hangs)."""
    missing = patch.missing()
    if missing["species"] or missing["moves"]:
        raise ProtocolError("poke-env lacks the Champions data (a Mega battle would hang): run `python -m xp.showdown.patch`")
    pools = {stage: load_pool(team_dir(cfg, stage)) for stage in ("tune", "eval")}
    assert_disjoint(pools["tune"], pools["eval"])
    for stage, pool in pools.items():
        pinned, actual = cfg["env"].get(f"{stage}_teams_sha"), pool_digest(pool)
        if pinned != actual:
            raise ProtocolError(f"{stage} team files changed: config pins {pinned!r}, files hash to {actual!r}")
    if not check_server:
        return
    try:
        socket.create_connection(("127.0.0.1", 8000), timeout=3).close()
    except OSError:
        raise ProtocolError("no Showdown server on 127.0.0.1:8000: start it (see RESUME.md) before running")
    showdown_dir = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))
    node = os.environ.get("XP_NODE", "node")
    for pool in pools.values():
        for text in pool.values():
            validate_team(text, cfg["env"]["format"], showdown_dir, node=node)


def tune_job(cfg: dict, out_dir, agent: str, params: dict, seed: int) -> float:
    """Train on the tune team pool with one tune seed (lower score is better): score = 1 - training AUC of the win rate."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    cached = read_tune_job(out_dir, phase, agent, params, seed)
    if cached is not None:
        return cached
    learner = build_learner(cfg, agent, params, seed)
    env = make_env(cfg, "tune", seed)
    try:
        run = learner.train(env, cfg["budget"]["steps"], target=cfg["budget"]["target_win"])
    finally:
        env.close()
    auc = training_auc(run["curve"], "win_rate_100")
    extra = {"auc": auc, "steps_to_target": run["steps_to_target"], "curve": curve_points(run["curve"], "win_rate_100")}
    write_tune_job(out_dir, phase, agent, params, seed, 1.0 - auc, extra=extra)
    print(f"[{phase.name}] tune {agent} {params} seed {seed} score {1.0 - auc:.3f}", flush=True)
    return 1.0 - auc


def seed_job(cfg: dict, out_dir, agent: str, params: dict, provisional: bool, seed: int) -> None:
    """Train one frozen-hyperparameter agent on the eval team pool, then evaluate the frozen policy on fresh battles."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_eval_seeds([seed])
    if seed_job_done(out_dir, phase, agent, seed):
        return
    steps = cfg["budget"]["steps"]
    pct = int(round(100 * cfg["budget"]["target_win"]))
    learner = build_learner(cfg, agent, params, seed)
    env = make_env(cfg, "eval", seed)
    try:
        run = learner.train(env, steps, target=cfg["budget"]["target_win"])
        frozen = learner.fingerprint()
        held_out = learner.evaluate(env, cfg["eval"]["battles"])
        if learner.fingerprint() != frozen:
            raise ProtocolError("a frozen agent changed during evaluation")
    finally:
        env.close()
    reached = run["steps_to_target"]
    entries = [
        (phase.familiar, f"steps_to_{pct}", float(steps if reached is None else reached)),  # censored at the budget
        (phase.familiar, f"reached_{pct}", float(reached is not None)),
        (phase.familiar, "auc", training_auc(run["curve"], "win_rate_100")),
        (phase.familiar, "win_final", float(run["curve"][-1]["win_rate_100"])),
        (phase.familiar, "win_eval", float(held_out["win_rate"])),
    ]
    curve_path = Path(out_dir) / phase.name / "curves" / f"{agent}__seed-{seed}.json"  # not *.csv: load_results ignores it
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(curve_path, lambda tmp: tmp.write_text(json.dumps(curve_points(run["curve"], "win_rate_100"))))  # before the done-marker
    write_seed_job(out_dir, phase, agent, seed, rows(phase, agent, seed, provisional, entries))
    print(f"[{phase.name}] seed job {agent} seed {seed} done (win_eval {held_out['win_rate']:.2f})", flush=True)


def run_phase(cfg: dict, out_dir, workers: int = 1, allow_code_change: bool = False) -> None:
    sizes = {a: grid_size(cfg["agents"][a]) for a in AGENTS}
    if sizes["kairi"] != sizes["kaiser"]:
        raise ProtocolError(f"kairi and kaiser tuning grids must have equal size, got {sizes}")
    phase = Phase.from_config(cfg)
    preflight(cfg)
    import gymnasium
    import poke_env

    versions = {"torch": torch.__version__, "gymnasium": gymnasium.__version__, "poke_env": getattr(poke_env, "__version__", "unknown")}
    write_manifest(cfg, phase, out_dir, versions, strict_code=True, allow_code_change=allow_code_change)
    tuned = {"kairi": tune_agent(cfg, phase, out_dir, "kairi", workers, tune_job)}
    extra = matched_entropy(cfg, tuned["kairi"])
    tuned["kaiser"] = tune_agent(cfg, phase, out_dir, "kaiser", workers, tune_job, extra)
    jobs = [(cfg, out_dir, a, tuned[a].best_params, tuned[a].provisional, s) for a in AGENTS for s in phase.eval_seeds]
    run_jobs(seed_job, jobs, workers)
    print(f"[{phase.name}] finished {len(jobs)} seed jobs", flush=True)
