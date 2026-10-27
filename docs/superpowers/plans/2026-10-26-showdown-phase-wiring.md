# Showdown Phase Wiring and Kaiser Implementation Plan (Plan C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **The user approves each task before it is done** (standing preference); do not chain tasks or start the full run without a separate yes.

**Goal:** Finish the Showdown phase so that `python -m xp.run --phase showdown --config configs/showdown.yaml` tunes Kairi and Kaiser and runs the final seeds, resumably, and `xp.report` produces the tables: add the RND/Kaiser learner, the phase module, the configs, the launcher, and prove the whole pipeline on a minutes-long smoke config against the real server.

**Architecture:** The generic job machinery of the MiniGrid phase (process pool, tuning orchestration, tie-breaking, matched entropy, result rows) is lifted unchanged into `xp/protocol/jobs.py` and shared; `xp/showdown/phase.py` supplies the Showdown `tune_job` and `seed_job`, a `preflight` that refuses to start unless the data patch, team files, validator and server are in order, and `run_phase`. `Learner` gains `rnd_coef` (Kaiser = Kairi plus `VectorRND`, mirroring the MiniGrid design) and `evaluate`/`fingerprint`.

**Tech Stack:** Python 3.9, PyTorch (CPU), `poke-env==0.10.0`, multiprocessing (spawn), pytest, a local Showdown server on Node >= 22.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md` section 5.6 and its two 2026-10-26 amendments (formats/teams/seeds; the user's opponent and budget decisions). Gate result: `docs/spikes/showdown_gate.md`. Earlier plans: `2026-10-26-showdown-phase-foundations.md` and `2026-10-26-showdown-learner-and-gate.md` (both done).

## Decisions needed from the user (nothing below starts without them)

1. **APPROVED by the user 2026-10-26: steps-to-target threshold 0.30** (`budget.target_win` in `configs/showdown.yaml`). The amendment requires it to be declared before any RND result exists. 0.30 is reached by plain PPO against DamageHeuristic after roughly 25k to 50k decisions (gate curve: 0.26 at 25.6k, 0.36 at 51k); 0.80 is never reached. The window is 100 battles, so each reading has a standard error of about 0.05.
2. **APPROVED by the user 2026-10-26 (build with the lift). Task 1 changes shared code.** It moves generic helpers out of `xp/minigrid/phase.py` into a new `xp/protocol/jobs.py` and re-imports them under their old names. No behaviour change (the 198 existing tests, including 27 MiniGrid phase, calibration and report tests, are the proof), but it touches the MiniGrid phase module and adds a file to `xp/protocol/`. The alternative is copying about 150 lines into `xp/showdown/phase.py`.
3. **Parallel workers:** 2 is the recommendation (about 12 h for the phase; 1 worker is about 15 h, plus RND overhead either way). More than 2 was never measured.
4. **The full run itself** (Task 5): the user said to ask again right before launching. It is a separate approval: it takes the laptop for about 12 to 15 hours, so likely two nights. See "Before launching".

## Global Constraints

- Python floor 3.9 (no backslashes or nested same-type quotes in f-string expressions); every module starts with `from __future__ import annotations` (after any module docstring).
- Tune and eval seed pools are disjoint (`ProtocolError`); the tune stage uses tune teams only, the eval stage eval teams only; both agents get equal-sized grids and budgets; a tuned optimum on a grid edge makes the run provisional (never in a final table).
- No RND result is read before the threshold (decision 1) is approved. The smoke run's numbers are plumbing checks, not results.
- Never connect to the public Showdown ladder; the server listens on `127.0.0.1` only. Account names are short, alphanumeric and fresh per run.
- Tests must not need the server except those marked `showdown`.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Pushes need the user's yes. `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md`.

## What was verified while preparing this plan (scratch clone, then the real server)

- Lift: the 198 existing tests still pass (212 with the new ones). RND: the learner still learns the toy battle with the bonus; the predictor's error falls at least 5x on seen observations and stays at least 3x higher on novel ones; plain PPO has no intrinsic head (same parameters and random streams as before); `evaluate` leaves the fingerprint unchanged.
- Smoke run on the real server, `--workers 2`: 12 jobs (8 tuning, 4 eval) finished in 72 s; Kaiser's entropy was matched to Kairi's tuned 0.01; rerunning the same command finished in 3 s from cache; `xp.report` wrote its tables and figure. The win rates (about 0.0 to 0.1) are meaningless after 3 PPO updates.
- Real-run time estimate from the gate's measured 166 decisions/s per run at 2 workers: about 25 min per run, 56 runs, about 11.7 h; RND overhead is not measured.
- Not understood: one server message from the opponent's side (`[Invalid choice] There's nothing to choose`) appears occasionally when runs are concurrent (3 times so far across the gate probe and the smoke runs). It has not stopped a run.

## File structure

| Path | Responsibility |
|---|---|
| `xp/protocol/jobs.py` (new), `xp/minigrid/phase.py` (modified) | shared job helpers; the MiniGrid phase re-imports them |
| `xp/showdown/rnd.py` (new), `ppo.py` (modified) | `VectorRND`; `Learner(rnd_coef=...)`, `evaluate`, `fingerprint` |
| `xp/showdown/teams.py` (modified) | `pool_digest` |
| `xp/showdown/phase.py` (new) | `make_env`, `build_learner`, `preflight`, `tune_job`, `seed_job`, `run_phase` |
| `xp/run.py` (modified) | `--phase showdown` |
| `configs/showdown.yaml`, `configs/showdown-smoke.yaml` (new) | the real and the smoke config |
| `scripts/run_showdown.sh` (new) | start or reuse the server, keep awake, log, resume |
| `tests/toy_showdown.py`, `tests/test_showdown_ppo.py` (modified), `tests/test_showdown_phase.py` (new) | offline tests |
| `docs/spikes/showdown_smoke.md` (new) | the smoke result and the launch checklist |

---

### Task 1: Lift the generic job helpers

**Files:**
- Create: `xp/protocol/jobs.py`
- Modify: `xp/minigrid/phase.py`

**Interfaces:**
- Produces: `run_jobs(fn, arg_tuples, workers)`, `grid_size(spec)`, `matched_entropy(cfg, kairi)`, `break_ties(result, grid, natural_limits, auc_by_point, steps_by_point=None)`, `rows(phase, agent, seed, provisional, entries)`, `tune_agent(cfg, phase, out_dir, agent, workers, tune_job, extra=None)`, `training_auc(curve, key="success_100")`, `curve_points(curve, key="success_100")`. `xp.minigrid.phase` keeps the names `_run_jobs`, `_grid_size`, `_rows`, `_tune_agent(cfg, phase, out_dir, agent, workers, extra=None)`, `break_ties`, `matched_entropy`, `training_auc`, `curve_points`.

- [ ] **Step 1: Confirm the baseline**

Run: `pytest -q`
Expected: 198 passed, 4 deselected.

- [ ] **Step 2: Create `xp/protocol/jobs.py`**

```python
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
```

- [ ] **Step 3: Replace `xp/minigrid/phase.py`** (the moved functions are removed and imported back; `tune_job`, `seed_job`, `build_learner` and `run_phase` are unchanged)

```python
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
```

- [ ] **Step 4: Run the whole suite**

Run: `pytest -q`
Expected: 198 passed, 4 deselected (identical to Step 1). If any MiniGrid test fails, restore `xp/minigrid/phase.py` with `git checkout xp/minigrid/phase.py` and stop.

- [ ] **Step 5: Commit**

```bash
git add xp/protocol/jobs.py xp/minigrid/phase.py
git commit -m "refactor: lift phase-generic job helpers into xp/protocol/jobs.py

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: RND bonus and the Kaiser learner

**Files:**
- Create: `xp/showdown/rnd.py`
- Modify: `xp/showdown/ppo.py`, `tests/toy_showdown.py`, `tests/test_showdown_ppo.py`

**Interfaces:**
- Consumes: `xp.minigrid.rnd.RunningMeanStd` and `IntrinsicScaler` (numpy and torch only; `xp/minigrid/__init__.py` is empty, so this does not import the `minigrid` package).
- Produces: `VectorRND(obs_dim=OBS_DIM, hidden=256, out=64)` with `.normalize(obs_np, device)`, `.error(x)`, `.pred`, `.target`; `Learner(ppo, ent_coef, seed, device="cpu", rnd_coef=None)` with new `.fingerprint() -> str` and `.evaluate(env, n_battles) -> {"win_rate", "battles"}`; `ActorCritic(..., intrinsic=False).heads_int(x) -> (logits0, logits1, value, value_int or None)`. The `ppo` dict gains optional `rnd_lr` (default 0.0001); when `rnd_coef` is set, `train` requires `env.n == ppo["n_envs"]` and raises `ValueError` otherwise.

- [ ] **Step 1: Update the tests first**

Replace `tests/toy_showdown.py` (adds `close`):

```python
"""Shared toy pieces for the Showdown learner tests: PPO settings and a one-decision battle env."""
from __future__ import annotations

from collections import deque

import numpy as np

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS

PPO = {"n_steps": 16, "epochs": 4, "minibatches": 4, "lr": 0.003, "gamma": 0.99, "lam": 0.95, "clip": 0.2, "vf_coef": 0.5}
GOOD = 3  # the action that wins in the toy env


class ToyVec:
    """One-decision battles: win (+1) when slot 0 plays action GOOD, else lose (-1). Actions 50..106 are never legal."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.results: deque = deque(maxlen=100)
        self._draw()

    def _draw(self):
        self.obs = self.rng.normal(size=(self.n, OBS_DIM)).astype(np.float32)
        self.masks = np.zeros((self.n, 2, N_ACTIONS), dtype=bool)
        self.masks[:, :, :50] = True

    def step(self, actions):
        assert self.masks[np.arange(self.n), 0, actions[:, 0]].all(), "slot 0 played an illegal action"
        rewards = np.where(actions[:, 0] == GOOD, 1.0, -1.0).astype(np.float32)
        self.results.extend((rewards > 0).astype(float).tolist())
        self._draw()
        return rewards, np.ones(self.n, dtype=np.float32)

    def recent_stats(self):
        return (float(np.mean(self.results)) if self.results else float("nan")), len(self.results)

    def recent_window_full(self):
        return len(self.results) == self.results.maxlen

    def close(self):
        pass
```

Replace `tests/test_showdown_ppo.py` (the original four tests plus six for RND, evaluate and fingerprint):

```python
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.toy_showdown import PPO, ToyVec  # noqa: E402

PPO_N = {**PPO, "n_envs": 16, "rnd_lr": 0.0001}
from xp.showdown.masks import N_ACTIONS  # noqa: E402
from xp.showdown.ppo import Learner, masked  # noqa: E402


def test_masked_categorical_gives_illegal_actions_zero_probability():
    logits = torch.zeros(2, N_ACTIONS)
    mask = torch.zeros(2, N_ACTIONS, dtype=torch.bool)
    mask[:, [5, 9]] = True
    probs = masked(logits, mask).probs
    assert probs[:, [5, 9]].sum().item() == pytest.approx(2.0) and probs[~mask].sum().item() == 0.0


def test_act_only_samples_legal_actions_and_conditions_the_second_slot():
    learner = Learner(PPO, 0.01, seed=0)
    env = ToyVec(8)
    env.masks[:, 1, :] = False
    env.masks[:, 1, 4] = env.masks[:, 1, 7] = True
    env.masks[:, 0, :] = False
    env.masks[:, 0, 4] = True  # slot 0 can only switch in "4", so slot 1 must not also pick switch target 4
    for _ in range(20):
        actions, logp, value, cond = learner.act(env.obs, env.masks)
        assert (actions[:, 0] == 4).all() and (actions[:, 1] == 7).all()
        assert logp.shape == (8,) and value.shape == (8,) and cond.shape == (8, 2, N_ACTIONS)


def test_training_is_deterministic_per_seed():
    runs = []
    for _ in range(2):
        learner = Learner(PPO, 0.01, seed=7)
        learner.train(ToyVec(8, seed=1), steps=8 * 16 * 3, log_every=1)
        runs.append([p.detach().numpy().copy() for p in learner.net.parameters()])
    assert all(np.array_equal(a, b) for a, b in zip(*runs))


def test_ppo_learns_the_toy_battle_with_illegal_actions_never_played():
    learner = Learner(PPO, 0.01, seed=0)
    env = ToyVec(16)
    out = learner.train(env, steps=16 * 16 * 100, log_every=5, target=0.9)
    assert out["curve"][-1]["win_rate_100"] > 0.9
    assert out["steps_to_target"] is not None and out["steps_to_target"] <= out["steps"]
    assert out["curve"][0]["win_rate_100"] < out["curve"][-1]["win_rate_100"]


def test_plain_ppo_has_no_intrinsic_head_and_kaiser_has_one():
    assert Learner(PPO_N, 0.01, seed=0).net.v_int is None and Learner(PPO_N, 0.01, seed=0).rnd is None
    kaiser = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    assert kaiser.net.v_int is not None and kaiser.rnd is not None


def test_rnd_error_falls_on_seen_observations_and_stays_higher_on_novel_ones():
    import torch as t
    from xp.showdown.encode import OBS_DIM
    from xp.showdown.rnd import VectorRND

    rnd = VectorRND()
    opt = t.optim.Adam(rnd.pred.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    seen = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    novel = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    x_seen, x_novel = t.as_tensor(seen), t.as_tensor(novel)
    start = rnd.error(x_seen).mean().item()
    for _ in range(200):
        loss = rnd.error(x_seen).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert rnd.error(x_seen).mean().item() < 0.2 * start
    assert rnd.error(x_novel).mean().item() > 3 * rnd.error(x_seen).mean().item()
    assert not any(p.requires_grad for p in rnd.target.parameters())


def test_kaiser_still_learns_the_toy_battle_and_never_plays_illegal_actions():
    learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    out = learner.train(ToyVec(16), steps=16 * 16 * 100, log_every=5)
    assert out["curve"][-1]["win_rate_100"] > 0.9


def test_evaluate_reports_a_win_rate_and_changes_nothing():
    for rnd_coef in (None, 0.1):
        learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=rnd_coef)
        learner.train(ToyVec(16), steps=16 * 16 * 5, log_every=5)
        before = learner.fingerprint()
        res = learner.evaluate(ToyVec(16, seed=3), n_battles=40)
        assert 0.0 <= res["win_rate"] <= 1.0 and res["battles"] >= 40
        assert learner.fingerprint() == before


def test_fingerprint_changes_when_training_happens():
    learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    before = learner.fingerprint()
    learner.train(ToyVec(16), steps=16 * 16, log_every=1)
    assert learner.fingerprint() != before


def test_a_mismatched_env_size_is_a_clear_error_for_rnd():
    learner = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1)
    with pytest.raises(ValueError, match="n_envs=16"):
        learner.train(ToyVec(8), steps=8 * 16, log_every=1)
```

- [ ] **Step 2: Run them to verify the new ones fail**

Run: `pytest tests/test_showdown_ppo.py -q`
Expected: failures or errors from the missing `rnd_coef` argument and the missing `xp.showdown.rnd`.

- [ ] **Step 3: Create `xp/showdown/rnd.py`**

```python
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from xp.minigrid.rnd import IntrinsicScaler, RunningMeanStd  # numpy/torch only; shared with the MiniGrid phase
from xp.showdown.encode import OBS_DIM

__all__ = ["VectorRND", "IntrinsicScaler"]


class VectorRND(nn.Module):
    """Random-network-distillation novelty bonus on the fixed feature vector (the deep analog of UCB's confidence bonus)."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, out: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.target = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
        self.pred = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms = RunningMeanStd((obs_dim,))

    def normalize(self, obs_np: np.ndarray, device) -> torch.Tensor:
        """Update the running statistics with this batch, then standardise and clip to [-5, 5]."""
        flat = obs_np.reshape(-1, self.obs_dim).astype(np.float64)
        self.obs_rms.update(flat)
        x = ((flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)).clip(-5, 5)
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)
```

- [ ] **Step 4: Replace `xp/showdown/ppo.py`**

```python
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS, restrict_second
from xp.showdown.rnd import IntrinsicScaler, VectorRND

NEG = -1e9


class ActorCritic(nn.Module):
    """MLP on the fixed feature vector with two 107-way heads (one per active slot) and a value head."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256, intrinsic: bool = False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.ModuleList([nn.Linear(hidden, N_ACTIONS), nn.Linear(hidden, N_ACTIONS)])
        self.v = nn.Linear(hidden, 1)
        # Separate head for the non-episodic intrinsic return; built only for RND agents, so plain PPO has exactly
        # the same parameters and random streams with or without this code path.
        self.v_int = nn.Linear(hidden, 1) if intrinsic else None

    def heads(self, x: torch.Tensor):
        h = self.body(x)
        return self.pi[0](h), self.pi[1](h), self.v(h).squeeze(-1)

    def heads_int(self, x: torch.Tensor):
        """Like `heads`, plus the intrinsic value (None when the net has no intrinsic head)."""
        h = self.body(x)
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi[0](h), self.pi[1](h), self.v(h).squeeze(-1), v_int


def masked(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Categorical over the legal actions only (illegal logits pushed to -1e9, so their probability is exactly 0)."""
    return Categorical(logits=logits.masked_fill(~mask, NEG))


def gae(rewards, values, next_value, nonterm, gamma: float, lam: float):
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(rewards.shape[0])):
        nv = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm[t] - values[t]
        last = delta + gamma * lam * nonterm[t] * last
        adv[t] = last
    return adv


class Learner:
    """PPO with action masks. Kairi = `rnd_coef` None; Kaiser = the same PPO plus an RND bonus (`rnd_coef` a float).

    Slot 1's mask is conditioned on slot 0's sampled action (no duplicate switch, one Mega).
    """

    def __init__(self, ppo: dict, ent_coef: float, seed: int, device: str = "cpu", rnd_coef: Optional[float] = None):
        self.ppo, self.ent_coef, self.seed, self.rnd_coef = ppo, ent_coef, int(seed), rnd_coef
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = ActorCritic(intrinsic=rnd_coef is not None).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo["lr"], eps=1e-5)
        self.rnd = self.rnd_opt = self.scaler = None
        if rnd_coef is not None:
            self.rnd = VectorRND().to(self.device)
            self.rnd_opt = torch.optim.Adam(self.rnd.pred.parameters(), lr=ppo.get("rnd_lr", 0.0001))
            self.scaler = IntrinsicScaler(ppo["n_envs"], ppo["gamma"])
        self.steps_trained = 0

    def fingerprint(self) -> str:
        """Hash of everything an update could change; used to prove evaluation never learns."""
        h = hashlib.sha256()
        tensors = list(self.net.parameters()) + (list(self.rnd.parameters()) if self.rnd else [])
        for p in tensors:
            h.update(p.detach().cpu().numpy().tobytes())
        h.update(str(self.steps_trained).encode())
        if self.rnd is not None:
            h.update(str(self.rnd.obs_rms.count).encode())
        return h.hexdigest()

    def evaluate(self, env, n_battles: int) -> dict:
        """Win rate of the current (frozen) policy over `n_battles` finished battles on `env`. Never updates anything."""
        wins = finished = 0
        while finished < n_battles:
            actions, _, _, _ = self.act(env.obs, env.masks)
            rewards, dones = env.step(actions)
            finished += int(dones.sum())
            wins += int(((rewards > 0) & (dones == 1)).sum())
        return {"win_rate": wins / finished, "battles": finished}

    def act(self, obs: np.ndarray, masks: np.ndarray):
        """Sample one joint action per env. Returns (actions (n, 2), logp (n,), value (n,), conditional masks (n, 2, 107))."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m0 = torch.as_tensor(masks[:, 0], device=self.device)
        with torch.no_grad():
            l0, l1, value = self.net.heads(x)
            d0 = masked(l0, m0)
            a0 = d0.sample()
            m1 = np.stack([restrict_second(masks[i, 1], int(a0[i])) for i in range(len(obs))])
            m1_t = torch.as_tensor(m1, device=self.device)
            d1 = masked(l1, m1_t)
            a1 = d1.sample()
            logp = d0.log_prob(a0) + d1.log_prob(a1)
        cond = np.stack([masks[:, 0], m1], axis=1)
        return torch.stack([a0, a1], dim=1).cpu().numpy(), logp, value, cond

    def train(self, env, steps: int, log_every: int = 10, target: Optional[float] = None) -> dict:
        """Train for about `steps` agent decisions. Returns the win-rate curve and the steps to reach `target` win rate."""
        # Reseed from (seed, steps so far) so training is deterministic however Learners are interleaved.
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, device, rnd = self.ppo, self.device, self.rnd
        n, t_len = env.n, c["n_steps"]
        if rnd is not None and n != c["n_envs"]:
            raise ValueError(f"the RND scaler was built for n_envs={c['n_envs']} but the env has {n} processes")
        batch = n * t_len
        updates = max(1, steps // batch)
        obs_buf = np.zeros((t_len, n, OBS_DIM), dtype=np.float32)
        mask_buf = np.zeros((t_len, n, 2, N_ACTIONS), dtype=bool)
        act_buf = torch.zeros((t_len, n, 2), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        val_buf = torch.zeros((t_len, n), device=device)
        vint_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        curve, steps_to_target = [], None
        for u in range(updates):
            for t in range(t_len):
                obs_buf[t] = env.obs
                actions, logp, value, cond = self.act(env.obs, env.masks)
                mask_buf[t], act_buf[t], logp_buf[t], val_buf[t] = cond, torch.as_tensor(actions, device=device), logp, value
                if rnd is not None:
                    with torch.no_grad():
                        vint_buf[t] = self.net.heads_int(torch.as_tensor(env.obs, dtype=torch.float32, device=device))[3]
                rew_buf[t], done_buf[t] = env.step(actions)

            if rnd is not None:
                next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape(-1, OBS_DIM)
                x = rnd.normalize(next_seq, device)
                with torch.no_grad():
                    raw = rnd.error(x).cpu().numpy().reshape(t_len, n)
                int_rewards = self.scaler(raw)
                perm = np.random.permutation(len(next_seq))
                for s in range(0, len(perm), 256):
                    idx = perm[s:s + 256]
                    loss = rnd.error(x[idx]).mean()
                    self.rnd_opt.zero_grad()
                    loss.backward()
                    self.rnd_opt.step()

            with torch.no_grad():
                _, _, next_value, next_vint = self.net.heads_int(torch.as_tensor(env.obs, dtype=torch.float32, device=device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv_ext = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            ret, adv = adv_ext + val_buf, adv_ext
            if rnd is not None:
                # The intrinsic return is not cut at episode ends (RND paper): reaching a new state costs nothing at the end.
                adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint, torch.ones_like(nonterm), c["gamma"], c["lam"])
                ret_int = adv_int + vint_buf
                adv = adv_ext + self.rnd_coef * adv_int
            b_obs = torch.as_tensor(obs_buf.reshape(-1, OBS_DIM), device=device)
            b_mask = torch.as_tensor(mask_buf.reshape(-1, 2, N_ACTIONS), device=device)
            b_act, b_logp = act_buf.reshape(-1, 2), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            b_ret_int = ret_int.reshape(-1) if rnd is not None else None
            for _ in range(c["epochs"]):
                perm_t = torch.randperm(batch, device=device)
                for s in range(0, batch, batch // c["minibatches"]):
                    idx = perm_t[s:s + batch // c["minibatches"]]
                    l0, l1, value, value_int = self.net.heads_int(b_obs[idx])
                    d0, d1 = masked(l0, b_mask[idx, 0]), masked(l1, b_mask[idx, 1])
                    new_logp = d0.log_prob(b_act[idx, 0]) + d1.log_prob(b_act[idx, 1])
                    ratio = (new_logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                    vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    if rnd is not None:
                        vf = vf + 0.5 * (value_int - b_ret_int[idx]).pow(2).mean()
                    ent = (d0.entropy() + d1.entropy()).mean()
                    loss = pg + c["vf_coef"] * vf - self.ent_coef * ent
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()
            self.steps_trained += batch
            if (u + 1) % log_every == 0 or u == updates - 1:
                win, _ = env.recent_stats()
                done_steps = (u + 1) * batch
                curve.append({"steps": done_steps, "win_rate_100": win})
                if target is not None and steps_to_target is None and env.recent_window_full() and win >= target:
                    steps_to_target = done_steps
        return {"curve": curve, "steps_to_target": steps_to_target, "steps": updates * batch}
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_showdown_ppo.py tests/test_showdown_gate.py -q`
Expected: 12 passed (10 learner tests and the 2 gate tests, which still call `Learner(PPO, 0.01, seed=0)`).

- [ ] **Step 6: Commit**

```bash
git add xp/showdown/rnd.py xp/showdown/ppo.py tests/toy_showdown.py tests/test_showdown_ppo.py
git commit -m "feat(showdown): RND bonus and the Kaiser learner, evaluate and fingerprint

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Phase module, configs and run wiring

**Files:**
- Create: `xp/showdown/phase.py`, `configs/showdown.yaml`, `configs/showdown-smoke.yaml`, `tests/test_showdown_phase.py`
- Modify: `xp/showdown/teams.py`, `xp/run.py`

**Interfaces:**
- Consumes: Task 1 (`xp.protocol.jobs`), Task 2 (`Learner`), foundations (`ShowdownVecEnv`, `load_pool`, `validate_team`, `assert_disjoint`, `patch.missing`).
- Produces: `pool_digest(pool) -> str` (12 hex characters); in `xp.showdown.phase`: `make_env(cfg, stage, seed)` (stage `tune` or `eval`), `build_learner(cfg, agent, params, seed)`, `preflight(cfg, check_server=True)`, `tune_job(cfg, out_dir, agent, params, seed) -> float`, `seed_job(cfg, out_dir, agent, params, provisional, seed)`, `run_phase(cfg, out_dir, workers=1, allow_code_change=False)`. Result metrics per eval seed: `auc`, `steps_to_30`, `reached_30`, `win_final`, `win_eval` (30 is the rounded percent of `budget.target_win`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_phase.py`:

```python
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("poke_env")

from tests.toy_showdown import ToyVec  # noqa: E402
from xp.errors import ProtocolError  # noqa: E402
from xp.protocol.jobs import grid_size  # noqa: E402
from xp.protocol.phase import Phase, load_config  # noqa: E402
from xp.protocol.results import load_results  # noqa: E402
from xp.showdown import phase as sp  # noqa: E402
from xp.showdown.teams import load_pool, pool_digest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """The smoke config shrunk to toy-env size, with make_env replaced by the toy battle env."""
    cfg = load_config(ROOT / "configs" / "showdown-smoke.yaml")
    cfg["ppo"].update({"n_envs": 16, "n_steps": 16})
    cfg["budget"]["steps"] = 16 * 16 * 3
    cfg["eval"]["battles"] = 20
    made = []

    def fake_make_env(c, stage, seed):
        made.append((stage, seed))
        return ToyVec(c["ppo"]["n_envs"], seed)

    monkeypatch.setattr(sp, "make_env", fake_make_env)
    monkeypatch.setattr(sp, "preflight", lambda c: None)
    return cfg, tmp_path, made


def test_real_config_matches_the_user_decisions_and_the_protocol():
    cfg = load_config(ROOT / "configs" / "showdown.yaml")
    phase = Phase.from_config(cfg)
    assert cfg["env"]["opponent"] == "damage" and cfg["budget"]["steps"] == 250000
    assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 6
    assert len(phase.tune_seeds) == 3 and len(phase.eval_seeds) == 10 and not set(phase.tune_seeds) & set(phase.eval_seeds)
    tune_jobs = 2 * 6 * len(phase.tune_seeds)
    assert tune_jobs + 2 * len(phase.eval_seeds) == 56
    assert cfg["ppo"]["n_steps"] * cfg["ppo"]["n_envs"] % cfg["ppo"]["minibatches"] == 0


def test_config_team_hashes_match_the_files_in_the_repo():
    for name in ("showdown", "showdown-smoke"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        for stage in ("tune", "eval"):
            assert cfg["env"][f"{stage}_teams_sha"] == pool_digest(load_pool(sp.team_dir(cfg, stage)))


def test_preflight_rejects_unpatched_data_changed_teams_and_a_missing_server(monkeypatch):
    cfg = load_config(ROOT / "configs" / "showdown.yaml")
    sp.preflight(cfg, check_server=False)  # the repo's real state passes (needs `python -m xp.showdown.patch`)
    changed = copy.deepcopy(cfg)
    changed["env"]["tune_teams_sha"] = "000000000000"
    with pytest.raises(ProtocolError, match="team files changed"):
        sp.preflight(changed, check_server=False)
    monkeypatch.setattr(sp.patch, "missing", lambda: {"species": ["golisopodmega"], "moves": []})
    with pytest.raises(ProtocolError, match="xp.showdown.patch"):
        sp.preflight(cfg, check_server=False)
    monkeypatch.undo()

    def refuse(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(sp.socket, "create_connection", refuse)
    with pytest.raises(ProtocolError, match="no Showdown server"):
        sp.preflight(cfg)


def test_build_learner_makes_plain_ppo_for_kairi_and_for_a_zero_rnd_coefficient():
    cfg = load_config(ROOT / "configs" / "showdown-smoke.yaml")
    assert sp.build_learner(cfg, "kairi", {"ent_coef": 0.01}, 0).rnd is None
    assert sp.build_learner(cfg, "kaiser", {"rnd_coef": 0, "ent_coef": 0.02}, 0).rnd is None
    kaiser = sp.build_learner(cfg, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.02}, 0)
    assert kaiser.rnd is not None and kaiser.rnd_coef == 0.3 and kaiser.ent_coef == 0.02
    with pytest.raises(ValueError):
        sp.build_learner(cfg, "tag-team", {}, 0)


def test_tune_job_trains_on_the_tune_pool_scores_one_minus_auc_and_caches(tiny):
    cfg, out, made = tiny
    score = sp.tune_job(cfg, out, "kairi", {"ent_coef": 0.01}, 0)
    assert 0.0 <= score <= 1.0 and made == [("tune", 0)]
    assert sp.tune_job(cfg, out, "kairi", {"ent_coef": 0.01}, 0) == score and len(made) == 1  # cached: no new env
    with pytest.raises(ProtocolError):
        sp.tune_job(cfg, out, "kairi", {"ent_coef": 0.01}, 100)  # an eval seed


def test_seed_job_trains_on_the_eval_pool_writes_every_metric_and_is_idempotent(tiny):
    cfg, out, made = tiny
    sp.seed_job(cfg, out, "kaiser", {"rnd_coef": 0.1, "ent_coef": 0.01}, False, 100)
    assert made == [("eval", 100)]
    df = load_results(out, "showdown")
    assert set(df["metric"]) == {"steps_to_30", "reached_30", "auc", "win_final", "win_eval"}
    assert set(df["agent"]) == {"kaiser"} and set(df["seed"]) == {100} and not df["provisional"].any()
    assert (out / "showdown" / "curves" / "kaiser__seed-100.json").exists()
    sp.seed_job(cfg, out, "kaiser", {"rnd_coef": 0.1, "ent_coef": 0.01}, False, 100)
    assert len(made) == 1
    with pytest.raises(ProtocolError):
        sp.seed_job(cfg, out, "kairi", {"ent_coef": 0.01}, False, 0)  # a tune seed


def test_run_phase_end_to_end_on_the_toy_env(tiny):
    cfg, out, made = tiny
    sp.run_phase(cfg, out)
    df = load_results(out, "showdown")
    phase = Phase.from_config(cfg)
    for agent in ("kairi", "kaiser"):
        sub = df[(df["agent"] == agent) & (df["metric"] == "auc")]
        assert sorted(sub["seed"]) == list(phase.eval_seeds)
    stages = {s for s, _ in made}
    assert stages == {"tune", "eval"}
    tune_seeds = {seed for stage, seed in made if stage == "tune"}
    eval_seeds = {seed for stage, seed in made if stage == "eval"}
    assert tune_seeds <= set(phase.tune_seeds) and eval_seeds <= set(phase.eval_seeds)
    before = len(made)
    sp.run_phase(cfg, out)  # resumable: everything is cached
    assert len(made) == before


def test_run_phase_rejects_unequal_tuning_grids(tiny):
    cfg, out, _ = tiny
    cfg["agents"]["kaiser"]["grid"]["rnd_coef"] = [0.1]
    with pytest.raises(ProtocolError, match="equal size"):
        sp.run_phase(cfg, out)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_showdown_phase.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.phase'`.

- [ ] **Step 3: Replace `xp/showdown/teams.py`** (adds `pool_digest`; everything else unchanged)

```python
from __future__ import annotations

import hashlib
import random
import subprocess
from pathlib import Path
from typing import Dict, List

from poke_env.teambuilder import Teambuilder

from xp.errors import ProtocolError


def normalise(text: str) -> str:
    """Team identity ignores trailing spaces and blank-line differences."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def load_pool(directory) -> Dict[str, str]:
    """name (file stem) -> export text, for every *.txt in `directory`, sorted by name."""
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
        raise ProtocolError(f"no team files in {directory}")
    return {p.stem: p.read_text() for p in paths}


def pool_digest(pool: Dict[str, str]) -> str:
    """Short hash of a pool's names and normalised team texts; a config pins it so edited team files cannot reuse old caches."""
    h = hashlib.sha256()
    for name in sorted(pool):
        h.update(name.encode() + b"\0" + normalise(pool[name]).encode() + b"\0")
    return h.hexdigest()[:12]


def assert_disjoint(tune: Dict[str, str], evals: Dict[str, str]) -> None:
    """Tune and eval pools must share no team (same text up to whitespace) and no file name."""
    same_name = set(tune) & set(evals)
    if same_name:
        raise ProtocolError(f"tune and eval pools share team names {sorted(same_name)}")
    seen = {normalise(t) for t in tune.values()}
    dup = [n for n, t in evals.items() if normalise(t) in seen]
    if dup:
        raise ProtocolError(f"eval teams {dup} also appear in the tune pool")


def validate_team(text: str, format_id: str, showdown_dir, node: str = "node") -> None:
    """Run the Showdown validator on one team. `poke-env` hangs silently on a rejected team, so call this first."""
    proc = subprocess.run([node, "pokemon-showdown", "validate-team", format_id], input=text, cwd=str(showdown_dir),
                          capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or out:
        raise ProtocolError(f"team rejected for {format_id}: {out or 'exit code ' + str(proc.returncode)}")


class PoolTeambuilder(Teambuilder):
    """Draws one team per battle, only from its own pool, with a seeded RNG. `drawn` logs the names."""

    def __init__(self, teams: Dict[str, str], seed: int):
        if not teams:
            raise ProtocolError("PoolTeambuilder needs at least one team")
        self.teams = dict(teams)
        self.names: List[str] = sorted(self.teams)
        self.drawn: List[str] = []
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        name = self._rng.choice(self.names)
        self.drawn.append(name)
        return self.join_team(self.parse_showdown_team(self.teams[name]))
```

- [ ] **Step 4: Create the phase module**

Create `xp/showdown/phase.py`:

```python
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
```

- [ ] **Step 5: Create the configs**

`configs/showdown.yaml` (the grids are the MultiRoom ones; the opponent and budget are the user's decisions; `target_win` is the proposal awaiting decision 1):

```yaml
# Phase 4: Pokemon Showdown VGC doubles (spec 5.6 and its amendments). Single familiar level (DamageHeuristic), no shift knob.
# Phase name "showdown"; the knob block is the protocol's minimum for one level (value 1 = the familiar opponent).
phase: showdown
knob:
  name: opponent
  familiar: 1
  sweep: [1]
env:
  format: gen9championsvgc2026regmc
  opponent: damage              # user decision 2026-10-26 (docs/spikes/showdown_gate.md): plain PPO 0.36 after 123k decisions
  tune_teams: configs/showdown_teams/tune
  eval_teams: configs/showdown_teams/eval
  tune_teams_sha: 4ebb9de5ef49  # pins the team files: `python -c "from xp.showdown.teams import *"` pool_digest of the directory
  eval_teams_sha: bcc8a297c758
tuning:
  objective: auc
seeds:
  tune: [0, 3]
  eval: [100, 110]
  chunk: 1
  agent_seed: 0
budget:
  steps: 250000                 # user decision 2026-10-26: the cap S; the learner runs 244 updates = 249,856 decisions
  target_win: 0.30              # user decision 2026-10-26: steps-to-target threshold, declared before any RND result
eval:
  battles: 200                  # fresh battles with the frozen policy after training, teams from the eval pool
ppo:
  n_envs: 8
  n_steps: 128
  epochs: 4
  minibatches: 4
  lr: 0.0003
  gamma: 0.99
  lam: 0.95
  clip: 0.2
  vf_coef: 0.5
  rnd_lr: 0.0001
agents:
  kairi:
    grid:
      ent_coef: [0.003, 0.006, 0.01, 0.02, 0.05, 0.1]
    natural_limits:
      ent_coef: [0.0, null]
  kaiser:
    fixed:
      ent_coef: 0.01            # default entropy: used only if a rnd_coef of 0 is tuned; tuning uses Kairi's tuned value
      ent_coef_from: kairi
    grid:
      rnd_coef: [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    natural_limits:
      rnd_coef: [0.0, null]
report:
  metric: auc
  lower_is_better: false
  agents: [kairi, kaiser]
  extra_metrics:
    win_eval: false
    steps_to_30: true
    reached_30: false
  xlabel: Showdown VGC doubles vs DamageHeuristic (single familiar level)
  ylabel: training AUC of win rate (95% CI)
  title: Showdown, tuned Kairi vs Kaiser
```

`configs/showdown-smoke.yaml` (3 PPO updates, 2 grid values, 2 tune and 2 eval seeds; the grid ends are declared natural limits only so a 2-value grid is not provisional; generated from the real config, comments aside):

```yaml
# SMOKE config for the Showdown phase: same pipeline, 3 PPO updates, 2 grid values, 2 tune and 2 eval seeds. The numbers mean nothing.
# The grid ends are declared natural limits ONLY so a 2-value grid is not 'provisional' (every point of a 2-value grid is an edge).
phase: showdown
knob:
  name: opponent
  familiar: 1
  sweep:
  - 1
env:
  format: gen9championsvgc2026regmc
  opponent: damage
  tune_teams: configs/showdown_teams/tune
  eval_teams: configs/showdown_teams/eval
  tune_teams_sha: 4ebb9de5ef49
  eval_teams_sha: bcc8a297c758
tuning:
  objective: auc
seeds:
  tune:
  - 0
  - 2
  eval:
  - 100
  - 102
  chunk: 1
  agent_seed: 0
budget:
  steps: 3072
  target_win: 0.3
eval:
  battles: 20
ppo:
  n_envs: 8
  n_steps: 128
  epochs: 4
  minibatches: 4
  lr: 0.0003
  gamma: 0.99
  lam: 0.95
  clip: 0.2
  vf_coef: 0.5
  rnd_lr: 0.0001
agents:
  kairi:
    grid:
      ent_coef:
      - 0.01
      - 0.05
    natural_limits:
      ent_coef:
      - 0.01
      - 0.05
  kaiser:
    fixed:
      ent_coef: 0.01
      ent_coef_from: kairi
    grid:
      rnd_coef:
      - 0.1
      - 0.5
    natural_limits:
      rnd_coef:
      - 0.1
      - 0.5
report:
  metric: auc
  lower_is_better: false
  agents:
  - kairi
  - kaiser
  extra_metrics:
    win_eval: false
    steps_to_30: true
    reached_30: false
  xlabel: Showdown VGC doubles vs DamageHeuristic (single familiar level)
  ylabel: training AUC of win rate (95% CI)
  title: Showdown SMOKE (3 updates, not a result)
```

- [ ] **Step 6: Wire `--phase showdown` into `xp/run.py`**

```bash
python3 - <<'EOF'
p = "xp/run.py"
s = open(p).read()
pairs = [
    ('PHASES = {"bandit": "xp.bandit.phase", "minigrid": "xp.minigrid.phase"}',
     'PHASES = {"bandit": "xp.bandit.phase", "minigrid": "xp.minigrid.phase", "showdown": "xp.showdown.phase"}'),
    ('help="parallel training jobs (minigrid only)"', 'help="parallel training jobs (minigrid and showdown)"'),
    ('help="minigrid only: allow resuming a run started under a different git commit"',
     'help="minigrid and showdown: allow resuming a run started under a different git commit"'),
    ('    if args.phase == "minigrid":', '    if args.phase in ("minigrid", "showdown"):'),
]
for old, new in pairs:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w").write(s)
EOF
```

- [ ] **Step 7: Run the phase tests and the whole suite**

Run: `pytest tests/test_showdown_phase.py -q && pytest -q`
Expected: 8 passed, then the full default suite passes (212 passed, 4 deselected). The preflight test needs the data patch (`python -m xp.showdown.patch`, already applied in this `.venv`).

- [ ] **Step 8: Commit**

```bash
git add xp/showdown/phase.py xp/showdown/teams.py xp/run.py configs/showdown.yaml configs/showdown-smoke.yaml tests/test_showdown_phase.py
git commit -m "feat(showdown): phase module, configs and xp.run wiring

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Launcher and the real-server smoke run

**Files:**
- Create: `scripts/run_showdown.sh`, `docs/spikes/showdown_smoke.md`
- Modify: `CLAUDE.md` (Commands), `RESUME.md`, `STATUS.md`

**Interfaces:**
- Produces: `scripts/run_showdown.sh [config] [out_dir] [workers]` (defaults `configs/showdown.yaml results/showdown 2`); environment switches `ALLOW_CODE_CHANGE=1` and `XP_SPIKE_DIR`.

- [ ] **Step 1: Create the launcher**

`scripts/run_showdown.sh`:

```bash
#!/usr/bin/env bash
# Run, or resume, the Showdown phase: starts the local server if it is not up, keeps the Mac awake, logs to <out>/run.log.
# Usage: scripts/run_showdown.sh [config] [out_dir] [workers]      (defaults: configs/showdown.yaml results/showdown 2)
# Resumable: every tuning and eval job is cached, so rerunning the same command continues where it stopped.
# NOTE: a resume refuses to run if git HEAD moved since the run started (even a docs commit); do not commit during a run,
# or resume with ALLOW_CODE_CHANGE=1 if the change is deliberate.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/showdown.yaml}
OUT=${2:-results/showdown}
WORKERS=${3:-2}
SPIKE=${XP_SPIKE_DIR:-$HOME/xp-showdown-spike}
export PATH="$SPIKE/node-v22.23.4-darwin-arm64/bin:$PATH"
export XP_NODE="$(command -v node)" XP_SHOWDOWN_DIR="$SPIKE/pokemon-showdown"
source .venv/bin/activate
python -m xp.showdown.patch > /dev/null

STARTED=0
if ! lsof -iTCP:8000 -sTCP:LISTEN > /dev/null 2>&1; then
  (cd "$XP_SHOWDOWN_DIR" && nohup node pokemon-showdown start --no-security > "$SPIKE/server.log" 2>&1 &)
  STARTED=1
  until lsof -iTCP:8000 -sTCP:LISTEN > /dev/null 2>&1; do sleep 2; done
fi
cleanup() { if [ "$STARTED" = 1 ]; then pkill -f "pokemon-showdown start" || true; fi; }
trap cleanup EXIT

mkdir -p "$OUT"
KEEP_AWAKE=""
if command -v caffeinate > /dev/null 2>&1; then KEEP_AWAKE="caffeinate -i"; fi
EXTRA=""
if [ "${ALLOW_CODE_CHANGE:-0}" = 1 ]; then EXTRA="--allow-code-change"; fi
echo "[$(date '+%F %T')] start $CFG -> $OUT (workers $WORKERS)" | tee -a "$OUT/run.log"
$KEEP_AWAKE python -u -m xp.run --phase showdown --config "$CFG" --out "$OUT" --workers "$WORKERS" $EXTRA 2>&1 | tee -a "$OUT/run.log"
echo "[$(date '+%F %T')] finished" | tee -a "$OUT/run.log"
```

Then run `chmod +x scripts/run_showdown.sh && bash -n scripts/run_showdown.sh`.
Expected: no output (syntax ok).

- [ ] **Step 2: Run the smoke config end to end on the real server (about 75 s)**

```bash
rm -rf results/showdown-smoke
scripts/run_showdown.sh configs/showdown-smoke.yaml results/showdown-smoke 2
scripts/run_showdown.sh configs/showdown-smoke.yaml results/showdown-smoke 2   # resume: should finish in a few seconds
python -m xp.report --phase showdown --config configs/showdown-smoke.yaml --out results/showdown-smoke --figs figs/showdown-smoke
```

Expected: 12 jobs finish (8 tuning, 4 eval) ending with `finished 4 seed jobs`; the second command finishes in a few seconds; the report prints `wrote tables and figure`. The launcher starts the server only if port 8000 is free and stops it only if it started it. `results/` and `figs/` are gitignored. If a job hangs, run `pkill -f "pokemon-showdown start"`, restart and rerun (stale accounts from a hung run).

- [ ] **Step 3: Add the Commands note to `CLAUDE.md`**

```bash
python3 - <<'EOF'
p = "CLAUDE.md"
s = open(p).read()
old = "Showdown tests that need the server:"
assert s.count(old) == 1
new = ("Showdown phase: `scripts/run_showdown.sh [config] [out] [workers]` (starts the local server if needed, resumable; "
       "`configs/showdown-smoke.yaml` is a 75-second end-to-end check, `configs/showdown.yaml` is the real run of about 12 to 15 hours; "
       "a resume needs the same git commit, or `ALLOW_CODE_CHANGE=1`), then `python -m xp.report --phase showdown ...` as for MiniGrid.\n")
s = s.replace(old, new + old)
open(p, "w").write(s)
EOF
```

- [ ] **Step 4: Write `docs/spikes/showdown_smoke.md`, update the handoff, commit**

The smoke note records the 12-job run time, the report output, the resume time, and copies the "Before launching" list below. Update `RESUME.md` and `STATUS.md` (Plan C built; the full run awaits the user's approval and the threshold decision), then:

```bash
git add scripts/run_showdown.sh CLAUDE.md docs/spikes/showdown_smoke.md RESUME.md STATUS.md
git commit -m "feat(showdown): launcher script, smoke run, launch checklist

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

Then the STATUS-only marker commit. Do not push without the user's yes.

---

### Task 5: The full run (separate approval; not part of building)

Only after the user approves decision 1 and this task. The user has said they want the testing done overnight.

- [ ] **Step 1:** Go through the "Before launching" list below with the user.
- [ ] **Step 2:** `scripts/run_showdown.sh configs/showdown.yaml results/showdown 2` (in a terminal that stays open, or under `nohup`). Expected total: about 12 to 15 h; Kairi's tuning finishes first (18 jobs), then Kaiser's (18), then the 20 seed jobs.
- [ ] **Step 3:** If it is interrupted (sleep, crash, battery), rerun the same command: finished jobs are cached.
- [ ] **Step 4:** When it prints `finished 20 seed jobs`, run `python -m xp.report --phase showdown --config configs/showdown.yaml --out results/showdown --figs figs/showdown` and write up. A `provisional` result (tuned optimum on a grid edge) must be widened and rerun, not reported.

## Before launching (checklist for decision time)

- Laptop on AC power and not set to sleep on lid close (`caffeinate -i` stops idle sleep only; closing the lid sleeps a MacBook unless an external display is attached). About 12 to 15 h of heavy CPU: expect fans and a busy machine; the Python processes are the bottleneck, not the server.
- Do not commit during the run. A resume refuses to continue if git `HEAD` moved (even for a docs commit); use `ALLOW_CODE_CHANGE=1` only for a deliberate code change.
- Two workers, each with 8 env processes: about 18 Python processes plus the server. Close other heavy programs.
- The first thing to check in the morning: `tail results/showdown/run.log` and `ls results/showdown/showdown/tuning | wc -l`.
- Threshold (`target_win`) approved and committed before launch, and the working tree clean.

## Self-review

**Spec coverage (5.6 and amendments):** agents Kairi/Kaiser with identical encoding and masks: Task 2. Protocol (3 tune seeds, 10 eval seeds, tune then freeze, equal 6-value grids, matched entropy, AUC objective, tie-break by steps to target, provisional on edge): Task 3 via the shared helpers. Metrics (AUC, steps to threshold censored at S, held-out win rate): Task 3 `seed_job`. Disjoint team pools per stage and pinned team files: Task 3 `preflight`. Opponent and S: the configs. Server-backed tests excluded from the default run: unchanged. Report: `xp.report` works unchanged (checked on the smoke results). Paired bootstrap CIs: provided by `xp.report` and `xp.stats` as for MiniGrid. **Not covered:** a Kairi-versus-Kaiser conclusion (that is the run), Colab support (the server would have to be installed there), the unexplained opponent-side server message.

**Placeholder scan:** none. Every file in a "Create" or "Replace" block was executed in a scratch clone (212 default tests, 4 server tests, smoke run, resume, report).

**Type consistency:** `Learner(ppo, ent_coef, seed, device="cpu", rnd_coef=None)` is the same in `ppo.py`, `phase.build_learner`, the gate and the tests. `tune_agent(..., tune_job, extra)` in `jobs.py` matches the calls in `xp/showdown/phase.py` and the `_tune_agent` wrapper in `xp/minigrid/phase.py`. Metric names `steps_to_30` and `reached_30` in `seed_job` match the `report.extra_metrics` keys in both configs. The curve key is `win_rate_100` throughout the Showdown code and is passed explicitly to `training_auc` and `curve_points`.

**Known risks:** (1) `target_win` = 0.30 is fixed (decision 1); it is noisy (window of 100 battles). (2) Runs are single-machine, about 12 to 15 h, with RND overhead unmeasured; the estimate could be off by tens of percent. (3) A stale-account or hung-battle failure needs a server restart and a rerun (jobs are cached, so little is lost). (4) The grids are copied from MultiRoom, not tuned for this problem; if Kairi's or Kaiser's optimum lands on an edge the run is provisional and the grid must be widened and rerun (another 6 to 7 h per widened agent). (5) With three tune teams and three eval teams, win rates are noisy across seeds.
