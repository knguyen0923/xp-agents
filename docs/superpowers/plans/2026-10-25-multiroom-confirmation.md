# MultiRoom RND Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the protocol-legal single-level MultiRoom confirmation (4 rooms, max room size 4) of the post-hoc finding that RND at coef 0.1 reaches 95% training success in about half the steps of plain PPO.

**Architecture:** Generalise the existing MiniGrid phase rather than fork it: a config-selected env level, a config-selected tuning objective (`auc`), and an optional Kairi-to-Kaiser entropy link. Every new config key is optional with a default that reproduces today's DoorKey behaviour, so the DoorKey config (and its config hash and cached results) is untouched. `xp.protocol` and `xp.stats` are not modified unless Task 5 finds a one-point-sweep bug.

**Tech Stack:** Python 3.9 floor, PyTorch, MiniGrid (`MultiRoomEnv`), pandas, pytest. Run tests with the venv: `source .venv/bin/activate`.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md`, section 5.4 (read it first; this plan implements it). Background: `docs/FINDINGS.md` "5M-step probe", `docs/superpowers/plans/2026-10-24-phase2-step-a.md` (shelved matched-entropy design).

## Global Constraints

- Python floor 3.9; every module starts with `from __future__ import annotations`; no `X | None` at runtime (use `Optional`).
- `xp.protocol` must never import from a phase package; `xp.minigrid` is never imported from `xp.protocol`.
- Tune and eval seed pools are disjoint (`ProtocolError`); config-hash mismatch raises; do not weaken either.
- Both agents get equal-sized tuning grids (6 values each); an optimum on a grid edge is `provisional`.
- Existing behaviour must not change: `configs/minigrid.yaml` is not edited, and the existing 136 fast tests keep passing after every task.
- Tuning objective for this phase: minimise `1 - auc`, `auc` = mean of `success_100` (NaN counted as 0) over the full logged training curve. `Learner.train` always trains the full budget, so no padding.
- Grids: Kairi `ent_coef` [0.003, 0.006, 0.01, 0.02, 0.05, 0.1]; Kaiser `rnd_coef` [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]; Kaiser `ent_coef` = Kairi's tuned value. Seeds: tune `[0, 3]`, eval `[100, 110]`. Target success 0.95, cap 5,000,000.
- Tie-break (pre-declared): lowest mean steps to 95% (censored at the budget), then grid order.
- Pushing needs the user's explicit yes. Launching the long run (Task 7) needs the user's go-ahead.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## File Structure

- Modify `xp/minigrid/env.py`: level-aware `make_env(size, spec=None)`; `VecEnv` and `evaluate_policy` take `spec`.
- Modify `xp/minigrid/ppo.py`: `Learner(..., env_spec=None)` passes it to `VecEnv` / `evaluate_policy`.
- Modify `xp/minigrid/phase.py`: `training_auc`, objective switch in `tune_job`, `break_ties(..., steps_by_point=None)`, `matched_entropy`, `_tune_agent(..., extra)`, `run_phase` ordering, target-named metrics in `seed_job`.
- Create `configs/minigrid-multiroom.yaml`.
- Modify `tests/conftest.py` (new `tiny_mr_cfg` fixture), `tests/test_minigrid_env.py`, `tests/test_minigrid_ppo.py`, `tests/test_minigrid_phase.py`, `tests/test_minigrid_config.py`, `tests/test_minigrid_report.py`.
- Docs at the end: `docs/FINDINGS.md`, `README.md`, `docs/WRITEUP.md`, `STATUS.md`, `RESUME.md`.

---

### Task 1: Level-aware environment

**Files:**
- Modify: `xp/minigrid/env.py` (`make_env`, `VecEnv.__init__`, `evaluate_policy`)
- Modify: `xp/minigrid/ppo.py` (`Learner.__init__`, `evaluate`, `train`)
- Test: `tests/test_minigrid_env.py`, `tests/test_minigrid_ppo.py`

**Interfaces:**
- Produces: `make_env(size: int, spec: Optional[dict] = None)`; `spec` is `None` or `{"level": "doorkey"}` (default) or `{"level": "multiroom", "max_room_size": int}`; for multiroom, `size` is the room count. `VecEnv(size, n, seed, first_episode=0, limit=..., spec=None)`. `evaluate_policy(net, size, n_layouts, device, base=TEST_LAYOUT_BASE, spec=None)`. `Learner(ppo, ent_coef, rnd_coef, seed, device="cpu", env_spec=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_minigrid_env.py` (check its existing imports first and add `make_env`, `VecEnv`, `ProtocolError` if missing):

```python
MR = {"level": "multiroom", "max_room_size": 4}


def test_make_env_multiroom_uses_room_count_and_room_size():
    env = make_env(3, MR).unwrapped
    assert (env.minNumRooms, env.maxNumRooms, env.maxRoomSize) == (3, 3, 4)


def test_make_env_defaults_to_doorkey_and_rejects_unknown_levels():
    assert type(make_env(5).unwrapped).__name__ == "DoorKeyEnv"
    assert type(make_env(5, {"level": "doorkey"}).unwrapped).__name__ == "DoorKeyEnv"
    with pytest.raises(ProtocolError):
        make_env(5, {"level": "nope"})


def test_vecenv_multiroom_layouts_are_deterministic_per_seed():
    a, b = VecEnv(2, 2, seed=4, spec=MR), VecEnv(2, 2, seed=4, spec=MR)
    assert (a.obs == b.obs).all()
```

Append to `tests/test_minigrid_ppo.py` (it already imports `Learner`; add a local ppo dict by reusing the fixture style of that file — if it has a `ppo_cfg`-like fixture use it, otherwise use `tiny_mg_cfg["ppo"]`):

```python
def test_learner_trains_and_evaluates_on_multiroom(tiny_mg_cfg):
    spec = {"level": "multiroom", "max_room_size": 4}
    learner = Learner(tiny_mg_cfg["ppo"], 0.01, None, 0, env_spec=spec)
    learner.train(2, 64)
    res = learner.evaluate(2, 3)
    assert 0.0 <= res["success"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_minigrid_env.py tests/test_minigrid_ppo.py -v`
Expected: the four new tests FAIL (TypeError on `make_env(3, MR)` / `env_spec`).

- [ ] **Step 3: Implement**

In `xp/minigrid/env.py` add `from typing import Optional` and `from minigrid.envs import DoorKeyEnv, MultiRoomEnv`, then replace `make_env`:

```python
def make_env(size: int, spec: Optional[dict] = None):
    """DoorKey (default; `size` is the grid size) or MultiRoom (`size` is the room count)."""
    level = (spec or {}).get("level", "doorkey")
    if level == "doorkey":
        return ImgObsWrapper(DoorKeyEnv(size=int(size)))
    if level == "multiroom":
        n = int(size)
        return ImgObsWrapper(MultiRoomEnv(minNumRooms=n, maxNumRooms=n, maxRoomSize=int(spec["max_room_size"])))
    raise ProtocolError(f"unknown env level {level!r}")


def _build_env(size: int, spec: Optional[dict]):
    # `make_env(size)` stays one-argument-callable when spec is None: the scripts in scripts/spikes/ monkeypatch it with a lambda.
    return make_env(size) if spec is None else make_env(size, spec)
```

`VecEnv.__init__(self, size, n, seed, first_episode=0, limit=FEWSHOT_EPISODE_OFFSET, spec=None)` and use `[_build_env(size, spec) for _ in range(n)]`. `evaluate_policy(net, size, n_layouts, device, base=TEST_LAYOUT_BASE, spec=None)` and `env = _build_env(size, spec)`.

In `xp/minigrid/ppo.py`: `Learner.__init__(..., seed: int, device: str = "cpu", env_spec: Optional[dict] = None)` storing `self.env_spec = env_spec`; `evaluate` returns `evaluate_policy(self.net, size, n_layouts, self.device, base, self.env_spec)`; in `train`, `env = VecEnv(size, n, self.seed, first_episode, limit, self.env_spec)`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_minigrid_env.py tests/test_minigrid_ppo.py -v` then `pytest -q`
Expected: all PASS (existing DoorKey tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add xp/minigrid/env.py xp/minigrid/ppo.py tests/test_minigrid_env.py tests/test_minigrid_ppo.py
git commit -m "feat(minigrid): level-aware env factory (DoorKey default, MultiRoom)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: AUC tuning objective and steps-to-target tie-break

**Files:**
- Modify: `xp/minigrid/phase.py` (`build_learner`, `tune_job`, `break_ties`, `_tune_agent`, new `training_auc`)
- Modify: `tests/conftest.py` (add `tiny_mr_cfg`)
- Test: `tests/test_minigrid_phase.py`

**Interfaces:**
- Consumes: Task 1's `Learner(..., env_spec=...)`.
- Produces: `training_auc(curve: list) -> float`; config keys `env` (`{level, max_room_size}`) and `tuning.objective` (`"validation_error"` default or `"auc"`); `break_ties(result, grid, natural_limits, auc_by_point, steps_by_point=None)`; `build_learner` passes `env_spec=cfg.get("env")`.

- [ ] **Step 1: Add the fixture and write failing tests**

Append to `tests/conftest.py` (after the `tiny_mg_cfg` fixture):

```python
def _tiny_mr():
    cfg = copy.deepcopy(_TINY_MG)
    cfg["knob"] = {"name": "rooms", "familiar": 2, "sweep": [2]}
    cfg["env"] = {"level": "multiroom", "max_room_size": 4}
    cfg["tuning"] = {"objective": "auc"}
    cfg["budget"] = {"steps": 256, "target_success": 0.95, "cap": 512}
    cfg["fewshot"] = {"sizes": [], "steps": 0}
    cfg["agents"] = {
        "kairi": {"grid": {"ent_coef": [0.01]}, "natural_limits": {"ent_coef": [0.0, None]}},
        "kaiser": {"fixed": {"ent_coef": 0.01, "ent_coef_from": "kairi"}, "grid": {"rnd_coef": [0.01]},
                   "natural_limits": {"rnd_coef": [0.0, None]}},
    }
    cfg["report"] = {"metric": "auc", "lower_is_better": False, "agents": ["kairi", "kaiser"],
                     "extra_metrics": {"success": False, "steps_to_95": True}}
    return cfg


@pytest.fixture
def tiny_mr_cfg():
    return _tiny_mr()
```

Append to `tests/test_minigrid_phase.py` (add `training_auc` to the import from `xp.minigrid.phase`):

```python
def test_training_auc_counts_nan_as_zero_and_averages_the_whole_curve():
    nan = float("nan")
    assert training_auc([{"success_100": nan}, {"success_100": 1.0}]) == 0.5
    assert training_auc([]) == 0.0


def test_auc_objective_scores_one_minus_auc_and_rejects_unknown_objectives(tiny_mr_cfg, tmp_path):
    phase = Phase.from_config(tiny_mr_cfg)
    score = tune_job(tiny_mr_cfg, tmp_path, "kairi", {"ent_coef": 0.01}, seed=0)
    extra = read_tune_job_extra(tmp_path, phase, "kairi", {"ent_coef": 0.01}, 0)
    assert score == pytest.approx(1.0 - extra["auc"])
    tiny_mr_cfg["tuning"]["objective"] = "bogus"
    with pytest.raises(ProtocolError):
        tune_job(tiny_mr_cfg, tmp_path / "other", "kairi", {"ent_coef": 0.01}, seed=0)


def test_break_ties_by_steps_prefers_fewest_steps_then_grid_order():
    grid = {"rnd_coef": [0.01, 0.03, 0.1]}
    result = _tied_result([0.2, 0.2, 0.2], grid)
    picked = break_ties(result, grid, None, [0.9, 0.1, 0.5], steps_by_point=[3.0, 1.0, 1.0])
    assert picked.best_params == {"rnd_coef": 0.03}  # steps 1.0 ties between points 1 and 2; grid order wins
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_minigrid_phase.py -v`
Expected: the three new tests FAIL (ImportError for `training_auc`; then TypeError for `steps_by_point`).

- [ ] **Step 3: Implement**

In `xp/minigrid/phase.py`:

```python
def training_auc(curve: list) -> float:
    """Mean logged training success over the whole curve (NaN, before the first full window, counts as 0)."""
    succ = [0.0 if row["success_100"] != row["success_100"] else row["success_100"] for row in curve]
    return float(np.mean(succ)) if succ else 0.0
```

`build_learner`: pass `env_spec=cfg.get("env")` as the last argument of all three `Learner(...)` calls.

`tune_job`: replace the score and extra computation with

```python
    objective = cfg.get("tuning", {}).get("objective", "validation_error")
    if objective not in ("validation_error", "auc"):
        raise ProtocolError(f"unknown tuning objective {objective!r}")
    ...  # (build learner and train as before)
    auc = training_auc(run["curve"])
    if objective == "auc":
        score = 1.0 - auc
    else:
        score = 1.0 - learner.evaluate(int(phase.familiar), cfg["eval"]["layouts"], VALIDATION_LAYOUT_BASE)["success"]
    extra = {"auc": auc, "steps_to_target": run["steps_to_target"]}
```

(Do the objective check before the cache read and training so a bad config fails fast.)

`break_ties`: add the parameter `steps_by_point=None`; when it is not `None`, `pick = min(tied, key=lambda i: (float(steps_by_point[i]), i))`, else the existing AUC rule. Update the docstring.

`_tune_agent`: when `cfg.get("tuning", {}).get("objective") == "auc"`, compute

```python
    def mean_steps(params: dict) -> float:
        budget = cfg["budget"]["steps"]
        vals = []
        for s in phase.tune_seeds:
            st = (read_tune_job_extra(out_dir, phase, agent, params, s) or {}).get("steps_to_target")
            vals.append(budget if st is None else st)
        return float(np.mean(vals))
```

and pass `steps_by_point=[mean_steps(p) for p in grid_points(spec["grid"])]` to `break_ties`; otherwise `None`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_minigrid_phase.py -v` then `pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add xp/minigrid/phase.py tests/conftest.py tests/test_minigrid_phase.py
git commit -m "feat(minigrid): AUC tuning objective and steps-to-target tie-break

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Matched entropy by construction

**Files:**
- Modify: `xp/minigrid/phase.py` (`build_learner`, `matched_entropy`, `_tune_agent`, `run_phase`)
- Test: `tests/test_minigrid_phase.py`

**Interfaces:**
- Consumes: Task 2's `tiny_mr_cfg` and `_tune_agent`.
- Produces: `matched_entropy(cfg: dict, kairi: TuneResult) -> dict` (returns `{}` unless `agents.kaiser.fixed.ent_coef_from == "kairi"`, then `{"ent_coef": kairi.best_params["ent_coef"]}`; raises `ProtocolError` if Kairi's result is provisional); `_tune_agent(cfg, phase, out_dir, agent, workers, extra=None)`; Kaiser's `best_params` then includes `ent_coef`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_minigrid_phase.py` (add `matched_entropy` to the import):

```python
def test_kaiser_uses_ent_coef_from_params_when_given(tiny_mg_cfg):
    kaiser = build_learner(tiny_mg_cfg, "kaiser", {"rnd_coef": 0.03, "ent_coef": 0.05}, seed=1)
    assert kaiser.ent_coef == 0.05 and kaiser.rnd_coef == 0.03


def test_matched_entropy_follows_the_config_and_refuses_provisional_kairi(tiny_mr_cfg, tiny_mg_cfg):
    ok = TuneResult({"ent_coef": 0.02}, 0.1, [], pd.DataFrame())
    assert matched_entropy(tiny_mr_cfg, ok) == {"ent_coef": 0.02}
    assert matched_entropy(tiny_mg_cfg, ok) == {}  # DoorKey config: Kaiser keeps its fixed entropy
    edge = TuneResult({"ent_coef": 0.1}, 0.1, ["ent_coef"], pd.DataFrame())
    with pytest.raises(ProtocolError):
        matched_entropy(tiny_mr_cfg, edge)


def test_run_phase_tunes_kairi_first_and_gives_kaiser_its_entropy(tiny_mr_cfg, tmp_path):
    run_phase(tiny_mr_cfg, tmp_path)
    phase = Phase.from_config(tiny_mr_cfg)
    kairi, kaiser = read_tune(tmp_path, phase, "kairi"), read_tune(tmp_path, phase, "kaiser")
    assert kaiser.best_params["ent_coef"] == kairi.best_params["ent_coef"]
    assert "rnd_coef" in kaiser.best_params
    df = load_results(tmp_path, phase.name)
    assert set(df["agent"]) == {"kairi", "kaiser"}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_minigrid_phase.py -v`
Expected: new tests FAIL (ImportError `matched_entropy`; then the `ent_coef` assertions).

- [ ] **Step 3: Implement**

`build_learner`, Kaiser branch: `fixed_ent = params.get("ent_coef", cfg["agents"]["kaiser"]["fixed"]["ent_coef"])`.

```python
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
```

`_tune_agent(cfg, phase, out_dir, agent, workers, extra=None)`: `extra = extra or {}`; add `full = lambda p: {**p, **extra}`; use `full(p)` for the job params, in `score`, `mean_auc` and `mean_steps`; after tie-breaking and before `write_tune`, `result.best_params.update(extra)`.

`run_phase`:

```python
    tuned = {"kairi": _tune_agent(cfg, phase, out_dir, "kairi", workers)}
    extra = matched_entropy(cfg, tuned["kairi"])
    tuned["kaiser"] = _tune_agent(cfg, phase, out_dir, "kaiser", workers, extra)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_minigrid_phase.py -v` then `pytest -q`
Expected: all PASS. The existing DoorKey `run_phase` tests still pass (no `ent_coef_from`, so `extra == {}` and behaviour is as before).

- [ ] **Step 5: Commit**

```bash
git add xp/minigrid/phase.py tests/test_minigrid_phase.py
git commit -m "feat(minigrid): matched entropy, Kaiser tunes with Kairi's tuned ent_coef

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Target-named metrics and AUC in seed jobs

**Files:**
- Modify: `xp/minigrid/phase.py` (`seed_job`)
- Test: `tests/test_minigrid_phase.py`

**Interfaces:**
- Consumes: `training_auc` (Task 2).
- Produces: per-seed rows with metrics `steps_to_<pct>`, `reached_<pct>` (pct = `round(100 * budget.target_success)`; 80 for DoorKey, 95 here), `auc` (only when `tuning.objective == "auc"`), plus `success` / `return` per sweep knob as before. `fewshot` may be absent or have empty `sizes`.

- [ ] **Step 1: Write the failing tests**

```python
def test_seed_job_names_metrics_after_the_target_and_adds_auc_for_the_auc_objective(tiny_mr_cfg, tmp_path):
    phase = Phase.from_config(tiny_mr_cfg)
    seed_job(tiny_mr_cfg, tmp_path, "kairi", {"ent_coef": 0.01}, False, 10)
    df = load_results(tmp_path, phase.name)
    assert {"steps_to_95", "reached_95", "auc", "success", "return"} <= set(df["metric"])
    assert not any(m.endswith("_fewshot") for m in df["metric"])  # fewshot.sizes is empty


def test_seed_job_keeps_doorkey_metric_names(tiny_mg_cfg, tmp_path):
    phase = Phase.from_config(tiny_mg_cfg)
    seed_job(tiny_mg_cfg, tmp_path, "kairi", {"ent_coef": 0.01}, False, 10)
    metrics = set(load_results(tmp_path, phase.name)["metric"])
    assert {"steps_to_80", "reached_80"} <= metrics and "auc" not in metrics
```

(Add `seed_job` to the import from `xp.minigrid.phase`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_minigrid_phase.py -v`
Expected: the first test FAILS (`steps_to_95` missing); the second passes already (it pins existing behaviour).

- [ ] **Step 3: Implement**

In `seed_job`:

```python
    pct = int(round(100 * cfg["budget"]["target_success"]))
    entries = [
        (phase.familiar, f"steps_to_{pct}", float(steps if reached is None else reached)),  # censored at the budget
        (phase.familiar, f"reached_{pct}", float(reached is not None)),
    ]
    if cfg.get("tuning", {}).get("objective") == "auc":
        entries.append((phase.familiar, "auc", training_auc(run["curve"])))
```

and `fs = cfg.get("fewshot", {"sizes": [], "steps": 0})` (the loop over `fs["sizes"]` is then empty). Update the final print to not claim few-shot.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_minigrid_phase.py tests/test_minigrid_report.py -v` then `pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add xp/minigrid/phase.py tests/test_minigrid_phase.py
git commit -m "feat(minigrid): target-named metrics and auc in seed jobs

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: One-point-sweep report and the MultiRoom config

**Files:**
- Create: `configs/minigrid-multiroom.yaml`
- Test: `tests/test_minigrid_report.py`, `tests/test_minigrid_config.py`
- Modify (only if the first test fails): `xp/report.py`, `xp/plots/curves.py`, `xp/stats/summary.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `configs/minigrid-multiroom.yaml` (valid `Phase`, phase name `minigrid` so `xp.run --phase minigrid` accepts it; results go to a separate `--out`); a verified report on a single-knob sweep (`curves.csv`, `gaps.csv`, `crossovers.csv`, `degradation.png`, `curves__success.csv`, `crossovers__steps_to_95.csv`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_minigrid_report.py`:

```python
def test_report_runs_on_a_one_point_sweep(tiny_mr_cfg, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_mr_cfg))
    run_phase(tiny_mr_cfg, tmp_path / "res")
    report_main(["--phase", "minigrid", "--config", str(cfg_path), "--out", str(tmp_path / "res"),
                 "--figs", str(tmp_path / "figs"), "--allow-provisional"])
    names = {p.name for p in (tmp_path / "figs").iterdir()}
    assert {"curves.csv", "gaps.csv", "crossovers.csv", "degradation.png",
            "curves__success.csv", "crossovers__steps_to_95.csv"} <= names
```

Append to `tests/test_minigrid_config.py`:

```python
MR_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "minigrid-multiroom.yaml"


def test_multiroom_config_matches_spec_5_4():
    cfg = load_config(MR_CONFIG)
    phase = Phase.from_config(cfg)
    assert phase.name == "minigrid" and phase.knob_name == "rooms"  # name must stay "minigrid" for xp.run dispatch
    assert phase.familiar == 4.0 and phase.sweep == (4.0,)
    assert cfg["env"] == {"level": "multiroom", "max_room_size": 4}
    assert cfg["tuning"]["objective"] == "auc"
    assert cfg["budget"]["target_success"] == 0.95 and cfg["budget"]["cap"] == 5_000_000
    assert phase.tune_seeds == (0, 1, 2) and phase.eval_seeds == tuple(range(100, 110))
    assert cfg["agents"]["kairi"]["grid"]["ent_coef"] == [0.003, 0.006, 0.01, 0.02, 0.05, 0.1]
    assert cfg["agents"]["kaiser"]["grid"]["rnd_coef"] == [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    assert cfg["agents"]["kaiser"]["fixed"]["ent_coef_from"] == "kairi"
    assert cfg["agents"]["kaiser"]["fixed"]["ent_coef"] == 0.01  # calibration default entropy


def test_multiroom_config_grids_are_equal_size_and_rnd_interior_contains_the_probe_value():
    cfg = load_config(MR_CONFIG)
    sizes = {a: int(np.prod([len(v) for v in cfg["agents"][a]["grid"].values()])) for a in ("kairi", "kaiser")}
    assert sizes["kairi"] == sizes["kaiser"] == 6
    rnd = cfg["agents"]["kaiser"]["grid"]["rnd_coef"]
    assert min(rnd) < 0.1 < max(rnd)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_minigrid_report.py tests/test_minigrid_config.py -v`
Expected: the config tests FAIL (file missing). The report test either FAILS (a one-point-sweep bug: fix in Step 3b) or passes (then Step 3b is skipped, and say so in the commit message).

- [ ] **Step 3a: Create the config**

`configs/minigrid-multiroom.yaml`:

```yaml
# Phase 2b: MultiRoom confirmation (spec 5.4). Phase name stays "minigrid" so `xp.run --phase minigrid` accepts it;
# run it with a separate --out (results/multiroom). budget.steps is a PLACEHOLDER (the cap) until calibration sets S (plan Task 7).
phase: minigrid
knob:
  name: rooms
  familiar: 4
  sweep: [4]
env:
  level: multiroom
  max_room_size: 4
tuning:
  objective: auc
seeds:
  tune: [0, 3]
  eval: [100, 110]
  chunk: 1
  agent_seed: 0
budget:
  steps: 5000000
  target_success: 0.95
  cap: 5000000
eval:
  layouts: 200
fewshot:
  sizes: []
  steps: 0
ppo:
  n_envs: 8
  n_steps: 128
  epochs: 4
  minibatches: 4
  lr: 0.00025
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
      ent_coef: 0.01          # default entropy: used only by calibration; tuning uses Kairi's tuned value
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
    success: false
    steps_to_95: true
    reached_95: false
  xlabel: MultiRoom, 4 rooms (single familiar level)
  ylabel: training AUC of success (95% CI)
  title: MultiRoom, tuned Kairi vs Kaiser
```

- [ ] **Step 3b: Fix one-point-sweep failures (only if the report test failed)**

Read the traceback; fix the smallest thing in `xp/report.py`, `xp/plots/curves.py` or `xp/stats/summary.py` (typical suspects: `crossings` on a one-row table, plotting a one-point line, `gap_table` against the familiar value). Keep multi-point behaviour identical; add a focused unit test next to the fix.

- [ ] **Step 4: Run to verify pass**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/minigrid-multiroom.yaml tests/test_minigrid_report.py tests/test_minigrid_config.py
git add -u xp   # only if Step 3b changed anything
git commit -m "feat(minigrid): MultiRoom confirmation config and one-point-sweep report test

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Dry run and review gate

**Files:** none (verification).

- [ ] **Step 1: Full suite and CI parity**

Run: `pytest -q` (expect all PASS, count above the earlier 136), then `git status --short` (expect clean).

- [ ] **Step 2: Real-code smoke test, laptop-sized**

Run a tiny real run with the committed config but shrunk, outside the repo's results:

```bash
python - <<'EOF'
import copy, tempfile
from xp.protocol.phase import load_config
from xp.minigrid.phase import run_phase
cfg = load_config("configs/minigrid-multiroom.yaml")
cfg["budget"].update(steps=20000, cap=20000)
cfg["seeds"].update(tune=[0, 1], eval=[100, 101])
for a in ("kairi", "kaiser"):
    k = next(iter(cfg["agents"][a]["grid"]))
    cfg["agents"][a]["grid"][k] = cfg["agents"][a]["grid"][k][:2]
run_phase(cfg, tempfile.mkdtemp())
EOF
```

Expected: finishes in a few minutes and prints tune lines for both agents, then "finished 4 seed jobs". Kairi's tiny grid may legitimately raise the matched-entropy `ProtocolError` if its optimum lands on the edge; if so, rerun with `grid[k] = [0.01]` for both agents to exercise the rest.

- [ ] **Step 3: Request review**

Use superpowers:requesting-code-review on the diff since commit `0959b04`; fix findings. Do not proceed to Task 7 until the user has seen the result.

---

### Task 7: Calibrate, freeze S, run (needs the user's go-ahead; about 11 to 13 hours)

**Files:**
- Modify: `configs/minigrid-multiroom.yaml` (`budget.steps` only)
- Create: `calibration-multiroom.json`, results under `results/multiroom/`

- [ ] **Step 1: Ask the user for the go-ahead and confirm the machine can stay awake** (`caffeinate -i`). Do not launch without a yes.

- [ ] **Step 2: Calibrate S (about 40 to 60 minutes)**

```bash
python -m xp.minigrid.calibrate --config configs/minigrid-multiroom.yaml --out calibration-multiroom.json --workers 3
```

The script prints the suggested S (default-entropy Kairi, tune seeds, 95% mean training success, capped at 5M) and says if the cap was hit.

- [ ] **Step 3: Freeze S**

Set `budget.steps` in `configs/minigrid-multiroom.yaml` to the printed value (comment which value and whether the cap was hit). Commit config and `calibration-multiroom.json` before any tuning run:

```bash
git add configs/minigrid-multiroom.yaml calibration-multiroom.json
git commit -m "chore: freeze MultiRoom budget S from calibration

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Before launching**

Run from a dedicated `git worktree` at the frozen commit (or leave the working tree untouched for the whole run): each stage (Kairi tuning, Kaiser tuning, seed jobs) starts a new spawn pool that re-imports `xp` from the tree, and the commit guard only runs at start. Make no commits touching `xp/` or `configs/` during the run. Checkpoint commits of `STATUS.md`/`RESUME.md` also change HEAD, so on a resume confirm `git diff <launch-commit>..HEAD -- xp configs` is empty before passing `--allow-code-change`.

- [ ] **Step 5: Launch the run in the background (resumable)**

```bash
caffeinate -i python -m xp.run --phase minigrid --config configs/minigrid-multiroom.yaml --out results/multiroom --workers 3
```

Use a background watcher; do not poll. On a Kairi grid-edge `ProtocolError`: widen the Kairi `ent_coef` grid (keeping 6 values, and widening Kaiser's grid to 6 as well if it changes size) and say so in the write-up. Widening changes the config hash, so rerunning into the same `--out` raises `ConfigMismatchError`: use a fresh `--out` (e.g. `results/multiroom-v2`) or archive `results/multiroom` first, and say so in the write-up.

Risks to budget for:
- A Kairi optimum on a grid edge aborts the run after about a third of it (after Kairi tuning).
- A Kaiser optimum at the 1.0 edge is plausible (the probe never tried above 0.1) and makes the run provisional.
- Budget time for a full rerun.

- [ ] **Step 6: Report**

```bash
python -m xp.report --phase minigrid --config configs/minigrid-multiroom.yaml --out results/multiroom --figs figs/multiroom
```

No `--allow-provisional` for the final table. If a result is provisional, widen and rerun instead.

The training curves are kept: per-seed sidecars under `results/multiroom/minigrid/curves/` (`<agent>__seed-<seed>.json`) and the `curve` entries inside the tuning job JSONs. They are the source for a Kairi-vs-Kaiser success-vs-steps figure (the figure itself is Task 8).

---

### Task 8: Write up (whatever the outcome)

**Files:** Modify `docs/FINDINGS.md`, `README.md`, `docs/WRITEUP.md`, `STATUS.md`, `RESUME.md`.

- [ ] **Step 1:** In `docs/FINDINGS.md` add a "MultiRoom confirmation" section: tuned values for both agents, tuning curves, the paired differences with 95% CIs for `auc`, `steps_to_95` (censored count, `reached_95` per agent), and held-out `success`; the calibrated S (and whether the cap was hit); the stated limitation (level and grid range chosen after the probe). State that test layouts are held-out layout seeds, not necessarily unseen layouts (as for DoorKey). Note that `success_100` uses a partial-window mean (NaN only for an empty window), which has no effect here since every logged MultiRoom window holds at least 128 episodes.
- [ ] **Step 2:** Replace the post-hoc caveat in `FINDINGS.md`, the README summary and `docs/WRITEUP.md` with the confirmed or unconfirmed result. Kairi tying or winning is reported as such, with the probe kept as history, not deleted.
- [ ] **Step 3:** Update `RESUME.md` and `STATUS.md` (commit the content first, then a STATUS-only marker commit so `STATUS_COMMIT` matches; see the Stop hook note in `RESUME.md`). Ask the user before pushing.

---

## Self-Review

- **Spec coverage (5.4):** level/knob (T1, T5); seeds and fresh-eval limitation (T5 config, T8 write-up); AUC objective (T2); grids and matched entropy (T3, T5); budget S (T7 via the existing calibrate script, which already honours `target_success` and `cap`); tie-break (T2); metrics `auc`, `steps_to_95`, `reached_95`, held-out `success` (T4); provisional handling (T3 guard for Kairi; the existing `provisional` flag for Kaiser); one-point sweep check (T5); compute and go-ahead (T7); out-of-scope items are absent.
- **Placeholders:** none; `budget.steps` in the config is an intentional, commented placeholder that Task 7 Step 3 replaces before any run.
- **Type consistency:** `make_env(size, spec)`, `env_spec`, `training_auc`, `break_ties(..., steps_by_point)`, `matched_entropy(cfg, kairi)`, `_tune_agent(..., extra)` and the metric names `steps_to_95`, `reached_95`, `auc` are used identically across tasks.
- **Known risks:** the calibration may hit the 5M cap (then the write-up says so); default-entropy Kairi's 3-seed mean may reach 95% late, so S can be near the cap and the run time near the 13 h end; the Task 6 smoke test may trip the matched-entropy guard by design.
