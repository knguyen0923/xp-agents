# Crafter reopened: calibrated budget and a Colab-ready phase: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `xp/crafter/` phase (env, stats, PPO + RND learner, calibration, tune/seed jobs), its configs, and a resumable Colab notebook, so the user can calibrate S, approve it, and run the 36 tuning + 20 evaluation jobs on Colab.

**Architecture:** A new self-contained package `xp/crafter/` with its own PPO and RND (the Showdown precedent), reusing only the phase-agnostic `xp.protocol` helpers (`run_jobs`, `tune_agent`, `matched_entropy`, results store, manifest). Nothing in `xp/minigrid/` or `xp/showdown/` is touched. One small backward-compatible parameter is added to `xp.protocol.jobs.tune_agent` (`tie_key`) so the spec's tie-break (highest mean final `ach_100`) can be honoured.

**Tech Stack:** Python 3.9, PyTorch, `crafter` 1.8.3, NumPy, pytest, Google Colab (GPU or CPU runtime).

**Spec:** `docs/superpowers/specs/2026-11-02-crafter-reopen-design.md` (amends `docs/superpowers/specs/2026-10-22-xp-design.md` section 5.5). Read both. The closed spike script `scripts/spikes/crafter_spike.py` is the source of the env worker, the CNN and `EpisodeStats`; it stays untouched.

## Global Constraints

- Every module starts with `from __future__ import annotations`; Python floor 3.9 (no `X | Y` types, no `match`). Ruff config: line length 200, rules `E9,F`.
- `xp/protocol/` never imports `xp.crafter`; `xp/crafter/` never imports `xp.minigrid` or `xp.showdown` (copy the few lines needed; duplication is deliberate, spec Change 2).
- With nothing Crafter-specific in a config, existing behaviour is unchanged: config hashes must stay `minigrid` b41211b3b5c6, `minigrid-multiroom` 39ce53712b8a, `minigrid-a` 51260035781b, `minigrid-entfair` e41af4a9d320, `showdown` fdac28308815, `showdown-control` 028d33113f41, `showdown-matchup` 7ca893d2d935, `showdown-tempo-base` 2083c9b9737b, `showdown-tempo` 435175fb4ec3, `bandit` 50ea69897efe. Do not edit any existing config.
- Area 32, 64x64x3 observations, 17 actions, 22 achievements. World seed per episode is `hash((env_seed, episode)) % (2**31 - 1)` (verified against crafter 1.8.3 `env.py` line 74).
- Protocol (spec 5.5, unchanged): 3 tune seeds `[0, 3)`, 10 eval seeds `[100, 110)`, disjoint pools guarded with `ProtocolError`; Kairi tuned first, Kaiser's `ent_coef` = Kairi's tuned value (`ent_coef_from: kairi`); equal 6-value grids, Kairi `ent_coef` {0.003, 0.006, 0.01, 0.02, 0.05, 0.1}, Kaiser `rnd_coef` {0.01, 0.03, 0.1, 0.3, 0.6, 1.0}; a grid-edge optimum makes the run provisional.
- Objective: training AUC of `ach_100` over the logged curve in [0, S], divided by 22; tuner minimises `1 - auc`; NaN before 100 episodes counts as 0; tie-break highest mean final `ach_100`, then grid order. Logging grid 10,240 env steps (10 updates of 8 envs x 128 steps).
- Budget (config `budget`): `cap: 3000000`, `target_ach: 5`, `sustain: 3`, `min_deep: 3`; `budget.steps` is unset until the user sets it from the calibration; **no tuning or seed job may start while `budget.steps` is unset** (guard in code). Calibration: default-entropy Kairi (`ent_coef` 0.01) on the 3 tune seeds for the whole cap, no early stopping; S = first logged step from which the 3-seed mean `ach_100` is >= 5 at that step and the next 2 logged steps; G1 clause 2 (>= 5 of 22 achievements with mean rate < 5% at S), G2 (mean `ach_100` at S exceeds the mean over log points with step <= 0.1 S), deep set = achievements with mean rate in [0.05, 0.95] at S, must have >= 3; any failure stops the phase.
- The user approves S and the printed cost before any tuning run; runs, pushes and merges need the user's yes each time (project memory: approve each step). Full runs go to Colab, never the laptop. Do not run any training while the Showdown run 1 is going unless the task says it is cheap (unit tests with fake envs are fine).
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. The Stop hook needs `STATUS.md`'s `STATUS_COMMIT` to equal the commit whose diff to HEAD touches only `STATUS.md` (commit everything else first, then a STATUS-only commit).
- Run the fast suite with `pytest -q` from the work tree (the `showdown` and `crafter` markers are excluded by default after Task 1).

## Working location

Showdown run 1 is running from the main work tree (`/Users/knguyen/VSC/xp-agents`, branch `showdown-tempo`) and its env workers import `xp` from disk. **Do not switch branches or edit files in that tree while it runs.** Work in a git worktree instead (`.worktrees/` is git-ignored):

```bash
cd /Users/knguyen/VSC/xp-agents
git worktree add .worktrees/crafter -b crafter-reopen showdown-tempo
cd .worktrees/crafter
```

The branch starts from `showdown-tempo` because the spec, CLAUDE.md edits and this plan live there; merge order is the user's call. Use the main venv by absolute path and force the worktree's `xp`:

```bash
export PY=/Users/knguyen/VSC/xp-agents/.venv/bin/python
PYTHONPATH=$PWD $PY -c "import xp, crafter, torch; print(xp.__file__)"   # must print .../.worktrees/crafter/xp/__init__.py
```

In every command below `pytest` means `PYTHONPATH=$PWD $PY -m pytest` and `python` means `PYTHONPATH=$PWD $PY`, run from `.worktrees/crafter`.

## File Structure

| File | Responsibility |
|---|---|
| `xp/crafter/__init__.py` (new) | Empty package marker. |
| `xp/crafter/stats.py` (new) | `ACHIEVEMENTS`, `EpisodeStats`, `crafter_score`, `nan_to_zero`. Pure Python/NumPy. |
| `xp/crafter/env.py` (new) | `world_seed`, env-seed pools, `assert_world_pools_disjoint`, `CrafterVec` (subprocess vector env). |
| `xp/crafter/rnd.py` (new) | `RunningMeanStd`, `IntrinsicScaler`, image `RND` (copied from MiniGrid, adapted to 64x64x3). |
| `xp/crafter/ppo.py` (new) | `ActorCritic`, `gae`, `Learner` (PPO, plus RND when `rnd_coef` is not None), `resolve_device`. |
| `xp/crafter/phase.py` (new) | `build_learner`, `make_env`, `require_budget`, `tune_job`, `seed_job`, `run_phase`. |
| `xp/crafter/calibrate.py` (new) | `suggest_budget`, `gate_checks`, `implied_hours`, `calibration_run`, CLI. |
| `xp/protocol/jobs.py` | `tune_agent(..., tie_key="auc")`: backward-compatible parameter. |
| `xp/run.py` | `PHASES` gains `crafter`. |
| `requirements-crafter.txt`, `pyproject.toml` (new / edit) | Optional dependency set; `crafter` pytest marker. |
| `configs/crafter.yaml`, `configs/crafter-smoke.yaml` (new) | The phase and a tiny end-to-end check. |
| `notebooks/run_crafter.ipynb` (new) | Colab: benchmark, smoke, calibrate, run, status, report. |
| `tests/test_crafter_*.py` (new) | Unit tests; real `crafter.Env` only in tests marked `crafter`. |

---

### Task 1: Package scaffold, markers and `stats.py`

**Files:**
- Create: `xp/crafter/__init__.py`, `xp/crafter/stats.py`, `requirements-crafter.txt`, `tests/test_crafter_stats.py`
- Modify: `pyproject.toml` (extras, marker, addopts)

**Interfaces:**
- Produces (used by every later task):
  - `ACHIEVEMENTS: tuple` of the 22 names, sorted alphabetically (the order of `crafter.constants.achievements`).
  - `EpisodeStats(window: int = 100)` with `.add(unlocked: Iterable[str]) -> None`, `.ach() -> float` (**NaN until the window holds `window` episodes**), `.rates(names=ACHIEVEMENTS) -> dict`, `.total: int`, `.full() -> bool`.
  - `nan_to_zero(x: float) -> float`.
  - `crafter_score(rates: dict) -> float`: official Crafter score, `exp(mean(log(1 + 100 r))) - 1` over the rates (percent).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crafter_stats.py
from __future__ import annotations

import math

import pytest

from xp.crafter.stats import ACHIEVEMENTS, EpisodeStats, crafter_score, nan_to_zero


def test_achievement_list_has_22_unique_names():
    assert len(ACHIEVEMENTS) == 22 and len(set(ACHIEVEMENTS)) == 22
    assert list(ACHIEVEMENTS) == sorted(ACHIEVEMENTS)


def test_episode_stats_window_rates_and_nan_before_full():
    st = EpisodeStats(window=2)
    assert math.isnan(st.ach()) and not st.full()
    st.add(["collect_wood"])
    assert math.isnan(st.ach())  # one episode: window not full yet
    st.add(["collect_wood", "eat_cow", "wake_up"])
    assert st.full() and st.ach() == 2.0
    st.add(["eat_cow"])  # the first episode falls out of the window
    assert st.ach() == 2.0 and st.total == 3
    r = st.rates(["collect_wood", "eat_cow", "wake_up", "place_table"])
    assert r == {"collect_wood": 0.5, "eat_cow": 1.0, "wake_up": 0.5, "place_table": 0.0}


def test_rates_before_any_episode_are_zero():
    assert set(EpisodeStats(5).rates().values()) == {0.0}


def test_nan_to_zero():
    assert nan_to_zero(float("nan")) == 0.0 and nan_to_zero(3.5) == 3.5


def test_crafter_score_is_zero_for_no_unlocks_and_100_for_all():
    assert crafter_score({a: 0.0 for a in ACHIEVEMENTS}) == 0.0
    assert crafter_score({a: 1.0 for a in ACHIEVEMENTS}) == pytest.approx(100.0)
    half = crafter_score({a: 0.5 for a in ACHIEVEMENTS})
    assert half == pytest.approx(50.0)  # geometric mean of 1 + 50 is 51, minus 1


@pytest.mark.crafter
def test_achievement_names_match_the_installed_crafter():
    from crafter import constants

    assert tuple(constants.achievements) == ACHIEVEMENTS
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_crafter_stats.py -v`
Expected: FAIL (`ModuleNotFoundError: xp.crafter`).

- [ ] **Step 3: Implement**

`xp/crafter/__init__.py`: empty file.

```python
# xp/crafter/stats.py
from __future__ import annotations

import collections
from typing import Dict, Iterable

import numpy as np

# The 22 Crafter achievements, in crafter.constants.achievements order (checked by a test marked `crafter`).
ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron", "collect_sapling", "collect_stone", "collect_wood",
    "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe",
    "make_stone_sword", "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant", "place_stone", "place_table", "wake_up",
)


def nan_to_zero(x: float) -> float:
    return 0.0 if x != x else float(x)


class EpisodeStats:
    """Distinct achievements per finished episode over a sliding window of the last `window` episodes.

    `ach()` is NaN until the window is full (counted as 0 in every AUC and budget rule, as in MiniGrid's `success_100`).
    """

    def __init__(self, window: int = 100):
        self.window = int(window)
        self.eps = collections.deque(maxlen=self.window)
        self.total = 0

    def add(self, unlocked: Iterable[str]) -> None:
        self.eps.append(frozenset(unlocked))
        self.total += 1

    def full(self) -> bool:
        return len(self.eps) >= self.window

    def ach(self) -> float:
        return float(np.mean([len(e) for e in self.eps])) if self.full() else float("nan")

    def rates(self, names: Iterable[str] = ACHIEVEMENTS) -> Dict[str, float]:
        n = max(1, len(self.eps))
        return {a: sum(a in e for e in self.eps) / n for a in names}


def crafter_score(rates: Dict[str, float]) -> float:
    """Official Crafter score: geometric mean of (1 + success rate in percent) over the achievements, minus 1."""
    vals = np.array([100.0 * float(r) for r in rates.values()])
    return float(np.exp(np.mean(np.log(1.0 + vals))) - 1.0)
```

`requirements-crafter.txt`:
```
-r requirements.txt
torch>=2.0
crafter
```

`pyproject.toml`: in `[project.optional-dependencies]` add `crafter = ["torch>=2.0", "crafter"]`; replace the pytest block with:
```toml
addopts = "-q -m 'not showdown and not crafter'"
markers = [
    "showdown: needs a running local Pokemon Showdown server (excluded from the default run)",
    "crafter: steps a real crafter.Env (needs the crafter package; excluded from the default run)",
]
```

- [ ] **Step 4: Run to verify pass, and the whole suite**

Run: `pytest tests/test_crafter_stats.py -v && pytest -q`
Expected: the stats tests pass (the `crafter`-marked one is deselected); full suite green.
Run (optional, crafter is installed here): `pytest -m crafter tests/test_crafter_stats.py -v` Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xp/crafter/__init__.py xp/crafter/stats.py tests/test_crafter_stats.py requirements-crafter.txt pyproject.toml
git commit -m "feat(crafter): package scaffold, EpisodeStats, official score, crafter marker

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `env.py`: world seeds, pool guard, vector env

**Files:**
- Create: `xp/crafter/env.py`, `tests/test_crafter_env.py`

**Interfaces:**
- Consumes: `xp.errors.ProtocolError`.
- Produces:
  - `world_seed(env_seed: int, episode: int) -> int`.
  - `train_env_seeds(seed: int, n: int) -> list` (`seed * 1000 + i`), `heldout_env_seeds(seed: int, n: int) -> list` (`HELDOUT_BASE + seed * 1000 + i`), `HELDOUT_BASE = 1_000_000`; both raise `ProtocolError` for `n > 1000` or `seed < 0`.
  - `assert_world_pools_disjoint(pools: dict, n_envs: int, episodes: int = 2000, max_overlap: float = 0.001) -> None`: `pools` maps a pool name to its list of seeds; checks every pair (train pools via `train_env_seeds`, a pool named `"heldout"` via `heldout_env_seeds`); raises `ProtocolError` if two pools share an env seed or if more than `max_overlap` of one pool's world seeds appear in another.
  - `CrafterVec(env_seeds: list, area: int = 32)` with `.n`, `.obs` (`uint8`, `(n, 64, 64, 3)`), `.step(actions) -> (obs, rewards float32 (n,), dones bool (n,), episodes list)` where each `episodes[i]` is `None` or the list of achievement names unlocked in the episode that just finished (the env auto-resets), `.close()`.

Why a fractional overlap and not exact world disjointness: a world seed is a hash mod 2^31 - 1, so two different `(env_seed, episode)` pairs can collide by chance (a handful in 10^5 worlds). That is harmless; what the guard must catch is a shared env seed or a systematic overlap.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crafter_env.py
from __future__ import annotations

import numpy as np
import pytest

from xp.crafter.env import HELDOUT_BASE, CrafterVec, assert_world_pools_disjoint, heldout_env_seeds, train_env_seeds, world_seed
from xp.errors import ProtocolError


def test_world_seed_is_deterministic_and_in_range():
    assert world_seed(5, 3) == world_seed(5, 3)
    assert world_seed(5, 3) != world_seed(5, 4)
    assert 0 <= world_seed(123456, 99) < 2**31 - 1


def test_env_seed_pools():
    assert train_env_seeds(3, 4) == [3000, 3001, 3002, 3003]
    assert heldout_env_seeds(3, 2) == [HELDOUT_BASE + 3000, HELDOUT_BASE + 3001]
    with pytest.raises(ProtocolError):
        train_env_seeds(0, 1001)
    with pytest.raises(ProtocolError):
        train_env_seeds(-1, 4)


def test_real_pools_pass_the_guard():
    assert_world_pools_disjoint({"tune": [0, 1, 2], "eval": list(range(100, 110)), "heldout": list(range(100, 110))}, n_envs=8, episodes=500)


def test_guard_rejects_a_shared_seed():
    with pytest.raises(ProtocolError, match="share"):
        assert_world_pools_disjoint({"tune": [0, 1], "eval": [1, 2]}, n_envs=4, episodes=10)


def test_guard_rejects_systematic_world_overlap(monkeypatch):
    import xp.crafter.env as env

    monkeypatch.setattr(env, "world_seed", lambda s, e: e)  # every env sees the same worlds: pools overlap completely
    with pytest.raises(ProtocolError, match="overlap"):
        env.assert_world_pools_disjoint({"tune": [0], "eval": [1]}, n_envs=2, episodes=50)


@pytest.mark.crafter
def test_real_env_steps_and_finishes_an_episode():
    vec = CrafterVec(train_env_seeds(0, 2), area=32)
    try:
        assert vec.obs.shape == (2, 64, 64, 3) and vec.obs.dtype == np.uint8
        finished = []
        for _ in range(12000):  # an episode is at most 10,000 steps
            obs, rew, done, eps = vec.step([0, 0])  # noop: the player eventually dies or times out
            assert obs.shape == (2, 64, 64, 3) and rew.shape == (2,) and done.shape == (2,)
            finished += [e for e in eps if e is not None]
            if finished:
                break
        assert finished and all(isinstance(a, str) for a in finished[0])
    finally:
        vec.close()


@pytest.mark.crafter
def test_same_env_seed_gives_the_same_first_world():
    a, b = CrafterVec([7], area=32), CrafterVec([7], area=32)
    try:
        assert np.array_equal(a.obs, b.obs)
    finally:
        a.close()
        b.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_crafter_env.py -v`
Expected: FAIL (`ModuleNotFoundError: xp.crafter.env`).

- [ ] **Step 3: Implement**

```python
# xp/crafter/env.py
from __future__ import annotations

import multiprocessing as mp
from itertools import combinations
from typing import Dict, Iterable, List

import numpy as np

from xp.errors import ProtocolError

HELDOUT_BASE = 1_000_000  # held-out evaluation env seeds start here; every training env seed (seed * 1000 + i) is far below it
_MAX_ENVS = 1000


def world_seed(env_seed: int, episode: int) -> int:
    """The world seed `crafter.Env` uses for an episode (crafter 1.8.3 `env.py` line 74)."""
    return hash((env_seed, episode)) % (2**31 - 1)


def _check(seed: int, n: int) -> None:
    if seed < 0 or not 0 <= n <= _MAX_ENVS:
        raise ProtocolError(f"seed {seed} or env count {n} is outside the allowed range (seed >= 0, n <= {_MAX_ENVS})")


def train_env_seeds(seed: int, n: int) -> List[int]:
    _check(seed, n)
    return [int(seed) * 1000 + i for i in range(n)]


def heldout_env_seeds(seed: int, n: int) -> List[int]:
    _check(seed, n)
    return [HELDOUT_BASE + int(seed) * 1000 + i for i in range(n)]


def _pool_env_seeds(name: str, seeds: Iterable[int], n_envs: int) -> List[int]:
    fn = heldout_env_seeds if name == "heldout" else train_env_seeds
    return [e for s in seeds for e in fn(int(s), n_envs)]


def assert_world_pools_disjoint(pools: Dict[str, Iterable[int]], n_envs: int, episodes: int = 2000, max_overlap: float = 0.001) -> None:
    """No two pools may share an env seed, and their world seeds may overlap only by hash chance (<= `max_overlap` of a pool)."""
    env_seeds = {name: _pool_env_seeds(name, seeds, n_envs) for name, seeds in pools.items()}
    for a, b in combinations(env_seeds, 2):
        shared = set(env_seeds[a]) & set(env_seeds[b])
        if shared:
            raise ProtocolError(f"world pools {a!r} and {b!r} share env seeds, e.g. {sorted(shared)[:3]}")
    worlds = {name: {world_seed(e, ep) for e in es for ep in range(1, episodes + 1)} for name, es in env_seeds.items()}
    for a, b in combinations(worlds, 2):
        frac = len(worlds[a] & worlds[b]) / max(1, min(len(worlds[a]), len(worlds[b])))
        if frac > max_overlap:
            raise ProtocolError(f"world pools {a!r} and {b!r} overlap: {frac:.4%} of worlds are shared (limit {max_overlap:.4%})")


def _worker(conn, area, env_seed):
    import crafter

    env = crafter.Env(area=(area, area), seed=env_seed)
    conn.send(env.reset())
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            obs, reward, done, info = env.step(data)
            ep = None
            if done:
                ep = [k for k, v in info["achievements"].items() if v > 0]
                obs = env.reset()
            conn.send((obs, float(reward), bool(done), ep))
    finally:
        conn.close()


class CrafterVec:
    """One crafter env per entry of `env_seeds`, each in its own process; finished episodes auto-reset.

    `step` returns (obs, rewards, dones, episodes) where `episodes[i]` is None, or the achievements unlocked in the
    episode env i just finished.
    """

    def __init__(self, env_seeds: List[int], area: int = 32):
        ctx = mp.get_context("spawn")  # fork is unsafe with torch
        self.n = len(env_seeds)
        self.conns, self.procs = [], []
        for seed in env_seeds:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, int(area), int(seed)), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.obs = np.stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        out = [c.recv() for c in self.conns]
        self.obs = np.stack([o[0] for o in out])
        return self.obs, np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out]), [o[3] for o in out]

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_crafter_env.py -v` Expected: 4 pass (2 `crafter` tests deselected).
Run: `pytest -m crafter tests/test_crafter_env.py -v` Expected: PASS (spawns 2 crafter processes; about 10 s to a minute, light on CPU; acceptable while run 1 is going).

- [ ] **Step 5: Commit**

```bash
git add xp/crafter/env.py tests/test_crafter_env.py
git commit -m "feat(crafter): world-seed pools with a disjointness guard and a subprocess vector env

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `rnd.py` and `ppo.py`: the learner

**Files:**
- Create: `xp/crafter/rnd.py`, `xp/crafter/ppo.py`, `tests/test_crafter_ppo.py`

**Interfaces:**
- Consumes: `xp.crafter.stats.EpisodeStats`, `ACHIEVEMENTS`.
- Produces:
  - `xp.crafter.rnd`: `RunningMeanStd`, `IntrinsicScaler(n_envs, gamma)` (`__call__(raw (t, n)) -> float32 (t, n)`), `RND` with `.target` (frozen CNN), `.pred`, `.obs_rms`, `.normalize(obs_uint8_np (N,64,64,3), device) -> Tensor (N,3,64,64)`, `.error(x) -> Tensor (N,)`.
  - `xp.crafter.ppo.resolve_device(name: str) -> torch.device` (`"auto"` picks cuda, then mps, then cpu).
  - `xp.crafter.ppo.Learner(ppo: dict, ent_coef: float, rnd_coef: Optional[float], seed: int, device="cpu")` with:
    - `.train(env, steps: int, log_every: int = 10) -> dict` returning `{"curve": [{"steps", "ach_100", "episodes", "rates"}], "first_unlock": {name: steps}, "episodes": int, "final_rates": dict, "steps": int, "secs": float}`. `env` is any object with `.obs`, `.n`, `.step(actions)` as in `CrafterVec`. A curve row is logged every `log_every` updates (one update = `n_envs * n_steps` steps) and at the last update.
    - `.evaluate(env, n_episodes: int) -> {"ach": float, "rates": dict, "episodes": int}`: sampled-action rollout, no learning.
    - `.fingerprint() -> str`.
  - `ppo` dict keys used: `n_envs, n_steps, epochs, minibatches, lr, gamma, lam, clip, vf_coef, rnd_lr`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crafter_ppo.py
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xp.crafter.ppo import ActorCritic, Learner, gae, resolve_device  # noqa: E402
from xp.crafter.rnd import RND, IntrinsicScaler  # noqa: E402

PPO = {"n_envs": 2, "n_steps": 8, "epochs": 1, "minibatches": 2, "lr": 0.00025, "gamma": 0.99, "lam": 0.95,
       "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 0.0001}


class FakeVec:
    """Random 64x64x3 frames; every env finishes an episode every `period` steps, having unlocked `collect_wood` (and `eat_cow` on odd episodes)."""

    def __init__(self, n=2, period=5, seed=0):
        self.n, self.period, self.t = n, period, 0
        self.rng = np.random.RandomState(seed)
        self.obs = self.rng.randint(0, 255, (n, 64, 64, 3), dtype=np.uint8)
        self.closed = False

    def step(self, actions):
        assert len(actions) == self.n
        self.t += 1
        self.obs = self.rng.randint(0, 255, (self.n, 64, 64, 3), dtype=np.uint8)
        done = np.full(self.n, self.t % self.period == 0)
        eps = [(["collect_wood"] + (["eat_cow"] if (self.t // self.period) % 2 else [])) if d else None for d in done]
        return self.obs, np.zeros(self.n, np.float32), done, eps

    def close(self):
        self.closed = True


def test_network_shapes_and_intrinsic_head_only_for_rnd():
    plain, rnd = ActorCritic(False), ActorCritic(True)
    logits, v, v_int = rnd.heads(torch.zeros(3, 3, 64, 64))
    assert logits.shape == (3, 8) and v.shape == (3,) and v_int.shape == (3,)
    assert plain.v_int is None and plain.heads(torch.zeros(1, 3, 64, 64))[2] is None


def test_rnd_target_is_frozen_and_error_shape():
    rnd = RND()
    assert not any(p.requires_grad for p in rnd.target.parameters())
    assert any(p.requires_grad for p in rnd.pred.parameters())
    x = rnd.normalize(np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8), "cpu")
    assert x.shape == (5, 3, 64, 64) and rnd.error(x).shape == (5,)


def test_intrinsic_scaler_shape_and_finite():
    out = IntrinsicScaler(2, 0.99)(np.abs(np.random.randn(8, 2)))
    assert out.shape == (8, 2) and np.isfinite(out).all() and out.dtype == np.float32


def test_gae_matches_a_hand_computed_value():
    r, v = torch.tensor([[1.0]]), torch.tensor([[0.5]])
    adv = gae(r, v, torch.tensor([2.0]), torch.tensor([[1.0]]), 0.9, 0.95)
    assert adv.item() == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")


def test_train_returns_curve_rates_and_first_unlocks():
    learner = Learner(PPO, 0.01, None, seed=0)
    env = FakeVec(period=5)
    run = learner.train(env, steps=16 * 4, log_every=2)  # batch = 16, 4 updates
    assert run["steps"] == 64 and learner.steps_trained == 64
    assert [row["steps"] for row in run["curve"]] == [32, 64]  # every 2 updates, last update included
    row = run["curve"][-1]
    assert row["ach_100"] != row["ach_100"]  # NaN: fewer than 100 episodes
    assert set(row["rates"]) >= {"collect_wood", "eat_cow"} and row["episodes"] == run["episodes"] > 0
    assert 0 < run["first_unlock"]["collect_wood"] <= 64 and "make_iron_sword" not in run["first_unlock"]
    assert run["final_rates"]["collect_wood"] == 1.0 and run["secs"] >= 0


def test_train_is_deterministic_per_seed():
    a, b = Learner(PPO, 0.01, None, seed=3), Learner(PPO, 0.01, None, seed=3)
    a.train(FakeVec(seed=1), 32, log_every=1)
    b.train(FakeVec(seed=1), 32, log_every=1)
    assert a.fingerprint() == b.fingerprint()


def test_kaiser_trains_rnd_and_kairi_has_none():
    kairi, kaiser = Learner(PPO, 0.01, None, seed=0), Learner(PPO, 0.01, 0.3, seed=0)
    assert kairi.rnd is None and kaiser.rnd is not None
    before = kaiser.fingerprint()
    run = kaiser.train(FakeVec(), 32, log_every=1)
    assert kaiser.fingerprint() != before and len(run["curve"]) == 2


def test_evaluate_never_learns():
    learner = Learner(PPO, 0.01, 0.3, seed=0)
    frozen = learner.fingerprint()
    res = learner.evaluate(FakeVec(period=3), n_episodes=4)
    assert learner.fingerprint() == frozen
    assert res["episodes"] >= 4 and res["ach"] >= 1.0 and res["rates"]["collect_wood"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_crafter_ppo.py -v` Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `rnd.py`**

```python
# xp/crafter/rnd.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update_moments(self, bm, bv, bc) -> None:
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def update(self, x: np.ndarray) -> None:
        self.update_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])


def _cnn(out: int) -> nn.Sequential:
    """The same conv stack as the policy encoder (64x64x3 in, 1024 features), ending in a linear layer of width `out`."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 4 * 4, out),
    )


class RND(nn.Module):
    """Random-network-distillation novelty bonus on 64x64x3 frames (the deep analog of UCB's confidence bonus)."""

    def __init__(self):
        super().__init__()
        self.target = _cnn(64)
        self.pred = nn.Sequential(_cnn(64), nn.ReLU(), nn.Linear(64, 64))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms = RunningMeanStd((3,))

    def normalize(self, obs_np: np.ndarray, device) -> torch.Tensor:
        """uint8 (N, 64, 64, 3) -> per-channel running-normalised, clipped float (N, 3, 64, 64)."""
        x = obs_np.astype(np.float32)
        self.obs_rms.update_moments(x.mean(axis=(0, 1, 2)), x.var(axis=(0, 1, 2)), x.shape[0] * x.shape[1] * x.shape[2])
        x = ((x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)).clip(-5, 5)
        return torch.as_tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)


class IntrinsicScaler:
    """Scales raw RND errors by the std of their discounted return (RND paper's forward filter); same as the MiniGrid scaler (5.2)."""

    def __init__(self, n_envs: int, gamma: float):
        self.gamma = gamma
        self.ret = np.zeros(n_envs)
        self.rms = RunningMeanStd(())

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        """raw: (t_len, n_envs) -> scaled float32 of the same shape."""
        for t in range(raw.shape[0]):
            self.ret = self.ret * self.gamma + raw[t]
            self.rms.update(self.ret)
        return (raw / np.sqrt(self.rms.var + 1e-8)).astype(np.float32)
```

- [ ] **Step 4: Implement `ppo.py`**

```python
# xp/crafter/ppo.py
from __future__ import annotations

import hashlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.crafter.rnd import RND, IntrinsicScaler
from xp.crafter.stats import ACHIEVEMENTS, EpisodeStats, nan_to_zero  # noqa: F401 (nan_to_zero re-exported for callers)

N_ACTIONS = 8
FRAME = (64, 64, 3)


def resolve_device(name: str) -> torch.device:
    """`auto` picks cuda, then mps, then cpu; anything else is passed to torch."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def prep(obs_np: np.ndarray, device) -> torch.Tensor:
    """uint8 (N, 64, 64, 3) -> float (N, 3, 64, 64) in [0, 1]."""
    return torch.as_tensor(obs_np, device=device).permute(0, 3, 1, 2).float() / 255.0


class ActorCritic(nn.Module):
    """The spike's small CNN encoder (identical for both agents), an actor head, and a critic head; RND agents add an intrinsic critic."""

    def __init__(self, intrinsic: bool = False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 4 * 4, 512), nn.ReLU(),
        )
        self.pi = nn.Linear(512, N_ACTIONS)
        self.v = nn.Linear(512, 1)
        # Built only for RND agents, so the plain-PPO parameter and random streams are identical with or without this path.
        self.v_int = nn.Linear(512, 1) if intrinsic else None

    def heads(self, x):
        h = self.enc(x)
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi(h), self.v(h).squeeze(-1), v_int


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
    """PPO, plus an RND bonus when `rnd_coef` is not None. Kairi = rnd_coef None; Kaiser = rnd_coef a float."""

    def __init__(self, ppo: dict, ent_coef: float, rnd_coef: Optional[float], seed: int, device="cpu"):
        self.ppo, self.ent_coef, self.rnd_coef, self.seed = ppo, ent_coef, rnd_coef, int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = ActorCritic(intrinsic=rnd_coef is not None).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo["lr"], eps=1e-5)
        self.rnd = self.rnd_opt = self.scaler = None
        if rnd_coef is not None:
            self.rnd = RND().to(self.device)
            self.rnd_opt = torch.optim.Adam(self.rnd.pred.parameters(), lr=ppo["rnd_lr"])
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

    def evaluate(self, env, n_episodes: int) -> dict:
        """Roll the frozen policy (sampled actions) until `n_episodes` episodes have finished; mean distinct achievements and unlock rates."""
        stats = EpisodeStats(window=max(1, n_episodes))
        done_eps = 0
        obs = env.obs
        while done_eps < n_episodes:
            with torch.no_grad():
                logits, _, _ = self.net.heads(prep(obs, self.device))
                action = Categorical(logits=logits).sample()
            obs, _, _, eps = env.step(action.cpu().numpy())
            for e in eps:
                if e is not None:
                    stats.add(e)
                    done_eps += 1
        return {"ach": float(np.mean([len(e) for e in stats.eps])), "rates": stats.rates(), "episodes": done_eps}

    def train(self, env, steps: int, log_every: int = 10) -> dict:
        """Train on `env` for about `steps` env steps (whole updates). Logs a curve row every `log_every` updates."""
        t0 = time.time()
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, rnd, device = self.ppo, self.rnd, self.device
        n, t_len = c["n_envs"], c["n_steps"]
        batch = n * t_len
        updates = max(1, steps // batch)
        net, opt = self.net, self.opt
        stats, first_unlock = EpisodeStats(100), {}
        obs_buf = np.zeros((t_len, n) + FRAME, dtype=np.uint8)
        act_buf = torch.zeros((t_len, n), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        val_buf = torch.zeros((t_len, n), device=device)
        vint_buf = torch.zeros((t_len, n), device=device)

        curve = []
        for u in range(updates):
            for t in range(t_len):
                obs_buf[t] = env.obs
                with torch.no_grad():
                    logits, value, value_int = net.heads(prep(env.obs, device))
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                act_buf[t], logp_buf[t], val_buf[t] = action, dist.log_prob(action), value
                if rnd is not None:
                    vint_buf[t] = value_int
                _, rew_buf[t], done_buf[t], eps = env.step(action.cpu().numpy())
                for e in eps:
                    if e is not None:
                        stats.add(e)
                        for name in e:
                            first_unlock.setdefault(name, (u * t_len + t + 1) * n)  # env steps so far in this call

            if rnd is not None:
                next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape((-1,) + FRAME)
                x = rnd.normalize(next_seq, device)
                with torch.no_grad():
                    raw = rnd.error(x).cpu().numpy().reshape(t_len, n)
                int_rewards = self.scaler(raw)
                perm = np.random.permutation(len(next_seq))
                for s in range(0, len(perm), 256):
                    idx = torch.as_tensor(perm[s:s + 256], device=device)
                    loss = rnd.error(x[idx]).mean()
                    self.rnd_opt.zero_grad()
                    loss.backward()
                    self.rnd_opt.step()

            with torch.no_grad():
                _, next_value, next_vint = net.heads(prep(env.obs, device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv_ext = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            returns, adv = adv_ext + val_buf, adv_ext
            if rnd is not None:
                # The intrinsic return is not cut at episode ends (RND paper), so dying costs no bonus.
                adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint, torch.ones_like(nonterm), c["gamma"], c["lam"])
                returns_int = adv_int + vint_buf
                adv = adv_ext + self.rnd_coef * adv_int

            b_obs = prep(obs_buf.reshape((-1,) + FRAME), device)
            b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
            b_ret_int = returns_int.reshape(-1) if rnd is not None else None
            for _ in range(c["epochs"]):
                perm_t = torch.randperm(batch, device=device)
                for s in range(0, batch, batch // c["minibatches"]):
                    idx = perm_t[s:s + batch // c["minibatches"]]
                    logits, value, value_int = net.heads(b_obs[idx])
                    dist = Categorical(logits=logits)
                    ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                    vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    if rnd is not None:
                        vf = vf + 0.5 * (value_int - b_ret_int[idx]).pow(2).mean()
                    loss = pg + c["vf_coef"] * vf - self.ent_coef * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()

            self.steps_trained += batch
            if (u + 1) % log_every == 0 or u == updates - 1:
                curve.append({"steps": (u + 1) * batch, "ach_100": stats.ach(), "episodes": stats.total, "rates": stats.rates()})
        return {"curve": curve, "first_unlock": first_unlock, "episodes": stats.total, "final_rates": stats.rates(),
                "steps": updates * batch, "secs": time.time() - t0}
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_crafter_ppo.py -v` Expected: all pass. If `test_train_is_deterministic_per_seed` fails, the cause is a data-dependent source of randomness (check that `torch.manual_seed` and `np.random.seed` are both reset at the top of `train`, and that `FakeVec(seed=1)` is rebuilt for each learner); do not loosen the test.

- [ ] **Step 6: Commit**

```bash
git add xp/crafter/rnd.py xp/crafter/ppo.py tests/test_crafter_ppo.py
git commit -m "feat(crafter): PPO and RND learner on 64x64 frames with a logged achievement curve

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `phase.py`, protocol `tie_key`, config, `run.py`

**Files:**
- Create: `xp/crafter/phase.py`, `configs/crafter.yaml`, `tests/test_crafter_phase.py`, `tests/test_crafter_config.py`
- Modify: `xp/protocol/jobs.py` (`tune_agent`), `xp/run.py`

**Interfaces:**
- Consumes: Task 1-3 symbols; `xp.protocol.jobs` (`grid_size`, `matched_entropy`, `rows`, `run_jobs`, `training_auc`, `curve_points`, `tune_agent`); `xp.protocol.results` (`read_tune_job`, `write_tune_job`, `seed_job_done`, `write_seed_job`, `atomic_write`); `xp.protocol.manifest.write_manifest`.
- Produces:
  - `xp.protocol.jobs.tune_agent(cfg, phase, out_dir, agent, workers, tune_job, extra=None, tie_key="auc")`: ties are broken by the highest mean of `read_tune_job_extra(...)[tie_key]` (default unchanged).
  - `xp.crafter.phase`: `AGENTS`, `require_budget(cfg) -> int` (returns `budget.steps`; raises `ProtocolError` if `budget.steps` is missing/None; also validates `budget.deep_set` entries are achievement names and that it is present), `build_learner(cfg, agent, params, seed) -> Learner`, `make_env(cfg, stage: str, seed: int)` (`stage` in `tune`, `eval`, `heldout`), `tune_job(cfg, out_dir, agent, params, seed) -> float`, `seed_job(cfg, out_dir, agent, params, provisional, seed) -> None`, `run_phase(cfg, out_dir, workers=1, allow_code_change=False)`.
  - Seed-job result rows (knob = familiar area): metrics `auc`, `ach_final`, `crafter_score`, `deep_first_unlock`, `heldout_ach`, `heldout_crafter_score`, and per achievement `rate_<name>` and `first_<name>` (steps, censored at S).
- Config keys used: `env.area`, `budget.{steps, cap, target_ach, sustain, min_deep, deep_set}`, `eval.episodes`, `ppo.*` plus `ppo.device` and `ppo.log_every`, `seeds`, `agents`.

**Decision recorded (plan-level, compatible with the spec):** the deep set is also copied into the config as `budget.deep_set` (the calibration prints it), because `seed_job` needs it for `deep_first_unlock` and the config hash then pins it before any RND result. Both keys are required before any job starts. The held-out evaluation (`eval.episodes` frozen-policy episodes on `heldout_env_seeds`) is how spec 5.5's "held-out score" is measured; `heldout_*` are secondary metrics.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crafter_config.py
from __future__ import annotations

import pytest

from xp.crafter.stats import ACHIEVEMENTS
from xp.errors import ProtocolError
from xp.protocol.jobs import grid_size
from xp.protocol.phase import Phase, load_config

EXISTING_HASHES = {
    "minigrid": "b41211b3b5c6", "minigrid-multiroom": "39ce53712b8a", "minigrid-a": "51260035781b", "minigrid-entfair": "e41af4a9d320",
    "showdown": "fdac28308815", "showdown-control": "028d33113f41", "showdown-matchup": "7ca893d2d935",
    "showdown-tempo-base": "2083c9b9737b", "showdown-tempo": "435175fb4ec3", "bandit": "50ea69897efe",
}


@pytest.mark.parametrize("name,expected", sorted(EXISTING_HASHES.items()))
def test_existing_config_hashes_are_unchanged(name, expected):
    assert Phase.from_config(load_config(f"configs/{name}.yaml")).config_hash == expected


def test_crafter_config_declares_the_protocol():
    cfg = load_config("configs/crafter.yaml")
    phase = Phase.from_config(cfg)
    assert phase.name == "crafter" and phase.familiar == 32.0 and phase.sweep == (32.0,)
    assert phase.tune_seeds == (0, 1, 2) and phase.eval_seeds == tuple(range(100, 110))
    assert cfg["agents"]["kairi"]["grid"]["ent_coef"] == [0.003, 0.006, 0.01, 0.02, 0.05, 0.1]
    assert cfg["agents"]["kaiser"]["grid"]["rnd_coef"] == [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 6
    assert cfg["agents"]["kaiser"]["fixed"]["ent_coef_from"] == "kairi"
    b = cfg["budget"]
    assert (b["cap"], b["target_ach"], b["sustain"], b["min_deep"]) == (3000000, 5, 3, 3)
    assert "steps" not in b and "deep_set" not in b  # unset until the user sets them from the calibration


def test_smoke_config_is_small_and_valid():
    cfg = load_config("configs/crafter-smoke.yaml")
    Phase.from_config(cfg)
    assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 2
    assert cfg["budget"]["steps"] <= 20000 and set(cfg["budget"]["deep_set"]) <= set(ACHIEVEMENTS)


def test_crafter_does_not_leak_into_other_packages():
    import pathlib

    for pkg in ("minigrid", "showdown", "protocol"):
        for path in pathlib.Path("xp", pkg).glob("*.py"):
            assert "xp.crafter" not in path.read_text(), path
    for path in pathlib.Path("xp", "crafter").glob("*.py"):
        text = path.read_text()
        assert "xp.minigrid" not in text and "xp.showdown" not in text, path


def test_require_budget_blocks_until_set():
    from xp.crafter.phase import require_budget

    cfg = load_config("configs/crafter.yaml")
    with pytest.raises(ProtocolError, match="budget.steps"):
        require_budget(cfg)
    cfg["budget"]["steps"] = 100000
    with pytest.raises(ProtocolError, match="deep_set"):
        require_budget(cfg)
    cfg["budget"]["deep_set"] = ["collect_wood", "not_an_achievement"]
    with pytest.raises(ProtocolError, match="not_an_achievement"):
        require_budget(cfg)
    cfg["budget"]["deep_set"] = ["collect_wood", "eat_cow", "collect_drink"]
    assert require_budget(cfg) == 100000
```

```python
# tests/test_crafter_phase.py
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from tests.test_crafter_ppo import FakeVec  # noqa: E402
from xp.crafter import phase as cp  # noqa: E402
from xp.errors import ConfigMismatchError, ProtocolError  # noqa: E402
from xp.protocol.phase import Phase  # noqa: E402
from xp.protocol.results import load_results, read_tune_job, read_tune_job_extra  # noqa: E402

TINY = {
    "phase": "crafter",
    "knob": {"name": "area", "familiar": 32, "sweep": [32]},
    "env": {"area": 32},
    "seeds": {"tune": [0, 2], "eval": [100, 102], "chunk": 1, "agent_seed": 0},
    "budget": {"steps": 64, "cap": 128, "target_ach": 5, "sustain": 3, "min_deep": 3, "deep_set": ["collect_wood", "eat_cow", "collect_drink"]},
    "eval": {"episodes": 3},
    "ppo": {"device": "cpu", "log_every": 2, "n_envs": 2, "n_steps": 8, "epochs": 1, "minibatches": 2, "lr": 0.00025, "gamma": 0.99,
            "lam": 0.95, "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 0.0001},
    "agents": {
        "kairi": {"grid": {"ent_coef": [0.01, 0.05]}, "natural_limits": {"ent_coef": [0.01, 0.05]}},
        "kaiser": {"fixed": {"ent_coef": 0.01, "ent_coef_from": "kairi"}, "grid": {"rnd_coef": [0.1, 0.3]},
                   "natural_limits": {"rnd_coef": [0.1, 0.3]}},
    },
    "report": {"metric": "auc", "lower_is_better": False, "agents": ["kairi", "kaiser"]},
}


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(cp, "make_env", lambda c, stage, seed: FakeVec(n=c["ppo"]["n_envs"], period=4, seed=seed))
    return copy.deepcopy(TINY)


def test_kairi_and_kaiser_differ_only_by_the_rnd_path(cfg):
    kairi = cp.build_learner(cfg, "kairi", {"ent_coef": 0.03}, seed=1)
    kaiser = cp.build_learner(cfg, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.03}, seed=1)
    assert kairi.rnd is None and kairi.ent_coef == 0.03
    assert kaiser.rnd is not None and kaiser.rnd_coef == 0.3 and kaiser.ent_coef == 0.03
    assert cp.build_learner(cfg, "kaiser", {"rnd_coef": 0.3}, seed=1).ent_coef == 0.01  # falls back to the fixed default


def test_tune_job_guards_seeds_budget_and_caches(cfg, tmp_path):
    phase = Phase.from_config(cfg)
    with pytest.raises(ProtocolError):
        cp.tune_job(cfg, tmp_path, "kairi", {"ent_coef": 0.01}, seed=100)  # an eval seed
    unset = copy.deepcopy(cfg)
    del unset["budget"]["steps"]
    with pytest.raises(ProtocolError, match="budget.steps"):
        cp.tune_job(unset, tmp_path, "kairi", {"ent_coef": 0.01}, seed=0)
    score = cp.tune_job(cfg, tmp_path, "kairi", {"ent_coef": 0.01}, seed=0)
    assert 0.0 <= score <= 1.0 and read_tune_job(tmp_path, phase, "kairi", {"ent_coef": 0.01}, 0) == score
    extra = read_tune_job_extra(tmp_path, phase, "kairi", {"ent_coef": 0.01}, 0)
    assert 0.0 <= extra["auc"] <= 1.0 and "final" in extra and extra["curve"]


def test_seed_job_writes_every_metric_and_the_curve(cfg, tmp_path):
    cp.seed_job(cfg, tmp_path, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.01}, False, seed=100)
    df = load_results(tmp_path, "crafter")
    got = set(df["metric"])
    assert {"auc", "ach_final", "crafter_score", "deep_first_unlock", "heldout_ach", "heldout_crafter_score"} <= got
    assert "rate_collect_wood" in got and "first_collect_wood" in got and len(df) == 6 + 44
    assert set(df["seed"]) == {100} and set(df["knob"]) == {32.0}
    first = df[df["metric"] == "first_make_iron_sword"]["value"].item()
    assert first == cfg["budget"]["steps"]  # never unlocked: censored at S
    assert (tmp_path / "crafter" / "curves" / "kaiser__seed-100.json").exists()
    with pytest.raises(ProtocolError):
        cp.seed_job(cfg, tmp_path, "kairi", {"ent_coef": 0.01}, False, seed=0)  # a tune seed


def test_run_phase_end_to_end_and_resume(cfg, tmp_path):
    cp.run_phase(cfg, tmp_path)
    df = load_results(tmp_path, "crafter")
    auc = df[df["metric"] == "auc"]
    assert len(auc) == 4 and set(auc["agent"]) == {"kairi", "kaiser"} and set(auc["seed"]) == {100, 101}
    assert not auc["provisional"].any()  # grid ends are declared natural limits
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "crafter").rglob("*.csv")}
    cp.run_phase(cfg, tmp_path)  # resume: every job cached
    assert before == {p: p.stat().st_mtime_ns for p in (tmp_path / "crafter").rglob("*.csv")}
    changed = copy.deepcopy(cfg)
    changed["budget"]["steps"] = 128
    with pytest.raises(ConfigMismatchError):
        cp.run_phase(changed, tmp_path)


def test_run_phase_refuses_unequal_grids_and_unset_budget(cfg, tmp_path):
    bad = copy.deepcopy(cfg)
    bad["agents"]["kaiser"]["grid"]["rnd_coef"] = [0.1]
    with pytest.raises(ProtocolError, match="equal size"):
        cp.run_phase(bad, tmp_path)
    unset = copy.deepcopy(cfg)
    del unset["budget"]["steps"]
    with pytest.raises(ProtocolError, match="budget.steps"):
        cp.run_phase(unset, tmp_path)


def test_tie_break_uses_final_ach_when_asked(cfg, tmp_path):
    """Two grid points with identical scores: tune_agent must pick the one with the higher mean `final`."""
    from xp.protocol.jobs import tune_agent
    from xp.protocol.results import write_tune_job

    phase = Phase.from_config(cfg)

    def fake_job(c, out, agent, params, seed):
        final = 3.0 if params["ent_coef"] == 0.05 else 1.0
        write_tune_job(out, phase, agent, params, seed, 0.5, extra={"auc": 0.1, "final": final, "curve": []})
        return 0.5

    res = tune_agent(cfg, phase, tmp_path, "kairi", 1, fake_job, tie_key="final")
    assert res.best_params == {"ent_coef": 0.05}
    res_default = tune_agent(cfg, phase, tmp_path / "b", "kairi", 1, fake_job)  # default tie key `auc` is equal: grid order wins
    assert res_default.best_params == {"ent_coef": 0.01}
```

(The tie-break test relies on `tune_agent` calling `tune_job(cfg, out_dir, agent, params, seed)`; `write_tune_job` signature is `write_tune_job(out_dir, phase, agent, params, seed, score, extra=None)`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_crafter_config.py tests/test_crafter_phase.py -v` Expected: FAIL (missing modules and configs).

- [ ] **Step 3: Add `tie_key` to the protocol**

In `xp/protocol/jobs.py`, change the `tune_agent` signature and `mean_auc`:

```python
def tune_agent(cfg: dict, phase: Phase, out_dir, agent: str, workers: int, tune_job: Callable, extra: Optional[dict] = None,
               tie_key: str = "auc") -> TuneResult:
```
and inside, replace `.get("auc", -1.0)` in `mean_auc` with `.get(tie_key, -1.0)`. Nothing else changes (the docstring of `break_ties` already says "highest mean"; add `tie_key` to `tune_agent`'s behaviour in one comment line above `def mean_auc`: `# tie_key: the extra field whose highest mean breaks exact score ties (AUC by default; Crafter uses final ach_100)`).

- [ ] **Step 4: Write `configs/crafter.yaml`**

```yaml
# Crafter phase (spec 5.5 and the 2026-11-02 reopen amendment): single familiar world (area 32), no shift knob.
# `budget.steps` and `budget.deep_set` are UNSET on purpose: run `python -m xp.crafter.calibrate`, read S, the verdicts and the hours,
# then (with the user's approval) add both keys here, push, and reclone. No tuning job starts while they are missing.
phase: crafter
knob:
  name: area
  familiar: 32
  sweep: [32]
env:
  area: 32
seeds:
  tune: [0, 3]
  eval: [100, 110]
  chunk: 1
  agent_seed: 0
budget:
  cap: 3000000          # calibration runs default-entropy Kairi this long on each tune seed, no early stopping
  target_ach: 5         # S = first logged step from which the tune-seed mean ach_100 is >= 5 ...
  sustain: 3            # ... at that step and the next 2 logged steps
  min_deep: 3           # the deep achievement set (plain-PPO unlock rate 5% to 95% at S) needs at least this many
eval:
  episodes: 100         # frozen-policy episodes on the held-out world pool after training
ppo:
  device: auto          # cuda, then mps, then cpu (resolved at run time, so the config hash is the same on every machine)
  log_every: 10         # updates between curve rows: 10 * 8 envs * 128 steps = 10,240 env steps
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
      ent_coef: 0.01          # used only if a rnd_coef of 0 is tuned; tuning uses Kairi's tuned value
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
    ach_final: false
    crafter_score: false
    heldout_ach: false
    deep_first_unlock: true
  xlabel: Crafter, area 32 (single familiar world)
  ylabel: training AUC of ach_100 / 22 (95% CI)
  title: Crafter, tuned Kairi vs Kaiser
```

- [ ] **Step 5: Implement `xp/crafter/phase.py`**

```python
# xp/crafter/phase.py
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
```

- [ ] **Step 6: Wire `run.py`**

In `xp/run.py`: add `"crafter": "xp.crafter.phase"` to `PHASES`; change `if args.phase in ("minigrid", "showdown"):` to `if args.phase in ("minigrid", "showdown", "crafter"):`; update the two help strings to mention crafter.

- [ ] **Step 7: Run to verify pass**

Run: `pytest tests/test_crafter_config.py tests/test_crafter_phase.py -v`
Expected: all pass except `test_smoke_config_is_small_and_valid`, which fails until Task 6 writes `configs/crafter-smoke.yaml`; to keep this commit green mark that one test `@pytest.mark.skip(reason="smoke config arrives in Task 6")` now and remove the skip in Task 6.
Run: `pytest -q` Expected: whole suite green (proves the protocol change and hashes are intact).

- [ ] **Step 8: Commit**

```bash
git add xp/crafter/phase.py xp/protocol/jobs.py xp/run.py configs/crafter.yaml tests/test_crafter_phase.py tests/test_crafter_config.py
git commit -m "feat(crafter): tune and seed jobs, run_phase with the unset-budget guard, crafter config, tie_key in tune_agent

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `calibrate.py`

**Files:**
- Create: `xp/crafter/calibrate.py`, `tests/test_crafter_calibrate.py`

**Interfaces:**
- Consumes: `xp.crafter.phase.build_learner, make_env`; `xp.crafter.stats.ACHIEVEMENTS, nan_to_zero`; `xp.protocol.jobs.run_jobs`; `xp.protocol.phase.Phase, load_config`.
- Produces:
  - `suggest_budget(curves, target: float, cap: int, sustain: int) -> dict` with keys `steps`, `reached`, `index` (log index of S, last index if not reached), `mean_ach`, `log_steps`. `curves` is a list (one per seed) of lists of rows `{"steps", "ach_100", "rates"}` logged at identical steps.
  - `gate_checks(suggestion: dict, curves, min_rare: int = 5, min_deep: int = 3) -> dict` with keys `g1_target` (= `reached`), `g1_rare` (count of achievements with mean rate < 0.05 at S, bool `rare_ok`), `g2`, `early_ach`, `final_ach`, `rates_at_s`, `deep_set` (sorted list), `deep_ok`, `go` (all true), `reasons` (list of failure strings).
  - `phase_runs(cfg) -> int` (`2 * grid_size * n_tune_seeds + 2 * n_eval_seeds`, i.e. 56).
  - `implied_hours(S: int, steps_per_s: float, cfg: dict) -> dict` with `full_phase_hours` = `phase_runs * S / sps / 3600` and `calibration_hours` = `n_tune_seeds * cap / sps / 3600`.
  - `calibration_run(cfg, seed) -> dict` `{"curve": rows, "secs": float, "steps": int}`; CLI `python -m xp.crafter.calibrate --config C --out J [--workers 1]`; JSON keys: `suggestion`, `gates`, `throughput`, `hours`, `curves`, `go`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crafter_calibrate.py
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch")

from xp.crafter.calibrate import gate_checks, implied_hours, phase_runs, suggest_budget  # noqa: E402
from xp.crafter.stats import ACHIEVEMENTS  # noqa: E402

RARE = {a: 0.0 for a in ACHIEVEMENTS}


def curve(achs, rates=None, step=10240):
    return [{"steps": step * (i + 1), "ach_100": a, "rates": dict(rates or RARE)} for i, a in enumerate(achs)]


def test_suggest_budget_needs_three_consecutive_points():
    c = curve([float("nan"), 3.0, 5.2, 4.0, 5.1, 5.3, 5.4, 6.0])
    s = suggest_budget([c, c], target=5.0, cap=10240 * 8, sustain=3)
    assert s["reached"] and s["steps"] == 10240 * 5 and s["index"] == 4  # the lone 5.2 at index 2 does not count


def test_suggest_budget_ignores_a_crossing_in_the_last_two_points_and_falls_back_to_cap():
    c = curve([1.0, 2.0, 3.0, 4.0, 4.5, 5.5, 5.5])
    s = suggest_budget([c], target=5.0, cap=10240 * 7, sustain=3)
    assert not s["reached"] and s["steps"] == 10240 * 7 and s["index"] == 6


def test_suggest_budget_averages_over_seeds_and_counts_nan_as_zero():
    a = curve([float("nan"), 6.0, 6.0, 6.0])
    b = curve([float("nan"), 4.0, 4.0, 4.0])
    s = suggest_budget([a, b], target=5.0, cap=10240 * 4, sustain=2)
    assert s["mean_ach"][0] == 0.0 and s["reached"] and s["steps"] == 10240 * 2


def test_suggest_budget_rejects_mismatched_log_steps():
    a, b = curve([1.0, 2.0]), curve([1.0, 2.0], step=5000)
    with pytest.raises(ValueError):
        suggest_budget([a, b], 5.0, 20480, 1)


def _rates(rare_count, deep_count):
    r = {a: 0.99 for a in ACHIEVEMENTS}
    names = list(ACHIEVEMENTS)
    for a in names[:rare_count]:
        r[a] = 0.0
    for a in names[rare_count:rare_count + deep_count]:
        r[a] = 0.5
    return r


def test_gate_checks_pass_and_set_the_deep_set():
    c = curve([2.0, 3.0, 5.0, 5.5, 6.0], rates=_rates(rare_count=6, deep_count=4))
    s = suggest_budget([c], 5.0, 10240 * 5, 3)
    g = gate_checks(s, [c])
    assert g["go"] and g["rare_ok"] and g["g2"] and g["deep_ok"] and len(g["deep_set"]) == 4 and g["reasons"] == []


def test_gate_checks_stop_on_too_few_rare_deep_or_no_growth():
    flat = curve([6.0] * 6, rates=_rates(rare_count=2, deep_count=1))
    g = gate_checks(suggest_budget([flat], 5.0, 10240 * 6, 3), [flat])
    assert not g["go"] and not g["rare_ok"] and not g["deep_ok"] and not g["g2"]
    assert len(g["reasons"]) == 3


def test_gate_checks_stop_when_the_target_is_not_reached():
    c = curve([1.0, 2.0, 3.0, 3.5], rates=_rates(6, 4))
    g = gate_checks(suggest_budget([c], 5.0, 10240 * 4, 3), [c])
    assert not g["go"] and any("target" in r for r in g["reasons"])


def test_cost_arithmetic():
    cfg = {"seeds": {"tune": [0, 3], "eval": [100, 110]}, "budget": {"cap": 3000000},
           "agents": {"kairi": {"grid": {"ent_coef": [1] * 6}}, "kaiser": {"grid": {"rnd_coef": [1] * 6}}}}
    assert phase_runs(cfg) == 56
    h = implied_hours(1000000, 500.0, cfg)
    assert h["full_phase_hours"] == pytest.approx(56 * 1000000 / 500.0 / 3600)
    assert h["calibration_hours"] == pytest.approx(3 * 3000000 / 500.0 / 3600)


def test_calibrate_cli_writes_the_json(tmp_path, monkeypatch):
    from tests.test_crafter_ppo import FakeVec
    from xp.crafter import calibrate as cal
    from xp.crafter import phase as cp
    from tests.test_crafter_phase import TINY

    cfg = copy.deepcopy(TINY)
    cfg["budget"].update({"cap": 64, "target_ach": 0.5, "sustain": 1})
    del cfg["budget"]["steps"]  # calibration must run with the budget unset
    monkeypatch.setattr(cp, "make_env", lambda c, stage, seed: FakeVec(n=2, period=2, seed=seed))
    import yaml

    cfg_path, out = tmp_path / "c.yaml", tmp_path / "cal.json"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cal.main(["--config", str(cfg_path), "--out", str(out)])
    data = json.loads(out.read_text())
    assert set(data) >= {"suggestion", "gates", "throughput", "hours", "curves", "go"}
    assert len(data["curves"]) == 2 and data["throughput"]["steps_per_s"] > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_crafter_calibrate.py -v` Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# xp/crafter/calibrate.py
from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import torch

from xp.crafter import phase as cp
from xp.crafter.stats import ACHIEVEMENTS, nan_to_zero
from xp.protocol.jobs import grid_size, run_jobs
from xp.protocol.phase import Phase, load_config

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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate the Crafter budget S from tune-seed Kairi runs (spec 2026-11-02, Change 1).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="crafter_calibration.json")
    parser.add_argument("--workers", type=int, default=1, help="keep 1 for a clean throughput measurement")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    runs = run_jobs(calibration_run, [(cfg, s) for s in phase.tune_seeds], args.workers)
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_crafter_calibrate.py -v && pytest -q` Expected: all pass.
If `test_gate_checks_stop_on_too_few_rare_deep_or_no_growth` fails on `g2`: the flat curve starts at 6.0, so the early mean equals the final and `final > early` is False; that is the intended case.

- [ ] **Step 5: Commit**

```bash
git add xp/crafter/calibrate.py tests/test_crafter_calibrate.py
git commit -m "feat(crafter): budget calibration (sustained crossing, G1/G2, deep set, cost printout)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Smoke config and a real end-to-end smoke run

**Files:**
- Create: `configs/crafter-smoke.yaml`
- Modify: `tests/test_crafter_config.py` (remove the skip added in Task 4)

**Interfaces:** Consumes everything above. Produces the config the notebook's smoke cell runs.

- [ ] **Step 1: Write `configs/crafter-smoke.yaml`**

A copy of `configs/crafter.yaml` with these changes (keep every other key identical; comments at the top say it is a smoke check and the numbers mean nothing; grid ends are declared natural limits only so 2-value grids are not provisional):

```yaml
# Crafter SMOKE: every path (calibration, tune, seed job, held-out eval, report) in a few minutes. The numbers mean nothing.
# Grid ends are declared natural limits ONLY so a 2-value grid is not 'provisional' (every point of a 2-value grid is an edge).
phase: crafter
knob: {name: area, familiar: 32, sweep: [32]}
env: {area: 32}
seeds: {tune: [0, 1], eval: [100, 102], chunk: 1, agent_seed: 0}
budget:
  steps: 4096
  cap: 4096
  target_ach: 0.5       # tiny, so the calibration's rule code path runs; the verdict is meaningless at this size
  sustain: 2
  min_deep: 1
  deep_set: [collect_wood, collect_drink, eat_cow]
eval: {episodes: 4}
ppo: {device: auto, log_every: 2, n_envs: 4, n_steps: 64, epochs: 2, minibatches: 4, lr: 0.00025, gamma: 0.99, lam: 0.95, clip: 0.2, vf_coef: 0.5, rnd_lr: 0.0001}
agents:
  kairi:
    grid: {ent_coef: [0.01, 0.05]}
    natural_limits: {ent_coef: [0.01, 0.05]}
  kaiser:
    fixed: {ent_coef: 0.01, ent_coef_from: kairi}
    grid: {rnd_coef: [0.1, 0.3]}
    natural_limits: {rnd_coef: [0.1, 0.3]}
report:
  metric: auc
  lower_is_better: false
  agents: [kairi, kaiser]
  extra_metrics: {ach_final: false, crafter_score: false, heldout_ach: false, deep_first_unlock: true}
  xlabel: Crafter smoke
  ylabel: training AUC of ach_100 / 22 (95% CI)
  title: Crafter smoke
```

Batch is 4 x 64 = 256 steps, so 4096 steps is 16 updates and `log_every: 2` logs 8 rows.

- [ ] **Step 2: Remove the skip and run the config tests**

Remove the `@pytest.mark.skip` on `test_smoke_config_is_small_and_valid`. Run: `pytest tests/test_crafter_config.py -v` Expected: PASS.

- [ ] **Step 3: Real smoke run (ask the user first; light but real)**

This spawns real `crafter` processes and trains for a few minutes on CPU; it competes with Showdown run 1 for CPU and memory, so **ask the user for a yes, and prefer running it after run 1 finishes.** Check `sysctl vm.swapusage` first (stop if swap use is above 2 GB).

```bash
python -m xp.crafter.calibrate --config configs/crafter-smoke.yaml --out <scratchpad dir>/crafter_smoke_cal.json
python -m xp.run --phase crafter --config configs/crafter-smoke.yaml --out results/crafter-smoke
python -m xp.report --phase crafter --config configs/crafter-smoke.yaml --out results/crafter-smoke --figs figs/crafter-smoke --allow-provisional
```
(Write the calibration JSON to the session scratchpad directory, not the repo.)
Expected: calibration prints S, rare count, deep set, throughput and either GO or PHASE STOPS (either is fine at smoke size); the run prints 4 tune jobs per agent... precisely 2 grid points x 1 tune seed x 2 agents = 4 `tune` lines, then 4 `seed job` lines and `finished 4 seed jobs`; the report writes `figs/crafter-smoke/degradation.png` and CSVs. Re-running the `xp.run` command finishes instantly (all cached). Record the measured steps/s in the task report. Delete `results/crafter-smoke` and `figs/crafter-smoke` afterwards (both are git-ignored).

- [ ] **Step 4: Commit**

```bash
git add configs/crafter-smoke.yaml tests/test_crafter_config.py
git commit -m "feat(crafter): smoke config exercising every path

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Colab notebook

**Files:**
- Create: `notebooks/run_crafter.ipynb`, `tests/test_crafter_notebook.py`

**Interfaces:** Consumes the CLIs: `python -m xp.crafter.calibrate`, `python -m xp.run --phase crafter`, `python -m xp.report --phase crafter`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_crafter_notebook.py
from __future__ import annotations

import json


def test_notebook_has_the_planned_cells_and_literal_paths():
    nb = json.load(open("notebooks/run_crafter.ipynb"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ("drive.mount", "git clone", "requirements-crafter.txt", "torch.cuda.is_available", "crafter-smoke.yaml",
                   "xp.crafter.calibrate", "xp.run --phase crafter", "xp.report --phase crafter", "xp-results", "xp-figs-crafter"):
        assert needle in src, needle
    assert "$" not in "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")  # literal paths: `$VAR` came out empty in Colab
    assert all(c.get("outputs") == [] for c in nb["cells"] if c["cell_type"] == "code")  # committed clean
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_crafter_notebook.py -v` Expected: FAIL (`FileNotFoundError`).

- [ ] **Step 3: Generate the notebook**

Write a throwaway script in the scratchpad (not the repo) that builds the notebook with `json` (nbformat 4, kernelspec `python3`) from these cells, then run it:

1. markdown: title "XP Phase 3 (Crafter) on Colab"; use a GPU runtime (Runtime > Change runtime type); results go to Drive; every unit is resumable, so after a disconnect rerun the cell you were on (at most one in-flight job restarts from zero, up to about an hour at S near the cap); literal paths on purpose; order: setup, benchmark, smoke, calibrate (stop and read), then the user edits `budget.steps` and `budget.deep_set` in `configs/crafter.yaml`, pushes, rerun the clone cell, then run, status, report.
2. code: `from google.colab import drive; drive.mount('/content/drive')`
3. code: `%cd /content`, `!rm -rf xp-agents && git clone --depth 1 -b crafter-reopen https://github.com/knguyen0923/xp-agents xp-agents`, `%cd /content/xp-agents`, `!pip install -q -r requirements-crafter.txt -e .`. (The branch name is `crafter-reopen`; if the user merges first, change `-b` to `main`: say so in the markdown cell.)
4. code: device check: `import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); import os; print(os.cpu_count(), 'cpus')`
5. markdown: "Benchmark (compare with the spike: area 32 CPU 589 steps/s, MPS 781)".
6. code: `!python -m xp.crafter.calibrate --config configs/crafter-smoke.yaml --out /content/drive/MyDrive/xp-results/crafter_smoke_calibration.json` with a comment that its printed `throughput ... steps/s` line is the benchmark (it is a 2-run, 4096-step plain-PPO run on the smoke config; the number includes start-up, as the spike's benchmark did).
7. markdown + code: smoke: `!python -m xp.run --phase crafter --config configs/crafter-smoke.yaml --out /content/drive/MyDrive/xp-results-smoke` then `!python -m xp.report --phase crafter --config configs/crafter-smoke.yaml --out /content/drive/MyDrive/xp-results-smoke --figs /content/drive/MyDrive/xp-figs-crafter-smoke --allow-provisional`.
8. code: calibration: `!python -m xp.crafter.calibrate --config configs/crafter.yaml --out /content/drive/MyDrive/xp-results/crafter_calibration.json --workers 1` (comment: up to 3 x 3M steps, about 3 hours at 781 steps/s; resumable only by rerunning from zero, so keep the tab alive).
9. markdown: "STOP HERE. Read S, the G1/G2 and deep-set verdicts and the hours estimate printed above and tell the user. If it says PHASE STOPS, write the calibration up as the result; do not continue. Otherwise, only after the user approves S and the cost: edit `budget.steps` and `budget.deep_set` in `configs/crafter.yaml`, commit and push, then rerun the clone cell."
10. code: full run: `!python -m xp.run --phase crafter --config configs/crafter.yaml --out /content/drive/MyDrive/xp-results --workers 1`
11. code: status: count `*.json` job files under `/content/drive/MyDrive/xp-results/crafter/` (tuning jobs under `tuning/jobs/`, seed jobs under `jobs/`; read `xp/protocol/results.py` `_job_path` and `_tune_job_path` for the exact folders and print `n done of 56`).
12. code: report: `!python -m xp.report --phase crafter --config configs/crafter.yaml --out /content/drive/MyDrive/xp-results --figs /content/drive/MyDrive/xp-figs-crafter` then display `/content/drive/MyDrive/xp-figs-crafter/degradation.png`.

Code cells must have `"outputs": []` and `"execution_count": null`; no `$` in code cells (use literal paths). Save as `notebooks/run_crafter.ipynb`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_crafter_notebook.py -v` Expected: PASS. Also `python -c "import json; json.load(open('notebooks/run_crafter.ipynb'))"`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/run_crafter.ipynb tests/test_crafter_notebook.py
git commit -m "feat(crafter): resumable Colab notebook (benchmark, smoke, calibrate, run, status, report)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Docs, whole-branch verification, handoff

**Files:**
- Modify: `CLAUDE.md` (Commands and Architecture bullets), `STATUS.md`, `RESUME.md`, `README.md` only if it lists phases (check with `grep -n -i "showdown" README.md`)

- [ ] **Step 1: Update CLAUDE.md**

Add to the Commands block (after the Showdown paragraph) a Crafter paragraph: `pip install -r requirements-crafter.txt -e .`; `python -m xp.crafter.calibrate --config configs/crafter.yaml --out crafter_calibration.json` (Colab only; up to about 3 hours); after the user approves S set `budget.steps` and `budget.deep_set` in `configs/crafter.yaml`; `python -m xp.run --phase crafter ...` (Colab, `notebooks/run_crafter.ipynb`); `pytest -m crafter` for the tests that step a real `crafter.Env`. Add an Architecture bullet: `xp/crafter/` is PPO/RND on 64x64 frames, own learner, never imports `xp.minigrid`/`xp.showdown`, never imported by `xp.protocol`; calibration gates S; no tuning job starts while `budget.steps` is unset.

- [ ] **Step 2: Full verification**

Run, from the worktree: `pytest -q` (whole fast suite), `pytest -m crafter -v` (real-env tests), `ruff check .` (if ruff is installed in the venv; otherwise `$PY -m pip install ruff` is acceptable). Expected: all green. Then confirm the existing config hashes with `pytest tests/test_crafter_config.py -v` (already part of the suite). Use superpowers:verification-before-completion: report the actual counts.

- [ ] **Step 3: Whole-branch review**

Request a code review of `git diff showdown-tempo..crafter-reopen` (superpowers:requesting-code-review) against the spec; fix findings with tests.

- [ ] **Step 4: Update STATUS.md and RESUME.md**

RESUME.md section 2: Crafter code is built and reviewed on `crafter-reopen` (worktree `.worktrees/crafter`), nothing run on Colab; **next, in order (each needs the user's yes):** push `crafter-reopen` (Colab clones from GitHub), run the notebook's benchmark and smoke cells, run the calibration cell, then read S, the verdicts and the hours and ask the user to approve S before `budget.steps` and `budget.deep_set` are set. STATUS.md: last-updated line, then a STATUS-only marker commit per the Stop hook rule.

- [ ] **Step 5: Commit docs, then the marker**

```bash
git add CLAUDE.md RESUME.md STATUS.md
git commit -m "docs: Crafter build recorded in CLAUDE.md, RESUME and STATUS

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
# then set STATUS_COMMIT to `git rev-parse --short HEAD`, and commit STATUS.md alone:
git commit -am "docs: STATUS marker and last-updated line

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Stop and report**

Do not push, merge or start a run. Tell the user the branch and worktree, the test counts, the measured smoke throughput (if the smoke was run), and the next steps listed in RESUME.md; merging or pushing `crafter-reopen` and removing the worktree (`git worktree remove .worktrees/crafter`) are the user's calls (superpowers:finishing-a-development-branch).

---

## Self-review (spec coverage)

- Change 1 (calibration): rule with 3 consecutive points, cap fallback, NaN as 0, G1 clause 2, G2, deep set with the >= 3 floor, `crafter_calibration.json` keys, cost printout with calibration hours and the RND floor note, `budget.steps` guard: Tasks 4 and 5.
- Change 2 (package): `env.py` (Task 2), `stats.py` (Task 1), `ppo.py`/`rnd.py` (Task 3), `calibrate.py` (Task 5), `phase.py` (Task 4), `run.py`, `requirements-crafter.txt`, configs (Tasks 1, 4, 6); isolation test (Task 4).
- Protocol: seeds, grids, matched entropy, objective, tie-break (`tie_key`), provisional-on-edge (via `tune_agent`/`matched_entropy`), 36 + 20 runs (`phase_runs`): Tasks 4 and 5. Secondary metrics (official score, per-achievement rates, time to first unlock of the deep set, held-out score): Task 4 `seed_job`.
- Notebook: Task 7. Tests list in the spec (seed pools, `suggest_budget`, `EpisodeStats`, shapes and frozen RND target, gate helpers, config guards, hash stability, one `crafter`-marked real-env test): Tasks 1 to 5. Pushing needs the user's yes: Task 8 and Global Constraints.
- Known approximations, stated in the code or here: first-unlock steps are stamped when the episode that contains the first unlock FINISHES (up to one episode late, small against S, and relative to the current train call); throughput is the per-run mean of `steps / secs` (clean at `--workers 1`); the held-out evaluation uses sampled, not greedy, actions.
