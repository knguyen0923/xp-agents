# XP Foundation and Phase 1 (Bandit + Tag-Team) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared protocol layer (phase spec, tuner, evaluator, results store, bootstrap stats, plots, CLI, Colab launcher) and the complete Phase 1 changing-bandit study, including the three tag-team handoff rules, producing final degradation curves, gap tables and crossover tables.

**Architecture:** A `Phase` spec (data from YAML) drives a generic tune → freeze → sweep-evaluate procedure that writes one tidy CSV per (agent, knob). Phase 1 is vectorized NumPy: the environment realization is pre-generated per seed so results are deterministic and independent of batching, and agents run vectorized across seeds. Stats and plotting read only the tidy results table.

**Tech Stack:** Python >=3.9, NumPy, pandas, PyYAML, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md` (read it first; this plan implements Sections 3, 4, 5.1, 6, 7 and 9, and amends Sections 4 and 4.1 as noted in Task 1).

**Out of scope for this plan:** MiniGrid / Phase 2 (separate plan, written after a timing spike once this plan is done), CoinRun, Crafter, Showdown.

## Global Constraints

- Python floor is 3.9 (amends the spec's "3.11+"; the laptop has 3.9.6, Colab has 3.10+). Every module starts with `from __future__ import annotations`. No `match`, no `X | Y` in runtime-evaluated annotations, no `zip(strict=)`.
- Dependencies are pinned in `requirements.txt`: numpy==2.0.2, pandas==2.2.3, pyyaml==6.0.2, matplotlib==3.9.2, pytest==8.3.3.
- Bandit: K=10 arms, noise sd 1, arm means drawn from N(0,1), horizon T=2000, knob = jump rate λ, sweep `[0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]`, familiar λ=0.001.
- Seeds: tune pool `[0, 100)`, eval pool `[10000, 11000)`, chunk size 250, agent seed 777. Tune and eval pools must be disjoint.
- Metric: per-step cumulative dynamic regret against the per-step best arm (mean regret per step per seed). Gap = regret(λ) − regret(familiar). Paired seeds.
- Kairi = ε-greedy with constant step size α. Kaiser = discounted UCB (γ, c). Tag-team block length 50.
- Protocol rules raise exceptions (never warn): disjoint seed pools; boundary optimum marks a run `provisional` and blocks final reports; config-hash mismatch on a resumable unit raises.
- Both agents get tuning grids of equal size.
- 95% bootstrap CIs, resampling the independent seed unit, paired differences use shared eval seeds. No number is quoted without its CI.
- Everything must run unattended from a CLI so it works in Colab. Local runs are limited to the fast test suite; full runs go to Colab.
- Follow `CLAUDE.md` once Task 1 creates it. Commit after every task. Do not push to a remote without asking the user first.

## File Structure

```
pyproject.toml  requirements.txt  .gitignore
CLAUDE.md  STATUS.md  RESUME.md  README.md
.claude/settings.json  .claude/hooks/{status-stop-check.sh,post-compact-note.sh}
.github/workflows/ci.yml
configs/bandit.yaml
notebooks/run_phase.ipynb
xp/__init__.py  xp/errors.py  xp/run.py  xp/report.py
xp/protocol/__init__.py  phase.py  tune.py  evaluate.py  results.py
xp/stats/__init__.py  ci.py  summary.py
xp/plots/__init__.py  curves.py
xp/bandit/__init__.py  world.py  agents.py  tagteam.py  phase.py
tests/conftest.py  test_phase.py  test_ci.py  test_world.py  test_agents.py
tests/test_results.py  test_tune.py  test_evaluate.py  test_tagteam.py
tests/test_bandit_phase.py  test_summary.py  test_plots.py
```

Responsibilities: `protocol/*` is phase-agnostic (it never imports `xp.bandit`). `bandit/*` owns everything specific to the bandit. `stats/*` and `plots/*` read only the tidy results DataFrame.

## Shared interfaces (referenced by several tasks)

Tidy results columns (`xp.protocol.results.COLUMNS`): `phase, agent, knob, seed, metric, value, provisional, config_hash`.

Bandit agent protocol (all bandit agents, including the tag-team agent, implement it):
```python
def reset(self, n_envs: int) -> None
def select(self, t: int) -> np.ndarray      # int actions, shape (n_envs,)
def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None
```
`PerSeedFn = Callable[[np.ndarray, float, dict, int], np.ndarray]`, called as `fn(seeds, knob, params, chunk_index)` and returning one metric value per seed.

---

### Task 1: Scaffold, CLAUDE.md, handoff hooks, CI, spec amendments

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `CLAUDE.md`, `STATUS.md`, `RESUME.md`, `.claude/settings.json`, `.claude/hooks/status-stop-check.sh`, `.claude/hooks/post-compact-note.sh`, `.github/workflows/ci.yml`, `xp/__init__.py`, `xp/errors.py`, `xp/protocol/__init__.py`, `xp/stats/__init__.py`, `xp/plots/__init__.py`, `xp/bandit/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Modify: `docs/superpowers/specs/2026-10-22-xp-design.md` (Section 4 stack line, Section 4.1 resumable-unit line)

**Interfaces:**
- Produces: `xp.errors.ProtocolError`, `ConfigMismatchError(ProtocolError)`, `ProvisionalResultError(ProtocolError)`; `tests/conftest.py::tiny_cfg` fixture (dict) used by many later tests.

- [ ] **Step 1: Create the feature branch from the spec branch**

```bash
cd /Users/knguyen/VSC/xp-agents
git status --short   # expect clean
git checkout -b feat/foundation-and-bandit
```

- [ ] **Step 2: Write `pyproject.toml`, `requirements.txt`, `.gitignore`**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "xp"
version = "0.1.0"
description = "XP: explore-vs-exploit study of Kairi (reward-driven) vs Kaiser (uncertainty-driven)"
requires-python = ">=3.9"
dependencies = ["numpy", "pandas", "pyyaml", "matplotlib"]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
include = ["xp*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`requirements.txt`:
```
numpy==2.0.2
pandas==2.2.3
pyyaml==6.0.2
matplotlib==3.9.2
pytest==8.3.3
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
results/
figs/
.ipynb_checkpoints/
.worktrees/
.coverage
.vscode/
```

- [ ] **Step 3: Write package stubs and `xp/errors.py`**

Create empty `xp/__init__.py`, `xp/protocol/__init__.py`, `xp/stats/__init__.py`, `xp/plots/__init__.py`, `xp/bandit/__init__.py`, `tests/__init__.py`.

`xp/errors.py`:
```python
from __future__ import annotations


class ProtocolError(Exception):
    """A research-protocol rule was violated."""


class ConfigMismatchError(ProtocolError):
    """An existing result was produced under a different config."""


class ProvisionalResultError(ProtocolError):
    """A provisional (boundary-optimum) result was used in a final output."""
```

- [ ] **Step 4: Write `tests/conftest.py` with the tiny config fixture**

```python
from __future__ import annotations

import copy

import pytest

_TINY = {
    "phase": "bandit",
    "k": 3,
    "sigma": 1.0,
    "horizon": 50,
    "knob": {"name": "jump_rate", "familiar": 0.01, "sweep": [0.0, 0.01, 0.05]},
    "seeds": {"tune": [0, 10], "eval": [100, 120], "chunk": 10, "agent_seed": 7},
    "agents": {
        "kairi": {
            "grid": {"epsilon": [0.05, 0.2], "alpha": [0.1, 0.5]},
            "natural_limits": {"alpha": [None, 1.0]},
        },
        "kaiser": {
            "grid": {"gamma": [0.9, 0.99], "c": [0.5, 1.0]},
            "natural_limits": {"gamma": [None, 1.0]},
        },
    },
    "tagteam": {
        "fixed": {"block": 10},
        "triggered": {
            "grid": {"delta": [0.8, 1.2], "h": [2.0, 4.0], "window": [5, 10]},
            "natural_limits": {},
        },
        "meta": {
            "block": 10,
            "grid": {"gamma": [0.9, 1.0], "c": [0.5, 1.0]},
            "natural_limits": {"gamma": [None, 1.0]},
        },
    },
}


@pytest.fixture
def tiny_cfg():
    return copy.deepcopy(_TINY)
```

- [ ] **Step 5: Write `CLAUDE.md`, `STATUS.md`, `RESUME.md`**

`CLAUDE.md`:
```markdown
# XP — Instructions for Claude

## What this is

XP studies the explore-vs-exploit tradeoff by comparing two agents across worlds that
shift progressively away from the conditions they were tuned on.

- **Kairi** — model-free, reward-driven learner (ε-greedy in the bandit, plain PPO in MiniGrid).
- **Kaiser** — Kairi plus an explicit uncertainty-driven exploration term (discounted UCB in the
  bandit, PPO + RND bonus in MiniGrid). Defining principle: optimism in the face of uncertainty.

Claim: a map of *when each agent wins and where it breaks*, built on a reusable protocol.
Kairi winning some regimes is a valid, expected result. A small, finished, well-measured result
beats a sprawling one. Design: `docs/superpowers/specs/2026-10-22-xp-design.md`.
Plans: `docs/superpowers/plans/`.

## Session handoff files

- `STATUS.md` — project state snapshot: what's done, what's left, pointers to docs
- `RESUME.md` — exact in-flight state when a session paused mid-task

## Token-budget checkpoint (read this every session)

Watch the `<total_tokens>N tokens left</total_tokens>` figure that appears in
`<system-reminder>` tags. When it drops to roughly **10% of the session's
starting budget** (i.e. ~90% used), stop the current task at the next safe
point and, before doing anything else:

1. Update `RESUME.md` with exactly what's in progress — done so far, what's
   still half-finished (with file paths), and the precise next step. Be
   specific enough that a fresh session with no memory of this conversation
   could pick it up correctly.
2. Update `STATUS.md`'s "Last updated" line, commit hash, and the
   `STATUS_COMMIT` marker comment to match current `git rev-parse --short HEAD`.
3. Tell the user you've paused and saved a resume point, in one sentence.

Do not wait for a hook to force this. A `PostCompact` hook and a `Stop` hook in
`.claude/settings.json` are safety nets (both need `jq` installed).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .
pytest                                   # fast suite, seconds; fine to run locally
python -m xp.run --phase bandit --config configs/bandit.yaml --out results   # full run: use Colab
python -m xp.report --phase bandit --config configs/bandit.yaml --out results --figs figs
```

Full runs go to Colab via `notebooks/run_phase.ipynb`; do not burn the laptop on them.

## Architecture

- `xp/protocol/` is phase-agnostic: `Phase` spec, tuner, evaluator, results store. It must never
  import from a phase package (`xp.bandit`, later `xp.minigrid`).
- `xp/bandit/` is vectorized NumPy. The world realization is pre-generated per seed, so results
  are deterministic regardless of batching. Agents run vectorized across seeds.
- `xp/stats/` and `xp/plots/` read only the tidy results table
  (`phase, agent, knob, seed, metric, value, provisional, config_hash`).

## Research protocol (invariants; code enforces them, never weaken them)

1. Tune on the familiar knob value using tune seeds only. 2. Freeze. 3. Evaluate at every knob
value on eval seeds. 4. Report degradation curves, gaps, crossovers with 95% bootstrap CIs.

- Tune and eval seed pools are disjoint; violating this raises `ProtocolError`.
- A tuned optimum on a grid edge makes the run `provisional`: widen the grid and rerun. Provisional
  results never go in a final plot or table. (Edges at a natural limit, e.g. α=1, γ=1, are exempt.)
- Both agents get equal-sized tuning grids and budgets.
- The unit of resampling is the independent seed; paired comparisons share eval seeds.
- Runs are resumable (one CSV per agent and knob), keyed by config hash. A hash mismatch raises.
- Python floor 3.9; every module starts with `from __future__ import annotations`.
- Scope: bandit and MiniGrid are the core. Do not build Crafter or Showdown.
```

`STATUS.md`:
```markdown
# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 0000000 -->

## Current state

Design spec approved (`docs/superpowers/specs/2026-10-22-xp-design.md`). Implementing the
foundation + Phase 1 plan (`docs/superpowers/plans/2026-10-22-xp-foundation-and-bandit.md`).
No results yet. MiniGrid plan is written after Phase 1 and a timing spike.

## Last updated

2026-10-22
```

`RESUME.md`:
```markdown
# XP — Resume

Paused-at: nothing in progress.
```

- [ ] **Step 6: Copy and adapt the handoff hooks from PikaRAG**

```bash
mkdir -p .claude/hooks
cp ~/VSC/PikaRAG/PikaRAG/.claude/hooks/status-stop-check.sh .claude/hooks/
cp ~/VSC/PikaRAG/PikaRAG/.claude/hooks/post-compact-note.sh .claude/hooks/
chmod +x .claude/hooks/*.sh
cp ~/VSC/PikaRAG/PikaRAG/.claude/settings.json .claude/settings.json
command -v jq || echo "jq missing: install with 'brew install jq' (hooks need it)"
```
Expected: both scripts and `settings.json` copied unchanged (they are project-agnostic). `settings.json` must contain only the `Stop` and `PostCompact` hooks shown in PikaRAG; if it contains anything else, remove it.

- [ ] **Step 7: Write the CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt -e .
      - run: pytest
```

- [ ] **Step 8: Amend the spec for the two decisions made while planning**

In `docs/superpowers/specs/2026-10-22-xp-design.md`:
- Section 4 "Stack" line: replace `Python 3.11+` with `Python >=3.9 (Colab-compatible, laptop has 3.9)`.
- Section 4.1 first bullet: replace `one result file per (agent, config, seed)` with `one result file per (agent, knob) for the bandit, where a whole unit takes seconds; per (agent, config, seed) for deep phases`.

- [ ] **Step 9: Verify the scaffold installs and an empty test run works**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt -e .
pytest
```
Expected: `no tests ran` (exit code 5 is fine here).

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: scaffold project, CLAUDE.md, handoff hooks, CI

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Phase spec, seed pools, config hash

**Files:**
- Create: `xp/protocol/phase.py`, `tests/test_phase.py`

**Interfaces:**
- Consumes: `xp.errors.ProtocolError`; `tiny_cfg` fixture.
- Produces:
  - `load_config(path) -> dict`
  - `config_hash(cfg: dict) -> str` (12 hex chars)
  - `Phase` frozen dataclass: `name: str, knob_name: str, familiar: float, sweep: tuple, tune_seeds: tuple, eval_seeds: tuple, chunk: int, agent_seed: int, config_hash: str, cfg: dict`; classmethod `Phase.from_config(cfg) -> Phase`; methods `assert_eval_seeds(seeds)`, `assert_tune_seeds(seeds)` (both raise `ProtocolError`).

- [ ] **Step 1: Write the failing tests**

`tests/test_phase.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from xp.errors import ProtocolError
from xp.protocol.phase import Phase, config_hash


def test_from_config_builds_seed_pools(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    assert phase.name == "bandit"
    assert phase.tune_seeds == tuple(range(0, 10))
    assert phase.eval_seeds == tuple(range(100, 120))
    assert phase.sweep == (0.0, 0.01, 0.05)
    assert phase.familiar == 0.01
    assert phase.chunk == 10 and phase.agent_seed == 7


def test_overlapping_pools_rejected(tiny_cfg):
    tiny_cfg["seeds"]["eval"] = [5, 20]
    with pytest.raises(ProtocolError):
        Phase.from_config(tiny_cfg)


def test_familiar_must_be_in_sweep(tiny_cfg):
    tiny_cfg["knob"]["familiar"] = 0.3
    with pytest.raises(ProtocolError):
        Phase.from_config(tiny_cfg)


def test_hash_changes_with_config(tiny_cfg):
    h1 = config_hash(tiny_cfg)
    tiny_cfg["horizon"] = 51
    assert config_hash(tiny_cfg) != h1
    assert len(h1) == 12


def test_hash_is_key_order_independent(tiny_cfg):
    reordered = dict(reversed(list(tiny_cfg.items())))
    assert config_hash(reordered) == config_hash(tiny_cfg)


def test_assert_eval_seeds_rejects_tune_seed(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    phase.assert_eval_seeds(np.array([100, 101]))
    with pytest.raises(ProtocolError):
        phase.assert_eval_seeds(np.array([3]))


def test_assert_tune_seeds_rejects_eval_seed(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    phase.assert_tune_seeds(np.array([0, 1]))
    with pytest.raises(ProtocolError):
        phase.assert_tune_seeds(np.array([105]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_phase.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.protocol.phase'`.

- [ ] **Step 3: Implement `xp/protocol/phase.py`**

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from xp.errors import ProtocolError


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Phase:
    name: str
    knob_name: str
    familiar: float
    sweep: tuple
    tune_seeds: tuple
    eval_seeds: tuple
    chunk: int
    agent_seed: int
    config_hash: str
    cfg: dict

    @classmethod
    def from_config(cls, cfg: dict) -> "Phase":
        knob = cfg["knob"]
        sweep = tuple(float(v) for v in knob["sweep"])
        familiar = float(knob["familiar"])
        if familiar not in sweep:
            raise ProtocolError(f"familiar value {familiar} is not in the sweep {sweep}")
        seeds = cfg["seeds"]
        tune = tuple(range(seeds["tune"][0], seeds["tune"][1]))
        evals = tuple(range(seeds["eval"][0], seeds["eval"][1]))
        overlap = set(tune) & set(evals)
        if overlap:
            raise ProtocolError(f"tune and eval seed pools overlap, e.g. {sorted(overlap)[:3]}")
        return cls(
            name=cfg["phase"],
            knob_name=knob["name"],
            familiar=familiar,
            sweep=sweep,
            tune_seeds=tune,
            eval_seeds=evals,
            chunk=int(seeds["chunk"]),
            agent_seed=int(seeds["agent_seed"]),
            config_hash=config_hash(cfg),
            cfg=cfg,
        )

    def assert_eval_seeds(self, seeds) -> None:
        bad = set(int(s) for s in np.asarray(seeds)) - set(self.eval_seeds)
        if bad:
            raise ProtocolError(f"seeds {sorted(bad)[:3]} are not in the eval pool")

    def assert_tune_seeds(self, seeds) -> None:
        bad = set(int(s) for s in np.asarray(seeds)) - set(self.tune_seeds)
        if bad:
            raise ProtocolError(f"seeds {sorted(bad)[:3]} are not in the tune pool")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_phase.py`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/protocol/phase.py tests/test_phase.py
git commit -m "feat: Phase spec with disjoint seed pools and config hash

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Bootstrap confidence intervals

**Files:**
- Create: `xp/stats/ci.py`, `tests/test_ci.py`

**Interfaces:**
- Produces: `bootstrap_ci(x, n_boot=5000, alpha=0.05, seed=0) -> (mean, lo, hi)` (floats); `paired_diff_ci(a, b, **kw) -> (mean, lo, hi)` of `a - b`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ci.py`:
```python
from __future__ import annotations

import numpy as np

from xp.stats.ci import bootstrap_ci, paired_diff_ci


def test_ci_contains_mean_and_is_ordered():
    x = np.random.default_rng(0).normal(2.0, 1.0, size=200)
    mean, lo, hi = bootstrap_ci(x, n_boot=1000)
    assert lo < mean < hi
    assert abs(mean - x.mean()) < 1e-12


def test_ci_is_deterministic_given_seed():
    x = np.arange(50, dtype=float)
    assert bootstrap_ci(x, n_boot=500, seed=3) == bootstrap_ci(x, n_boot=500, seed=3)


def test_ci_width_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    _, lo1, hi1 = bootstrap_ci(rng.normal(size=30), n_boot=1000)
    _, lo2, hi2 = bootstrap_ci(rng.normal(size=3000), n_boot=1000)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_ci_coverage_on_synthetic_data():
    rng = np.random.default_rng(42)
    hits = 0
    reps = 200
    for i in range(reps):
        sample = rng.normal(0.0, 1.0, size=40)
        _, lo, hi = bootstrap_ci(sample, n_boot=400, seed=i)
        hits += lo <= 0.0 <= hi
    assert 0.88 <= hits / reps <= 0.99


def test_paired_diff_ci_detects_shift():
    rng = np.random.default_rng(5)
    base = rng.normal(size=300)
    a = base + 0.5
    b = base
    mean, lo, hi = paired_diff_ci(a, b, n_boot=1000)
    assert abs(mean - 0.5) < 1e-9
    assert lo > 0.4 and hi < 0.6  # pairing removes the shared noise
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ci.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.stats.ci'`.

- [ ] **Step 3: Implement `xp/stats/ci.py`**

```python
from __future__ import annotations

import numpy as np

_BATCH = 500


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, _BATCH):
        m = min(_BATCH, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        means[start:start + m] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, **kw):
    """CI of the mean of a - b for paired samples."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), **kw)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ci.py`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/stats/ci.py tests/test_ci.py
git commit -m "feat: bootstrap and paired-difference confidence intervals

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Bandit world and episode runner

**Files:**
- Create: `xp/bandit/world.py`, `tests/test_world.py`

**Interfaces:**
- Produces:
  - `World` frozen dataclass: `means: np.ndarray` shape `(n, T, k)`, `noise: np.ndarray` shape `(n, T)`.
  - `make_world(seeds, lam: float, horizon: int, k: int, sigma: float) -> World`. Each seed uses its own generator `default_rng(seed)`; the noise stream and the uniform stream that decides jumps do not depend on `lam` (so worlds are coupled across knob values: a jump at λ also happens at any larger λ).
  - `run_agent(agent, world) -> np.ndarray` shape `(n,)`: mean per-step regret per seed. Agent follows the bandit agent protocol in "Shared interfaces".

- [ ] **Step 1: Write the failing tests**

`tests/test_world.py`:
```python
from __future__ import annotations

import numpy as np

from xp.bandit.world import make_world, run_agent


def test_world_shapes():
    w = make_world(np.arange(5), lam=0.01, horizon=40, k=3, sigma=1.0)
    assert w.means.shape == (5, 40, 3)
    assert w.noise.shape == (5, 40)


def test_stationary_when_lambda_zero():
    w = make_world(np.arange(4), lam=0.0, horizon=60, k=4, sigma=1.0)
    assert np.all(w.means == w.means[:, :1, :])


def test_jump_count_matches_rate():
    horizon, n = 2000, 200
    w = make_world(np.arange(n), lam=0.01, horizon=horizon, k=3, sigma=1.0)
    changes = (np.diff(w.means, axis=1) != 0).any(axis=2).sum(axis=1)
    assert abs(changes.mean() - 0.01 * horizon) < 3.0


def test_deterministic_per_seed_independent_of_batching():
    a = make_world(np.array([3, 4, 5]), 0.02, 50, 3, 1.0)
    b = make_world(np.array([4]), 0.02, 50, 3, 1.0)
    assert np.array_equal(a.means[1], b.means[0])
    assert np.array_equal(a.noise[1], b.noise[0])


def test_noise_and_jump_timing_coupled_across_lambda():
    lo = make_world(np.arange(3), 0.01, 200, 3, 1.0)
    hi = make_world(np.arange(3), 0.05, 200, 3, 1.0)
    assert np.array_equal(lo.noise, hi.noise)
    lo_jumps = (np.diff(lo.means, axis=1) != 0).any(axis=2)
    hi_jumps = (np.diff(hi.means, axis=1) != 0).any(axis=2)
    assert np.all(hi_jumps | ~lo_jumps)  # every low-rate jump also occurs at the high rate


class _Oracle:
    def __init__(self, world):
        self.world = world

    def reset(self, n_envs):
        pass

    def select(self, t):
        return self.world.means[:, t, :].argmax(axis=1)

    def update(self, t, actions, rewards):
        pass


def test_oracle_has_zero_regret():
    w = make_world(np.arange(6), 0.02, 80, 4, 1.0)
    regret = run_agent(_Oracle(w), w)
    assert regret.shape == (6,)
    assert np.allclose(regret, 0.0)


class _FixedArm:
    def reset(self, n_envs):
        self.n = n_envs

    def select(self, t):
        return np.zeros(self.n, dtype=int)

    def update(self, t, actions, rewards):
        pass


def test_regret_of_fixed_arm_is_nonnegative_and_positive_on_average():
    w = make_world(np.arange(30), 0.0, 100, 5, 1.0)
    regret = run_agent(_FixedArm(), w)
    assert np.all(regret >= 0)
    assert regret.mean() > 0.1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_world.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.bandit.world'`.

- [ ] **Step 3: Implement `xp/bandit/world.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class World:
    means: np.ndarray  # (n, T, k) true arm means at each step
    noise: np.ndarray  # (n, T) reward noise (already scaled by sigma)


def make_world(seeds, lam: float, horizon: int, k: int, sigma: float) -> World:
    """Pre-generate one realization per seed. Draw order is fixed and independent
    of `lam`, so the same seed gives the same noise and coupled jump times."""
    seeds = np.asarray(seeds)
    n = len(seeds)
    means = np.empty((n, horizon, k))
    noise = np.empty((n, horizon))
    steps = np.arange(1, horizon + 1)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        m0 = rng.normal(size=k)
        u = rng.random(horizon)
        new_means = rng.normal(size=(horizon, k))
        noise[i] = sigma * rng.normal(size=horizon)
        jump = u < lam
        idx = np.maximum.accumulate(np.where(jump, steps, 0))
        source = np.vstack([m0[None, :], new_means])
        means[i] = source[idx]
    return World(means=means, noise=noise)


def run_agent(agent, world: World) -> np.ndarray:
    """Run `agent` on `world`; return mean per-step dynamic regret per seed."""
    n, horizon, _ = world.means.shape
    rows = np.arange(n)
    best = world.means.max(axis=2)
    total = np.zeros(n)
    agent.reset(n)
    for t in range(horizon):
        actions = agent.select(t)
        mu = world.means[rows, t, actions]
        rewards = mu + world.noise[:, t]
        total += best[:, t] - mu
        agent.update(t, actions, rewards)
    return total / horizon
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_world.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/bandit/world.py tests/test_world.py
git commit -m "feat: bandit world generator and vectorized episode runner

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Kairi and Kaiser bandit agents

**Files:**
- Create: `xp/bandit/agents.py`, `tests/test_agents.py`

**Interfaces:**
- Consumes: bandit agent protocol; `make_world`, `run_agent`.
- Produces:
  - `KairiAgent(k: int, epsilon: float, alpha: float, seed: int = 0)`, with attribute `q` of shape `(n, k)` after `reset`.
  - `KaiserAgent(k: int, gamma: float, c: float, seed: int = 0)`, with attributes `cnt`, `sum` of shape `(n, k)` after `reset`.
  Both implement `reset / select / update`. Kairi's `reset` re-seeds its RNG so a run is reproducible.

- [ ] **Step 1: Write the failing tests**

`tests/test_agents.py`:
```python
from __future__ import annotations

import numpy as np

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.world import make_world, run_agent


def test_kairi_greedy_picks_argmax_q():
    agent = KairiAgent(k=3, epsilon=0.0, alpha=0.5, seed=0)
    agent.reset(2)
    agent.q = np.array([[0.0, 1.0, 0.5], [2.0, 0.0, 1.0]])
    assert agent.select(0).tolist() == [1, 0]


def test_kairi_alpha_one_tracks_last_reward():
    agent = KairiAgent(k=3, epsilon=0.0, alpha=1.0, seed=0)
    agent.reset(2)
    agent.update(0, np.array([2, 0]), np.array([0.7, -0.3]))
    assert agent.q[0, 2] == 0.7 and agent.q[1, 0] == -0.3


def test_kairi_constant_step_update():
    agent = KairiAgent(k=2, epsilon=0.0, alpha=0.5, seed=0)
    agent.reset(1)
    agent.update(0, np.array([0]), np.array([1.0]))
    agent.update(1, np.array([0]), np.array([1.0]))
    assert np.isclose(agent.q[0, 0], 0.75)


def test_kairi_epsilon_one_explores_all_arms():
    agent = KairiAgent(k=4, epsilon=1.0, alpha=0.1, seed=1)
    agent.reset(500)
    assert set(agent.select(0).tolist()) == {0, 1, 2, 3}


def test_kairi_reset_makes_runs_reproducible():
    w = make_world(np.arange(10), 0.01, 100, 5, 1.0)
    agent = KairiAgent(5, 0.1, 0.2, seed=3)
    assert np.array_equal(run_agent(agent, w), run_agent(agent, w))


def test_kaiser_pulls_each_arm_once_first():
    agent = KaiserAgent(k=3, gamma=1.0, c=1.0)
    agent.reset(1)
    pulled = []
    for t in range(3):
        a = agent.select(t)
        pulled.append(int(a[0]))
        agent.update(t, a, np.array([0.0]))
    assert sorted(pulled) == [0, 1, 2]


def test_kaiser_with_zero_bonus_exploits_best_arm():
    agent = KaiserAgent(k=3, gamma=1.0, c=0.0)
    agent.reset(1)
    means = np.array([0.0, 1.0, 0.5])
    chosen = []
    for t in range(10):
        a = agent.select(t)
        chosen.append(int(a[0]))
        agent.update(t, a, means[a])
    assert chosen[3:] == [1] * 7


def test_kaiser_discount_forgets_old_evidence():
    agent = KaiserAgent(k=2, gamma=0.5, c=0.0)
    agent.reset(1)
    agent.update(0, np.array([0]), np.array([1.0]))
    before = agent.cnt[0, 0]
    agent.update(1, np.array([1]), np.array([0.0]))
    assert agent.cnt[0, 0] == before * 0.5


def test_both_beat_random_on_stationary_world():
    w = make_world(np.arange(40), 0.0, 500, 5, 1.0)
    rand = run_agent(KairiAgent(5, 1.0, 0.1, seed=0), w).mean()
    kairi = run_agent(KairiAgent(5, 0.1, 0.1, seed=0), w).mean()
    kaiser = run_agent(KaiserAgent(5, 1.0, 1.0), w).mean()
    assert kairi < rand and kaiser < rand
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_agents.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.bandit.agents'`.

- [ ] **Step 3: Implement `xp/bandit/agents.py`**

```python
from __future__ import annotations

import numpy as np


class KairiAgent:
    """Model-free, reward-driven: epsilon-greedy with a constant step-size value update."""

    def __init__(self, k: int, epsilon: float, alpha: float, seed: int = 0):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.seed = seed

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.rng = np.random.default_rng(self.seed)
        self.q = np.zeros((n_envs, self.k))

    def select(self, t: int) -> np.ndarray:
        greedy = np.argmax(self.q + 1e-9 * self.rng.random(self.q.shape), axis=1)
        explore = self.rng.random(self.n) < self.epsilon
        random_arm = self.rng.integers(0, self.k, self.n)
        return np.where(explore, random_arm, greedy)

    def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None:
        rows = np.arange(self.n)
        self.q[rows, actions] += self.alpha * (rewards - self.q[rows, actions])


class KaiserAgent:
    """Optimism under uncertainty: discounted UCB (Garivier & Moulines)."""

    def __init__(self, k: int, gamma: float, c: float, seed: int = 0):
        self.k = k
        self.gamma = gamma
        self.c = c
        self.seed = seed  # unused; kept so factories share one signature

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.cnt = np.zeros((n_envs, self.k))
        self.sum = np.zeros((n_envs, self.k))

    def select(self, t: int) -> np.ndarray:
        total = np.maximum(self.cnt.sum(axis=1, keepdims=True), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            index = self.sum / self.cnt + self.c * np.sqrt(np.log(total) / self.cnt)
        index = np.where(self.cnt <= 0, np.inf, index)
        return np.argmax(index, axis=1)

    def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None:
        self.cnt *= self.gamma
        self.sum *= self.gamma
        rows = np.arange(self.n)
        self.cnt[rows, actions] += 1.0
        self.sum[rows, actions] += rewards
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_agents.py`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/bandit/agents.py tests/test_agents.py
git commit -m "feat: Kairi (epsilon-greedy) and Kaiser (discounted UCB) bandit agents

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Results store (resumable units, tune records, provisional guard)

**Files:**
- Create: `xp/protocol/results.py`, `tests/test_results.py`
- Modify: none (the `TuneResult` class it persists is defined here to avoid a circular import; `tune.py` imports it from here)

**Interfaces:**
- Consumes: `Phase`, `ConfigMismatchError`, `ProvisionalResultError`.
- Produces:
  - `COLUMNS: list[str]`
  - `TuneResult` dataclass: `best_params: dict`, `best_score: float`, `boundary_dims: list`, `table: pd.DataFrame`; property `provisional -> bool` (true iff `boundary_dims` non-empty); classmethod `TuneResult.fixed(params: dict) -> TuneResult` (no tuning, `best_score=nan`, empty `boundary_dims`, empty table).
  - `unit_done(out_dir, phase, agent, knob) -> bool` (raises `ConfigMismatchError` if the file exists with a different hash)
  - `write_unit(out_dir, phase, agent, knob, df) -> None`
  - `load_results(out_dir, phase_name) -> pd.DataFrame`
  - `write_tune(out_dir, phase, agent, result) -> None`
  - `read_tune(out_dir, phase, agent) -> TuneResult | None` (raises `ConfigMismatchError` on hash mismatch)
  - `require_final(df, allow_provisional=False) -> None` (raises `ProvisionalResultError`)
- Layout on disk: `out_dir/<phase>/<agent>__<knob_name>-<knob:g>.csv` for units, `out_dir/<phase>/tuning/<agent>.json` and `<agent>.csv` for tuning.

- [ ] **Step 1: Write the failing tests**

`tests/test_results.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xp.errors import ConfigMismatchError, ProvisionalResultError
from xp.protocol.phase import Phase
from xp.protocol.results import (
    COLUMNS,
    TuneResult,
    load_results,
    read_tune,
    require_final,
    unit_done,
    write_tune,
    write_unit,
)


def _df(phase, agent="kairi", knob=0.01, provisional=False):
    return pd.DataFrame(
        {
            "phase": phase.name,
            "agent": agent,
            "knob": knob,
            "seed": [100, 101],
            "metric": "regret",
            "value": [0.5, 0.6],
            "provisional": provisional,
            "config_hash": phase.config_hash,
        }
    )


def test_unit_roundtrip_and_done(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    assert not unit_done(tmp_path, phase, "kairi", 0.01)
    write_unit(tmp_path, phase, "kairi", 0.01, _df(phase))
    assert unit_done(tmp_path, phase, "kairi", 0.01)
    out = load_results(tmp_path, "bandit")
    assert list(out.columns) == COLUMNS
    assert len(out) == 2


def test_hash_mismatch_raises(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    write_unit(tmp_path, phase, "kairi", 0.01, _df(phase))
    tiny_cfg["horizon"] = 99
    other = Phase.from_config(tiny_cfg)
    with pytest.raises(ConfigMismatchError):
        unit_done(tmp_path, other, "kairi", 0.01)


def test_write_unit_rejects_wrong_columns(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    with pytest.raises(ValueError):
        write_unit(tmp_path, phase, "kairi", 0.01, _df(phase).drop(columns=["value"]))


def test_tune_roundtrip(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    table = pd.DataFrame({"epsilon": [0.05, 0.2], "score": [0.4, 0.5]})
    result = TuneResult({"epsilon": 0.05}, 0.4, ["epsilon"], table)
    assert result.provisional
    assert read_tune(tmp_path, phase, "kairi") is None
    write_tune(tmp_path, phase, "kairi", result)
    back = read_tune(tmp_path, phase, "kairi")
    assert back.best_params == {"epsilon": 0.05}
    assert back.boundary_dims == ["epsilon"] and back.provisional
    assert len(back.table) == 2


def test_tune_hash_mismatch_raises(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    write_tune(tmp_path, phase, "kairi", TuneResult.fixed({"block": 10}))
    tiny_cfg["horizon"] = 99
    with pytest.raises(ConfigMismatchError):
        read_tune(tmp_path, Phase.from_config(tiny_cfg), "kairi")


def test_fixed_tune_result_is_not_provisional():
    r = TuneResult.fixed({"block": 50})
    assert not r.provisional and np.isnan(r.best_score)


def test_require_final_blocks_provisional(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    require_final(_df(phase, provisional=False))
    with pytest.raises(ProvisionalResultError):
        require_final(_df(phase, provisional=True))
    require_final(_df(phase, provisional=True), allow_provisional=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_results.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.protocol.results'`.

- [ ] **Step 3: Implement `xp/protocol/results.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from xp.errors import ConfigMismatchError, ProvisionalResultError
from xp.protocol.phase import Phase

COLUMNS = ["phase", "agent", "knob", "seed", "metric", "value", "provisional", "config_hash"]


@dataclass
class TuneResult:
    best_params: dict
    best_score: float
    boundary_dims: list
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def provisional(self) -> bool:
        return bool(self.boundary_dims)

    @classmethod
    def fixed(cls, params: dict) -> "TuneResult":
        return cls(dict(params), float("nan"), [], pd.DataFrame())


def _unit_path(out_dir, phase: Phase, agent: str, knob: float) -> Path:
    return Path(out_dir) / phase.name / f"{agent}__{phase.knob_name}-{knob:g}.csv"


def _tune_paths(out_dir, phase: Phase, agent: str):
    base = Path(out_dir) / phase.name / "tuning"
    return base / f"{agent}.json", base / f"{agent}.csv"


def unit_done(out_dir, phase: Phase, agent: str, knob: float) -> bool:
    path = _unit_path(out_dir, phase, agent, knob)
    if not path.exists():
        return False
    hashes = list(pd.read_csv(path, usecols=["config_hash"])["config_hash"].unique())
    if hashes != [phase.config_hash]:
        raise ConfigMismatchError(
            f"{path} was written under config {hashes}, current config is {phase.config_hash}"
        )
    return True


def write_unit(out_dir, phase: Phase, agent: str, knob: float, df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError(f"results columns must be {COLUMNS}, got {list(df.columns)}")
    path = _unit_path(out_dir, phase, agent, knob)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(out_dir, phase_name: str) -> pd.DataFrame:
    files = sorted((Path(out_dir) / phase_name).glob("*__*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)[COLUMNS]


def write_tune(out_dir, phase: Phase, agent: str, result: TuneResult) -> None:
    json_path, csv_path = _tune_paths(out_dir, phase, agent)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "config_hash": phase.config_hash,
        "best_params": result.best_params,
        "best_score": None if pd.isna(result.best_score) else result.best_score,
        "boundary_dims": result.boundary_dims,
    }
    json_path.write_text(json.dumps(record, indent=2))
    result.table.to_csv(csv_path, index=False)


def read_tune(out_dir, phase: Phase, agent: str):
    json_path, csv_path = _tune_paths(out_dir, phase, agent)
    if not json_path.exists():
        return None
    record = json.loads(json_path.read_text())
    if record["config_hash"] != phase.config_hash:
        raise ConfigMismatchError(
            f"{json_path} was written under config {record['config_hash']}, "
            f"current config is {phase.config_hash}"
        )
    try:
        table = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        table = pd.DataFrame()
    score = float("nan") if record["best_score"] is None else record["best_score"]
    return TuneResult(record["best_params"], score, record["boundary_dims"], table)


def require_final(df: pd.DataFrame, allow_provisional: bool = False) -> None:
    if allow_provisional or df.empty:
        return
    bad = df.loc[df["provisional"].astype(bool), "agent"].unique()
    if len(bad):
        raise ProvisionalResultError(
            f"provisional (grid-boundary) results for agents {sorted(bad)}: "
            "widen their grids and rerun, or pass allow_provisional for a draft"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_results.py`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/protocol/results.py tests/test_results.py
git commit -m "feat: resumable results store with config-hash and provisional guards

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Tuner with grid-boundary flag

**Files:**
- Create: `xp/protocol/tune.py`, `tests/test_tune.py`

**Interfaces:**
- Consumes: `Phase`, `TuneResult`, `PerSeedFn`.
- Produces:
  - `grid_points(grid: dict[str, list]) -> list[dict]` (Cartesian product, deterministic order, dict-insertion order of `grid`)
  - `boundary_dims(best: dict, grid: dict[str, list], natural_limits: dict | None = None) -> list[str]`. `natural_limits[dim] = [lo_or_None, hi_or_None]`; an edge value equal to a non-None natural limit is exempt. Dimensions with fewer than 2 distinct values are skipped.
  - `tune(score_fn, grid, natural_limits=None) -> TuneResult` (lower score is better)
  - `score_params(phase, per_seed_fn, params) -> float`: mean of `per_seed_fn(chunk, phase.familiar, params, chunk_index)` over the tune pool in chunks of `phase.chunk`; asserts each chunk is in the tune pool.

- [ ] **Step 1: Write the failing tests**

`tests/test_tune.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from xp.errors import ProtocolError
from xp.protocol.phase import Phase
from xp.protocol.tune import boundary_dims, grid_points, score_params, tune


def test_grid_points_is_cartesian_product():
    pts = grid_points({"a": [1, 2], "b": [10, 20, 30]})
    assert len(pts) == 6
    assert pts[0] == {"a": 1, "b": 10} and pts[-1] == {"a": 2, "b": 30}


def test_boundary_detected_on_min_and_max_edges():
    grid = {"a": [1, 2, 3], "b": [10, 20, 30]}
    assert boundary_dims({"a": 2, "b": 20}, grid) == []
    assert boundary_dims({"a": 1, "b": 20}, grid) == ["a"]
    assert boundary_dims({"a": 2, "b": 30}, grid) == ["b"]
    assert boundary_dims({"a": 3, "b": 10}, grid) == ["a", "b"]


def test_natural_limit_is_exempt_but_other_edge_is_not():
    grid = {"alpha": [0.1, 0.5, 1.0]}
    limits = {"alpha": [None, 1.0]}
    assert boundary_dims({"alpha": 1.0}, grid, limits) == []
    assert boundary_dims({"alpha": 0.1}, grid, limits) == ["alpha"]


def test_single_valued_dimension_is_ignored():
    assert boundary_dims({"a": 5}, {"a": [5]}) == []


def test_tune_finds_interior_optimum_without_flag():
    grid = {"eps": [0.01, 0.1, 0.3], "alpha": [0.05, 0.3, 0.9]}
    result = tune(lambda p: (p["eps"] - 0.1) ** 2 + (p["alpha"] - 0.3) ** 2, grid)
    assert result.best_params == {"eps": 0.1, "alpha": 0.3}
    assert not result.provisional
    assert len(result.table) == 9 and "score" in result.table.columns


def test_tune_flags_edge_optimum_as_provisional():
    grid = {"eps": [0.01, 0.1, 0.3]}
    result = tune(lambda p: p["eps"], grid)  # best is the smallest value
    assert result.best_params == {"eps": 0.01}
    assert result.provisional and result.boundary_dims == ["eps"]


def test_tune_best_params_are_plain_python_types():
    result = tune(lambda p: p["x"], {"x": [3, 1, 2]})
    assert type(result.best_params["x"]) is int


def test_score_params_uses_only_tune_seeds_in_chunks(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    calls = []

    def fn(seeds, knob, params, offset):
        calls.append((list(seeds), knob, offset))
        return np.full(len(seeds), 2.0)

    assert score_params(phase, fn, {}) == 2.0
    assert calls == [(list(range(10)), phase.familiar, 0)]


def test_score_params_rejects_per_seed_fn_that_is_handed_wrong_pool(tiny_cfg):
    tiny_cfg["seeds"]["chunk"] = 4
    phase = Phase.from_config(tiny_cfg)
    seen = []

    def fn(seeds, knob, params, offset):
        seen.append(offset)
        return np.zeros(len(seeds))

    score_params(phase, fn, {})
    assert seen == [0, 1, 2]
    with pytest.raises(ProtocolError):
        phase.assert_tune_seeds(np.array(phase.eval_seeds[:2]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_tune.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.protocol.tune'`.

- [ ] **Step 3: Implement `xp/protocol/tune.py`**

```python
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from xp.protocol.phase import Phase
from xp.protocol.results import TuneResult

PerSeedFn = Callable[[np.ndarray, float, dict, int], np.ndarray]


def grid_points(grid: Dict[str, list]) -> List[dict]:
    names = list(grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*(grid[n] for n in names))]


def boundary_dims(best: dict, grid: Dict[str, list], natural_limits: Optional[dict] = None) -> List[str]:
    limits = natural_limits or {}
    flagged = []
    for dim, values in grid.items():
        lo, hi = min(values), max(values)
        if lo == hi:
            continue
        nat_lo, nat_hi = (limits.get(dim) or [None, None])
        value = best[dim]
        at_lo_edge = value == lo and nat_lo != lo
        at_hi_edge = value == hi and nat_hi != hi
        if at_lo_edge or at_hi_edge:
            flagged.append(dim)
    return flagged


def tune(score_fn: Callable[[dict], float], grid: Dict[str, list], natural_limits: Optional[dict] = None) -> TuneResult:
    """Exhaustive grid search; lower score is better."""
    points = grid_points(grid)
    rows = [{**p, "score": float(score_fn(p))} for p in points]
    table = pd.DataFrame(rows)
    best_idx = int(table["score"].idxmin())
    best_params = dict(points[best_idx])
    return TuneResult(
        best_params=best_params,
        best_score=float(table.loc[best_idx, "score"]),
        boundary_dims=boundary_dims(best_params, grid, natural_limits),
        table=table,
    )


def score_params(phase: Phase, per_seed_fn: PerSeedFn, params: dict) -> float:
    """Mean metric at the familiar knob value over the tune-seed pool only."""
    seeds = np.array(phase.tune_seeds)
    values = []
    for chunk_index, start in enumerate(range(0, len(seeds), phase.chunk)):
        chunk = seeds[start:start + phase.chunk]
        phase.assert_tune_seeds(chunk)
        values.append(np.asarray(per_seed_fn(chunk, phase.familiar, params, chunk_index)))
    return float(np.concatenate(values).mean())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_tune.py`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/protocol/tune.py tests/test_tune.py
git commit -m "feat: grid tuner that flags grid-boundary optima as provisional

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Evaluator

**Files:**
- Create: `xp/protocol/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Phase`, `COLUMNS`, `PerSeedFn`.
- Produces: `evaluate_unit(phase, agent, params, knob, provisional, per_seed_fn, metric="regret") -> pd.DataFrame` with columns `COLUMNS` and one row per eval seed. Chunks the eval pool by `phase.chunk`, asserts each chunk is in the eval pool, passes the chunk index as the last argument.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate.py`:
```python
from __future__ import annotations

import numpy as np

from xp.protocol.evaluate import evaluate_unit
from xp.protocol.phase import Phase
from xp.protocol.results import COLUMNS


def test_evaluate_unit_shape_columns_and_chunking(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    calls = []

    def fn(seeds, knob, params, offset):
        calls.append((len(seeds), knob, offset, params))
        return seeds.astype(float) / 1000.0

    df = evaluate_unit(phase, "kairi", {"epsilon": 0.1}, 0.05, True, fn)
    assert list(df.columns) == COLUMNS
    assert len(df) == len(phase.eval_seeds)
    assert df["seed"].tolist() == list(phase.eval_seeds)
    assert set(df["agent"]) == {"kairi"} and set(df["metric"]) == {"regret"}
    assert df["provisional"].all() and set(df["knob"]) == {0.05}
    assert set(df["config_hash"]) == {phase.config_hash}
    assert calls == [(10, 0.05, 0, {"epsilon": 0.1}), (10, 0.05, 1, {"epsilon": 0.1})]
    assert np.allclose(df["value"], np.array(phase.eval_seeds) / 1000.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_evaluate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.protocol.evaluate'`.

- [ ] **Step 3: Implement `xp/protocol/evaluate.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from xp.protocol.phase import Phase
from xp.protocol.results import COLUMNS
from xp.protocol.tune import PerSeedFn


def evaluate_unit(
    phase: Phase,
    agent: str,
    params: dict,
    knob: float,
    provisional: bool,
    per_seed_fn: PerSeedFn,
    metric: str = "regret",
) -> pd.DataFrame:
    """Evaluate one frozen agent at one knob value over the whole eval-seed pool."""
    seeds = np.array(phase.eval_seeds)
    values = []
    for chunk_index, start in enumerate(range(0, len(seeds), phase.chunk)):
        chunk = seeds[start:start + phase.chunk]
        phase.assert_eval_seeds(chunk)
        values.append(np.asarray(per_seed_fn(chunk, knob, params, chunk_index)))
    df = pd.DataFrame(
        {
            "phase": phase.name,
            "agent": agent,
            "knob": knob,
            "seed": seeds,
            "metric": metric,
            "value": np.concatenate(values),
            "provisional": bool(provisional),
            "config_hash": phase.config_hash,
        }
    )
    return df[COLUMNS]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_evaluate.py`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/protocol/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluator that sweeps a frozen agent over the eval-seed pool

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Tag-team controllers and agent

**Files:**
- Create: `xp/bandit/tagteam.py`, `tests/test_tagteam.py`

**Interfaces:**
- Consumes: `KairiAgent`, `KaiserAgent` (attribute `KairiAgent.q`).
- Produces:
  - `FixedSchedule(block: int)`: `reset(n)`, `who(t) -> int array (n,)` (0=Kairi, 1=Kaiser; `(t // block) % 2`), `observe(t, who, rewards, pred)`.
  - `ChangeTriggered(delta: float, h: float, window: int)`: CUSUM on `|reward - pred|`: `g = max(0, g + |e| - delta)`; when `g > h`, Kaiser controls for the next `window` steps and `g` resets. Default controller is Kairi.
  - `MetaBandit(gamma: float, c: float, block: int)`: discounted UCB over {Kairi, Kaiser} at block granularity, using mean block reward.
  - `TagTeamAgent(kairi, kaiser, controller)` implementing the bandit agent protocol. Both agents' `select` is called every step; the controller picks whose action is used; both agents `update` with every (action, reward); `pred` for the controller is Kairi's value estimate for the chosen arm *before* Kairi updates.
- Controller protocol: `reset(n_envs)`, `who(t)`, `observe(t, who, rewards, pred)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tagteam.py`:
```python
from __future__ import annotations

import numpy as np

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.tagteam import ChangeTriggered, FixedSchedule, MetaBandit, TagTeamAgent
from xp.bandit.world import make_world, run_agent


def test_fixed_schedule_alternates_in_blocks():
    c = FixedSchedule(block=3)
    c.reset(2)
    seq = [int(c.who(t)[0]) for t in range(12)]
    assert seq == [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    assert c.who(0).shape == (2,)


def test_change_triggered_starts_with_kairi_and_fires_on_big_residuals():
    c = ChangeTriggered(delta=1.0, h=3.0, window=4)
    c.reset(2)
    assert c.who(0).tolist() == [0, 0]
    pred = np.zeros(2)
    # env 0 sees huge residuals, env 1 sees tiny ones
    for t in range(3):
        c.observe(t, c.who(t), np.array([5.0, 0.1]), pred)
    assert c.who(3).tolist() == [1, 0]


def test_change_triggered_returns_to_kairi_after_window():
    c = ChangeTriggered(delta=0.0, h=1.0, window=2)
    c.reset(1)
    c.observe(0, c.who(0), np.array([2.0]), np.zeros(1))  # fires immediately
    assert c.who(1).tolist() == [1]
    c.observe(1, c.who(1), np.array([0.0]), np.zeros(1))
    c.observe(2, c.who(2), np.array([0.0]), np.zeros(1))
    assert c.who(3).tolist() == [0]


def test_meta_bandit_tries_both_then_prefers_better_agent():
    c = MetaBandit(gamma=1.0, c=0.01, block=2)
    c.reset(1)
    chosen = []
    for t in range(20):
        w = c.who(t)
        chosen.append(int(w[0]))
        reward = 1.0 if w[0] == 1 else 0.0  # Kaiser is better
        c.observe(t, w, np.array([reward]), np.zeros(1))
    assert chosen[0] == 0 and chosen[2] == 1  # explores both first
    assert chosen[-4:] == [1, 1, 1, 1]


class _Spy(KairiAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.updates = 0

    def update(self, t, actions, rewards):
        self.updates += 1
        super().update(t, actions, rewards)


def test_tagteam_updates_both_agents_every_step_even_when_other_chooses():
    kairi = _Spy(k=3, epsilon=0.0, alpha=1.0, seed=0)
    kaiser = KaiserAgent(k=3, gamma=1.0, c=1.0)
    team = TagTeamAgent(kairi, kaiser, FixedSchedule(block=1000))  # Kairi always in charge
    w = make_world(np.arange(4), 0.0, 30, 3, 1.0)
    run_agent(team, w)
    assert kairi.updates == 30
    assert kaiser.cnt.sum() > 0  # Kaiser learned from actions it did not choose


def test_tagteam_passes_pre_update_kairi_prediction_to_controller():
    seen = []

    class Probe(FixedSchedule):
        def observe(self, t, who, rewards, pred):
            seen.append(pred.copy())

    kairi = KairiAgent(k=2, epsilon=0.0, alpha=1.0, seed=0)
    team = TagTeamAgent(kairi, KaiserAgent(2, 1.0, 1.0), Probe(block=1000))
    team.reset(1)
    a = team.select(0)
    team.update(0, a, np.array([0.9]))
    assert seen[0][0] == 0.0  # q was still 0 before the update


def test_tagteam_runs_reproducibly_with_each_rule():
    w = make_world(np.arange(8), 0.02, 120, 4, 1.0)
    for ctrl in (FixedSchedule(10), ChangeTriggered(1.0, 4.0, 10), MetaBandit(0.95, 0.5, 10)):
        team = TagTeamAgent(KairiAgent(4, 0.1, 0.3, seed=2), KaiserAgent(4, 0.95, 1.0), ctrl)
        r1 = run_agent(team, w)
        r2 = run_agent(team, w)
        assert np.array_equal(r1, r2)
        assert r1.shape == (8,)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_tagteam.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.bandit.tagteam'`.

- [ ] **Step 3: Implement `xp/bandit/tagteam.py`**

```python
from __future__ import annotations

import numpy as np

KAIRI, KAISER = 0, 1


class FixedSchedule:
    """Baseline: alternate control between Kairi and Kaiser in fixed blocks."""

    def __init__(self, block: int):
        self.block = block

    def reset(self, n_envs: int) -> None:
        self.n = n_envs

    def who(self, t: int) -> np.ndarray:
        return np.full(self.n, (t // self.block) % 2, dtype=int)

    def observe(self, t, who, rewards, pred) -> None:
        pass


class ChangeTriggered:
    """Kairi by default; a CUSUM change detector on |reward - pred| hands control to
    Kaiser for `window` steps."""

    def __init__(self, delta: float, h: float, window: int):
        self.delta = delta
        self.h = h
        self.window = int(window)

    def reset(self, n_envs: int) -> None:
        self.g = np.zeros(n_envs)
        self.remaining = np.zeros(n_envs, dtype=int)

    def who(self, t: int) -> np.ndarray:
        return (self.remaining > 0).astype(int)

    def observe(self, t, who, rewards, pred) -> None:
        self.g = np.maximum(0.0, self.g + np.abs(rewards - pred) - self.delta)
        fire = self.g > self.h
        self.remaining = np.where(fire, self.window, np.maximum(self.remaining - 1, 0))
        self.g = np.where(fire, 0.0, self.g)


class MetaBandit:
    """Learned meta-controller: discounted UCB over {Kairi, Kaiser}, one choice per block,
    rewarded by the mean reward earned during the block."""

    def __init__(self, gamma: float, c: float, block: int):
        self.gamma = gamma
        self.c = c
        self.block = int(block)

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.cnt = np.zeros((n_envs, 2))
        self.sum = np.zeros((n_envs, 2))
        self.cur = np.zeros(n_envs, dtype=int)
        self.acc = np.zeros(n_envs)

    def who(self, t: int) -> np.ndarray:
        if t % self.block == 0:
            total = np.maximum(self.cnt.sum(axis=1, keepdims=True), 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                index = self.sum / self.cnt + self.c * np.sqrt(np.log(total) / self.cnt)
            index = np.where(self.cnt <= 0, np.inf, index)
            self.cur = np.argmax(index, axis=1)
        return self.cur

    def observe(self, t, who, rewards, pred) -> None:
        self.acc += rewards
        if (t + 1) % self.block == 0:
            rows = np.arange(self.n)
            self.cnt *= self.gamma
            self.sum *= self.gamma
            self.cnt[rows, self.cur] += 1.0
            self.sum[rows, self.cur] += self.acc / self.block
            self.acc = np.zeros(self.n)


class TagTeamAgent:
    """One shared arm choice per step. Both agents propose every step and learn from every
    (arm, reward) pair, whoever chose the arm; the controller decides whose proposal is used."""

    def __init__(self, kairi, kaiser, controller):
        self.kairi = kairi
        self.kaiser = kaiser
        self.controller = controller

    def reset(self, n_envs: int) -> None:
        self.n = n_envs
        self.kairi.reset(n_envs)
        self.kaiser.reset(n_envs)
        self.controller.reset(n_envs)
        self._who = np.zeros(n_envs, dtype=int)

    def select(self, t: int) -> np.ndarray:
        proposal_kairi = self.kairi.select(t)
        proposal_kaiser = self.kaiser.select(t)
        self._who = self.controller.who(t)
        return np.where(self._who == KAISER, proposal_kaiser, proposal_kairi)

    def update(self, t: int, actions: np.ndarray, rewards: np.ndarray) -> None:
        pred = self.kairi.q[np.arange(self.n), actions].copy()
        self.kairi.update(t, actions, rewards)
        self.kaiser.update(t, actions, rewards)
        self.controller.observe(t, self._who, rewards, pred)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_tagteam.py`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/bandit/tagteam.py tests/test_tagteam.py
git commit -m "feat: tag-team agent with fixed, change-triggered and meta-bandit handoff

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Bandit phase orchestration, CLI, real config

**Files:**
- Create: `xp/bandit/phase.py`, `xp/run.py`, `configs/bandit.yaml`, `tests/test_bandit_phase.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces:
  - `build_factories(cfg) -> dict[str, Callable[[dict, int], agent]]` for `"kairi"`, `"kaiser"`.
  - `make_per_seed_fn(cfg, build_agent) -> PerSeedFn` (builds the world for the chunk at the given knob, builds the agent with seed `cfg["seeds"]["agent_seed"] + chunk_index`, returns `run_agent` result).
  - `build_tagteam(cfg, rule, base) -> Callable[[dict, int], TagTeamAgent]` where `rule` is `"fixed" | "triggered" | "meta"` and `base = {"kairi": {...}, "kaiser": {...}}` holds the frozen single-agent params.
  - `run_phase(cfg: dict, out_dir) -> None`: tunes kairi and kaiser (resumable), tunes the tag-team controllers using those frozen params, then evaluates all five agents (`kairi, kaiser, tt_fixed, tt_triggered, tt_meta`) at every knob value; each (agent, knob) unit is skipped if already done. A tag-team agent is provisional if its own tuning is provisional **or** either base agent's tuning is.
  - CLI: `python -m xp.run --phase bandit --config PATH --out DIR`.

- [ ] **Step 1: Write the real config `configs/bandit.yaml`**

```yaml
phase: bandit
k: 10
sigma: 1.0
horizon: 2000
knob:
  name: jump_rate
  familiar: 0.001
  sweep: [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
seeds:
  tune: [0, 100]
  eval: [10000, 11000]
  chunk: 250
  agent_seed: 777
agents:
  kairi:
    grid:
      epsilon: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
      alpha: [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    natural_limits:
      alpha: [null, 1.0]
  kaiser:
    grid:
      gamma: [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
      c: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    natural_limits:
      gamma: [null, 1.0]
tagteam:
  fixed:
    block: 50
  triggered:
    grid:
      delta: [0.8, 1.0, 1.2, 1.5, 2.0]
      h: [2.0, 4.0, 8.0, 16.0]
      window: [10, 25, 50, 100]
    natural_limits: {}
  meta:
    block: 50
    grid:
      gamma: [0.8, 0.9, 0.95, 0.99, 1.0]
      c: [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    natural_limits:
      gamma: [null, 1.0]
```

Note: Kairi and Kaiser grids are both 7×7 = 49 points (equal size, per the protocol).

- [ ] **Step 2: Write the failing tests**

`tests/test_bandit_phase.py`:
```python
from __future__ import annotations

import os

import numpy as np
import pytest

from xp.bandit.phase import build_factories, build_tagteam, make_per_seed_fn, run_phase
from xp.bandit.tagteam import TagTeamAgent
from xp.errors import ConfigMismatchError
from xp.protocol.phase import Phase, load_config
from xp.protocol.results import load_results, read_tune

AGENTS = {"kairi", "kaiser", "tt_fixed", "tt_triggered", "tt_meta"}


def test_real_config_is_valid_and_grids_are_equal_size():
    cfg = load_config("configs/bandit.yaml")
    Phase.from_config(cfg)
    n = lambda spec: int(np.prod([len(v) for v in spec["grid"].values()]))
    assert n(cfg["agents"]["kairi"]) == n(cfg["agents"]["kaiser"])


def test_per_seed_fn_is_deterministic_and_per_seed_independent_of_batching(tiny_cfg):
    f = build_factories(tiny_cfg)["kairi"]
    fn = make_per_seed_fn(tiny_cfg, f)
    params = {"epsilon": 0.1, "alpha": 0.3}
    a = fn(np.array([100, 101, 102]), 0.01, params, 0)
    b = fn(np.array([100, 101, 102]), 0.01, params, 0)
    assert np.array_equal(a, b)


def test_build_tagteam_returns_agent_for_each_rule(tiny_cfg):
    base = {"kairi": {"epsilon": 0.1, "alpha": 0.3}, "kaiser": {"gamma": 0.95, "c": 1.0}}
    for rule, params in (
        ("fixed", {"block": 10}),
        ("triggered", {"delta": 1.0, "h": 4.0, "window": 5}),
        ("meta", {"gamma": 0.9, "c": 0.5}),
    ):
        agent = build_tagteam(tiny_cfg, rule, base)(params, 0)
        assert isinstance(agent, TagTeamAgent)


def test_run_phase_end_to_end_then_resume_then_mismatch(tiny_cfg, tmp_path):
    run_phase(tiny_cfg, tmp_path)
    phase = Phase.from_config(tiny_cfg)
    df = load_results(tmp_path, "bandit")
    assert set(df["agent"]) == AGENTS
    assert set(df["knob"]) == set(phase.sweep)
    assert len(df) == len(AGENTS) * len(phase.sweep) * len(phase.eval_seeds)
    assert np.isfinite(df["value"]).all() and (df["value"] >= 0).all()
    for name in ("kairi", "kaiser", "tt_triggered", "tt_meta"):
        assert read_tune(tmp_path, phase, name) is not None
    # tiny grids always put the optimum on an edge, so everything tuned must be provisional
    assert df.loc[df["agent"] == "kairi", "provisional"].all()

    unit = tmp_path / "bandit" / "kairi__jump_rate-0.01.csv"
    before = os.path.getmtime(unit)
    run_phase(tiny_cfg, tmp_path)  # resume: nothing recomputed
    assert os.path.getmtime(unit) == before

    tiny_cfg["horizon"] = 60
    with pytest.raises(ConfigMismatchError):
        run_phase(tiny_cfg, tmp_path)


def test_tagteam_inherits_provisional_from_base_agents(tiny_cfg, tmp_path):
    run_phase(tiny_cfg, tmp_path)
    df = load_results(tmp_path, "bandit")
    assert df.loc[df["agent"] == "tt_fixed", "provisional"].all()  # no own tuning, base is provisional
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_bandit_phase.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.bandit.phase'`.

- [ ] **Step 4: Implement `xp/bandit/phase.py`**

```python
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.tagteam import ChangeTriggered, FixedSchedule, MetaBandit, TagTeamAgent
from xp.bandit.world import make_world, run_agent
from xp.protocol.evaluate import evaluate_unit
from xp.protocol.phase import Phase
from xp.protocol.results import TuneResult, read_tune, unit_done, write_tune, write_unit
from xp.protocol.tune import PerSeedFn, score_params, tune

AgentBuilder = Callable[[dict, int], object]


def build_factories(cfg: dict) -> Dict[str, AgentBuilder]:
    k = cfg["k"]
    return {
        "kairi": lambda p, s: KairiAgent(k, p["epsilon"], p["alpha"], s),
        "kaiser": lambda p, s: KaiserAgent(k, p["gamma"], p["c"], s),
    }


def make_per_seed_fn(cfg: dict, build_agent: AgentBuilder) -> PerSeedFn:
    def fn(seeds, knob, params, chunk_index):
        world = make_world(seeds, knob, cfg["horizon"], cfg["k"], cfg["sigma"])
        agent = build_agent(params, cfg["seeds"]["agent_seed"] + chunk_index)
        return run_agent(agent, world)

    return fn


def build_tagteam(cfg: dict, rule: str, base: dict) -> AgentBuilder:
    k = cfg["k"]

    def build(params: dict, seed: int) -> TagTeamAgent:
        kairi = KairiAgent(k, base["kairi"]["epsilon"], base["kairi"]["alpha"], seed)
        kaiser = KaiserAgent(k, base["kaiser"]["gamma"], base["kaiser"]["c"], seed)
        if rule == "fixed":
            controller = FixedSchedule(params["block"])
        elif rule == "triggered":
            controller = ChangeTriggered(params["delta"], params["h"], params["window"])
        elif rule == "meta":
            controller = MetaBandit(params["gamma"], params["c"], cfg["tagteam"]["meta"]["block"])
        else:
            raise ValueError(f"unknown tag-team rule {rule!r}")
        return TagTeamAgent(kairi, kaiser, controller)

    return build


def _tune_or_load(phase, out_dir, name, per_seed_fn, spec) -> TuneResult:
    cached = read_tune(out_dir, phase, name)
    if cached is not None:
        return cached
    result = tune(lambda p: score_params(phase, per_seed_fn, p), spec["grid"], spec.get("natural_limits"))
    write_tune(out_dir, phase, name, result)
    return result


def run_phase(cfg: dict, out_dir) -> None:
    phase = Phase.from_config(cfg)
    factories = build_factories(cfg)

    tuned: Dict[str, TuneResult] = {}
    for name in ("kairi", "kaiser"):
        fn = make_per_seed_fn(cfg, factories[name])
        tuned[name] = _tune_or_load(phase, out_dir, name, fn, cfg["agents"][name])
    base = {name: tuned[name].best_params for name in tuned}
    base_provisional = any(tuned[name].provisional for name in tuned)

    tt = cfg["tagteam"]
    tag_results: Dict[str, TuneResult] = {"tt_fixed": TuneResult.fixed(tt["fixed"])}
    for rule in ("triggered", "meta"):
        name = f"tt_{rule}"
        fn = make_per_seed_fn(cfg, build_tagteam(cfg, rule, base))
        tag_results[name] = _tune_or_load(phase, out_dir, name, fn, tt[rule])
    write_tune(out_dir, phase, "tt_fixed", tag_results["tt_fixed"])

    plan = [(name, make_per_seed_fn(cfg, factories[name]), tuned[name]) for name in ("kairi", "kaiser")]
    for rule in ("fixed", "triggered", "meta"):
        name = f"tt_{rule}"
        plan.append((name, make_per_seed_fn(cfg, build_tagteam(cfg, rule, base)), tag_results[name]))

    for name, fn, result in plan:
        is_tag_team = name.startswith("tt_")
        provisional = result.provisional or (is_tag_team and base_provisional)
        for knob in phase.sweep:
            if unit_done(out_dir, phase, name, knob):
                continue
            df = evaluate_unit(phase, name, result.best_params, knob, provisional, fn)
            write_unit(out_dir, phase, name, knob, df)
            print(f"[{phase.name}] {name} {phase.knob_name}={knob:g} mean regret {df['value'].mean():.4f}", flush=True)
```

- [ ] **Step 5: Implement `xp/run.py`**

```python
from __future__ import annotations

import argparse

from xp.protocol.phase import load_config

PHASES = {"bandit": "xp.bandit.phase"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run an XP phase (tune, freeze, sweep-evaluate).")
    parser.add_argument("--phase", required=True, choices=sorted(PHASES))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if cfg["phase"] != args.phase:
        raise SystemExit(f"config phase {cfg['phase']!r} does not match --phase {args.phase!r}")
    module = __import__(PHASES[args.phase], fromlist=["run_phase"])
    module.run_phase(cfg, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_bandit_phase.py`
Expected: 5 passed (takes a few seconds; the tiny config is small).

- [ ] **Step 7: Run the whole suite**

Run: `pytest`
Expected: all tests pass (about 60), nothing slow.

- [ ] **Step 8: Commit**

```bash
git add xp/bandit/phase.py xp/run.py configs/bandit.yaml tests/test_bandit_phase.py
git commit -m "feat: bandit phase orchestration, resumable CLI, real config

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Summary statistics, plots, report CLI

**Files:**
- Create: `xp/stats/summary.py`, `xp/plots/curves.py`, `xp/report.py`, `tests/test_summary.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: `bootstrap_ci`, `paired_diff_ci`, results DataFrame (`COLUMNS`), `require_final`.
- Produces (`xp.stats.summary`; all take the tidy results DataFrame `df`, filter `metric == "regret"` by default, and require each (agent, knob) to cover the same seed set):
  - `curve_table(df, n_boot=2000, seed=0) -> DataFrame[agent, knob, mean, lo, hi]`
  - `gap_table(df, familiar, n_boot=2000, seed=0) -> DataFrame[agent, knob, gap, lo, hi]` (per-seed `value(knob) - value(familiar)`, paired by seed; the familiar row is included and is exactly 0 with CI [0, 0])
  - `pair_table(df, a, b, n_boot=2000, seed=0) -> DataFrame[knob, diff, lo, hi, winner]` where `diff = a - b` paired by seed; `winner` is `a` if `hi < 0`, `b` if `lo > 0`, else `"tie"` (lower regret wins)
  - `crossings(pair: DataFrame) -> list[tuple[float, float]]`: knob intervals between consecutive non-tie winners that differ
  - `pair_vs_best_single(df, team, singles=("kairi","kaiser"), n_boot=2000, seed=0) -> DataFrame[knob, diff, lo, hi, winner]` comparing `team` against whichever single agent has the lower mean regret at that knob (`winner` is `team`, the best single's name, or `"tie"`)
- Produces (`xp.plots.curves`): `plot_degradation(curves, familiar, out_path, title)` writes a PNG with categorical x positions, a CI band per agent, and a dashed line at the familiar knob.
- Produces (`xp.report`): CLI `python -m xp.report --phase bandit --config PATH --out RESULTS_DIR --figs FIGS_DIR [--allow-provisional]` writing `curves.csv`, `gaps.csv`, `crossovers.csv`, `pair_vs_best_single.csv`, `degradation.png` into `FIGS_DIR`.

- [ ] **Step 1: Write the failing tests**

`tests/test_summary.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from xp.stats.summary import crossings, curve_table, gap_table, pair_table, pair_vs_best_single

SEEDS = np.arange(200)


def _rows(agent, knob, values):
    return pd.DataFrame(
        {
            "phase": "bandit", "agent": agent, "knob": knob, "seed": SEEDS,
            "metric": "regret", "value": values, "provisional": False, "config_hash": "x",
        }
    )


def _synthetic():
    rng = np.random.default_rng(0)
    shared = {k: rng.normal(size=len(SEEDS)) * 0.05 for k in (0.0, 0.1, 0.2)}
    parts = []
    # A is better at 0.0 and 0.1, B is better at 0.2 (crossover between 0.1 and 0.2)
    means_a = {0.0: 0.2, 0.1: 0.3, 0.2: 0.9}
    means_b = {0.0: 0.5, 0.1: 0.4, 0.2: 0.5}
    for k in (0.0, 0.1, 0.2):
        parts.append(_rows("A", k, means_a[k] + shared[k]))
        parts.append(_rows("B", k, means_b[k] + shared[k]))
    return pd.concat(parts, ignore_index=True)


def test_curve_table_means_and_ordering():
    t = curve_table(_synthetic(), n_boot=300)
    assert set(t.columns) == {"agent", "knob", "mean", "lo", "hi"}
    assert len(t) == 6
    assert (t["lo"] <= t["mean"]).all() and (t["mean"] <= t["hi"]).all()
    a0 = t[(t.agent == "A") & (t.knob == 0.0)].iloc[0]
    assert abs(a0["mean"] - 0.2) < 0.02


def test_gap_table_is_zero_at_familiar_and_positive_for_growing_regret():
    t = gap_table(_synthetic(), familiar=0.0, n_boot=300)
    zero = t[(t.agent == "A") & (t.knob == 0.0)].iloc[0]
    assert zero["gap"] == 0.0 and zero["lo"] == 0.0 and zero["hi"] == 0.0
    big = t[(t.agent == "A") & (t.knob == 0.2)].iloc[0]
    assert big["gap"] > 0.5 and big["lo"] > 0.5


def test_pair_table_winners_and_crossing():
    t = pair_table(_synthetic(), "A", "B", n_boot=300)
    assert t.sort_values("knob")["winner"].tolist() == ["A", "A", "B"]
    assert crossings(t) == [(0.1, 0.2)]


def test_crossings_skips_ties_and_reports_none_when_one_agent_dominates():
    t = pd.DataFrame({"knob": [0.0, 0.1, 0.2], "winner": ["A", "tie", "A"]})
    assert crossings(t) == []
    t2 = pd.DataFrame({"knob": [0.0, 0.1, 0.2, 0.3], "winner": ["A", "tie", "tie", "B"]})
    assert crossings(t2) == [(0.0, 0.3)]


def test_pair_vs_best_single_picks_better_single_per_knob():
    df = _synthetic()
    team_vals = np.where(True, 0.35, 0.0) + np.zeros(len(SEEDS))
    team = pd.concat([_rows("T", k, team_vals) for k in (0.0, 0.1, 0.2)], ignore_index=True)
    out = pair_vs_best_single(pd.concat([df, team], ignore_index=True), "T", singles=("A", "B"), n_boot=300)
    row0 = out[out.knob == 0.0].iloc[0]  # best single is A (0.2); team (0.35) is worse
    assert row0["winner"] == "A"
    row2 = out[out.knob == 0.2].iloc[0]  # best single is B (0.5); team (0.35) is better
    assert row2["winner"] == "T"
```

`tests/test_plots.py`:
```python
from __future__ import annotations

import pandas as pd

from xp.plots.curves import plot_degradation


def test_plot_degradation_writes_png(tmp_path):
    curves = pd.DataFrame(
        {
            "agent": ["A", "A", "B", "B"],
            "knob": [0.0, 0.1, 0.0, 0.1],
            "mean": [0.2, 0.3, 0.5, 0.4],
            "lo": [0.19, 0.28, 0.48, 0.38],
            "hi": [0.21, 0.32, 0.52, 0.42],
        }
    )
    out = tmp_path / "plot.png"
    plot_degradation(curves, familiar=0.0, out_path=out, title="t")
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_summary.py tests/test_plots.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'xp.stats.summary'`.

- [ ] **Step 3: Implement `xp/stats/summary.py`**

```python
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from xp.stats.ci import bootstrap_ci, paired_diff_ci


def _values(df: pd.DataFrame, metric: str = "regret") -> Dict[Tuple[str, float], np.ndarray]:
    """{(agent, knob): per-seed values sorted by seed}; checks every cell has the same seeds."""
    out: Dict[Tuple[str, float], np.ndarray] = {}
    seed_sets = []
    for (agent, knob), g in df[df["metric"] == metric].groupby(["agent", "knob"]):
        g = g.sort_values("seed")
        out[(agent, float(knob))] = g["value"].to_numpy(dtype=float)
        seed_sets.append(tuple(g["seed"]))
    if len(set(seed_sets)) > 1:
        raise ValueError("every (agent, knob) must cover the same eval seeds to be paired")
    return out


def curve_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for (agent, knob), v in sorted(_values(df).items()):
        mean, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
        rows.append({"agent": agent, "knob": knob, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def gap_table(df: pd.DataFrame, familiar: float, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    vals = _values(df)
    rows = []
    for (agent, knob), v in sorted(vals.items()):
        ref = vals[(agent, float(familiar))]
        mean, lo, hi = paired_diff_ci(v, ref, n_boot=n_boot, seed=seed)
        rows.append({"agent": agent, "knob": knob, "gap": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _winner(lo: float, hi: float, a: str, b: str) -> str:
    if hi < 0:
        return a
    if lo > 0:
        return b
    return "tie"


def pair_table(df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    vals = _values(df)
    knobs = sorted({k for (_, k) in vals})
    rows = []
    for knob in knobs:
        mean, lo, hi = paired_diff_ci(vals[(a, knob)], vals[(b, knob)], n_boot=n_boot, seed=seed)
        rows.append({"knob": knob, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, a, b)})
    return pd.DataFrame(rows)


def crossings(pair: pd.DataFrame) -> List[Tuple[float, float]]:
    decided = [(float(r.knob), r.winner) for r in pair.sort_values("knob").itertuples() if r.winner != "tie"]
    return [(k0, k1) for (k0, w0), (k1, w1) in zip(decided, decided[1:]) if w0 != w1]


def pair_vs_best_single(
    df: pd.DataFrame, team: str, singles=("kairi", "kaiser"), n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Compare `team` with whichever single agent has the lower mean regret at each knob."""
    vals = _values(df)
    knobs = sorted({k for (_, k) in vals})
    rows = []
    for knob in knobs:
        best = min(singles, key=lambda s: vals[(s, knob)].mean())
        mean, lo, hi = paired_diff_ci(vals[(team, knob)], vals[(best, knob)], n_boot=n_boot, seed=seed)
        rows.append(
            {"knob": knob, "best_single": best, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, team, best)}
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement `xp/plots/curves.py`**

Before writing the final styling, load the `dataviz` skill and follow its palette, axis and legend rules. The code below is the working structure that skill's rules should be applied to.

```python
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_degradation(curves, familiar: float, out_path, title: str = "Regret vs jump rate") -> None:
    knobs = sorted(curves["knob"].unique())
    pos = {k: i for i, k in enumerate(knobs)}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for agent, g in curves.groupby("agent"):
        g = g.sort_values("knob")
        x = [pos[k] for k in g["knob"]]
        ax.plot(x, g["mean"], marker="o", label=agent)
        ax.fill_between(x, g["lo"], g["hi"], alpha=0.2)
    ax.axvline(pos[familiar], linestyle="--", linewidth=1, color="gray")
    ax.set_xticks(range(len(knobs)))
    ax.set_xticklabels([f"{k:g}" for k in knobs])
    ax.set_xlabel("jump rate λ (dashed line = familiar)")
    ax.set_ylabel("mean per-step regret (95% CI)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 5: Implement `xp/report.py`**

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xp.plots.curves import plot_degradation
from xp.protocol.phase import Phase, load_config
from xp.protocol.results import load_results, require_final
from xp.stats.summary import crossings, curve_table, gap_table, pair_table, pair_vs_best_single


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build final tables and figures from results.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="results directory written by xp.run")
    parser.add_argument("--figs", required=True, help="directory for tables and figures")
    parser.add_argument("--allow-provisional", action="store_true", help="draft only: include provisional results")
    args = parser.parse_args(argv)

    phase = Phase.from_config(load_config(args.config))
    df = load_results(args.out, args.phase)
    if df.empty:
        raise SystemExit(f"no results found under {args.out}/{args.phase}")
    require_final(df, allow_provisional=args.allow_provisional)

    figs = Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    curves = curve_table(df)
    curves.to_csv(figs / "curves.csv", index=False)
    gap_table(df, phase.familiar).to_csv(figs / "gaps.csv", index=False)

    pair = pair_table(df, "kairi", "kaiser")
    pair["crossover_intervals"] = str(crossings(pair))
    pair.to_csv(figs / "crossovers.csv", index=False)

    team_rows = []
    for team in sorted(a for a in df["agent"].unique() if a.startswith("tt_")):
        t = pair_vs_best_single(df, team)
        t.insert(0, "team", team)
        team_rows.append(t)
    if team_rows:
        pd.concat(team_rows).to_csv(figs / "pair_vs_best_single.csv", index=False)

    plot_degradation(curves, phase.familiar, figs / "degradation.png")
    print(f"wrote tables and figure to {figs}")
    print(f"kairi-vs-kaiser crossover intervals: {crossings(pair) or 'none (one agent dominates, or all ties)'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_summary.py tests/test_plots.py`
Expected: 6 passed.

- [ ] **Step 7: Add a report smoke test to `tests/test_bandit_phase.py` and run it**

Append:
```python
def test_report_cli_blocks_provisional_then_writes_with_flag(tiny_cfg, tmp_path):
    import yaml

    from xp.errors import ProvisionalResultError
    from xp.report import main as report_main

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_cfg))
    run_phase(tiny_cfg, tmp_path / "res")
    argv = ["--phase", "bandit", "--config", str(cfg_path), "--out", str(tmp_path / "res"), "--figs", str(tmp_path / "figs")]
    with pytest.raises(ProvisionalResultError):
        report_main(argv)
    report_main(argv + ["--allow-provisional"])
    for name in ("curves.csv", "gaps.csv", "crossovers.csv", "pair_vs_best_single.csv", "degradation.png"):
        assert (tmp_path / "figs" / name).exists()
```
Run: `pytest tests/test_bandit_phase.py`
Expected: 6 passed.

- [ ] **Step 8: Run the whole suite**

Run: `pytest`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add xp/stats/summary.py xp/plots/curves.py xp/report.py tests/test_summary.py tests/test_plots.py tests/test_bandit_phase.py
git commit -m "feat: curve, gap, crossover and pair-vs-best-single tables, plot, report CLI

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Colab launcher, README, final Phase 1 run

**Files:**
- Create: `notebooks/run_phase.ipynb`, `README.md`
- Modify: `STATUS.md` (end-of-task refresh)

**Interfaces:**
- Consumes: CLIs `xp.run` and `xp.report`.
- Produces: the finished Phase 1 results (`results/`, `figs/` on Drive; copies of the figure and tables committed under `docs/results/phase1/`) and a README that states the protocol and the findings.

- [ ] **Step 1: Ask the user for the GitHub remote, and confirm before pushing**

Pushing is outward-facing. Ask the user: "Create the GitHub repo and push `feat/foundation-and-bandit`?" Do not push without a yes. Once a remote exists, record its URL:

```bash
git remote get-url origin
```
If the user declines a remote, the notebook can instead be used by uploading the repo folder to Drive; note that choice in `STATUS.md` and skip Step 2's `REPO_URL` substitution.

- [ ] **Step 2: Write `notebooks/run_phase.ipynb`**

Write this exact JSON, then replace `REPO_URL_HERE` with the remote URL from Step 1 (or, if there is no remote, replace the clone cell as noted in Step 1):

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# XP: run a phase on Colab\n", "Clones the repo, installs pinned deps, runs a phase (resumable) with results on Drive, then builds the report.\n", "If the runtime disconnects, rerun all cells: finished units are skipped."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["from google.colab import drive\n", "drive.mount('/content/drive')\n", "RESULTS = '/content/drive/MyDrive/xp-results'\n", "FIGS = '/content/drive/MyDrive/xp-figs'\n", "PHASE = 'bandit'"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["REPO_URL = 'REPO_URL_HERE'\n", "!rm -rf xp-agents && git clone --depth 1 $REPO_URL xp-agents\n", "%cd xp-agents\n", "!pip install -q -r requirements.txt -e ."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python -m xp.run --phase $PHASE --config configs/$PHASE.yaml --out $RESULTS"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python -m xp.report --phase $PHASE --config configs/$PHASE.yaml --out $RESULTS --figs $FIGS\n", "from IPython.display import Image\n", "Image(FIGS + '/degradation.png')"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
 "nbformat": 4,
 "nbformat_minor": 5
}
```

Verify it parses:
```bash
python3 -c "import json; nb=json.load(open('notebooks/run_phase.ipynb')); print(len(nb['cells']), 'cells'); assert 'REPO_URL_HERE' not in open('notebooks/run_phase.ipynb').read() or print('REMINDER: set REPO_URL')"
```
Expected: `5 cells`, and no REMINDER once the URL is set.

- [ ] **Step 3: Commit the notebook, and (only after the user said yes in Step 1) push**

```bash
git add notebooks/run_phase.ipynb
git commit -m "feat: Colab launcher notebook

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
# only if the user approved a remote in Step 1:
git push -u origin feat/foundation-and-bandit
```

- [ ] **Step 4: Run Phase 1 on Colab**

Ask the user to open `notebooks/run_phase.ipynb` in Colab (CPU runtime is fine) and run all cells. Expected: progress lines `[bandit] kairi jump_rate=0 mean regret ...`, then `wrote tables and figure to ...`, then the figure.

**If `xp.report` raises `ProvisionalResultError`:** the tuned optimum for some agent is on a grid edge. Read `results/bandit/tuning/<agent>.json` (`boundary_dims`) from Drive, widen that agent's grid in `configs/bandit.yaml` on that dimension (keep Kairi's and Kaiser's grids equal in size by widening both, adding the same number of values), commit, and rerun the notebook. The old results are rejected automatically by the config-hash check, so delete `xp-results/bandit` on Drive first. Repeat until the report succeeds. Do not use `--allow-provisional` for the final numbers.

- [ ] **Step 5: Copy the final outputs into the repo**

```bash
mkdir -p docs/results/phase1
# copy curves.csv gaps.csv crossovers.csv pair_vs_best_single.csv degradation.png from Drive figs into docs/results/phase1/
# copy results/bandit/tuning/*.json from Drive into docs/results/phase1/tuning/
ls docs/results/phase1
```
Expected: the four CSVs, the PNG and a `tuning/` folder.

- [ ] **Step 6: Write `README.md` from the real outputs**

Sections: (1) one-paragraph purpose and the claim, (2) the protocol and its enforced rules, (3) how to run (local tests, Colab for full runs), (4) **Phase 1 results**: embed `docs/results/phase1/degradation.png`; state, from `gaps.csv`, `crossovers.csv` and `pair_vs_best_single.csv`, which agent wins at which jump rates with CIs, whether a crossover exists and between which λ values, and whether any tag-team rule beats the better single agent (a plain "no" is a valid answer); (5) **Where each agent loses** (mandatory section, written from the data, not from expectation); (6) limitations (single bandit family, tuned on one familiar value, discrete grids); (7) the planned MiniGrid phase and future work (Crafter, Showdown). Every number must come from the CSVs and carry its CI. Do not write a sentence you cannot trace to a table.

- [ ] **Step 7: Refresh `STATUS.md` and commit**

Update "Current state" (Phase 1 done, results in `docs/results/phase1/`, MiniGrid plan next, timing spike pending), set "Last updated", and set the marker to the current short hash:
```bash
git rev-parse --short HEAD    # put this value in the STATUS_COMMIT marker
git add README.md STATUS.md docs/results
git commit -m "docs: Phase 1 results, README, status refresh

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Close out**

Run `pytest` once more (expected: all pass), then use superpowers:finishing-a-development-branch to decide how to integrate `feat/foundation-and-bandit`.

---

## Self-Review (completed)

**Spec coverage.** §3 protocol rules → Tasks 2 (disjoint pools), 6–7 (provisional flag, natural-limit exemption), 8 (frozen: the evaluator takes fixed `params`). §4 layout → File Structure and Tasks 1–11 (`minigrid/` deliberately absent, see scope). §4.1 Colab → Tasks 6 (resumable, hash), 10 (CLI), 12 (notebook), Global Constraints (pinned deps, deterministic seeding via `default_rng(seed)`). §5.1 bandit and tag-team (3 rules) → Tasks 4, 5, 9, 10 with the spec's numbers copied into `configs/bandit.yaml`. §6 statistics and outputs → Tasks 3, 11, 12 (curves, gap table, crossover table with CIs, write-up with "where each agent loses"). §7 testing → tests in every task; CI in Task 1; smoke test in Task 10. §9 CLAUDE.md → Task 1. **Gaps by design:** §5.2–5.3 (MiniGrid) belong to the next plan; a note in Task 1 amends the spec where planning changed it (Python floor, resumable unit).

**Placeholder scan.** The only deliberate substitution is `REPO_URL_HERE` in the notebook, which Task 12 Step 1–2 resolves and verifies. No TBD/TODO steps.

**Type consistency.** `TuneResult` is defined in `results.py` (Task 6) and imported by `tune.py` (Task 7) and `bandit/phase.py` (Task 10). `PerSeedFn` is defined in `tune.py` and used in `evaluate.py` and `bandit/phase.py`. Agent method names `reset/select/update` and constructor signatures (`KairiAgent(k, epsilon, alpha, seed)`, `KaiserAgent(k, gamma, c, seed)`) match across Tasks 5, 9, 10. Controller protocol `reset/who/observe(t, who, rewards, pred)` matches `TagTeamAgent` and the tests. `Phase.assert_eval_seeds/assert_tune_seeds` are used consistently in Tasks 7–8.
