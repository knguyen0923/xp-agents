# XP Phase 2 (MiniGrid DoorKey: PPO vs PPO+RND) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Phase 2: Kairi (PPO) vs Kaiser (PPO + RND bonus) on MiniGrid DoorKey, tuned on the familiar size 6, frozen, then evaluated zero-shot at sizes {5, 6, 8, 12} and few-shot at sizes {8, 12}, producing success-rate degradation curves, gap and crossover tables with 95% bootstrap CIs.

**Architecture:** A new `xp/minigrid/` package holds the PyTorch PPO `Learner` (Kairi = PPO; Kaiser = the same class with an RND bonus), the DoorKey environment wrappers with explicit disjoint layout-seed pools, and a phase runner. The protocol layer is reused unchanged in spirit: `Phase` spec, `tune()` grid search with the boundary-optimum rule, config-hash guarded results. The unit of work is one trained agent per (agent, training seed) because training dominates cost; a trained agent is evaluated at every knob value, so per-seed results land in the existing tidy table. Stats, plots and the report are generalised from "regret, lower is better" to a configurable metric and direction.

**Tech Stack:** Python >=3.9, NumPy, pandas, PyYAML, matplotlib, PyTorch (CPU), Gymnasium/MiniGrid, pytest.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md` (read Sections 3, 5.2, 5.3, 6, 7 first; Task 1 amends Section 5.2 and 5.3 with the decisions below).

**Out of scope:** CoinRun, Crafter, Showdown, a tag-team for the deep phase, MultiRoom (only a fallback if sizes are at a 0% floor, which the spike ruled out).

## Facts from the Phase 2 spike (already measured; the design below relies on them)

- Throughput: about 850 env steps/s per Colab CPU core, about 2,000 to 2,800 steps/s on the laptop with 2 torch threads. The environment alone runs 5,000 steps/s; the PPO update dominates.
- PPO on DoorKey-6 is **high-variance across seeds**: one seed reached 80% success at 143k steps, another only 12% at 250k and another 47%. A single seed or a short budget says little, hence 3 tune seeds, 10 eval seeds and a measured budget S.
- A second check with the packaged Learner at 600k steps on seeds 0, 2, 3: 0% success, 100% (80% reached at 276k steps) and 39%. So expect the calibrated S to be large, possibly the 2M cap, and expect wide CIs; that is a finding to report, not a reason to tune further.
- A policy trained on size 6 and evaluated zero-shot: size 5 and 6 about 100%, size 8 about 42% (one seed), size 12 about 0% (random policy 2%). So size 12 is a floor for this agent, size 8 is informative; the MultiRoom fallback is not needed.
- RND bug found and fixed in the spike: scaling the intrinsic reward by the std of the **per-step** error made the intrinsic value targets huge, and through the shared trunk plus the global gradient clip this stopped all learning, even at RND coefficient 0. The fix, used here: scale by the std of the **discounted intrinsic return**, keep the intrinsic return non-episodic, give it its own value head, and add `coef * adv_int` to the extrinsic advantage. With it, RND coefficients 0.001 and 0.01 learn as well as plain PPO and 0.1 does not (the bonus noise dominates the advantage), so the coefficient grid below spans that range.

## Global Constraints

- Python floor 3.9; every module starts with `from __future__ import annotations`; no `match`, no `X | Y` in runtime-evaluated annotations, no `zip(strict=)`.
- `xp/protocol/` is phase-agnostic and must never import from a phase package (`xp.bandit`, `xp.minigrid`).
- Protocol rules raise exceptions, never warn: tune and eval seed pools disjoint; a boundary optimum marks a run `provisional` and blocks final reports; a config-hash mismatch on a resumable unit raises; frozen agents never update during evaluation; Kairi and Kaiser tuning grids have equal size.
- Seeds (spec 5.2): 3 training seeds for tuning, 10 for final results; 200 test layouts per knob value; the independent unit for CIs is the training seed, test layouts are averaged within a seed. Paired comparisons share training-seed indices and the same test layouts.
- Knob: grid size, sweep {5, 6, 8, 12}, familiar 6. Few-shot: hyperparameters frozen, adapt online for 200k steps on the unfamiliar size, sizes 8 and 12 only.
- Training budget S: smallest step count at which Kairi reaches 80% success on the familiar size on tune seeds, capped at 2M steps, recorded in the config.
- Tuning: Kairi tunes its entropy coefficient, Kaiser tunes its RND coefficient (entropy fixed at 0.01); equal-size grids; tune on tune seeds only.
- Do not change `configs/bandit.yaml` (its config hash keys the Phase 1 results on Drive). The bandit report must keep producing identical files.
- Full runs go to Colab or, if the user agrees, the laptop (Task 9 asks). Fast tests only run in CI and locally. Commit after every task. Do not push without asking the user.

## File Structure

```
requirements-minigrid.txt                  new  torch + minigrid on top of requirements.txt
pyproject.toml, .github/workflows/ci.yml   mod  minigrid extra; CI installs torch (CPU)
configs/minigrid.yaml                      new  the Phase 2 config
xp/protocol/manifest.py                    new  write_manifest moved out of xp/bandit (shared by both phases)
xp/protocol/results.py                     mod  per-(agent, seed) job files and tuning-job cache
xp/bandit/phase.py                         mod  imports write_manifest from the protocol layer
xp/minigrid/__init__.py                    new
xp/minigrid/env.py                         new  DoorKey env, layout-seed pools, VecEnv, frozen-policy evaluation
xp/minigrid/rnd.py                         new  RND network, running stats, IntrinsicScaler
xp/minigrid/ppo.py                         new  ActorCritic, GAE, Learner (Kairi = PPO, Kaiser = PPO + RND)
xp/minigrid/phase.py                       new  tuning jobs, seed jobs, parallel runner, run_phase
xp/minigrid/calibrate.py                   new  training-budget S suggestion
xp/run.py                                  mod  minigrid phase + --workers
xp/stats/summary.py, xp/plots/curves.py, xp/report.py   mod  metric and direction are parameters
notebooks/run_minigrid.ipynb               new  Colab launcher (literal paths, no $VAR)
tests/...                                  one test file per new module
```

Interfaces shared by several tasks:

- `Learner(ppo: dict, ent_coef: float, rnd_coef: Optional[float], seed: int, device="cpu")`; `.train(size, steps, first_episode=0, limit=500_000, log_every=10, target=0.8) -> {"curve": [{"steps","success_100","return_100"}], "steps_to_target": Optional[int], "steps": int}`; `.evaluate(size, n_layouts, base=TEST_LAYOUT_BASE) -> {"success","mean_return"}`; `.fingerprint() -> str`.
- `build_learner(cfg, agent, params, seed) -> Learner`, `tune_job(cfg, out_dir, agent, params, seed) -> float`, `seed_job(cfg, out_dir, agent, params, provisional, seed) -> None`, `run_phase(cfg, out_dir, workers=1) -> None`.
- Tidy table for Phase 2 (same columns as Phase 1): per (agent, seed) rows with `knob` = grid size and `metric` in `success`, `return` (zero-shot, every size), `success_fewshot`, `return_fewshot` (sizes 8, 12), and at the familiar size `steps_to_80` (censored at S) and `reached_80`.

---

### Task 1: Dependencies, config, CI, spec amendments

**Files:**
- Create: `requirements-minigrid.txt`, `configs/minigrid.yaml`, `xp/minigrid/__init__.py` (empty), `tests/test_minigrid_config.py`
- Modify: `pyproject.toml`, `.github/workflows/ci.yml`, `tests/conftest.py`, `docs/superpowers/specs/2026-10-22-xp-design.md`

**Interfaces:**
- Produces: `configs/minigrid.yaml` (loaded by every later task); fixture `tiny_mg_cfg` (a tiny Phase 2 config dict with `report` block) in `tests/conftest.py`.

- [ ] **Step 1: Create a branch and the environment**

```bash
cd /Users/knguyen/VSC/xp-agents
git status --short   # expect clean
git checkout -b feat/phase2-minigrid
source .venv/bin/activate
```

- [ ] **Step 2: Write the failing config test**

Create `tests/test_minigrid_config.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from xp.protocol.phase import Phase, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "minigrid.yaml"


def test_committed_minigrid_config_is_a_valid_phase():
    cfg = load_config(CONFIG)
    phase = Phase.from_config(cfg)
    assert phase.name == "minigrid" and phase.knob_name == "grid_size"
    assert phase.familiar == 6.0 and phase.sweep == (5.0, 6.0, 8.0, 12.0)
    assert not set(phase.tune_seeds) & set(phase.eval_seeds)
    assert len(phase.tune_seeds) == 3 and len(phase.eval_seeds) == 10  # spec 5.2: 3 tuning seeds, 10 final seeds


def test_committed_grids_are_equal_size_and_in_the_fewshot_sizes():
    cfg = load_config(CONFIG)
    sizes = {a: int(np.prod([len(v) for v in cfg["agents"][a]["grid"].values()])) for a in ("kairi", "kaiser")}
    assert sizes["kairi"] == sizes["kaiser"]
    assert set(cfg["fewshot"]["sizes"]) <= set(cfg["knob"]["sweep"])  # spec 5.3: few-shot at sizes 8 and 12 only
    assert cfg["eval"]["layouts"] == 200
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `pytest tests/test_minigrid_config.py -v`
Expected: FAIL (FileNotFoundError for configs/minigrid.yaml).

- [ ] **Step 4: Create the config and the dependency files**

Create `configs/minigrid.yaml`:

```yaml
phase: minigrid
knob:
  name: grid_size
  familiar: 6
  sweep: [5, 6, 8, 12]
seeds:
  tune: [0, 3]
  eval: [100, 110]
  chunk: 1
  agent_seed: 0
budget:
  steps: 1000000
  target_success: 0.8
  cap: 2000000
eval:
  layouts: 200
fewshot:
  sizes: [8, 12]
  steps: 200000
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
      ent_coef: [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    natural_limits:
      ent_coef: [0.0, null]
  kaiser:
    fixed:
      ent_coef: 0.01
    grid:
      rnd_coef: [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
    natural_limits:
      rnd_coef: [0.0, null]
report:
  metric: success
  lower_is_better: false
  agents: [kairi, kaiser]
  extra_metrics:
    success_fewshot: false
    steps_to_80: true
  xlabel: grid size (dashed line = familiar)
  ylabel: success rate on unseen layouts (95% CI)
  title: DoorKey success vs grid size
```

`budget.steps: 1000000` is a provisional starting value; Task 9 replaces it with the calibrated S before the real run. `chunk: 1` and `agent_seed: 0` exist only because `Phase.from_config` requires them; the deep phase does not chunk.

Create `requirements-minigrid.txt`:

```text
-r requirements.txt
torch>=2.0
minigrid>=3.0
```

```bash
touch xp/minigrid/__init__.py
```

- [ ] **Step 5: Apply the pyproject, CI and conftest patches** (the conftest patch adds the `tiny_mg_cfg` fixture used by later tasks)

```bash
git apply <<'PATCH'
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -11,6 +11,7 @@
 
 [project.optional-dependencies]
 dev = ["pytest"]
+minigrid = ["torch>=2.0", "minigrid>=3.0"]
 
 [tool.setuptools.packages.find]
 include = ["xp*"]
PATCH
```

```bash
git apply <<'PATCH'
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -11,5 +11,5 @@
       - uses: actions/setup-python@v5
         with:
           python-version: ${{ matrix.python-version }}
-      - run: pip install -r requirements.txt -e .
+      - run: pip install -r requirements-minigrid.txt -e . --extra-index-url https://download.pytorch.org/whl/cpu
       - run: pytest
PATCH
```

```bash
git apply <<'PATCH'
--- a/tests/conftest.py
+++ b/tests/conftest.py
@@ -39,3 +39,26 @@
 @pytest.fixture
 def tiny_cfg():
     return copy.deepcopy(_TINY)
+
+
+_TINY_MG = {
+    "phase": "minigrid",
+    "knob": {"name": "grid_size", "familiar": 5, "sweep": [5, 6]},
+    "seeds": {"tune": [0, 2], "eval": [10, 12], "chunk": 1, "agent_seed": 0},
+    "budget": {"steps": 256, "target_success": 0.8, "cap": 2000},
+    "eval": {"layouts": 3},
+    "fewshot": {"sizes": [6], "steps": 64},
+    "ppo": {"n_envs": 2, "n_steps": 16, "epochs": 2, "minibatches": 2, "lr": 0.00025, "gamma": 0.99, "lam": 0.95,
+            "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 0.0001},
+    "agents": {
+        "kairi": {"grid": {"ent_coef": [0.0, 0.01]}, "natural_limits": {"ent_coef": [0.0, None]}},
+        "kaiser": {"fixed": {"ent_coef": 0.01}, "grid": {"rnd_coef": [0.0, 0.01]},
+                   "natural_limits": {"rnd_coef": [0.0, None]}},
+    },
+    "report": {"metric": "success", "lower_is_better": False, "agents": ["kairi", "kaiser"]},
+}
+
+
+@pytest.fixture
+def tiny_mg_cfg():
+    return copy.deepcopy(_TINY_MG)
PATCH
```

- [ ] **Step 6: Install the Phase 2 dependencies locally and run the test**

```bash
pip install -r requirements-minigrid.txt -e .
pytest tests/test_minigrid_config.py -v
pytest
```
Expected: the config test passes and the full suite passes (88 existing tests plus the 2 new ones).

- [ ] **Step 7: Amend the spec.** In `docs/superpowers/specs/2026-10-22-xp-design.md`, insert the following block directly after the paragraph that begins `Metrics: success rate and mean return per knob value` (end of Section 5.3):

```markdown
*Amendment (Phase 2 plan; design decisions and spike facts):*

- **Layout pools.** Training layout seeds are `seed * 1_000_000 + episode` (episodes below 500,000 for familiar-size training, 500,000 and above for few-shot adaptation). Tuning-time evaluation uses validation layouts with seeds from 2,000,000,000 up; final evaluation uses 200 test layouts with seeds from 1,000,000,000 up. The pools are disjoint by construction and guarded in code. All agents and training seeds share the same test layouts, so comparisons are paired.
- **Tuning objective.** 1 - success rate on the validation layouts at the familiar size after the full budget S, averaged over the 3 tune seeds (lower is better).
- **Budget S.** Because S must exist before either agent can be tuned, it is measured with default-entropy Kairi (entropy coefficient 0.01) on the tune seeds: the smallest logged step count at which mean training success over the tune seeds reaches 80%, capped at 2M, then written to the config. If the cap is reached, the write-up says so.
- **Grids.** Kairi: entropy coefficient {0, 0.001, 0.003, 0.01, 0.03, 0.1}. Kaiser: RND coefficient {0, 0.001, 0.003, 0.01, 0.03, 0.1} with entropy fixed at 0.01. Zero is a natural limit for both; a Kaiser optimum at 0 means Kaiser collapses to Kairi, which is a valid finding rather than a provisional result.
- **RND implementation.** The intrinsic reward is scaled by the std of its discounted return, the intrinsic return is non-episodic with its own value head, and the policy advantage is extrinsic + coef * intrinsic. Scaling by the per-step std instead made the intrinsic value targets huge and, through the shared trunk and the global gradient clip, stopped learning entirely.
- **Sample efficiency.** `steps_to_80` is right-censored at S for runs that never reach 80%, with a companion `reached_80` indicator.
- **Spike facts.** PPO on DoorKey-6 is high-variance across seeds (80% by 143k steps on one seed, 12% by 250k on another). A size-6 policy is about 100% at sizes 5 and 6, about 42% at size 8 and 0% at size 12 (random policy 2%), so the MultiRoom fallback is not triggered. Throughput is about 850 steps/s per Colab core and 2,000 to 2,800 on the laptop.
- **Code layout.** Kairi and Kaiser are one `Learner` class with and without RND (`xp.minigrid.phase.build_learner`), so there are no separate `kairi.py` / `kaiser.py`; `rnd.py` and `calibrate.py` are added, and `write_manifest` moves to `xp/protocol/manifest.py`.
```

- [ ] **Step 8: Commit**

```bash
git add requirements-minigrid.txt configs/minigrid.yaml xp/minigrid/__init__.py tests/test_minigrid_config.py tests/conftest.py pyproject.toml .github/workflows/ci.yml docs/superpowers/specs/2026-10-22-xp-design.md
git commit -m "feat: phase 2 config, minigrid deps, CI torch, spec amendments

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: DoorKey environment, layout pools, frozen evaluation

**Files:**
- Create: `xp/minigrid/env.py`, `tests/test_minigrid_env.py`

**Interfaces:**
- Produces: `make_env(size)`, `prep(obs_np, device)`, `train_layout_seed(seed, episode) -> int`, `eval_layout_seeds(n, base=TEST_LAYOUT_BASE) -> list`, `VecEnv(size, n, seed, first_episode=0, limit=500_000)` with `.obs`, `.step(actions)`, `.recent_stats(k=100)`, `evaluate_policy(net, size, n_layouts, device, base=TEST_LAYOUT_BASE) -> {"success","mean_return"}`; constants `N_ACTIONS, TRAIN_STRIDE, FEWSHOT_EPISODE_OFFSET, TEST_LAYOUT_BASE, VALIDATION_LAYOUT_BASE`.
- Consumes: `xp.errors.ProtocolError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_minigrid_env.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from xp.errors import ProtocolError  # noqa: E402
from xp.minigrid.env import (  # noqa: E402
    FEWSHOT_EPISODE_OFFSET, TEST_LAYOUT_BASE, TRAIN_STRIDE, VecEnv, eval_layout_seeds, train_layout_seed,
)


def test_train_and_test_layout_pools_are_disjoint():
    top_train = train_layout_seed(999, TRAIN_STRIDE - 1)
    assert top_train < TEST_LAYOUT_BASE <= min(eval_layout_seeds(5))


def test_training_layout_seed_cannot_reach_the_test_pool():
    with pytest.raises(ProtocolError):
        train_layout_seed(1000, 0)
    with pytest.raises(ProtocolError):
        train_layout_seed(0, TRAIN_STRIDE)


def test_validation_pool_is_disjoint_from_training_and_test():
    from xp.minigrid.env import VALIDATION_LAYOUT_BASE

    assert set(eval_layout_seeds(200)).isdisjoint(eval_layout_seeds(200, VALIDATION_LAYOUT_BASE))
    assert max(eval_layout_seeds(200)) < VALIDATION_LAYOUT_BASE
    with pytest.raises(ProtocolError):
        eval_layout_seeds(10, base=12345)


def test_vecenv_is_deterministic_and_resets_with_fresh_layouts():
    a, b = VecEnv(5, 3, seed=4), VecEnv(5, 3, seed=4)
    assert (a.obs == b.obs).all()
    rng = np.random.default_rng(0)
    for _ in range(400):
        act = rng.integers(0, 7, 3)
        oa, ra, da = a.step(act)
        ob, rb, db = b.step(act)
        assert (oa == ob).all() and (da == db).all()
    assert a.recent  # at least one episode finished (truncation at 10 * size**2 = 250 steps)
    assert a.next_episode > 3  # auto-resets consumed new layout seeds


def test_vecenv_raises_when_its_layout_stream_runs_out():
    env = VecEnv(5, 2, seed=1, first_episode=FEWSHOT_EPISODE_OFFSET - 2, limit=FEWSHOT_EPISODE_OFFSET)
    with pytest.raises(ProtocolError):
        for _ in range(600):
            env.step(np.zeros(2, dtype=int))
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_minigrid_env.py -v`
Expected: FAIL (ModuleNotFoundError: xp.minigrid.env).

- [ ] **Step 3: Implement**

Create `xp/minigrid/env.py`:

```python
from __future__ import annotations

import numpy as np
import torch
from minigrid.envs import DoorKeyEnv
from minigrid.wrappers import ImgObsWrapper
from torch.distributions import Categorical

from xp.errors import ProtocolError

N_ACTIONS = 7
TRAIN_STRIDE = 1_000_000          # training seed s owns layout seeds [s * STRIDE, (s + 1) * STRIDE)
FEWSHOT_EPISODE_OFFSET = 500_000  # episodes below this index are for familiar-size training, above for few-shot
TEST_LAYOUT_BASE = 1_000_000_000        # final-evaluation layouts: seeds >= this; every training layout seed is below it
VALIDATION_LAYOUT_BASE = 2_000_000_000  # tuning-time layouts, disjoint from training and from the test pool


def make_env(size: int):
    return ImgObsWrapper(DoorKeyEnv(size=int(size)))


def prep(obs_np: np.ndarray, device) -> torch.Tensor:
    """(N, 7, 7, 3) uint8 -> (N, 3, 7, 7) float in [0, 1]."""
    x = torch.as_tensor(obs_np, dtype=torch.float32, device=device) / 10.0
    return x.permute(0, 3, 1, 2)


def train_layout_seed(seed: int, episode: int) -> int:
    """Layout seed for the `episode`-th training episode of training seed `seed`; never in the test pool."""
    if not 0 <= episode < TRAIN_STRIDE:
        raise ProtocolError(f"training episode index {episode} is outside [0, {TRAIN_STRIDE})")
    value = int(seed) * TRAIN_STRIDE + int(episode)
    if value >= TEST_LAYOUT_BASE:
        raise ProtocolError(f"training layout seed {value} reaches the test pool (>= {TEST_LAYOUT_BASE})")
    return value


def eval_layout_seeds(n: int, base: int = TEST_LAYOUT_BASE) -> list:
    if base not in (TEST_LAYOUT_BASE, VALIDATION_LAYOUT_BASE) or not 0 <= n < VALIDATION_LAYOUT_BASE - TEST_LAYOUT_BASE:
        raise ProtocolError(f"unknown layout pool base {base} or size {n}")
    return [base + i for i in range(n)]


class VecEnv:
    """Synchronous vector env. Every episode (including after auto-reset) gets an explicit training layout seed."""

    def __init__(self, size: int, n: int, seed: int, first_episode: int = 0, limit: int = FEWSHOT_EPISODE_OFFSET):
        self.n, self.seed, self.limit = n, int(seed), limit
        self.next_episode = first_episode
        self.envs = [make_env(size) for _ in range(n)]
        self.obs = np.stack([self._reset(e) for e in self.envs])
        self.ep_ret = np.zeros(n)
        self.recent = []  # (success, return) of finished episodes

    def _reset(self, env):
        if self.next_episode >= self.limit:
            raise ProtocolError(f"training seed {self.seed} used up its layout stream (limit {self.limit})")
        layout = train_layout_seed(self.seed, self.next_episode)
        self.next_episode += 1
        return env.reset(seed=layout)[0]

    def step(self, actions):
        rews = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.float32)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, _ = env.step(int(a))
            self.ep_ret[i] += r
            rews[i] = r
            if term or trunc:
                dones[i] = 1.0
                self.recent.append((float(term and r > 0), float(self.ep_ret[i])))
                self.ep_ret[i] = 0.0
                o = self._reset(env)
            self.obs[i] = o
        return self.obs.copy(), rews, dones

    def recent_stats(self, k: int = 100):
        window = self.recent[-k:]
        if not window:
            return float("nan"), float("nan")
        arr = np.array(window)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())


@torch.no_grad()
def evaluate_policy(net, size: int, n_layouts: int, device, base: int = TEST_LAYOUT_BASE) -> dict:
    """Success rate and mean return of a frozen policy on a fixed layout pool. Never updates anything."""
    env = make_env(size)
    wins, rets = 0.0, []
    for layout in eval_layout_seeds(n_layouts, base):
        gen = torch.Generator().manual_seed(layout)  # sampling is deterministic per layout and leaves global RNG alone
        obs, _ = env.reset(seed=layout)
        done, ret, term, r = False, 0.0, False, 0.0
        while not done:
            logits, _ = net(prep(obs[None], device))
            action = int(torch.multinomial(Categorical(logits=logits).probs, 1, generator=gen).item())
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            done = term or trunc
        wins += float(term and r > 0)
        rets.append(ret)
    return {"success": wins / n_layouts, "mean_return": float(np.mean(rets))}
```

Notes for the implementer: training layouts always get an **explicit** seed (including after auto-reset), so train, validation and test layouts come from provably disjoint seed pools; `evaluate_policy` samples actions from a per-layout `torch.Generator`, so evaluation is deterministic and never touches the global RNG; `eval_layout_seeds` is named so pytest does not collect it as a test.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_minigrid_env.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/minigrid/env.py tests/test_minigrid_env.py
git commit -m "feat: DoorKey env with disjoint layout-seed pools and frozen evaluation

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: RND, intrinsic scaling and the PPO Learner

**Files:**
- Create: `xp/minigrid/rnd.py`, `xp/minigrid/ppo.py`, `tests/test_minigrid_ppo.py`

**Interfaces:**
- Consumes: Task 2 (`VecEnv`, `evaluate_policy`, `prep`, constants).
- Produces: `RunningMeanStd`, `RND` (`.normalize(obs_np, device)`, `.error(x)`), `IntrinsicScaler(n_envs, gamma)` (callable on a `(t_len, n_envs)` array), `ActorCritic(intrinsic=False)`, `gae(...)`, and `Learner` (see Interfaces above).

- [ ] **Step 1: Write the failing tests** (they include the regression test for the spike's RND scaling bug)

Create `tests/test_minigrid_ppo.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

import torch  # noqa: E402

from xp.minigrid.ppo import Learner  # noqa: E402
from xp.minigrid.rnd import IntrinsicScaler  # noqa: E402

PPO = {"n_envs": 2, "n_steps": 16, "epochs": 2, "minibatches": 2, "lr": 2.5e-4, "gamma": 0.99, "lam": 0.95,
       "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 1e-4}


def test_training_is_deterministic_per_seed():
    a, b = Learner(PPO, 0.01, None, seed=3), Learner(PPO, 0.01, None, seed=3)
    a.train(5, 128)
    b.train(5, 128)
    assert a.fingerprint() == b.fingerprint()
    c = Learner(PPO, 0.01, None, seed=4)
    c.train(5, 128)
    assert c.fingerprint() != a.fingerprint()


def test_evaluation_never_updates_anything():
    for rnd_coef in (None, 0.01):
        agent = Learner(PPO, 0.01, rnd_coef, seed=1)
        agent.train(5, 64)
        before = agent.fingerprint()
        res = agent.evaluate(5, 6)
        assert 0.0 <= res["success"] <= 1.0
        assert agent.fingerprint() == before


def test_evaluation_is_deterministic_for_fixed_weights():
    agent = Learner(PPO, 0.01, None, seed=2)
    assert agent.evaluate(5, 6) == agent.evaluate(5, 6)


def test_kaiser_changes_weights_differently_from_kairi():
    kairi, kaiser = Learner(PPO, 0.01, None, seed=5), Learner(PPO, 0.01, 0.01, seed=5)
    assert kairi.net.v_int is None and kaiser.net.v_int is not None
    kairi.train(5, 128)
    kaiser.train(5, 128)
    assert kairi.fingerprint() != kaiser.fingerprint()


def test_deepcopy_continues_independently():
    import copy

    base = Learner(PPO, 0.01, 0.01, seed=6)
    base.train(5, 64)
    before = base.fingerprint()
    clone = copy.deepcopy(base)
    assert clone.fingerprint() == before
    clone.train(6, 64, first_episode=500_000, limit=1_000_000)
    assert base.fingerprint() == before and clone.fingerprint() != before


def _max_discounted_return(rewards, gamma=0.99):
    ret, peak = np.zeros(rewards.shape[1]), 0.0
    for t in range(rewards.shape[0]):
        ret = ret * gamma + rewards[t]
        peak = max(peak, float(np.abs(ret).max()))
    return peak


def test_intrinsic_returns_stay_small():
    """Regression: scaling by the per-step std left discounted intrinsic returns in the hundreds or thousands."""
    rng = np.random.default_rng(0)
    raw = 5.0 + 0.1 * rng.standard_normal((2000, 4))
    scaled = IntrinsicScaler(4, 0.99)(raw)
    naive = (raw / raw.std()).astype(np.float32)
    assert _max_discounted_return(scaled) < 20.0
    assert _max_discounted_return(naive) > 1000.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_minigrid_ppo.py -v`
Expected: FAIL (ModuleNotFoundError: xp.minigrid.rnd).

- [ ] **Step 3: Implement the RND module**

Create `xp/minigrid/rnd.py`:

```python
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot


def _cnn(out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, 2), nn.ReLU(), nn.Conv2d(16, 32, 2), nn.ReLU(),
        nn.Conv2d(32, 32, 2), nn.ReLU(), nn.Flatten(), nn.Linear(32 * 4 * 4, out),
    )


class RND(nn.Module):
    """Random-network-distillation novelty bonus (the deep analog of UCB's confidence bonus)."""

    def __init__(self):
        super().__init__()
        self.target = _cnn(64)
        self.pred = nn.Sequential(_cnn(64), nn.ReLU(), nn.Linear(64, 64))
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms = RunningMeanStd((3,))

    def normalize(self, obs_np: np.ndarray, device) -> torch.Tensor:
        flat = obs_np.reshape(-1, 3).astype(np.float64)
        self.obs_rms.update(flat)
        x = ((obs_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)).clip(-5, 5)
        return torch.as_tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        return (self.pred(x) - self.target(x)).pow(2).mean(-1)


class IntrinsicScaler:
    """Scales raw RND errors by the std of their discounted return (RND paper's forward filter).

    Dividing by the std of the per-step error instead leaves the intrinsic value targets in the hundreds; through
    the shared trunk and the global grad clip that crushed the policy gradient (found in the Phase 2 spike).
    """

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

- [ ] **Step 4: Implement the Learner**

Create `xp/minigrid/ppo.py`:

```python
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.minigrid.env import FEWSHOT_EPISODE_OFFSET, N_ACTIONS, TEST_LAYOUT_BASE, VecEnv, evaluate_policy, prep
from xp.minigrid.rnd import RND, IntrinsicScaler


class ActorCritic(nn.Module):
    def __init__(self, intrinsic: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 2), nn.ReLU(), nn.Conv2d(16, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(), nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 64), nn.Tanh())
        self.pi = nn.Linear(64, N_ACTIONS)
        self.v = nn.Linear(64, 1)
        # Separate head for the non-episodic intrinsic return; only built for RND agents, so the plain-PPO
        # parameter and random streams are identical with or without this code path.
        self.v_int = nn.Linear(64, 1) if intrinsic else None

    def heads(self, x):
        h = self.fc(self.conv(x))
        v_int = self.v_int(h).squeeze(-1) if self.v_int is not None else None
        return self.pi(h), self.v(h).squeeze(-1), v_int

    def forward(self, x):
        logits, v, _ = self.heads(x)
        return logits, v


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
    """PPO, plus an RND bonus when `rnd_coef` is not None. Kairi = rnd_coef None; Kaiser = rnd_coef a float.

    One Learner holds its weights and optimizer state across `train` calls, so the few-shot phase can continue
    training a deep copy on a new size. `evaluate` never updates anything.
    """

    def __init__(self, ppo: dict, ent_coef: float, rnd_coef: Optional[float], seed: int, device: str = "cpu"):
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

    def evaluate(self, size: int, n_layouts: int, base: int = TEST_LAYOUT_BASE) -> dict:
        return evaluate_policy(self.net, size, n_layouts, self.device, base)

    def train(self, size: int, steps: int, first_episode: int = 0, limit: int = FEWSHOT_EPISODE_OFFSET,
              log_every: int = 10, target: float = 0.8) -> dict:
        """Train on fresh layouts of `size` for about `steps` env steps. Returns the curve and steps-to-target."""
        # Reseed from (seed, steps so far) so training is deterministic however Learners are interleaved.
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, rnd, device = self.ppo, self.rnd, self.device
        n, t_len = c["n_envs"], c["n_steps"]
        batch = n * t_len
        updates = max(1, steps // batch)
        env = VecEnv(size, n, self.seed, first_episode, limit)
        net, opt = self.net, self.opt
        obs_buf = np.zeros((t_len, n, 7, 7, 3), dtype=np.uint8)
        act_buf = torch.zeros((t_len, n), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        val_buf = torch.zeros((t_len, n), device=device)
        vint_buf = torch.zeros((t_len, n), device=device)

        curve, steps_to_target = [], None
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
                _, rew_buf[t], done_buf[t] = env.step(action.cpu().numpy())

            if rnd is not None:
                next_seq = np.concatenate([obs_buf[1:], env.obs[None]], axis=0).reshape(-1, 7, 7, 3)
                x = rnd.normalize(next_seq.astype(np.float32), device)
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
                _, next_value, next_vint = net.heads(prep(env.obs, device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv_ext = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            returns, adv = adv_ext + val_buf, adv_ext
            if rnd is not None:
                # The intrinsic return is not cut at episode ends (RND paper), so reaching the goal costs no bonus.
                adv_int = gae(torch.as_tensor(int_rewards, device=device), vint_buf, next_vint,
                              torch.ones_like(nonterm), c["gamma"], c["lam"])
                returns_int = adv_int + vint_buf
                adv = adv_ext + self.rnd_coef * adv_int

            b_obs = prep(obs_buf.reshape(-1, 7, 7, 3), device)
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
                succ, ret = env.recent_stats()
                done_steps = (u + 1) * batch
                curve.append({"steps": done_steps, "success_100": succ, "return_100": ret})
                if steps_to_target is None and succ == succ and succ >= target:
                    steps_to_target = done_steps
        return {"curve": curve, "steps_to_target": steps_to_target, "steps": updates * batch}
```

Notes for the implementer: `Learner(..., rnd_coef=None)` builds **no** RND modules and no intrinsic value head, so Kairi's parameter and random streams are exactly plain PPO. `train` reseeds torch and numpy from `(seed, steps_trained)`, so training is deterministic however Learners are interleaved and a `copy.deepcopy` continues deterministically for few-shot adaptation. `fingerprint()` hashes every parameter an update could change; evaluation must leave it unchanged.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_minigrid_ppo.py -v`
Expected: 6 passed (about 10 seconds).

- [ ] **Step 6: Commit**

```bash
git add xp/minigrid/rnd.py xp/minigrid/ppo.py tests/test_minigrid_ppo.py
git commit -m "feat: PPO Learner with RND (return-scaled, non-episodic intrinsic stream)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Protocol layer: shared manifest and per-job stores

**Files:**
- Create: `xp/protocol/manifest.py`
- Modify: `xp/bandit/phase.py`, `xp/protocol/results.py`, `tests/test_results.py`

**Interfaces:**
- Produces: `xp.protocol.manifest.write_manifest(cfg, phase, out_dir, extra_versions=None)` (moved from `xp/bandit/phase.py`, still importable from there); in `xp.protocol.results`: `seed_job_done(out_dir, phase, agent, seed) -> bool`, `write_seed_job(out_dir, phase, agent, seed, df)`, `read_tune_job(out_dir, phase, agent, params, seed) -> Optional[float]`, `write_tune_job(out_dir, phase, agent, params, seed, score)`.
- Per-seed job files are `<agent>__seed-<seed>.csv`, so the existing `load_results` glob (`*__*.csv`) already picks them up.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_results.py`)

```bash
git apply <<'PATCH'
--- a/tests/test_results.py
+++ b/tests/test_results.py
@@ -159,3 +159,40 @@
     assert sorted(p.name for p in (tmp_path / "bandit").rglob("*") if p.is_file()) == [
         "kairi.csv", "kairi.json", "kairi__jump_rate-0.01.csv"
     ]
+
+
+def test_seed_job_roundtrip_and_hash_guard(tmp_path, tiny_cfg):
+    import pandas as pd
+    import pytest
+
+    from xp.errors import ConfigMismatchError
+    from xp.protocol.phase import Phase
+    from xp.protocol.results import COLUMNS, seed_job_done, write_seed_job
+
+    phase = Phase.from_config(tiny_cfg)
+    df = pd.DataFrame([["bandit", "kairi", 1.0, 100, "success", 0.5, False, phase.config_hash]], columns=COLUMNS)
+    assert not seed_job_done(tmp_path, phase, "kairi", 100)
+    write_seed_job(tmp_path, phase, "kairi", 100, df)
+    assert seed_job_done(tmp_path, phase, "kairi", 100)
+    other = Phase.from_config({**tiny_cfg, "horizon": 51})
+    with pytest.raises(ConfigMismatchError):
+        seed_job_done(tmp_path, other, "kairi", 100)
+    with pytest.raises(ValueError):
+        write_seed_job(tmp_path, phase, "kairi", 101, df.rename(columns={"value": "v"}))
+
+
+def test_tune_job_roundtrip_and_hash_guard(tmp_path, tiny_cfg):
+    import pytest
+
+    from xp.errors import ConfigMismatchError
+    from xp.protocol.phase import Phase
+    from xp.protocol.results import read_tune_job, write_tune_job
+
+    phase = Phase.from_config(tiny_cfg)
+    params = {"ent_coef": 0.01, "rnd_coef": 0.0}
+    assert read_tune_job(tmp_path, phase, "kaiser", params, 3) is None
+    write_tune_job(tmp_path, phase, "kaiser", params, 3, 0.25)
+    assert read_tune_job(tmp_path, phase, "kaiser", params, 3) == 0.25
+    other = Phase.from_config({**tiny_cfg, "horizon": 51})
+    with pytest.raises(ConfigMismatchError):
+        read_tune_job(tmp_path, other, "kaiser", params, 3)
PATCH
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_results.py -v`
Expected: 2 new tests FAIL (ImportError: seed_job_done).

- [ ] **Step 3: Create the shared manifest module and re-point the bandit**

Create `xp/protocol/manifest.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from xp.errors import ConfigMismatchError
from xp.protocol.phase import Phase
from xp.protocol.results import atomic_write


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(cfg: dict, phase: Phase, out_dir, extra_versions: Optional[dict] = None) -> None:
    path = Path(out_dir) / phase.name / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text()).get("config_hash")
        if old != phase.config_hash:
            raise ConfigMismatchError(f"{path} was written under config {old}, current config is {phase.config_hash}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cfg": cfg,
        "config_hash": phase.config_hash,
        "git_commit": _git_commit(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            **(extra_versions or {}),
        },
    }
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str)))
```

Then apply this patch, which removes `_git_commit` / `write_manifest` from `xp/bandit/phase.py` and imports the shared one (the name stays importable from `xp.bandit.phase`):

```bash
git apply <<'PATCH'
--- a/xp/bandit/phase.py
+++ b/xp/bandit/phase.py
@@ -1,9 +1,5 @@
 from __future__ import annotations
 
-import json
-import subprocess
-import sys
-from pathlib import Path
 from typing import Callable, Dict
 
 import numpy as np
@@ -12,10 +8,11 @@
 from xp.bandit.agents import KaiserAgent, KairiAgent
 from xp.bandit.tagteam import ChangeTriggered, FixedSchedule, MetaBandit, TagTeamAgent
 from xp.bandit.world import make_world, run_agent
-from xp.errors import ConfigMismatchError, ProtocolError
+from xp.errors import ProtocolError
 from xp.protocol.evaluate import evaluate_unit
+from xp.protocol.manifest import write_manifest  # noqa: F401  (re-exported; tests and callers import it from here)
 from xp.protocol.phase import Phase
-from xp.protocol.results import atomic_write, TuneResult, read_tune, unit_done, write_tune, write_unit
+from xp.protocol.results import TuneResult, read_tune, unit_done, write_tune, write_unit
 from xp.protocol.tune import PerSeedFn, score_params, tune
 
 AgentBuilder = Callable[[dict, int], object]
@@ -70,34 +67,6 @@
     return int(np.prod([len(v) for v in spec["grid"].values()]))
 
 
-def _git_commit():
-    try:
-        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
-        return out.stdout.strip() or None
-    except (OSError, subprocess.CalledProcessError):
-        return None
-
-
-def write_manifest(cfg: dict, phase: Phase, out_dir) -> None:
-    path = Path(out_dir) / phase.name / "manifest.json"
-    if path.exists():
-        old = json.loads(path.read_text()).get("config_hash")
-        if old != phase.config_hash:
-            raise ConfigMismatchError(f"{path} was written under config {old}, current config is {phase.config_hash}")
-    path.parent.mkdir(parents=True, exist_ok=True)
-    manifest = {
-        "cfg": cfg,
-        "config_hash": phase.config_hash,
-        "git_commit": _git_commit(),
-        "versions": {
-            "python": sys.version.split()[0],
-            "numpy": np.__version__,
-            "pandas": pd.__version__,
-        },
-    }
-    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str)))
-
-
 def run_phase(cfg: dict, out_dir) -> None:
     sizes = {name: _grid_size(cfg["agents"][name]) for name in ("kairi", "kaiser")}
     if sizes["kairi"] != sizes["kaiser"]:
PATCH
```

- [ ] **Step 4: Add the job stores to the results module**

```bash
git apply <<'PATCH'
--- a/xp/protocol/results.py
+++ b/xp/protocol/results.py
@@ -126,3 +126,52 @@
             f"provisional (grid-boundary) results for agents {sorted(bad)}: "
             "widen their grids and rerun, or pass allow_provisional for a draft"
         )
+
+
+def _job_path(out_dir, phase: Phase, agent: str, seed: int) -> Path:
+    return Path(out_dir) / phase.name / f"{agent}__seed-{seed}.csv"
+
+
+def seed_job_done(out_dir, phase: Phase, agent: str, seed: int) -> bool:
+    """A per-(agent, training seed) job is done when its atomically written file exists under this config."""
+    path = _job_path(out_dir, phase, agent, seed)
+    if not path.exists():
+        return False
+    try:
+        hashes = list(pd.read_csv(path, usecols=["config_hash"], dtype={"config_hash": str})["config_hash"].unique())
+    except (pd.errors.EmptyDataError, ValueError):
+        return False
+    if hashes != [phase.config_hash]:
+        raise ConfigMismatchError(f"{path} was written under config {hashes}, current config is {phase.config_hash}")
+    return True
+
+
+def write_seed_job(out_dir, phase: Phase, agent: str, seed: int, df: pd.DataFrame) -> None:
+    if list(df.columns) != COLUMNS:
+        raise ValueError(f"results columns must be {COLUMNS}, got {list(df.columns)}")
+    path = _job_path(out_dir, phase, agent, seed)
+    path.parent.mkdir(parents=True, exist_ok=True)
+    atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))
+
+
+def _tune_job_path(out_dir, phase: Phase, agent: str, params: dict, seed: int) -> Path:
+    key = "_".join(f"{k}-{params[k]:g}" for k in sorted(params))
+    return Path(out_dir) / phase.name / "tuning" / "jobs" / f"{agent}__{key}__seed-{seed}.json"
+
+
+def read_tune_job(out_dir, phase: Phase, agent: str, params: dict, seed: int):
+    """Cached tuning score for one (agent, params, tune seed), or None."""
+    path = _tune_job_path(out_dir, phase, agent, params, seed)
+    if not path.exists():
+        return None
+    record = json.loads(path.read_text())
+    if record["config_hash"] != phase.config_hash:
+        raise ConfigMismatchError(f"{path} was written under config {record['config_hash']}, current config is {phase.config_hash}")
+    return float(record["score"])
+
+
+def write_tune_job(out_dir, phase: Phase, agent: str, params: dict, seed: int, score: float) -> None:
+    path = _tune_job_path(out_dir, phase, agent, params, seed)
+    path.parent.mkdir(parents=True, exist_ok=True)
+    record = {"config_hash": phase.config_hash, "params": params, "seed": seed, "score": float(score)}
+    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(record, default=_json_default)))
PATCH
```

- [ ] **Step 5: Run the whole suite**

Run: `pytest`
Expected: all pass (the bandit tests are unchanged and must still pass).

- [ ] **Step 6: Commit**

```bash
git add xp/protocol/manifest.py xp/bandit/phase.py xp/protocol/results.py tests/test_results.py
git commit -m "feat: shared manifest and per-job result stores in the protocol layer

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Phase runner: tuning jobs, seed jobs, parallel execution

**Files:**
- Create: `xp/minigrid/phase.py`, `tests/test_minigrid_phase.py`
- Modify: `xp/run.py`

**Interfaces:**
- Consumes: Tasks 3 and 4 (`Learner`, job stores, `write_manifest`), `xp.protocol.tune.tune/grid_points`, `Phase.from_config`.
- Produces: `build_learner`, `tune_job`, `seed_job`, `run_phase(cfg, out_dir, workers=1)`; CLI `python -m xp.run --phase minigrid --config ... --out ... --workers N`.

Design: `run_phase` checks the equal-grid rule, writes the manifest, tunes each agent (one cached job per grid point and tune seed, so an interrupted tuning loses at most the jobs in flight), freezes, then runs one `seed_job` per (agent, eval seed). A `seed_job` trains once at the familiar size, evaluates zero-shot at every size on the frozen weights (and raises if the weights changed), then for each few-shot size deep-copies the trained learner, adapts it on that size and evaluates. Parallelism is a `spawn` process pool (fork is unsafe with torch); workers set `torch.set_num_threads(1)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_minigrid_phase.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from xp.errors import ConfigMismatchError, ProtocolError  # noqa: E402
from xp.minigrid.phase import build_learner, run_phase, tune_job  # noqa: E402
from xp.protocol.phase import Phase  # noqa: E402
from xp.protocol.results import load_results, read_tune, read_tune_job  # noqa: E402


def test_kairi_and_kaiser_differ_only_by_the_rnd_path(tiny_mg_cfg):
    kairi = build_learner(tiny_mg_cfg, "kairi", {"ent_coef": 0.03}, seed=1)
    kaiser = build_learner(tiny_mg_cfg, "kaiser", {"rnd_coef": 0.03}, seed=1)
    assert kairi.rnd is None and kairi.ent_coef == 0.03
    assert kaiser.rnd is not None and kaiser.rnd_coef == 0.03 and kaiser.ent_coef == 0.01


def test_tune_job_rejects_eval_seeds_and_caches(tiny_mg_cfg, tmp_path):
    phase = Phase.from_config(tiny_mg_cfg)
    with pytest.raises(ProtocolError):
        tune_job(tiny_mg_cfg, tmp_path, "kairi", {"ent_coef": 0.0}, seed=10)  # 10 is an eval seed
    score = tune_job(tiny_mg_cfg, tmp_path, "kairi", {"ent_coef": 0.0}, seed=0)
    assert 0.0 <= score <= 1.0
    assert read_tune_job(tmp_path, phase, "kairi", {"ent_coef": 0.0}, 0) == score


def test_run_phase_end_to_end_resumes_and_guards(tiny_mg_cfg, tmp_path):
    run_phase(tiny_mg_cfg, tmp_path)
    phase = Phase.from_config(tiny_mg_cfg)
    df = load_results(tmp_path, "minigrid")
    assert set(df["agent"]) == {"kairi", "kaiser"}
    assert set(df["seed"]) == set(phase.eval_seeds)
    assert set(df["metric"]) == {"steps_to_80", "reached_80", "success", "return", "success_fewshot", "return_fewshot"}
    zero_shot = df[df["metric"] == "success"]
    assert set(zero_shot["knob"]) == {5.0, 6.0} and zero_shot["value"].between(0, 1).all()
    assert set(df[df["metric"] == "success_fewshot"]["knob"]) == {6.0}
    assert (df["config_hash"] == phase.config_hash).all()
    assert read_tune(tmp_path, phase, "kairi") is not None and read_tune(tmp_path, phase, "kaiser") is not None

    stamps = {p: p.stat().st_mtime_ns for p in (tmp_path / "minigrid").glob("*__seed-*.csv")}
    assert len(stamps) == 4  # 2 agents x 2 eval seeds
    run_phase(tiny_mg_cfg, tmp_path)
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps  # resumed: nothing recomputed

    changed = {**tiny_mg_cfg, "budget": {**tiny_mg_cfg["budget"], "steps": 512}}
    with pytest.raises(ConfigMismatchError):
        run_phase(changed, tmp_path)


def test_unequal_grids_are_refused(tiny_mg_cfg, tmp_path):
    tiny_mg_cfg["agents"]["kaiser"]["grid"]["rnd_coef"] = [0.0, 0.01, 0.03]
    with pytest.raises(ProtocolError):
        run_phase(tiny_mg_cfg, tmp_path)


def test_two_workers_match_one_worker(tiny_mg_cfg, tmp_path):
    run_phase(tiny_mg_cfg, tmp_path / "one", workers=1)
    run_phase(tiny_mg_cfg, tmp_path / "two", workers=2)
    cols = ["agent", "knob", "seed", "metric", "value"]
    a = load_results(tmp_path / "one", "minigrid")[cols].sort_values(cols[:4]).reset_index(drop=True)
    b = load_results(tmp_path / "two", "minigrid")[cols].sort_values(cols[:4]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_minigrid_phase.py -v`
Expected: FAIL (ModuleNotFoundError: xp.minigrid.phase).

- [ ] **Step 3: Implement the phase runner**

Create `xp/minigrid/phase.py`:

```python
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
```

- [ ] **Step 4: Register the phase and the `--workers` flag**

```bash
git apply <<'PATCH'
--- a/xp/run.py
+++ b/xp/run.py
@@ -4,7 +4,7 @@
 
 from xp.protocol.phase import load_config
 
-PHASES = {"bandit": "xp.bandit.phase"}
+PHASES = {"bandit": "xp.bandit.phase", "minigrid": "xp.minigrid.phase"}
 
 
 def main(argv=None) -> None:
@@ -12,12 +12,16 @@
     parser.add_argument("--phase", required=True, choices=sorted(PHASES))
     parser.add_argument("--config", required=True)
     parser.add_argument("--out", required=True)
+    parser.add_argument("--workers", type=int, default=1, help="parallel training jobs (minigrid only)")
     args = parser.parse_args(argv)
     cfg = load_config(args.config)
     if cfg["phase"] != args.phase:
         raise SystemExit(f"config phase {cfg['phase']!r} does not match --phase {args.phase!r}")
     module = __import__(PHASES[args.phase], fromlist=["run_phase"])
-    module.run_phase(cfg, args.out)
+    if args.phase == "minigrid":
+        module.run_phase(cfg, args.out, workers=args.workers)
+    else:
+        module.run_phase(cfg, args.out)
 
 
 if __name__ == "__main__":
PATCH
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_minigrid_phase.py -v`
Expected: 5 passed (about 20 seconds; the two-worker test spawns processes).

- [ ] **Step 6: Commit**

```bash
git add xp/minigrid/phase.py xp/run.py tests/test_minigrid_phase.py
git commit -m "feat: minigrid phase runner with cached tuning jobs, seed jobs and workers

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Training-budget calibration

**Files:**
- Create: `xp/minigrid/calibrate.py`, `tests/test_minigrid_calibrate.py`

**Interfaces:**
- Consumes: `build_learner`, `_run_jobs` from Task 5.
- Produces: `suggest_budget(curves, target, cap) -> {"steps","reached","mean_success","log_steps"}`; CLI `python -m xp.minigrid.calibrate --config ... --out file.json --workers N`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_minigrid_calibrate.py`:

```python
from __future__ import annotations

import math

import pytest

from xp.minigrid.calibrate import suggest_budget


def _curve(values):
    return [{"steps": 1000 * (i + 1), "success_100": v} for i, v in enumerate(values)]


def test_budget_is_first_step_where_mean_success_reaches_target():
    curves = [_curve([0.0, 0.5, 0.9, 1.0]), _curve([0.0, 0.2, 0.8, 1.0])]
    out = suggest_budget(curves, target=0.8, cap=10_000)
    assert out["steps"] == 3000 and out["reached"]  # mean at step 3000 is (0.9 + 0.8) / 2 = 0.85


def test_budget_falls_back_to_cap_and_says_so():
    out = suggest_budget([_curve([0.0, 0.1, 0.2])], target=0.8, cap=5000)
    assert out["steps"] == 5000 and not out["reached"]


def test_nan_success_counts_as_zero():
    out = suggest_budget([_curve([math.nan, 0.9])], target=0.8, cap=5000)
    assert out["steps"] == 2000


def test_mismatched_log_steps_are_rejected():
    with pytest.raises(ValueError):
        suggest_budget([_curve([0.0, 1.0]), _curve([0.0, 1.0, 1.0])], target=0.8, cap=5000)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_minigrid_calibrate.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

Create `xp/minigrid/calibrate.py`:

```python
from __future__ import annotations

import argparse
import json
from typing import List

import numpy as np
import torch

from xp.minigrid.phase import _run_jobs, build_learner
from xp.protocol.phase import Phase, load_config


def suggest_budget(curves: List[List[dict]], target: float, cap: int) -> dict:
    """Smallest logged step count at which mean training success over the seeds reaches `target` (else `cap`).

    Each curve is a list of {"steps", "success_100"} dicts logged at the same steps for every seed.
    """
    steps = [row["steps"] for row in curves[0]]
    for curve in curves:
        if [row["steps"] for row in curve] != steps:
            raise ValueError("all curves must be logged at the same steps")
    mean_success = np.mean([[0.0 if row["success_100"] != row["success_100"] else row["success_100"] for row in c]
                            for c in curves], axis=0)
    hits = [s for s, m in zip(steps, mean_success) if m >= target and s <= cap]
    return {"steps": int(hits[0]) if hits else int(cap), "reached": bool(hits), "mean_success": mean_success.tolist(),
            "log_steps": steps}


def calibration_run(cfg: dict, seed: int) -> list:
    """Default-entropy Kairi on one tune seed for the whole cap; returns its training-success curve."""
    torch.set_num_threads(1)
    phase = Phase.from_config(cfg)
    phase.assert_tune_seeds([seed])
    default_ent = cfg["agents"]["kaiser"]["fixed"]["ent_coef"]
    learner = build_learner(cfg, "kairi", {"ent_coef": default_ent}, seed)
    return learner.train(int(phase.familiar), cfg["budget"]["cap"], target=2.0)["curve"]  # target 2.0: never stops early


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Suggest the Phase 2 training budget S from tune-seed Kairi runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="JSON file for the curves and the suggestion")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    curves = _run_jobs(calibration_run, [(cfg, s) for s in phase.tune_seeds], args.workers)
    suggestion = suggest_budget(curves, cfg["budget"]["target_success"], cfg["budget"]["cap"])
    with open(args.out, "w") as f:
        json.dump({"suggestion": suggestion, "curves": curves}, f, indent=2)
    state = "reached" if suggestion["reached"] else "NOT reached; using the cap"
    print(f"suggested budget S = {suggestion['steps']} steps (80% target {state}); set budget.steps in the config")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_minigrid_calibrate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/minigrid/calibrate.py tests/test_minigrid_calibrate.py
git commit -m "feat: calibrate the Phase 2 training budget from tune-seed Kairi runs

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Stats, plots and report for a configurable metric

**Files:**
- Modify: `xp/stats/summary.py`, `xp/plots/curves.py`, `xp/report.py`, `tests/test_summary.py`
- Create: `tests/test_minigrid_report.py`

**Interfaces:**
- Produces: `curve_table(df, n_boot, seed, metric="regret")`, `gap_table(df, familiar, n_boot, seed, metric="regret")`, `pair_table(df, a, b, n_boot, seed, metric="regret", lower_is_better=True)`, `pair_vs_best_single(df, team, singles, n_boot, seed, metric="regret", lower_is_better=True)`; `plot_degradation(..., xlabel=, ylabel=)` with the Phase 1 labels as defaults; `xp.report.report_settings(cfg)` reading an optional `report:` config block. The bandit config has no `report` block, so every Phase 1 default is unchanged and `configs/bandit.yaml` keeps its hash.
- Direction rule: `_winner` takes the CI of `mean(a - b)`; with `lower_is_better=False` a positive difference favours `a`.

- [ ] **Step 1: Write the failing tests** (the summary test checks the direction flip; the report test runs a tiny Phase 2 end to end and reads the report outputs)

```bash
git apply <<'PATCH'
--- a/tests/test_summary.py
+++ b/tests/test_summary.py
@@ -77,3 +77,12 @@
     b = _rows("B", 0.0, np.zeros(len(SEEDS))).iloc[:-1]
     with pytest.raises(ValueError, match="same eval seeds"):
         curve_table(pd.concat([a, b], ignore_index=True), n_boot=50)
+
+
+def test_higher_is_better_flips_the_winner():
+    df = pd.concat([_rows("A", 0.0, 0.9 + np.zeros(len(SEEDS)) + 0.01 * np.sin(SEEDS)),
+                    _rows("B", 0.0, 0.1 + np.zeros(len(SEEDS)) + 0.01 * np.sin(SEEDS))], ignore_index=True)
+    df["metric"] = "success"
+    lower = pair_table(df, "A", "B", n_boot=200, metric="success", lower_is_better=True)
+    higher = pair_table(df, "A", "B", n_boot=200, metric="success", lower_is_better=False)
+    assert lower.iloc[0]["winner"] == "B" and higher.iloc[0]["winner"] == "A"
PATCH
```

Create `tests/test_minigrid_report.py`:

```python
from __future__ import annotations

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from xp.minigrid.phase import run_phase  # noqa: E402
from xp.report import main as report_main  # noqa: E402


def test_report_runs_on_a_minigrid_results_table(tiny_mg_cfg, tmp_path):
    tiny_mg_cfg["report"]["extra_metrics"] = {"success_fewshot": False, "steps_to_80": True}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_mg_cfg))
    run_phase(tiny_mg_cfg, tmp_path / "res")
    report_main(["--phase", "minigrid", "--config", str(cfg_path), "--out", str(tmp_path / "res"),
                 "--figs", str(tmp_path / "figs"), "--allow-provisional"])
    names = {p.name for p in (tmp_path / "figs").iterdir()}
    assert {"curves.csv", "gaps.csv", "crossovers.csv", "degradation.png",
            "curves__success_fewshot.csv", "crossovers__steps_to_80.csv"} <= names
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_summary.py tests/test_minigrid_report.py -v`
Expected: FAIL (unexpected keyword `metric`).

- [ ] **Step 3: Generalise the stats, plot and report**

```bash
git apply <<'PATCH'
--- a/xp/stats/summary.py
+++ b/xp/stats/summary.py
@@ -21,16 +21,16 @@
     return out
 
 
-def curve_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
+def curve_table(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0, metric: str = "regret") -> pd.DataFrame:
     rows = []
-    for (agent, knob), v in sorted(_values(df).items()):
+    for (agent, knob), v in sorted(_values(df, metric).items()):
         mean, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
         rows.append({"agent": agent, "knob": knob, "mean": mean, "lo": lo, "hi": hi})
     return pd.DataFrame(rows)
 
 
-def gap_table(df: pd.DataFrame, familiar: float, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
-    vals = _values(df)
+def gap_table(df: pd.DataFrame, familiar: float, n_boot: int = 2000, seed: int = 0, metric: str = "regret") -> pd.DataFrame:
+    vals = _values(df, metric)
     rows = []
     for (agent, knob), v in sorted(vals.items()):
         ref = vals[(agent, float(familiar))]
@@ -39,21 +39,25 @@
     return pd.DataFrame(rows)
 
 
-def _winner(lo: float, hi: float, a: str, b: str) -> str:
+def _winner(lo: float, hi: float, a: str, b: str, lower_is_better: bool = True) -> str:
+    """CI [lo, hi] is for mean(a - b). With lower-is-better a negative difference favours a; higher-is-better flips it."""
     if hi < 0:
-        return a
+        return a if lower_is_better else b
     if lo > 0:
-        return b
+        return b if lower_is_better else a
     return "tie"
 
 
-def pair_table(df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
-    vals = _values(df)
+def pair_table(
+    df: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0, metric: str = "regret",
+    lower_is_better: bool = True,
+) -> pd.DataFrame:
+    vals = _values(df, metric)
     knobs = sorted({k for (_, k) in vals})
     rows = []
     for knob in knobs:
         mean, lo, hi = paired_diff_ci(vals[(a, knob)], vals[(b, knob)], n_boot=n_boot, seed=seed)
-        rows.append({"knob": knob, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, a, b)})
+        rows.append({"knob": knob, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, a, b, lower_is_better)})
     return pd.DataFrame(rows)
 
 
@@ -63,16 +67,18 @@
 
 
 def pair_vs_best_single(
-    df: pd.DataFrame, team: str, singles=("kairi", "kaiser"), n_boot: int = 2000, seed: int = 0
+    df: pd.DataFrame, team: str, singles=("kairi", "kaiser"), n_boot: int = 2000, seed: int = 0,
+    metric: str = "regret", lower_is_better: bool = True,
 ) -> pd.DataFrame:
-    """Compare `team` with whichever single agent has the lower mean regret at each knob."""
-    vals = _values(df)
+    """Compare `team` with whichever single agent is better (by mean `metric`) at each knob."""
+    vals = _values(df, metric)
     knobs = sorted({k for (_, k) in vals})
     rows = []
     for knob in knobs:
-        best = min(singles, key=lambda s: vals[(s, knob)].mean())
+        pick = min if lower_is_better else max
+        best = pick(singles, key=lambda s: vals[(s, knob)].mean())
         mean, lo, hi = paired_diff_ci(vals[(team, knob)], vals[(best, knob)], n_boot=n_boot, seed=seed)
         rows.append(
-            {"knob": knob, "best_single": best, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, team, best)}
+            {"knob": knob, "best_single": best, "diff": mean, "lo": lo, "hi": hi, "winner": _winner(lo, hi, team, best, lower_is_better)}
         )
     return pd.DataFrame(rows)
PATCH
```

```bash
git apply <<'PATCH'
--- a/xp/plots/curves.py
+++ b/xp/plots/curves.py
@@ -15,7 +15,10 @@
 AXIS = "#c3c2b7"
 
 
-def plot_degradation(curves, familiar: float, out_path, title: str = "Regret vs jump rate") -> None:
+def plot_degradation(
+    curves, familiar: float, out_path, title: str = "Regret vs jump rate",
+    xlabel: str = "jump rate λ (dashed line = familiar)", ylabel: str = "mean per-step regret (95% CI)",
+) -> None:
     knobs = sorted(curves["knob"].unique())
     pos = {k: i for i, k in enumerate(knobs)}
     agents = sorted(curves["agent"].unique())  # colour follows the entity (sorted name), not its rank
@@ -39,8 +42,8 @@
     ax.axvline(pos[familiar], linestyle="--", linewidth=1, color=MUTED)
     ax.set_xticks(range(len(knobs)))
     ax.set_xticklabels([f"{k:g}" for k in knobs])
-    ax.set_xlabel("jump rate λ (dashed line = familiar)", color=INK_2)
-    ax.set_ylabel("mean per-step regret (95% CI)", color=INK_2)
+    ax.set_xlabel(xlabel, color=INK_2)
+    ax.set_ylabel(ylabel, color=INK_2)
     ax.set_title(title, color=INK, loc="left", fontsize=12)
     ax.grid(axis="y", color=GRID, linewidth=0.8)
     ax.set_axisbelow(True)
PATCH
```

```bash
git apply <<'PATCH'
--- a/xp/report.py
+++ b/xp/report.py
@@ -15,15 +15,30 @@
 EXPECTED_AGENTS = ("kairi", "kaiser", "tt_fixed", "tt_triggered", "tt_meta")
 
 
-def validate_results(df: pd.DataFrame, phase: Phase) -> None:
+def report_settings(cfg: dict) -> dict:
+    """Per-phase report options; the bandit config has no `report` block, so these defaults reproduce Phase 1."""
+    r = cfg.get("report", {})
+    return {
+        "metric": r.get("metric", "regret"),
+        "lower_is_better": r.get("lower_is_better", True),
+        "agents": tuple(r.get("agents", EXPECTED_AGENTS)),
+        "extra_metrics": dict(r.get("extra_metrics", {})),
+        "xlabel": r.get("xlabel", "jump rate λ (dashed line = familiar)"),
+        "ylabel": r.get("ylabel", "mean per-step regret (95% CI)"),
+        "title": r.get("title", "Regret vs jump rate"),
+    }
+
+
+def validate_results(df: pd.DataFrame, phase: Phase, settings: dict = None) -> None:
     """Results must come from this config and cover every agent x knob cell completely."""
+    settings = settings or report_settings(phase.cfg)
     hashes = set(df["config_hash"])
     if hashes != {phase.config_hash}:
         raise ConfigMismatchError(f"results were written under configs {sorted(hashes)}, current config is {phase.config_hash}")
-    counts = df.groupby(["agent", "knob"]).size()
+    counts = df[df["metric"] == settings["metric"]].groupby(["agent", "knob"]).size()
     expected = len(phase.eval_seeds)
     problems = []
-    for agent in EXPECTED_AGENTS:
+    for agent in settings["agents"]:
         for knob in phase.sweep:
             n = int(counts.get((agent, knob), 0))
             if n != expected:
@@ -48,30 +63,41 @@
     df = load_results(args.out, args.phase)
     if df.empty:
         raise SystemExit(f"no results found under {args.out}/{args.phase}")
-    validate_results(df, phase)
+    settings = report_settings(cfg)
+    validate_results(df, phase, settings)
     require_final(df, allow_provisional=args.allow_provisional)
 
     figs = Path(args.figs)
     figs.mkdir(parents=True, exist_ok=True)
-    curves = curve_table(df)
+    metric, lower = settings["metric"], settings["lower_is_better"]
+    curves = curve_table(df, metric=metric)
     curves.to_csv(figs / "curves.csv", index=False)
-    gap_table(df, phase.familiar).to_csv(figs / "gaps.csv", index=False)
+    gap_table(df, phase.familiar, metric=metric).to_csv(figs / "gaps.csv", index=False)
 
-    pair = pair_table(df, "kairi", "kaiser")
+    first, second = settings["agents"][0], settings["agents"][1]
+    pair = pair_table(df, first, second, metric=metric, lower_is_better=lower)
     pair["crossover_intervals"] = str(crossings(pair))
     pair.to_csv(figs / "crossovers.csv", index=False)
 
     team_rows = []
     for team in sorted(a for a in df["agent"].unique() if a.startswith("tt_")):
-        t = pair_vs_best_single(df, team)
+        t = pair_vs_best_single(df, team, metric=metric, lower_is_better=lower)
         t.insert(0, "team", team)
         team_rows.append(t)
     if team_rows:
         pd.concat(team_rows).to_csv(figs / "pair_vs_best_single.csv", index=False)
 
-    plot_degradation(curves, phase.familiar, figs / "degradation.png")
+    for extra, extra_lower in settings["extra_metrics"].items():
+        curve_table(df, metric=extra).to_csv(figs / f"curves__{extra}.csv", index=False)
+        extra_pair = pair_table(df, first, second, metric=extra, lower_is_better=extra_lower)
+        extra_pair.to_csv(figs / f"crossovers__{extra}.csv", index=False)
+
+    plot_degradation(
+        curves, phase.familiar, figs / "degradation.png",
+        title=settings["title"], xlabel=settings["xlabel"], ylabel=settings["ylabel"],
+    )
     print(f"wrote tables and figure to {figs}")
-    print(f"kairi-vs-kaiser crossover intervals: {crossings(pair) or 'none (one agent dominates, or all ties)'}")
+    print(f"{first}-vs-{second} crossover intervals: {crossings(pair) or 'none (one agent dominates, or all ties)'}")
 
 
 if __name__ == "__main__":
PATCH
```

- [ ] **Step 4: Run the whole suite** (the bandit report tests must still pass unchanged)

Run: `pytest`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xp/stats/summary.py xp/plots/curves.py xp/report.py tests/test_summary.py tests/test_minigrid_report.py
git commit -m "feat: metric and direction are parameters of stats, plots and report

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Colab launcher, docs, spike cleanup

**Files:**
- Create: `notebooks/run_minigrid.ipynb`
- Modify: `CLAUDE.md`, `STATUS.md`, `RESUME.md`
- Delete: `spikes/`, `notebooks/spike_minigrid.ipynb` (the spike numbers are absorbed into this plan and the spec amendment)

- [ ] **Step 1: Create the launcher notebook.** Every shell command uses literal paths: `$VAR` expansion came out empty in Colab in this project.

Create `notebooks/run_minigrid.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# XP Phase 2 (MiniGrid DoorKey) on Colab\n",
    "Use a CPU runtime. Results go to Drive; every unit is resumable, so after a disconnect rerun the cell you were on.\n",
    "Shell commands below use literal paths on purpose: `$VAR` expansion came out empty in Colab.\n",
    "Order: setup, then calibrate (once), then edit `budget.steps` in `configs/minigrid.yaml` in the repo and push, then reclone and run, then report."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd /content\n",
    "!rm -rf xp-agents && git clone --depth 1 https://github.com/knguyen0923/xp-agents xp-agents\n",
    "%cd /content/xp-agents\n",
    "!pip install -q -r requirements-minigrid.txt -e ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calibration: default-entropy Kairi on the 3 tune seeds, up to the cap. Writes a suggested budget S.\n",
    "%cd /content/xp-agents\n",
    "!python -m xp.minigrid.calibrate --config configs/minigrid.yaml --out /content/drive/MyDrive/xp-results/minigrid_calibration.json --workers 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Full Phase 2 run (tune, freeze, evaluate zero-shot and few-shot). Resumable.\n",
    "%cd /content/xp-agents\n",
    "!python -m xp.run --phase minigrid --config configs/minigrid.yaml --out /content/drive/MyDrive/xp-results --workers 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%cd /content/xp-agents\n",
    "!python -m xp.report --phase minigrid --config configs/minigrid.yaml --out /content/drive/MyDrive/xp-results --figs /content/drive/MyDrive/xp-figs-minigrid\n",
    "from IPython.display import Image\n",
    "Image('/content/drive/MyDrive/xp-figs-minigrid/degradation.png')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "None",
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
```

- [ ] **Step 2: Check the notebook parses and contains no shell variables**

```bash
python -c "import json; json.load(open('notebooks/run_minigrid.ipynb'))"
grep -n '!python\|!pip' notebooks/run_minigrid.ipynb | grep '\$' || echo "no \$VAR in shell lines"
```
Expected: the second command prints `no $VAR in shell lines`.

- [ ] **Step 3: Update `CLAUDE.md`.** In the Commands code block, add after the `python -m xp.report ...` line:

```bash
pip install -r requirements-minigrid.txt -e .   # Phase 2 only: adds torch + minigrid
python -m xp.minigrid.calibrate --config configs/minigrid.yaml --out calibration.json --workers 3
python -m xp.run --phase minigrid --config configs/minigrid.yaml --out results --workers 3
python -m xp.report --phase minigrid --config configs/minigrid.yaml --out results --figs figs
```

In the Architecture list, add after the `xp/bandit/` bullet:

```markdown
- `xp/minigrid/` is PyTorch PPO (`Learner`; Kaiser adds RND). The unit of work is one trained agent per
  (agent, training seed); tuning scores and per-seed results are cached by config hash. Train, validation and
  test layouts come from disjoint seed pools (`xp/minigrid/env.py`). Never import it from `xp.protocol`.
```

- [ ] **Step 4: Remove the spike and update the handoff files**

```bash
git rm -r spikes notebooks/spike_minigrid.ipynb
```
Edit `STATUS.md`: replace the Phase 2 "plan not written" and spike paragraphs with: plan executed through Task 8, `xp/minigrid/` implemented and tested, remaining work = Task 9 (calibrate, run, report, README results). Refresh the `STATUS_COMMIT` marker and "Last updated" to the current `git rev-parse --short HEAD` after committing. Edit `RESUME.md` to describe the next step (Task 9, Step 1).

- [ ] **Step 5: Run the full suite and commit**

```bash
pytest
git add notebooks/run_minigrid.ipynb CLAUDE.md STATUS.md RESUME.md
git commit -m "docs: minigrid Colab launcher, CLAUDE.md commands, remove spike

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Run the study and write it up (operational)

This task produces results, not code. Report to the user at every decision point.

**Compute estimate** (from the spike; S is the calibrated budget in steps):

- Calibration: 3 tune seeds x up to 2M steps = up to 6M steps.
- Tuning: 6 grid points x 3 seeds x 2 agents = 36 trainings of S steps.
- Final: 2 agents x 10 seeds = 20 trainings of S steps, plus 2 agents x 10 seeds x 2 sizes = 40 few-shot adaptations of 200k steps (8M steps).
- Total about 56 x S + 8M steps. At S = 1M that is about 64M steps. Laptop with 5 workers (about 10,000 steps/s combined): about 2 hours. Colab free CPU with 2 workers (about 1,700 steps/s combined): about 10 hours, split across resumable sessions.

- [ ] **Step 1: Ask the user where to run it.** `CLAUDE.md` says full runs go to Colab; the laptop is about 5x faster here. Do not start a laptop run without an explicit yes. On the laptop use `--workers 3` or more; on Colab use `--workers 2`.

- [ ] **Step 2: Calibrate the budget**

```bash
python -m xp.minigrid.calibrate --config configs/minigrid.yaml --out calibration.json --workers 3
```
Expected: `suggested budget S = <steps> steps (80% target reached)`. If it prints `NOT reached; using the cap`, keep S = 2,000,000 and record in the write-up that tuned Kairi's familiar-size success at the budget is below 80%. Set `budget.steps` in `configs/minigrid.yaml` to the suggested value, commit (`git commit -am "config: calibrated Phase 2 budget S"`), and do not edit the config again unless tuning requires widening (Step 4): any edit changes the config hash.

- [ ] **Step 3: Run the phase**

```bash
python -m xp.run --phase minigrid --config configs/minigrid.yaml --out results --workers 3
```
Expected log lines: `[minigrid] tuned kairi: {'ent_coef': ...} score ... boundary=...`, the same for kaiser, then `[minigrid] finished 20 seed jobs`. The run is resumable; rerunning continues from the cached jobs. If a tuned optimum is on a grid edge the line shows `boundary=['ent_coef']` (or `rnd_coef`).

- [ ] **Step 4: Apply the provisional rule.** Run the report without `--allow-provisional`:

```bash
python -m xp.report --phase minigrid --config configs/minigrid.yaml --out results --figs figs
```
If it raises `ProvisionalResultError`, widen the grid of the flagged agent **and** the other agent to keep the grids equal in size (both grids get the same number of new points), edit `configs/minigrid.yaml`, commit, and rerun into a **fresh** `--out` directory (`results-b`); the old results carry the old config hash and will correctly raise `ConfigMismatchError`. Report the cost to the user first, since a rerun retrains everything.

- [ ] **Step 5: Read the tables.** `figs/crossovers.csv` (Kairi minus Kaiser success, paired CI, winner per size, `crossover_intervals`), `figs/gaps.csv` (gap vs size 6), `figs/curves.csv`, `figs/curves__success_fewshot.csv`, `figs/crossovers__success_fewshot.csv`, `figs/curves__steps_to_80.csv`, `figs/crossovers__steps_to_80.csv`, `figs/degradation.png`. Show the figure to the user. Kairi winning some regimes is a valid result; do not tune further to change the outcome.

- [ ] **Step 6: Write `README.md`** with the spec Section 6 structure: question, protocol (tune on size 6, freeze, evaluate at every size; pools; budget S and how it was chosen), results (every number with its 95% CI taken from the tables, the crossover statement or an explicit "curves do not cross", the few-shot result, sample efficiency with the censoring note), limitations (one familiar size, single seed-pool, `Learner` hyperparameters not tuned beyond the one coefficient each, spike-observed seed variance, size 12 at the floor for zero-shot), a "where each agent loses" section, and future work (Crafter, Showdown, deep-RL tag-team). Commit the README and `figs/` selection the user wants kept (figures only; `results/` is git-ignored).

- [ ] **Step 7: Close out.** Update `STATUS.md` (Phase 2 done, results location, commit hash, marker) and `RESUME.md`, run `pytest`, commit, and ask the user before pushing.
