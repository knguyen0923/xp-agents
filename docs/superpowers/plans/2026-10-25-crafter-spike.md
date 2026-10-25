# Crafter Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, with plain PPO only, whether Crafter has a usable difficulty range at world area 32 or 64, whether world seeds are controllable, and what the full Crafter phase would cost, so the user can make the go/no-go call in spec 5.5.

**Architecture:** One throwaway script, `scripts/spikes/crafter_spike.py` (a subprocess vector env over `crafter.Env`, a small CNN PPO, per-window achievement statistics, and a gate analyzer), plus tests for its pure helpers. No change to `xp/`, no RND, no protocol code. The write-up lands in `docs/spikes/crafter_spike.md`. The full phase gets its own plan only after the user says go.

**Tech Stack:** Python 3.9, PyTorch (CPU or MPS), `crafter` 1.8.3 (installed in `.venv`, not in `requirements*.txt`; the full phase would add `requirements-crafter.txt`), numpy.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md`, section 5.5 (the spike part). Binding.

## Global Constraints

- Python floor 3.9; every module starts with `from __future__ import annotations`.
- Plain PPO only. **No RND result is read** in this spike (spec 5.5). An RND smoke test is allowed but is not part of this plan.
- Gates (spec 5.5, G1 raised from 3 to 5 before any result was read, because a smoke test showed a near-random policy already averages about 3 distinct achievements): G1 final `ach_100` at least 5 and at least 5 of 22 achievements unlocked in under 5% of the last 100 episodes; G2 final `ach_100` exceeds the mean over the first 10% of training; G3 the full-phase cost is acceptable to the user.
- Areas considered: 32 and 64 only (96 is excluded on cost). Budget 1,000,000 steps, 1 seed (seed 0), per area. Tie-break between passing areas: the cheaper one, then 64.
- The spike runs on the laptop (`caffeinate -i`), but **the long runs (Task 4) need the user's go-ahead**. Never push without the user's yes.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Keep `STATUS.md` current (commit `RESUME.md` and other changes first, then a STATUS-only marker commit).
- Facts already established (do not redo): `crafter.Env` world seed for an episode is `hash((env_seed, episode)) % (2**31 - 1)` where `episode` counts resets from 1 (read from the installed source); the script was smoke-run for 40,960 steps at area 32: CPU 572 steps/s, MPS 701 steps/s, `ach_100` about 3 after 41k steps (near-random policy level).

---

### Task 1: Tests for the script's helpers and a commit of the script

**Files:**
- Already written, uncommitted: `scripts/spikes/crafter_spike.py`
- Create: `tests/test_crafter_spike.py`

**Interfaces:**
- Consumes: `crafter_spike.world_seed(env_seed, episode)`, `EpisodeStats(window)` with `.add(unlocked)`, `.ach()`, `.rates(names)`, `gates(curve, rates)`.
- Produces: a passing test file that CI can run without `crafter` installed (tests that need `crafter` use `pytest.importorskip`).

- [ ] **Step 1: Write the tests**

```python
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "spikes", "crafter_spike.py")
_SPEC = importlib.util.spec_from_file_location("crafter_spike", _PATH)
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)


def test_episode_stats_window_and_rates():
    st = cs.EpisodeStats(window=2)
    assert st.ach() == 0.0
    st.add(["a"])
    st.add(["a", "b", "c"])
    st.add(["b"])  # the first episode falls out of the window
    assert st.ach() == 2.0
    assert st.total == 3
    r = st.rates(["a", "b", "c", "d"])
    assert r == {"a": 0.5, "b": 1.0, "c": 0.5, "d": 0.0}


def test_gates_pass_and_fail():
    rates_ok = {k: 0.0 for k in "abcde"}
    rates_ok["f"] = 0.9
    g = cs.gates([(10240, 3.0), (500000, 4.0), (1000000, 6.0)], rates_ok)
    assert g["G1"] and g["G2"] and g["rare_achievements"] == 5
    low = cs.gates([(10240, 3.0), (1000000, 3.2)], rates_ok)
    assert not low["G1"]  # final below 5
    flat = cs.gates([(10240, 6.0), (1000000, 6.0)], rates_ok)
    assert flat["G1"] and not flat["G2"]  # no growth
    no_headroom = cs.gates([(10240, 3.0), (1000000, 7.0)], {k: 0.5 for k in "abcdef"})
    assert not no_headroom["G1"]  # fewer than 5 rare achievements


def test_world_seed_pools_are_disjoint():
    def pool(bases, episodes):
        return {cs.world_seed(b * 1000 + i, e) for b in bases for i in range(8) for e in range(1, episodes + 1)}

    train, ev = pool(range(0, 3), 2000), pool(range(100, 110), 200)
    assert len(train) > 0.99 * 3 * 8 * 2000  # almost no collisions inside a pool
    assert not (train & ev)


def test_same_env_seed_gives_same_first_world():
    crafter = pytest.importorskip("crafter")
    a = crafter.Env(area=(32, 32), seed=7)
    b = crafter.Env(area=(32, 32), seed=7)
    c = crafter.Env(area=(32, 32), seed=8)
    assert np.array_equal(a.reset(), b.reset())
    assert not np.array_equal(a.reset(), c.reset())
    assert cs.world_seed(7, 1) == hash((7, 1)) % (2 ** 31 - 1)


def test_achievement_names_has_22():
    pytest.importorskip("crafter")
    assert len(cs.achievement_names()) == 22
```

- [ ] **Step 2: Run the tests**

Run: `source .venv/bin/activate && pytest tests/test_crafter_spike.py -v`
Expected: 5 passed (3 passed and 2 skipped if `crafter` were missing). If `test_same_env_seed_gives_same_first_world` fails, **stop and report**: the seeding assumption in spec 5.5 is wrong, which changes the plan. If `test_world_seed_pools_are_disjoint` finds a collision, report the pair; a fix (different pool bases) is a ruling for the controller, not a silent change.

- [ ] **Step 3: Run the whole suite**

Run: `pytest -q`
Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/crafter_spike.py tests/test_crafter_spike.py docs/superpowers/specs/2026-10-22-xp-design.md
git commit -m "feat(spike): Crafter spike script (plain PPO) and helper tests

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Throughput at area 64 and the cost table

**Files:**
- Create: `docs/spikes/crafter_spike.md` (started here, completed in Task 5)

**Interfaces:**
- Consumes: the CLI `python scripts/spikes/crafter_spike.py --area A --steps N --device D --out DIR`, which prints a JSON with `steps_per_s` and writes `DIR/area{A}-seed{seed}/curve.csv` and `summary.json`.
- Produces: measured steps/s for (area, device) pairs and the full-phase cost estimate used by gate G3.

- [ ] **Step 1: Measure area 64 and re-measure area 32 (short runs, about 2 minutes each)**

```bash
source .venv/bin/activate
for area in 32 64; do for dev in cpu mps; do
  echo "== area $area $dev"
  python scripts/spikes/crafter_spike.py --area $area --steps 40960 --device $dev --out /tmp/crafter-bench 2>&1 | grep -E '"steps_per_s"|Error|Traceback'
done; done
```

Expected: four `steps_per_s` lines (area 32: about 570 on CPU and 700 on MPS from the earlier smoke run). If MPS errors, record the error text and continue with CPU numbers only.

- [ ] **Step 2: Write the cost table into the doc**

Create `docs/spikes/crafter_spike.md` with this content, filling the four measured numbers (the formula rows are arithmetic, not estimates of the unknown):

```markdown
# Crafter spike (spec 5.5)

Status: IN PROGRESS. Plain PPO only; no RND result was read. Script: `scripts/spikes/crafter_spike.py`; plan: `docs/superpowers/plans/2026-10-25-crafter-spike.md`.

## Throughput (40,960-step runs, 8 envs, PPO with a small CNN, includes start-up)
| area | CPU steps/s | MPS steps/s |
|---|---|---|
| 32 | <fill> | <fill> |
| 64 | <fill> | <fill> |

## Full-phase cost (spec 5.5: 36 tuning runs + 20 eval runs at 1M steps = 56M steps)
Hours = 56,000,000 / steps-per-second / 3600, using the faster device and one run at a time.
| area | steps/s used | hours (one run at a time) |
|---|---|---|
| 32 | <fill> | <fill> |
| 64 | <fill> | <fill> |
Concurrent runs would be faster only if cores are free; the spike did not measure that.
```

- [ ] **Step 3: Commit**

```bash
git add docs/spikes/crafter_spike.md
git commit -m "docs: Crafter spike throughput and cost table

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Ask the user for the go-ahead on the two 1M-step runs

- [ ] **Step 1: Report the Task 2 cost table and the expected spike time** (two runs of 1M steps; at about 700 steps/s area 32 is about 25 minutes, and area 64 is longer by the measured ratio). Ask: "May I launch the two 1M-step plain-PPO spike runs now (laptop, `caffeinate -i`)?" Do not launch without a yes. Mention that the spike reads no RND result.

---

### Task 4: Run the spike (needs the user's go-ahead from Task 3)

**Files:**
- Create (git-ignored results): `results/crafter-spike/area32-seed0/`, `results/crafter-spike/area64-seed0/`

**Interfaces:**
- Consumes: the CLI above. Seed 0, so env seeds are 0 to 7 (`base_seed * 1000 + i`).
- Produces: `curve.csv` (steps, ach_100, episodes, steps_per_s) and `summary.json` (area, steps, seed, device, secs, per-achievement rates over the last 100 episodes) per area.

- [ ] **Step 1: Launch both runs in the background, one after the other**

Use the faster device from Task 2.

```bash
source .venv/bin/activate
(caffeinate -i bash -c 'for a in 32 64; do python scripts/spikes/crafter_spike.py --area $a --steps 1000000 --seed 0 --device <DEVICE> --out results/crafter-spike; done' > crafter-spike.log 2>&1; echo "exit $?" >> crafter-spike.log) &
```

Do not poll; use a background watcher that waits for the `exit` line. Progress is printed every 10 curve rows. Do not edit `scripts/spikes/crafter_spike.py` while the runs are going.

- [ ] **Step 2: Confirm both finished**

Run: `tail -3 crafter-spike.log; ls results/crafter-spike/*/summary.json`
Expected: `exit 0` and two `summary.json` files. If a run crashed, report the traceback; a rerun is the same command for that area.

---

### Task 5: Analyze, decide, and write up

**Files:**
- Modify: `docs/spikes/crafter_spike.md`, `docs/FINDINGS.md` (a short section), `docs/superpowers/specs/2026-10-22-xp-design.md` (a "Spike results" paragraph at the end of 5.5), `RESUME.md`, `STATUS.md`
- Copy into the repo (small): `docs/spikes/crafter_spike_curves.csv` (the two `curve.csv` files, with an `area` column) and `docs/spikes/crafter_spike_rates.csv` (area, achievement, rate)

- [ ] **Step 1: Run the analyzer for each area**

```bash
source .venv/bin/activate
python scripts/spikes/crafter_spike.py --analyze results/crafter-spike/area32-seed0
python scripts/spikes/crafter_spike.py --analyze results/crafter-spike/area64-seed0
```

- [ ] **Step 2: Write the results into `docs/spikes/crafter_spike.md`**

Replace the status line with `Status: DONE <date>`. Add, with the real numbers: a table of G1, G2 and the values behind them per area (final `ach_100`, early `ach_100`, count of rare achievements); the per-achievement unlock rates (top and bottom five); the world-seed finding from Task 1 (world seeds are per `(env_seed, episode)`, deterministic, pools disjoint by construction and checked); the cost from Task 2 filled with the measured full-run times; the chosen area by the spec's rule (passes G1 and G2, cheaper, ties toward 64) or "no area passes"; the **deep achievement set** (achievements unlocked in 5% to 95% of the last 100 episodes at the chosen area); and one recommendation (go, go with a changed setting, or stop). State plainly that this is one seed per area and that no RND result was read.

- [ ] **Step 3: Update the spec and FINDINGS**

Add a "Spike results (date)" paragraph at the end of 5.5 with the chosen area, the gate outcomes, the fixed budget S and the deep set (or the stop decision). Add a short "Crafter spike" section to `docs/FINDINGS.md` pointing to the spike doc.

- [ ] **Step 4: Commit the doc changes, then the STATUS marker**

```bash
git add docs RESUME.md
git commit -m "docs: Crafter spike results and recommendation

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
# then update STATUS.md (marker = the commit above) and commit it alone
```

- [ ] **Step 5: Report to the user**

Give the gate table, the chosen area (or the stop reason), the measured full-phase cost, and the recommendation. Ask for the go/no-go on the full Crafter phase. A "go" leads to a new brainstorm-free plan for the full phase (generalising the learner per spec 5.5); a "no-go" closes the phase with the spike as the result, as Step B was. Do not push without the user's yes.
