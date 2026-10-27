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
pip install -r requirements-minigrid.txt -e .   # Phase 2 only: adds torch + minigrid
python -m xp.minigrid.calibrate --config configs/minigrid.yaml --out calibration.json --workers 3
python -m xp.run --phase minigrid --config configs/minigrid.yaml --out results --workers 3
python -m xp.report --phase minigrid --config configs/minigrid.yaml --out results --figs figs
```

After installing `poke-env`, run `python -m xp.showdown.patch` once (it lacks the Champions-only species; without the patch a Mega battle hangs, and a guard test fails).
Showdown phase: `scripts/run_showdown.sh [config] [out] [workers]` (starts the local server if needed, resumable; `configs/showdown-smoke.yaml` is a 75-second end-to-end check, `configs/showdown.yaml` is the real run of about 12 to 15 hours; a resume needs the same git commit, or `ALLOW_CODE_CHANGE=1`), then `python -m xp.report --phase showdown ...` as for MiniGrid.
Showdown tests that need the server: `XP_NODE=<node 22+ binary> pytest -m showdown` (server started as in `RESUME.md`; `pip install -r requirements-showdown.txt -e .` first).

Full runs go to Colab via `notebooks/run_phase.ipynb` (Phase 1) and `notebooks/run_minigrid.ipynb` (Phase 2); do not burn the laptop on them.

## Architecture

- `xp/protocol/` is phase-agnostic: `Phase` spec, tuner, evaluator, results store. It must never
  import from a phase package (`xp.bandit`, later `xp.minigrid`).
- `xp/bandit/` is vectorized NumPy. The world realization is pre-generated per seed, so the world
  is deterministic regardless of batching. Agent randomness is deterministic per config but
  depends on chunk composition. Agents run vectorized across seeds.
- `xp/minigrid/` is PyTorch PPO (`Learner`; Kaiser adds RND). The unit of work is one trained agent per
  (agent, training seed); tuning scores and per-seed results are cached by config hash. Train, validation and
  test layouts come from disjoint seed pools (`xp/minigrid/env.py`). Never import it from `xp.protocol`.
- `xp/showdown/` is the Pokemon Showdown phase (`poke-env` doubles env, fixed feature encoder, team pools). It needs a local Showdown server
  (outside the repo) only for tests marked `showdown`, which the default `pytest` run excludes. Never import it from `xp.protocol`.
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
- Scope: bandit and MiniGrid are the core. Crafter was spiked and then CLOSED as a documented spike on 2026-10-26 (user decision, for time; may be reconsidered later; plain PPO missed the usable-range gate, see `docs/spikes/crafter_spike.md`); reopening needs a spec amendment raising the budget S first. Showdown was spiked (`docs/spikes/showdown_spike.md`) and building was approved by the user on 2026-10-26 under the spec 5.6 amendment; the budget cap S needs the user's approval before any RND result is read.
