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
