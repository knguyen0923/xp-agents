# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 899ff48 -->

## Current state

- Design spec approved: `docs/superpowers/specs/2026-10-22-xp-design.md` (amended during Phase 1:
  Page-Hinkley trigger, 500 tune seeds, crossover reporting semantics).
- Foundation + Phase 1 (bandit, tag-team) implemented and reviewed: plan
  `docs/superpowers/plans/2026-10-22-xp-foundation-and-bandit.md`, fix-wave report
  `docs/superpowers/plans/2026-10-22-fix-wave-report.md`. 88 fast tests pass. Probe on disjoint
  seeds: no agent flagged provisional with the committed grids.
- **Not done:** the real Phase 1 run (Colab, `notebooks/run_phase.ipynb`) and the README results.
  First Colab attempt failed because notebook variables (`$PHASE`) were empty after a runtime
  restart; the notebook cells are now self-contained (commit `dd396d0`).
  Colab is on Python 3.13, where pinned `numpy==2.0.2` compiled from source and tripped resolver
  warnings. Fixed by loosening `requirements.txt` to `>=` lower bounds (commit `0734460`, pushed).
  Notebook shell commands now use literal paths (commit `c1b33f4`, local, not pushed) because
  `$PHASE` expanded empty in Colab. Phase 1 run and report completed on Colab (Kaiser best single agent everywhere; `tt_fixed`
  beats it only at lambda>=0.01; `tt_meta`/`tt_triggered` never robustly win). README results
  still to write; need `gaps.csv` and `crossovers.csv` CIs.
  Untested: whether results match across numpy versions; rerun the Colab setup cell to confirm.
- Phase 2 (MiniGrid DoorKey, PPO vs PPO+RND): spec section 5.2-5.3 approved, **plan written** (`docs/superpowers/plans/2026-10-23-phase2-minigrid.md`, 9 tasks; code prototyped and replayed task by task in a scratch copy, 103+ tests green; executing on branch `feat/phase2-minigrid` via subagents: Tasks 1-5 of 9 committed (`c4e1c9d` config/deps/CI/spec, `3aa575e` DoorKey env + layout pools, `501dc87` PPO Learner + RND, `f9aa66e` shared manifest + job stores, `5f57204` phase runner + CLI; Task 5 review in flight), Tasks 6-8 pending, Task 9 (calibrate + full run) needs the user's laptop-vs-Colab decision; ledger in `.superpowers/sdd/`, git-ignored).
  Spike results (seed 0, 500k steps): ~860 sps, ~10 min/run; PPO hit 80% at 143k steps; zero-shot
  from 6x6: size 5/6 = 100%, 8 = 42%, 12 = 0% (floor). PPO+RND got 0% everywhere (below random):
  suspected cause is a positive, episodic intrinsic bonus that penalizes reaching the goal; fix
  is a non-episodic intrinsic stream with its own value head. Diagnostic v1 (150k, stale clone, no ep_len/bonus) was inconclusive: even coef 0.0 had not learned
  by 150k, so it cannot separate bonus effect from slow sparse-reward learning (also single seed).
  v2 result: control PPO reaches 100% (80% at 143,360 steps); RND at coef 0.1 and 0.01 both 0%.
  Root cause found locally (scratch venv, ~2-3k sps, 2-3x Colab): intrinsic returns were scaled by
  per-step reward std, so the intrinsic value targets were huge and, via the shared trunk and global
  grad clip, crushed the policy gradient (even coef 0.0 stayed at 0%). Fix in `spikes/spike_minigrid.py`:
  non-episodic intrinsic stream, second value head, bonus scaled by std of the discounted return.
  Result (250k steps, seeds 0/1, eval success at 6x6): coef 0 = 88/56%, 0.001 = 46/98%, 0.01 = 42/88%,
  0.1 = 0/0% (intrinsic advantage noise dominates; needs tuning). Plain PPO is high-variance across
  seeds (seed 0 only 12% at 250k locally, seed 1 47%), so Phase 2 needs many seeds and longer budgets.
  The bonus decays to ~0 and RND shows no visible benefit over coef 0 on DoorKey-6.
  Superseded diagnostics v1-v3 cells remain in the notebook. v2 cells (control PPO + coef 0.1/0.01, 250k) added; rerun clone cell first. Diagnostic cells added (`79ceda6`,
  local, not pushed); run on Colab before writing the plan.
  A throwaway spike (`spikes/spike_minigrid.py`, `notebooks/spike_minigrid.ipynb`) is written and
  smoke-tested; notebook shell args hardcoded (`b0898e4`, pushed); its Colab results decide the plan's grids and budgets.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), default branch `main`.
  `origin/main` includes the requirements fix (`0734460`); the STATUS refresh and `c1b33f4` are local only (push needs the user's yes).

## Last updated

2026-10-23 (commit 899ff48)
