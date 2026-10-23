# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 79ceda6 -->

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
- Phase 2 (MiniGrid DoorKey, PPO vs PPO+RND): spec section 5.2-5.3 approved, **plan not written**.
  Spike results (seed 0, 500k steps): ~860 sps, ~10 min/run; PPO hit 80% at 143k steps; zero-shot
  from 6x6: size 5/6 = 100%, 8 = 42%, 12 = 0% (floor). PPO+RND got 0% everywhere (below random):
  suspected cause is a positive, episodic intrinsic bonus that penalizes reaching the goal; fix
  is a non-episodic intrinsic stream with its own value head. Diagnostic cells added (`79ceda6`,
  local, not pushed); run on Colab before writing the plan.
  A throwaway spike (`spikes/spike_minigrid.py`, `notebooks/spike_minigrid.ipynb`) is written and
  smoke-tested; notebook shell args hardcoded (`79ceda6`, local, not pushed); its Colab results decide the plan's grids and budgets.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), default branch `main`.
  `origin/main` includes the requirements fix (`0734460`); the STATUS refresh and `c1b33f4` are local only (push needs the user's yes).

## Last updated

2026-10-23 (commit 79ceda6)
