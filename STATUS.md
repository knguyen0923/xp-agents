# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 2430999 -->

## Current state

- Design spec approved: `docs/superpowers/specs/2026-10-22-xp-design.md` (amended during Phase 1:
  Page-Hinkley trigger, 500 tune seeds, crossover reporting semantics).
- Foundation + Phase 1 (bandit, tag-team) implemented and reviewed: plan
  `docs/superpowers/plans/2026-10-22-xp-foundation-and-bandit.md`, fix-wave report
  `docs/superpowers/plans/2026-10-22-fix-wave-report.md`. 88 fast tests pass. Probe on disjoint
  seeds: no agent flagged provisional with the committed grids.
- **Not done:** the real Phase 1 run (Colab, `notebooks/run_phase.ipynb`) and the README results.
- Phase 2 (MiniGrid DoorKey, PPO vs PPO+RND): spec section 5.2-5.3 approved, **plan not written**.
  A throwaway spike (`spikes/spike_minigrid.py`, `notebooks/spike_minigrid.ipynb`) is written and
  smoke-tested; its Colab results decide the plan's grids and budgets.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), default branch `main`.
  Local commits after `9e45451` are NOT pushed yet (push needs the user's yes).

## Last updated

2026-10-22
