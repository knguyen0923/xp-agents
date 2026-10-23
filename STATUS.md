# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: e13d3c1 -->

## Current state

- Design spec approved: `docs/superpowers/specs/2026-10-22-xp-design.md` (amended during Phase 1:
  Page-Hinkley trigger, 500 tune seeds, crossover reporting semantics).
- Foundation + Phase 1 (bandit, tag-team) implemented and reviewed: plan
  `docs/superpowers/plans/2026-10-22-xp-foundation-and-bandit.md`, fix-wave report
  `docs/superpowers/plans/2026-10-22-fix-wave-report.md`. 88 fast tests pass. Probe on disjoint
  seeds: no agent flagged provisional with the committed grids.
- **Not done:** Phase 1 README results are drafted in `README.md` (commit `9928aec`, TODOs for gap/crossover CIs from `gaps.csv`/`crossovers.csv`; the Colab run is complete).
  First Colab attempt failed because notebook variables (`$PHASE`) were empty after a runtime
  restart; the notebook cells are now self-contained (commit `dd396d0`).
  Colab is on Python 3.13, where pinned `numpy==2.0.2` compiled from source and tripped resolver
  warnings. Fixed by loosening `requirements.txt` to `>=` lower bounds (commit `0734460`, pushed).
  Notebook shell commands now use literal paths (commit `c1b33f4`) because
  `$PHASE` expanded empty in Colab. Phase 1 run and report completed on Colab (Kaiser best single agent everywhere; `tt_fixed`
  beats it only at lambda>=0.01; `tt_meta`/`tt_triggered` never robustly win). README results
  still to write; need `gaps.csv` and `crossovers.csv` CIs.
  Untested: whether results match across numpy versions; rerun the Colab setup cell to confirm.
- Phase 2 (MiniGrid DoorKey, PPO vs PPO+RND): plan `docs/superpowers/plans/2026-10-23-phase2-minigrid.md` executed through Task 8 and merged into `main` (pushed): `xp/minigrid/` implemented and tested (123 tests), Colab launcher `notebooks/run_minigrid.ipynb`, CLAUDE.md updated. Remaining work = Task 9 (calibrate the budget, full run, report, README results), which the user chose to run locally on the laptop later (go-ahead needed before starting; see RESUME.md). The spike scripts were removed; their numbers are in the plan and the spec amendment.
- Phase 2 calibration finished locally (2026-10-23): 80% target NOT reached, so `budget.steps` = 2,000,000 (the cap), committed in `e13d3c1`. `calibration.json` is untracked. Next: get the user's go-ahead, then the full run (~4h at `--workers 5`), report, README Phase 2 results. Write the unmet 80% target into the limitations.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), default branch `main`.
  `origin/main` includes the requirements fix (`0734460`); `c1b33f4` is on `origin/main`; the `feat/phase2-minigrid` branch is local only (push needs the user's yes).

## Last updated

2026-10-23 (commit e13d3c1)
