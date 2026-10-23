# XP — Resume

Paused-at: Phase 2 Tasks 1-8 complete, merged into `main` and pushed. Task 9 (the real run) not started.

Decision (user, 2026-10-23): run Phase 2 **locally on the laptop later** (not on Colab for now). CLAUDE.md still
says full runs go to Colab; the user's call overrides it for Phase 2, but ask for a go-ahead before starting.
Laptop is ~2-3x faster per core than Colab (about 2,000-2,800 steps/s with 2 torch threads); expect ~2 hours
at a 1M-step budget with `--workers 5`.

Next steps, in order:
1. When the user says go: `python -m xp.minigrid.calibrate --config configs/minigrid.yaml --out calibration.json --workers 3`,
   set `budget.steps` in `configs/minigrid.yaml` to the suggested S (commit it; any later config edit except the
   `report` block changes the hash).
2. `python -m xp.run --phase minigrid --config configs/minigrid.yaml --out results --workers 5`, then
   `python -m xp.report --phase minigrid ... --figs figs`. If it raises ProvisionalResultError, widen both grids
   equally and rerun into a fresh `--out`.
3. Write README results (Task 9 Step 6 of `docs/superpowers/plans/2026-10-23-phase2-minigrid.md`).
Note: a resume under a different git commit needs `--allow-code-change` (code-version guard).
