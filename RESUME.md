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

## Also pending: Phase 1 README results (bandit run is complete on Drive, `xp-results` / `xp-figs`)
- Still needed from the user: contents of `crossovers.csv` and `gaps.csv` (Colab: `pd.read_csv(...).round(4).to_string()`), to confirm the Kairi-vs-Kaiser gap CIs.
- Already known (point estimates + `pair_vs_best_single` CIs): Kaiser has the lowest regret of the single agents at every lambda (Kairi is never best; Kairi-Kaiser gap .033 at lambda 0, peak .072 at 0.005, .060 at 0.02). `tt_fixed` is worse than Kaiser for lambda <= 0.002, tied at 0.005, better at 0.01 (-0.020) and 0.02 (-0.029), CIs excluding 0. `tt_meta` never beats Kaiser. `tt_triggered` ties Kaiser except one win at 0.005 (-0.0098), likely noise (1 of 21 comparisons).
- Caveats for the write-up: `tt_*` agents are in the config but not described in CLAUDE.md/spec; ties between Kaiser and `tt_triggered` at low lambda.

## Phase 2 facts worth remembering when writing it up
- PPO on DoorKey-6 is very seed-dependent (600k steps: seeds 0/2/3 reached 0%, 100%, 39%); expect the calibrated budget near the 2M cap and wide CIs.
- Test layouts at sizes 5/6 are held-out seeds, not unseen layouts (47/164 distinct of 200 test seeds); truncation counts as terminal in extrinsic GAE; RND showed no benefit over plain PPO at size 6 in the spike.
