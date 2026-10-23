# XP — Resume

Paused-at: waiting on the user (nothing half-finished in the working tree).

Next steps, in order:
1. User pushes/approves push of local commits (spike script + notebook) to `origin/main`.
2. User runs `notebooks/run_phase.ipynb` (Phase 1) and `notebooks/spike_minigrid.ipynb` (spike) in
   Colab and reports the printed summaries. If the report raises `ProvisionalResultError`, widen the
   named agent's grid in `configs/bandit.yaml` (keep Kairi/Kaiser grids equal) and rerun.
3. From the spike numbers (steps/sec, steps to 80% success, zero-shot success at sizes 8 and 12 vs
   random), write the Phase 2 plan with the writing-plans skill. If sizes 8/12 are at 0% for every
   agent, switch the knob to MultiRoom room count (spec 5.2 fallback) and amend the spec.
4. Task 12 of the Phase 1 plan: copy outputs to `docs/results/phase1/`, write README from them
   (every number with its CI; "where each agent loses"; note tt_meta/tt_triggered findings).

SDD ledger (gitignored, local): `.superpowers/sdd/2026-10-22-xp-foundation-and-bandit/progress.md`
holds rulings and the 12 deferred minors.
