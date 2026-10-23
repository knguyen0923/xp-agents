# XP — when does uncertainty-driven exploration pay off?

XP studies the explore-vs-exploit tradeoff by comparing two agents across worlds that shift
progressively away from the conditions they were tuned on.

- **Kairi** — model-free, reward-driven learner (ε-greedy in the bandit, plain PPO in MiniGrid).
- **Kaiser** — Kairi plus an explicit uncertainty-driven exploration term (discounted UCB in the
  bandit, PPO + RND bonus in MiniGrid). Defining principle: optimism in the face of uncertainty.

The output is a map of *when each agent wins and where it breaks*, built on a reusable protocol.
Kairi winning some regimes is a valid result. Design: `docs/superpowers/specs/2026-10-22-xp-design.md`.

## Protocol

1. Tune each agent on the familiar knob value, using tune seeds only.
2. Freeze the tuned parameters.
3. Evaluate at every knob value on disjoint eval seeds.
4. Report degradation curves, gaps and crossovers with 95% bootstrap CIs (the independent seed is the
   resampling unit; paired comparisons share eval seeds).

Both agents get equal-sized tuning grids and budgets. A tuned optimum on a grid edge marks the run
`provisional` and it is excluded from final tables. Runs are resumable and keyed by config hash.

## Phase 1 — changing bandit

10-armed Gaussian bandit (σ = 1, horizon 2000) whose best arm jumps at rate λ. Tuned at
λ = 0.001; evaluated at λ ∈ {0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02}. 500 tune seeds, 1000 eval
seeds. Metric: regret per step (lower is better). Three tag-team agents (`tt_fixed`, `tt_triggered`,
`tt_meta`) that switch between Kairi and Kaiser are also evaluated.

### Results

- Kaiser has the lowest regret of the two single agents at every λ; Kairi is never best.
  Kairi − Kaiser gap: 0.033 at λ = 0, peak 0.072 at λ = 0.005, 0.060 at λ = 0.02.
- `tt_fixed` (fixed 50-step blocks) is worse than Kaiser for λ ≤ 0.002, tied at 0.005, and better
  at 0.01 (−0.020) and 0.02 (−0.029); those CIs exclude 0.
- `tt_meta` never beats Kaiser.
- `tt_triggered` ties Kaiser except one win at λ = 0.005 (−0.0098), likely noise (1 of 21 comparisons).

<!-- TODO: confirm Kairi-vs-Kaiser gap CIs and crossover statements from gaps.csv / crossovers.csv -->

## Phase 2 — MiniGrid DoorKey

PPO (Kairi) vs PPO + RND bonus (Kaiser). Tuned on size 6; evaluated zero-shot at sizes
{5, 6, 8, 12} and few-shot at {8, 12}. Metric: success rate.

### Results

<!-- TODO: fill from the Phase 2 run (plan Task 9 Step 6) -->

## Limitations

- Phase 1: the `tt_*` agents go beyond the original two-agent design; Kaiser and `tt_triggered` tie at low λ.
- Phase 1 numbers were produced on Colab (Python 3.13); cross-numpy-version agreement is untested.
- Phase 2: one familiar size; one seed pool; PPO hyperparameters are not tuned beyond one coefficient
  per agent; PPO on DoorKey-6 is very seed-dependent, so CIs are wide.
- Phase 2: test layouts at sizes 5 and 6 are held-out seeds, not unseen layouts; size 12 is at the
  floor for zero-shot.

## Where each agent loses

<!-- TODO: write after both phases' results are final -->

## Future work

Crafter, Showdown, and a deep-RL tag-team agent.

## Reproducing

See `CLAUDE.md` for commands. Full runs go to Colab (`notebooks/`), Phase 2 may also run locally.
