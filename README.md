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

- **Kaiser beats Kairi at every jump rate and the curves do not cross.** Kairi − Kaiser regret per step (95% CI):

  | λ | 0 | 0.0005 | 0.001 (familiar) | 0.002 | 0.005 | 0.01 | 0.02 |
  |---|---|---|---|---|---|---|---|
  | diff | 0.033 [0.028, 0.038] | 0.051 [0.044, 0.059] | 0.053 [0.045, 0.061] | 0.064 [0.055, 0.074] | 0.072 [0.061, 0.082] | 0.068 [0.058, 0.078] | 0.060 [0.051, 0.069] |

  All seven intervals exclude 0 (winner: Kaiser each time); Kairi is never best.
- **Degradation from the familiar λ** (regret increase over each agent's own value at λ = 0.001): at λ = 0.02, Kairi +0.552
  [0.540, 0.564], Kaiser +0.546 [0.538, 0.553], `tt_fixed` +0.498 [0.490, 0.506]; at λ = 0.005, Kairi +0.219 [0.206, 0.231]
  vs Kaiser +0.200 [0.193, 0.208]. These CIs are per agent against its own baseline; head-to-head comparisons are the table above.
- `tt_fixed` (fixed 50-step blocks) is worse than Kaiser for λ ≤ 0.002, tied at 0.005, and better
  at 0.01 (−0.020) and 0.02 (−0.029); those CIs exclude 0.
- `tt_meta` never beats Kaiser.
- `tt_triggered` ties Kaiser except one win at λ = 0.005 (−0.0098), likely noise (1 of 21 comparisons).


## Phase 2 — MiniGrid DoorKey

PPO (Kairi) vs PPO + RND bonus (Kaiser). Tuned on size 6; evaluated zero-shot at sizes
{5, 6, 8, 12} and few-shot at {8, 12}. Metric: success rate.

### Results

Tuned on size 6 (3 tune seeds): Kairi `ent_coef` = 0.03, Kaiser `rnd_coef` = 0.001; both interior optima
(not provisional). Budget S = 2M steps (the cap; calibration did not reach the 80% target, mean success ~0.67).
10 eval seeds per agent, 200 held-out layouts per size.

| size | Kairi | Kaiser | Kairi − Kaiser (95% CI) |
|---|---|---|---|
| 5 | 1.000 | 0.904 [0.71, 1.00] | 0.096 [0.00, 0.29] (tie) |
| 6 (familiar) | 1.000 | 0.900 [0.70, 1.00] | 0.100 [0.00, 0.30] (tie) |
| 8 | 0.640 [0.57, 0.71] | 0.496 [0.37, 0.58] | 0.143 [0.02, 0.29] (Kairi) |
| 12 | 0.083 [0.05, 0.12] | 0.038 [0.03, 0.05] | 0.046 [0.01, 0.08] (Kairi) |

- **The curves do not cross.** Kairi is at or above Kaiser at every size; with the tuned settings the difference excludes 0 at sizes 8 and 12 (but see the tie-sensitivity check below).
- **Few-shot (200k steps):** size 8, Kairi 1.00 vs Kaiser 0.90 [0.70, 1.00]; size 12, Kairi 0.72 [0.44, 0.97] vs
  Kaiser 0.50 [0.22, 0.78]. CIs overlap; no winner is claimed.
- **Sample efficiency (steps to 80% at size 6):** mean Kairi 279k [230k, 328k], Kaiser 543k [271k, 929k], but the Kaiser
  mean is inflated by one censored seed (seed 101 never reached 80% and enters at the 2M cap). Medians are about equal
  (Kairi 261k, Kaiser 246k); seven of ten Kaiser seeds reach 80% by 460k steps. No efficiency difference is claimed.
- **One Kaiser seed fails outright.** Kaiser seed 101 got 0.04 at size 5, 0.00 at size 6 and 0.00 at sizes 8 and 12 (and 0.00
  few-shot at size 8); every other Kaiser seed reached 1.0 at sizes 5 and 6, as did all ten Kairi seeds. That one seed
  explains Kaiser's 0.90 at sizes 5/6 and 0.90 few-shot at size 8. Excluding it, Kaiser is 0.552 at size 8 and 0.042 at size 12
  (still below Kairi's 0.640 and 0.083), so the size 8 and 12 ordering does not depend on that seed. (Post-hoc check,
  not part of the protocol.)

**Sensitivity to the tuning tie (post-hoc, outside the protocol).** Tuning tied three Kaiser settings (`rnd_coef` 0.001, 0.003, 0.01;
all 0.0) and two Kairi settings (`ent_coef` 0.03, 0.1); training AUC broke the ties to 0.001 and 0.03. Rerunning the eval on the
same 10 seeds at the other tied values (`scripts/tie_robustness.py`, `scripts/compare_robustness.py`) shows the size 8/12
advantage depends on that choice. Paired Kairi − Kaiser success (95% CI), Kairi at its tuned 0.03:

| Kaiser `rnd_coef` | size 8 | size 12 |
|---|---|---|
| 0.001 (tuned) | 0.143 [0.020, 0.295] | 0.046 [0.012, 0.082] |
| 0.003 | 0.121 [−0.002, 0.260] | 0.011 [−0.058, 0.063] |
| 0.01 | 0.030 [−0.086, 0.141] | −0.009 [−0.105, 0.052] |

At `rnd_coef` 0.01 Kaiser matches Kairi at every size (1.000, 1.000, 0.610, 0.092 vs Kairi 1.000, 1.000, 0.640, 0.083). With Kairi at its
other tied value (`ent_coef` 0.1: 1.000, 1.000, 0.636, 0.116) the tuned-Kaiser gap is similar (size 8: 0.140 [0.047, 0.239]; size 12:
0.078 [0.019, 0.160]). So the supported statement is: **Kaiser is never better than Kairi anywhere, and is clearly worse at sizes 8 and 12
only for the AUC-selected `rnd_coef`; under other tied settings the two are statistically indistinguishable.** Choosing a variant by its
eval score would not be protocol-legal, so the tuned result stays the headline and this is reported as sensitivity. One Kaiser seed fails
outright in two of the three settings (seed 101 at 0.001, seed 105 at 0.003); none fails at 0.01 or in any Kairi run.

Figure: `figs/degradation.png`.

## Limitations

- Phase 1: the `tt_*` agents go beyond the original two-agent design; Kaiser and `tt_triggered` tie at low λ.
- Phase 1 was run on Colab (Python 3.13) and rerun locally (Python 3.9.6, numpy 2.0.2): the `gaps.csv` and `crossovers.csv` tables agree to
  within the 4-decimal rounding of the Colab output (max difference 5e-05), so the results do not depend on the numpy/Python version.
- Phase 2: the calibrated budget is the 2M cap and the 80% calibration target was not reached; one familiar size; one seed pool; PPO hyperparameters are not tuned beyond one coefficient
  per agent; PPO on DoorKey-6 is very seed-dependent, so CIs are wide.
- Phase 2: the tuned-setting advantage of Kairi at sizes 8 and 12 is sensitive to which tied `rnd_coef` the AUC tie-break picks (see above); 10 eval seeds only.
- Phase 2: test layouts at sizes 5 and 6 are held-out seeds, not unseen layouts; size 12 is at the
  floor for zero-shot.

## Where each agent loses

- **Bandit (Phase 1):** Kairi (epsilon-greedy) is never the best single agent; Kaiser's discounted UCB wins at every jump rate.
  The only regime where something beats Kaiser is `tt_fixed` at high lambda (>= 0.01).
- **MiniGrid (Phase 2):** Kaiser (PPO + RND) never beats plain PPO. It ties at sizes 5 and 6 and, at its tuned setting, loses at 8 and 12 (indistinguishable from Kairi at other tied settings), with
  RND adding per-step cost (Kaiser seed jobs took ~2.5x as long). The RND bonus gave no benefit on DoorKey (a plausible but untested reason: extrinsic reward is reachable with plain exploration); the agents break down together as size grows (size 12 near the floor).
- So the exploration bonus helps in the nonstationary bandit and not in this sparse-but-learnable MiniGrid family.

## Future work

Crafter, Showdown, and a deep-RL tag-team agent.

## Reproducing

See `CLAUDE.md` for commands. Full runs go to Colab (`notebooks/`), Phase 2 may also run locally.
