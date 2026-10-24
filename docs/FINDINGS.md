# XP — Findings and analysis

Last updated 2026-10-24. This is the running record of what we concluded and why; the README holds the tables, `STATUS.md` the project state.
Numbers below come from the README results sections and `results/`; every claim carries its evidence and its caveat.

## Question
When does an explicit uncertainty-driven exploration term (Kaiser) beat the same learner without it (Kairi), as the world shifts away from the
conditions both were tuned on? Protocol: tune on the familiar knob value (tune seeds only), freeze, evaluate across the knob on disjoint eval seeds, report
paired 95% bootstrap CIs.

## Phase 1 — changing bandit (DONE)
**Conclusion: Kaiser (discounted UCB) beats Kairi (epsilon-greedy) at every jump rate; the curves do not cross.**
- Kairi − Kaiser regret/step is 0.033 to 0.072 across λ ∈ [0, 0.02]; all seven 95% CIs exclude 0. Evidence: README Phase 1 table, `gaps.csv`.
- Both agents degrade similarly away from the familiar λ = 0.001 (at λ = 0.02, Kairi +0.552, Kaiser +0.546 regret), so the Kaiser lead is a
  level shift, not better robustness.
- Tag-team switching adds little: `tt_fixed` wins only at λ ≥ 0.01 (−0.020, −0.029 vs Kaiser); `tt_meta` never wins; `tt_triggered` ties (one win in
  21 comparisons, likely noise).
- Robustness: Colab (Python 3.13) and a local rerun (3.9.6) agree within 5e-5.
- Caveats: one bandit family (k = 10, σ = 1, horizon 2000), one tuning point; tag-team agents go beyond the original two-agent design.
- Extra shifts: see the next section.

## Phase 1 extra shifts (DONE 2026-10-24; figures and tables in `figs/shifts/<name>/`)
Same bandit, agents, protocol and seed pools; only the swept knob (or the tuning point) changes. Configs: `configs/bandit-{reversed,sigma,k,drift}.yaml`.
Mean regret per step, Kairi vs Kaiser (Kairi − Kaiser > 0 means Kaiser better; winner declared when the paired 95% CI excludes 0):

| shift (familiar in bold) | result |
|---|---|
| **Reversed tuning** (tuned at λ = 0.02, swept λ 0 to 0.02) | Kaiser wins at all 7 λ; the gap grows with λ (0.0035 [0.0008, 0.0064] at λ = 0 to 0.101 [0.095, 0.107] at 0.02). No crossover. |
| **σ** 0.25, 0.5, **1.0**, 2.0, 4.0 | Kaiser wins at all 5; gap 0.158, 0.105, 0.053, 0.158, 0.325 (all CIs exclude 0). No crossover. |
| **k** 2, 5, **10**, 20, 50 arms | Kaiser wins at k ≤ 20 (0.009, 0.037, 0.053, 0.043). **Kairi wins at k = 50 (−0.054 [−0.071, −0.037]); crossover in (20, 50).** |
| **Drift** per-step std 0, 0.005, **0.01**, 0.02, 0.05, 0.1 (no jumps) | Kaiser wins up to 0.02 (gap 0.063 to 0.070); tie at 0.05 (−0.027 [−0.059, 0.002]); **Kairi wins at 0.1 (−0.209 [−0.283, −0.132]); crossover in (0.02, 0.1).** |

Conclusions:
- **Kaiser's bandit advantage holds under reversed tuning and under reward-noise shifts.** It is not an artifact of tuning at a low jump rate or at σ = 1.
- **We found where Kairi wins, in two regimes: many more arms than tuned for (k = 50 vs 10) and fast random-walk drift (0.1 vs 0.01).** These are the first
  defensible "Kairi wins" results in the project. Both are shifts away from the tuned point, so they fit the project's question (where does each agent break).
- Plausible explanations (untested): with k = 50 and horizon 2000, UCB's optimism spends many pulls on arms it has not yet tried, so exploration cost grows with k;
  under fast drift Kaiser tuned at drift 0.01 chose gamma = 1.0 (no discounting), so its estimates lag, while Kairi's constant step size (alpha 0.4) keeps tracking.
  Testing these would need extra ablations (e.g. retuning at each shift, or a discounted Kaiser at drift 0.01).
- Tag-team: `tt_fixed` and `tt_meta` beat the best single agent in the two regimes where Kairi wins (k = 50: vs Kairi by 0.07 to 0.08; drift 0.05 and 0.1) and at k = 2;
  `tt_triggered` mostly ties.
- Protocol notes: the first reversed run (Kaiser gamma at the grid's low edge) and the first drift run (Kairi epsilon, Kaiser c and `tt_meta` c at grid edges) were provisional;
  grids were widened (equal sizes kept) and rerun, final results are not provisional. The provisional runs are kept locally in `results/*-v1-provisional/`, not in plots.
  The sigma and k tuning point is the main run's (σ = 1, k = 10, λ = 0.001), so their familiar values match Phase 1.
- Caveats: frozen agents (tuned once at the familiar point, no retuning per shift); no multiplicity correction across knob values; one horizon (2000); the drift world is
  a Gaussian random walk without clipping, so arm means can wander far from N(0, 1); unequal knob grids differ per shift.

## Phase 2 — MiniGrid DoorKey (CLOSED 2026-10-24 by user decision)
**Conclusion: RND gave no measurable benefit on DoorKey. Kairi ≥ Kaiser everywhere, but the apparent Kairi win at sizes 8 and 12 is not robust.**
Evidence chain, in the order we learned it:
1. **Tuned run (10 seeds):** Kairi 1.00/1.00/0.640/0.083 vs Kaiser 0.904/0.900/0.496/0.038 at sizes 5/6/8/12. Differences exclude 0 only at sizes 8
   and 12 (0.143 [0.02, 0.29], 0.046 [0.01, 0.08]). One Kaiser seed (101) failed outright; excluding it Kaiser is still below Kairi at 8 and 12.
2. **Tie-sensitivity:** Kaiser's tuning was a 3-way tie (`rnd_coef` 0.001/0.003/0.01, all 0.0) broken by training AUC. At 0.003 the 8/12 gaps include 0;
   at 0.01 Kaiser matches Kairi at every size. So the headline depended on an arbitrary tie-break.
3. **Matched entropy:** Kaiser's `ent_coef` was fixed at 0.01 while Kairi tuned 0.03 (an unequal treatment). With Kaiser at 0.03 and `rnd_coef` re-tuned
   (5-way tie, AUC picks 0.01), paired Kairi − Kaiser is 0.000, 0.000, 0.043 [−0.053, 0.142], −0.037 [−0.159, 0.040]; no seed failures; few-shot about equal.
4. **Reading:** the deficit is mostly an entropy/tuning artifact. It is not evidence that RND helps either: the best case is a tie, and with n = 10 a
   small effect in either direction cannot be ruled out, especially at size 8/12 where CIs are wide.
- Why plausibly no benefit (untested): on DoorKey the extrinsic reward is reachable with plain PPO exploration, so an exploration bonus has little to add.
  This points to step B (a world where exploration is the bottleneck) as the right place to look for RND's benefit.
- Why we stopped: more DoorKey seeds (the shelved step A, ~6-8 h) would tighten a tie, not change the story. Plan kept at
  `docs/superpowers/plans/2026-10-24-phase2-step-a.md`.
- Caveats: budget is the 2M cap and the 80% calibration target was not reached (mean success ~0.67); one familiar size; PPO on DoorKey is very
  seed-dependent; test layouts at sizes 5/6 are held-out seeds, not necessarily unseen layouts; size 12 is at the floor zero-shot; the entropy check is post-hoc,
  outside the protocol (the tuned result stays the headline).

## Cross-phase picture ("where each agent loses")
- Kaiser's exploration term pays off in the nonstationary bandit near and moderately beyond the tuned conditions (jumps, noise, drift up to ~2x), and showed no measurable
  benefit in the stationary-but-sparse MiniGrid DoorKey family.
- **Kairi wins when the world moves far from the tuned point in a way that makes optimism expensive or stale: 5x more arms than tuned, or 10x faster drift** (bandit extra shifts).
  In MiniGrid Kairi only ties. So "a map of when each agent wins and where it breaks" now has both sides: Kaiser breaks under many arms and fast drift.
- Open for step B: does RND pay off in a world where exploration is the bottleneck, and is the "Kaiser breaks far from tuning" pattern also visible in deep RL?

## Methodological lessons (worth carrying forward)
- Give both agents equal treatment on every non-tuned hyperparameter, or tune it for both (entropy was fixed for one and tuned for the other).
- Pre-declare the tie-break rule and check how many grid points tie; a saturated tuning objective (5 of 6 values at 0.000) makes the "tuned" choice arbitrary.
- Report post-hoc robustness checks next to the headline, not instead of it; do not pick a variant by its eval score.
- Seed-level failures (one outright failure in 10) move means a lot; always report them with the CI.

## Next
Step B spike (exploration-heavy world, optionally an ablation of the k = 50 and drift = 0.1 explanations) -> polish (CI on push, README figures, short write-up).
