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
- **Kairi wins in two frozen-agent regimes: many more arms than tuned for (k = 50 vs 10) and fast random-walk drift (0.1 vs 0.01).** These are the first
  "Kairi wins" results in the project; the ablation section below shows they vanish when both agents are retuned for the shifted world.
- Explanations for the two Kairi wins were tested in the ablations below; the result changes how these wins should be read (they are frozen-tuning effects).
- Tag-team vs the best single agent (`pair_vs_best_single.csv`; difference = team − best single, negative = team better): at k = 50 only `tt_meta` beats the best single
  agent, Kairi (−0.069 [−0.085, −0.054]); `tt_fixed` (+0.080) and `tt_triggered` (+0.058) are worse than Kairi. At drift 0.05 and 0.1 `tt_fixed` (−0.067, −0.058) and `tt_meta`
  (−0.077, −0.074) beat Kairi, while `tt_triggered` ties at 0.05 and is worse at 0.1 (+0.115). At k = 2 all three teams beat Kaiser by a small margin (about −0.004 to −0.005).
  In the reversed and σ shifts the teams mostly lose to Kaiser. So switching rules help most where the single agents fail, and `tt_meta` is the most reliable of the three.
- Protocol notes: the first reversed run (Kaiser gamma at the grid's low edge) and the first drift run (Kairi epsilon, Kaiser c and `tt_meta` c at grid edges) were provisional;
  grids were widened (equal sizes kept) and rerun, final results are not provisional. The provisional runs are kept locally in `results/*-v1-provisional/`, not in plots.
  The sigma and k tuning point is the main run's (σ = 1, k = 10, λ = 0.001), so their familiar values match Phase 1.
- Caveats: frozen agents (tuned once at the familiar point, no retuning per shift); no multiplicity correction across knob values; one horizon (2000); the drift world is
  a Gaussian random walk without clipping, so arm means can wander far from N(0, 1); unequal knob grids differ per shift.

## Ablations of the two Kairi wins (post-hoc, outside the protocol; DONE 2026-10-24)
Tables in `figs/ablations/`; scripts `scripts/regret_trace.py`, `scripts/compare_pair.py`; configs `configs/bandit-{k-at50,drift-at01}.yaml`.

**1. Retune at the shift point.** Same protocol, but the familiar value is the shifted one (k = 50, drift = 0.1): both agents tuned on tune seeds there, evaluated on the disjoint eval seeds.
Kairi − Kaiser regret per step (95% paired CI; positive = Kaiser better), Kairi and Kaiser tuning optima interior (not provisional; the `tt_*` agents were provisional and are not reported):

| retuned at | knob values where Kaiser wins | where Kairi wins | at the original Kairi-win point |
|---|---|---|---|
| k = 50 (Kaiser `gamma` 0.995, `c` 0.1; Kairi epsilon 0.005, alpha 0.7) | k = 10, 20, 50 | k = 2, 5 | **k = 50: Kaiser wins by 0.086 [0.070, 0.103]** (was Kairi by 0.054) |
| drift 0.1 (Kaiser `gamma` 0.99, `c` 0.5; Kairi epsilon 0.05, alpha 0.7) | all 6 (0.129 to 0.168) | none | **drift 0.1: Kaiser wins by 0.168 [0.156, 0.179]** (was Kairi by 0.209) |

So both "Kairi wins" **disappear when the agents are retuned for the shifted world**. The frozen Kaiser configuration (`gamma` 0.98 at k = 10; `gamma` 1.0, `c` 1.0 at drift 0.01) was the problem: its
optimism and discount settings fit the tuned world and transfer worse than Kairi's epsilon-greedy settings. Retuned Kaiser used a longer memory at k = 50 (`gamma` 0.995 vs 0.98, effective window
200 vs 50 steps; with 50 arms a 50-step window forgets arm estimates between pulls, consistent with this but not tested directly) and a discount below 1 under fast drift (`gamma` 0.99 vs 1.0).

**2. Regret over time at k = 50 (agents frozen at their k = 10 tuning).** Kairi − Kaiser by window (positive = Kaiser better):
steps 0-100 −0.286 [−0.317, −0.254]; 100-200 +0.425 [+0.388, +0.462]; 200-500 −0.006 (tie); 500-1000 −0.075 [−0.099, −0.051]; 1000-2000 −0.082 [−0.104, −0.060].
Kaiser's initial sweep of the 50 arms costs it early regret, but it then recovers and is net ahead through step 500; its loss accrues after step 500 (about 110% of the total gap). **The
"UCB's initial exploration cost grows with k" explanation is not supported**; the loss is a steady-state cost of the frozen settings. (At k = 10 the Kaiser lead comes mostly late too, 58% of the total gap after step 1000.)

Reading for the project's question: the Kairi wins are best described as **Kaiser's tuned configuration being more brittle than Kairi's under large shifts**, not as exploration being harmful there. That is still
a finding about frozen agents ("breaks when conditions move far from the tuned point"), and it means the robust advice is to retune Kaiser when the world changes a lot. Caveats: retuning at the shift is an oracle
the protocol does not allow (tuning uses the shifted world's tune seeds); only these two shifts were ablated; the drift retune widened Kairi's epsilon grid after a provisional first run (kept locally in
`results/bandit-drift-at01-v1-provisional/`); `tt_*` retunes were provisional and are not reported; `compare_pair.py` ignores them.

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
- **Kairi wins only when frozen agents face a world far from the tuned point (5x more arms, 10x faster drift), and the ablations show this is tuning brittleness, not a cost of exploration:** retuned for the
  shifted world, Kaiser wins at both points. So the defensible map is "Kaiser's tuned settings break under large shifts; its approach does not", plus MiniGrid where RND gives no measurable benefit.
- Open: does RND pay off in a world where exploration is the bottleneck (step B), and is the frozen-tuning brittleness also visible in deep RL?

## Methodological lessons (worth carrying forward)
- Give both agents equal treatment on every non-tuned hyperparameter, or tune it for both (entropy was fixed for one and tuned for the other).
- Pre-declare the tie-break rule and check how many grid points tie; a saturated tuning objective (5 of 6 values at 0.000) makes the "tuned" choice arbitrary.
- Report post-hoc robustness checks next to the headline, not instead of it; do not pick a variant by its eval score.
- Seed-level failures (one outright failure in 10) move means a lot; always report them with the CI.

## Next
Step B (exploration-heavy world) is the one open item.
