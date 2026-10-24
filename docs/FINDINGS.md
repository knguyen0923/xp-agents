# XP — Findings and analysis

Last updated 2026-10-24. This is the running record of what we concluded and why; the README holds the tables, `STATUS.md` the project state.
Numbers below come from the README results sections and `results/`; every claim carries its evidence and its caveat.

## Question
When does an explicit uncertainty-driven exploration term (Kaiser) beat the same learner without it (Kairi), as the world shifts away from the
conditions both were tuned on? Protocol: tune on the familiar knob value (tune seeds only), freeze, evaluate across the knob on disjoint eval seeds, report
paired 95% bootstrap CIs.

## Phase 1 — changing bandit (DONE; extra shifts pending)
**Conclusion: Kaiser (discounted UCB) beats Kairi (epsilon-greedy) at every jump rate; the curves do not cross.**
- Kairi − Kaiser regret/step is 0.033 to 0.072 across λ ∈ [0, 0.02]; all seven 95% CIs exclude 0. Evidence: README Phase 1 table, `gaps.csv`.
- Both agents degrade similarly away from the familiar λ = 0.001 (at λ = 0.02, Kairi +0.552, Kaiser +0.546 regret), so the Kaiser lead is a
  level shift, not better robustness.
- Tag-team switching adds little: `tt_fixed` wins only at λ ≥ 0.01 (−0.020, −0.029 vs Kaiser); `tt_meta` never wins; `tt_triggered` ties (one win in
  21 comparisons, likely noise).
- Robustness: Colab (Python 3.13) and a local rerun (3.9.6) agree within 5e-5.
- Caveats: one bandit family (k = 10, σ = 1, horizon 2000), one tuning point; tag-team agents go beyond the original two-agent design.
- Pending extra shifts (cheap): reversed tuning direction (familiar λ = 0.02; running, `configs/bandit-reversed.yaml`, `results/bandit-reversed/`),
  reward noise σ, arm count k, slow drift. Results go in a section below when done.

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
- Kaiser's exploration term pays off in the nonstationary bandit (reward distribution changes, so uncertainty re-grows) and showed no measurable benefit in
  the stationary-but-sparse MiniGrid DoorKey family.
- Kairi never wins outright in either phase; in MiniGrid it ties. "Kairi winning some regimes" has not yet happened at a level we can defend.
  Open question for step B and the Phase 1 shifts: is there any regime where the plain learner clearly wins (e.g. very high noise, many arms)?

## Methodological lessons (worth carrying forward)
- Give both agents equal treatment on every non-tuned hyperparameter, or tune it for both (entropy was fixed for one and tuned for the other).
- Pre-declare the tie-break rule and check how many grid points tie; a saturated tuning objective (5 of 6 values at 0.000) makes the "tuned" choice arbitrary.
- Report post-hoc robustness checks next to the headline, not instead of it; do not pick a variant by its eval score.
- Seed-level failures (one outright failure in 10) move means a lot; always report them with the CI.

## Next
Phase 1 extra shifts (reversed direction running) -> step B spike (exploration-heavy world) -> polish (CI on push, README figures, short write-up).
