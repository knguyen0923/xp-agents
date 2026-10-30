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

## Few-shot across variants (post-hoc re-analysis of existing runs, no new training; `scripts/fewshot_compare.py`, `docs/spikes/fewshot_pairs.csv`)
This is the deep-RL analog of the bandit "frozen hyperparameters" setting: hyperparameters tuned at size 6 are frozen and the agent adapts online for 200k steps at size 8 or 12 (10 seeds).
Few-shot success at size 12 (95% CI): Kairi 0.72 [0.44, 0.97] (`ent_coef` 0.03) and 0.62 [0.40, 0.82] (0.1); Kaiser 0.50 [0.23, 0.78] (tuned `rnd_coef` 0.001, ent 0.01), 0.25 [0.05, 0.49] (0.003), 0.72 [0.43, 0.97] (0.01), 0.73 [0.46, 1.00]
(matched entropy: `rnd_coef` 0.01, ent 0.03). At size 8 everything except the tuned and 0.003 Kaiser variants is at about 1.00.
Paired Kairi − Kaiser at size 12 against Kairi's tuned run: 0.215 [0.016, 0.465] (tuned Kaiser; excludes 0), 0.468 [0.160, 0.764] (`rnd_coef` 0.003; excludes 0), 0.000 [−0.423, 0.416] (0.01), −0.016 [−0.085, 0.060] (matched entropy).
- **Corrected a README sentence:** it said the few-shot CIs overlap and no winner is claimed, but the project's rule is the paired CI, which at size 12 excludes 0 for the tuned Kaiser (the protocol's `crossovers__success_fewshot.csv` already said "kairi").
  The README now reports the paired result.
- **A hint of the bandit pattern in deep RL, not established:** Kaiser's few-shot score at size 12 swings from 0.25 to 0.73 across three `rnd_coef` values (and the AUC tie-break picked a middling one), while Kairi's moved 0.62 to 0.72 across its two tied `ent_coef` values.
  So Kaiser's few-shot result depends more on which tied setting is chosen, consistent with "Kaiser's tuned settings are more brittle". With matched entropy and `rnd_coef` 0.01 the difference vanishes.
- **Caveats:** 10 seeds, wide CIs (size 12 CIs are 0.3 to 0.5 wide), Kairi has only two variants, the variants differ in two coefficients at once in the matched run, and the `rnd_coef` swings may be seed noise (a single failing seed moves the size-12 mean by 0.1). A full retune at size 8 (config-only, about 8 to 10 hours) would be the direct test; we judged it low value.

## Step B spike: exploration-heavy MiniGrid (MultiRoom), spike 2026-10-24, probe 2026-10-24
Question: does the RND bonus pay off where exploration is the bottleneck? We ran a throwaway feasibility spike instead of a full protocol sweep: plain PPO vs PPO+RND, 2 seeds per cell, 2M steps, `Learner` unchanged
with the environment factory swapped to MultiRoom (`scripts/spikes/`, table in `docs/spikes/multiroom_spike.csv`; outside the protocol, no tuning). Test success after training (steps to 95% training success):

| setting | plain PPO | PPO+RND |
|---|---|---|
| 2 rooms, size ≤ 6 | 1.00, 1.00 (184k, 184k) | 1.00, 1.00 (195k, 328k), coef 0.003 |
| 3 rooms, size ≤ 6 | 0.00, 0.01 | 0.00, 0.00, coef 0.003 |
| 4 rooms, size ≤ 6 | 0.00, 0.00 | 0.00, 0.00, coef 0.003 |
| 3 rooms, size ≤ 4 | 1.00, 1.00 (195k, 461k) | 1.00, 1.00 (287k, 614k), coef 0.03 |
| 4 rooms, size ≤ 4 | 0.00, 1.00 (1.77M) | 0.00, 0.00, coef 0.03 |
| 2 rooms, size ≤ 8 | 1.00, 1.00 (297k, 696k) | 1.00, 0.99 (328k, 881k), coef 0.03 |
| 2 rooms, size ≤ 10 | 0.99, 1.00 (369k, 512k) | 0.99, 0.99 (1.12M, 881k), coef 0.03 |

- **In the 2M-step spike RND gave no benefit anywhere** (revised by the 5M-step probe below): in all 8 seed-matched pairs where both agents solved the level it needed more steps to reach 95%, it never solved a level plain PPO failed, and runs took about 1.2 to 1.8x as long. At the one borderline
  level (4 rooms, size ≤ 4) plain PPO solved 1 of 2 seeds and RND 0 of 2; with 2 seeds this is not evidence that RND is worse.
- **Difficulty axis:** room count with small rooms is the usable axis (3 rooms solved, 4 rooms borderline, probably the floor at 5); with size-6 rooms the wall is already at 3 rooms; bigger rooms at 2 rooms only slow PPO mildly.
- **Consistent with Phase 2 at 2M steps:** the bonus gave no measurable benefit on DoorKey either. Together this was evidence for "no RND benefit in sparse-reward MiniGrid at a 2M-step PPO budget"; the 5M-step probe below shows the answer changes with budget and coefficient.
- **Why we stopped:** a full sweep (environment code, spec, tuned RND coefficient, 10+ seeds, roughly 6 to 10 hours) would most likely add another tie with the top level at the floor. User decision: close as a documented spike.
- **Caveats:** 2 seeds per cell, one fixed untuned RND coefficient per setting (0.003 or 0.03), a 2M-step cap (published RND wins on MultiRoom use far larger budgets), one PPO configuration. This does not show that RND cannot help; a larger budget or a tuned
  coefficient at the 4-small-room level was the cheapest probe; it was run, see the next subsection.

### 5M-step probe (post-hoc, 2026-10-24; `scripts/spikes/multiroom_probe.py`, `scripts/spikes/probe_summary.py`, `docs/spikes/multiroom_probe.csv`)
Follow-up on the borderline level (4 rooms, max room size 4): 5M steps instead of 2M, plain PPO vs PPO+RND at `rnd_coef` 0.01, 0.03 and 0.1, seeds 0 to 4 (20 runs). Steps to 95% training success (censored at 5M if never reached), paired by seed:

| setting | reached 95% | median steps | mean steps | PPO − RND mean steps [95% CI] | RND faster on |
|---|---|---|---|---|---|
| plain PPO | 5/5 | 2.66M | 2.54M | | |
| RND 0.01 | 4/5 (seed 4 never) | 2.24M | 2.43M | +0.11M [−0.92M, +1.12M] | 3/5 seeds |
| RND 0.03 | 5/5 | 2.94M | 3.02M | −0.48M [−0.93M, +0.01M] | 1/5 seeds |
| **RND 0.1** | 5/5 | **1.44M** | **1.37M** | **+1.17M [+0.53M, +1.75M]** | **5/5 seeds** |

- **First RND benefit in the project:** at `rnd_coef` 0.1 and a 5M budget, RND reached 95% training success in about half the steps of plain PPO, faster on every seed, with a paired CI that excludes 0. Test success was about 1.00 for every run that reached the threshold
  (plain PPO seed 2 scored 0.96; RND 0.01 seed 4 scored 0.00), so the difference is speed, not final success.
- **The effect depends strongly on the coefficient:** neutral to slightly harmful at 0.03 (slower, CI just includes 0), mixed at 0.01 (fast on 3 of 4 solved seeds, one outright failure), clearly helpful at 0.1. The 2M-step spike used only 0.003 and 0.03, which likely explains why it saw no benefit.
- **This revises the spike conclusion:** "RND never helped" holds for the 2M-step spike, not for the larger budget with a larger coefficient. It fits the project's thesis (optimism helps where exploration is the bottleneck), and it contrasts with DoorKey, where no benefit appeared.
- **Caveats (important):**
  - Post-hoc selection: three coefficients were tried and 0.1 is the best, so its advantage is optimistic. A protocol-legal version would tune the coefficient on tune seeds and evaluate on disjoint eval seeds.
  - 0.1 is the top of the tried range, so the optimum may be higher (a grid-edge result, which the protocol would treat as provisional).
  - 5 seeds, one level, one PPO configuration; the seeds are from the tune pool and there is no disjoint eval pool here.
  - The metric is steps to 95% on the training curve (a recent-episodes window); wall-clock was not compared fairly (runs shared the CPU with other jobs).
  - The 0.01 failure on seed 4 shows seed-level failures matter.
- **Update 2026-10-25: this confirmation was run (see "Protocol-legal MultiRoom confirmation" below); the line that follows is the original plan.**
- **What a confirmation would need:** a full protocol phase on this level (familiar 4 small rooms or the nearest solvable level, a widened RND grid including values above 0.1, tune seeds disjoint from eval seeds, 10+ eval seeds, matched entropy by construction). Not run; user decision pending.


### Protocol-legal MultiRoom confirmation (2026-10-25; full write-up `docs/MULTIROOM_EVALUATION.md`, data `docs/spikes/multiroom_confirmation.csv`)
Spec amendment 5.4: MultiRoom, 4 rooms, max room size 4, single familiar level; Kairi `ent_coef` and Kaiser `rnd_coef` tuned on 3 tune seeds by training AUC (Kaiser's entropy matched to Kairi's tuned 0.01); S = 2,938,880 steps from calibration; 10 disjoint eval seeds (100 to 109); not provisional (Kairi 0.01 and Kaiser 0.6 are interior).

| metric (10 eval seeds) | Kairi | Kaiser | Kaiser − Kairi, paired 95% CI | Kaiser better on |
|---|---|---|---|---|
| training AUC | 0.258 | 0.778 | +0.520 [+0.312, +0.728] | 10/10 |
| steps to 95% (censored at S) | 2.23M | 0.70M | −1.52M [−2.12M, −0.93M] | 10/10 |
| seeds reaching 95% by S | 5/10 | 10/10 | +0.50 [+0.20, +0.80] | |
| held-out success (200 layouts) | 0.50 | 1.00 | +0.50 [+0.20, +0.80] | |

- **Result: Kaiser is faster and more reliable on this level under the full protocol.** First confirmed RND benefit in the project; the direction agrees with the post-hoc probe, and the size is larger (1.5M fewer steps against 1.17M).
- **Caveats:** one level; the level, target and grid range were chosen after the probe and tune seeds 0 to 2 overlap the probe seeds, so only eval seeds are fresh; S was fit to Kairi's tune seeds, so Kairi's 5/10 "reached 95%" and its success rate depend on the budget (the probe saw PPO reach 95% on all 5 seeds by 3.15M steps), and the steps gap is a lower bound because failures are censored at S; Kairi's tuned entropy is a noisy pick from 3 seeds; steps, not wall-clock, were compared.


### Wall-clock note (2026-10-26, from the logged `secs` in `docs/spikes/multiroom_probe.csv`; not a controlled timing)
Mean seconds per 5M-step run on 4 small rooms: plain PPO 2163, RND 0.01 1875, 0.03 1877, 0.1 1845 (5 seeds each; three runs shared the laptop, so load differed between runs). So the bonus did not visibly add time per step in the probe, and at coefficient 0.1, which reached 95% in a mean 1.37M steps against 2.54M, the wall-clock to 95% would be roughly half. The DoorKey seed jobs, by contrast, took about 2.5x as long for Kaiser. A fair comparison needs a dedicated run with equal machine load; the MultiRoom confirmation did not log per-job time.

## Phase 4 — Pokémon Showdown VGC doubles (DONE 2026-10-28; tables `figs/showdown/`, copied to `docs/spikes/showdown_results/`)
Generality test, one familiar level, no shift knob (spec 5.6 and its amendment). Format `gen9championsvgc2026regmc`, opponent DamageHeuristic, budget S = 250,000 decisions per run (244 PPO updates, 8 envs), steps-to-target threshold 0.30; opponent, S and threshold were the user's decisions (2026-10-26), declared before any RND result was read (`docs/spikes/showdown_gate.md`). Tuned on tune seeds 0 to 2 with equal 6-value grids (Kairi `ent_coef` 0.003 to 0.1; Kaiser `rnd_coef` 0.01 to 1.0 with Kairi's tuned entropy), frozen, then 10 disjoint eval seeds (100 to 109) per agent, each evaluated on 200 fresh battles with eval-pool teams.
Tuned values: Kairi `ent_coef` 0.01 (score 0.610), Kaiser `rnd_coef` 0.03 (score 0.612); both interior, so nothing is provisional. The tuning score is 1 − training AUC, lower is better.
**Result: a tie.** Training AUC of the win rate: Kairi 0.406 [0.397, 0.413], Kaiser 0.403 [0.394, 0.410]; paired difference (Kaiser − Kairi, same eval seeds) −0.003 [−0.015, +0.008]. Win rate of the frozen policy over 200 fresh battles: 0.455 [0.431, 0.479] vs 0.451 [0.426, 0.478], difference −0.004 [−0.047, +0.042]. Decisions to first reach a 30% win rate: Kairi 29,696 [24,576, 34,816], Kaiser 34,816 [31,744, 37,888], difference +5,120 [−1,024, +11,264] (all 20 seeds reached it). All intervals overlap zero; the direction of the steps-to-30% gap (Kairi sooner) is a hint, not a finding (1 seed favoured Kaiser, 4 favoured Kairi, 5 tied at the 1,024-decision update granularity).
Caveats: one opponent and one format, so this tests only that Kaiser's bonus neither helps nor hurts here, not how either agent breaks (no shift knob, no crossover); 10 seeds per agent; Kaiser's tuning scores were nearly flat across the grid (about 0.59 to 0.64 per seed, 3 tune seeds), so its `rnd_coef` pick is noisy; S is fixed at 250k decisions, so a larger budget could separate them; decisions, not wall-clock, are compared. Two tuning-grid widenings for Kairi were made on a misread of the score (lower is better) and reverted before any eval result existed; tuning jobs from the first run were reused (same config hash, `ALLOW_CODE_CHANGE=1`). The plot `figs/showdown/degradation.png` has a zoomed y-axis (0.395 to 0.413) that exaggerates a 0.003 gap; read the tables.

### Matchup-features run (2026-10-29 to 2026-10-30; tables `docs/spikes/showdown_matchup_results/`)
**Matchup-features run (2026-10-29 to 2026-10-30).** Same protocol with a switchable observation block (`env.matchup`, 72 extra floats: for each of my moves an ability-aware type multiplier and an estimated damage fraction of the target's HP, plus the incoming threat from the foe's revealed moves; spec `docs/superpowers/specs/2026-10-29-showdown-matchup-features-design.md`). The first attempt stopped at the protocol guard (Kairi's `ent_coef` 0.003 sat on the lower grid edge), so both grids were widened to 8 values (Kairi `ent_coef` 0.001 to 0.1, Kaiser `rnd_coef` 0.01 to 3.0) and the run restarted in a fresh folder. Tuned: Kairi `ent_coef` 0.006, Kaiser `rnd_coef` 1.0 (with Kairi's `ent_coef` 0.006); both interior, so nothing is provisional. Kaiser's tuning means were again nearly flat (0.597 to 0.612 across the 8 values), so its pick is noisy. The estimator reported 0 errors on every worker line. **Result: still a tie.** Training AUC Kairi 0.416 [0.408, 0.425], Kaiser 0.424 [0.417, 0.431], difference (Kairi − Kaiser) −0.008 [−0.021, +0.006]; win rate 0.480 [0.457, 0.504] vs 0.470 [0.458, 0.485], difference +0.009 [−0.022, +0.041]; decisions to a 30% win rate 26,624 vs 31,744, difference −5,120 [−10,240, +1,024] (all 20 seeds reached it). Against the original run, paired by eval seed (100 to 109): Kaiser's training AUC rose +0.022 [+0.012, +0.031] (9 of 10 seeds improved); every other change has an interval that includes 0 (Kairi AUC +0.011 [−0.002, +0.023]; win rate Kairi +0.025, Kaiser +0.019; steps to 30% about 3,000 sooner for both), and the gain is not reliably larger for Kaiser than for Kairi (difference in AUC gains +0.011 [−0.001, +0.026]). Caveats: the new inputs, the wider grids and the retuned values changed together, so the inputs are not isolated; six cross-run comparisons with no multiplicity correction, one of which excludes 0; the cross-run pairing assumes the same seed number means the same battles and teams, which was not verified. Tables and the paired comparison: `docs/spikes/showdown_matchup_results/` (script `scripts/showdown_matchup_compare.py`).

## Cross-phase picture ("where each agent loses")
- Kaiser's exploration term pays off in the nonstationary bandit near and moderately beyond the tuned conditions (jumps, noise, drift up to ~2x), showed no measurable
  benefit in the stationary-but-sparse MiniGrid DoorKey family or at 2M steps on MultiRoom, and in a post-hoc 5M-step probe on the exploration-heavy MultiRoom level (4 small rooms) reached 95% success in about half
  the steps of plain PPO at `rnd_coef` 0.1 (faster on 5/5 seeds), but not at 0.01 or 0.03: its benefit depends on budget and coefficient.
- **Kairi wins only when frozen agents face a world far from the tuned point (5x more arms, 10x faster drift), and the ablations show this is tuning brittleness, not a cost of exploration:** retuned for the
  shifted world, Kaiser wins at both points. So the defensible map is "Kaiser's tuned settings break under large shifts; its approach does not", plus MiniGrid where RND helps only where exploration is the bottleneck, at a large enough bonus and budget (and only post-hoc evidence so far).
- Showdown (new, Phase 4): on a single familiar opponent, tuned Kairi and tuned Kaiser tie (AUC 0.406 vs 0.403, CIs overlap), so the bonus is neither a win nor a cost there; the map does not extend to "where it breaks" for Showdown because there is no shift knob. Adding type-matchup inputs to the observation leaves it a tie (AUC 0.416 vs 0.424, CIs overlap); only Kaiser's AUC improves clearly against the original run (+0.022), confounded with retuning.
- The 5M-step RND benefit was confirmed under the full protocol on that one level (see the confirmation section above): Kaiser at `rnd_coef` 0.6 beat Kairi on 10/10 fresh eval seeds. Still open: other levels, wall-clock cost. The frozen-tuning brittleness has not been tested in deep RL beyond the few-shot re-analysis.

## Methodological lessons (worth carrying forward)
- Give both agents equal treatment on every non-tuned hyperparameter, or tune it for both (entropy was fixed for one and tuned for the other).
- Pre-declare the tie-break rule and check how many grid points tie; a saturated tuning objective (5 of 6 values at 0.000) makes the "tuned" choice arbitrary.
- Report post-hoc robustness checks next to the headline, not instead of it; do not pick a variant by its eval score.
- Seed-level failures (one outright failure in 10) move means a lot; always report them with the CI.
- Pre-register the spike gates and read nothing downstream before they are met: the Crafter gate was raised from 3 to 5 achievements after a smoke test showed a near-random policy already scores about 3, and the spike then correctly failed instead of passing on a trivial threshold.
- A spike can be the result: it is cheaper to find out that a world has no usable range (Crafter, MultiRoom at 2M steps) than to run the full protocol and read noise.

## Crafter spike (2026-10-26)
Plain PPO, one seed per area, 1M steps: final mean achievements per episode 4.52 (area 32) and 4.37 (area 64), below the pre-set G1 gate of 5; curves still rising. No gate passed; the user closed Crafter as a documented spike (2026-10-26), no full phase. We may reconsider Crafter in the future; the user decided against building it for now because of time, not because the question is answered. No RND result was read. See `docs/spikes/crafter_spike.md`.

## Next
Nothing required. Crafter is closed as a documented spike (reopening needs a spec amendment raising S). Showdown is done (above), including the matchup-features run. Deferred by the user: recording a few battles per agent to watch in the Showdown replay viewer (plan `docs/superpowers/plans/2026-10-27-showdown-battle-replays.md`; it needs retraining, since policies are not saved).
