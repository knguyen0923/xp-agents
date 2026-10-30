# XP — future-work sketch: Crafter and Pokémon Showdown

**Update 2026-10-30:** Showdown was built and run (original run 2026-10-28, matchup-features run 2026-10-29 to 2026-10-30): a tie in both, with only Kaiser's training AUC rising clearly against the original run, confounded with the wider grids and retuning (`docs/FINDINGS.md`, Phase 4). Sections below that call Showdown "not approved for building" are history. The open follow-ups are collected under "Also still open" at the end.

**Update 2026-10-26:** the Crafter spike ran and failed its usable-range gate (G1) at both areas; the user closed Crafter as a documented spike (`docs/spikes/crafter_spike.md`). We may reconsider Crafter in the future; the user decided against building it for now because of time, not because the question is answered.

**Update 2026-10-25:** both worlds are now specified as generality tests (single familiar level, no shift knob) in spec sections 5.5 (Crafter, in scope, spike first) and 5.6 (Showdown, not approved for building). The sketch below is the earlier thinking.

Status: a design sketch, **not a plan and not approved for building**. `CLAUDE.md` and the design spec list both worlds as non-goals; building either needs the user to lift that rule first.
Environment facts below come from general knowledge, not from the project; items marked (verify) must be checked against the current package docs before any spec is written.

## What lessons from the finished phases carry over
1. **Spike first.** DoorKey and MultiRoom both had levels where every agent sat at 0% or 100%; a 1-hour spike found the usable difficulty axis (MultiRoom needed small rooms). Any new world starts with a spike that finds where plain PPO is at the edge.
2. **Equal treatment by construction.** Our largest MiniGrid confound was Kaiser's entropy being fixed while Kairi's was tuned. In any new world, tune shared hyperparameters for both agents, or fix them identically.
3. **Pre-declare the tie-break and check for saturated tuning.** Tuning at an easy familiar level makes most grid points tie at the ceiling, so the "tuned" choice is arbitrary (5 of 6 `rnd_coef` values tied in MiniGrid).
4. **Report frozen-tuning brittleness separately from the approach.** The bandit ablations showed the Kairi wins came from Kaiser's frozen settings, not exploration. Plan a retune-at-the-shift ablation from the start.
5. **Seeds dominate in deep RL.** Plan for many seeds and wide CIs; one failing seed moves a mean by about 0.1.

## Crafter
**Question it would answer:** does an RND bonus pay off in a long-horizon, sparse-reward, procedurally generated world where exploration is the bottleneck (the question the MiniGrid spikes could not settle at 2M steps)?
- **Environment (verify):** 2D survival game, 64x64 image observations, 22 achievements, reward for unlocking an achievement once per episode plus small health terms, a `crafter` Python package with a Gym-style interface and an `area` argument for world size, standard budget about 1M steps.
- **Agents:** Kairi = PPO, Kaiser = PPO + RND, as in MiniGrid. Requires a larger convolutional network for 64x64 images than the 7x7 grid net, and likely GPU time (Colab); the existing `Learner` would need generalising (network, observation prep, vectorised env, reward handling), so this is a real new sub-project.
- **Candidate shift knob:** world area (smaller familiar world, sweep to larger worlds with sparser resources; the `area` constructor argument, verify). Alternatives: episode length, or a held-out set of world seeds (weak shift). Metric: achievement score (geometric mean of unlock rates) or number of distinct achievements unlocked.
- **Protocol mapping:** tune seeds and eval world seeds disjoint as now; tune on the familiar area, freeze, evaluate at every area, paired by seed.
- **Cost (rough estimate, not measured):** each run is millions of image-based steps, plausibly tens of GPU-hours for a full sweep with tuning grids; the laptop is not suitable, Colab is. Start with a one-seed spike at the standard budget.
- **Risks:** the floor/ceiling problem again (many achievements are rarely unlocked by PPO at 1M steps); RND may show no benefit, as on DoorKey (while on MultiRoom it helped only at a large coefficient and a 5M-step budget), which is a valid result; compute budget; engineering of the vectorised env and reward logging.
- **Go/no-go for a full phase:** the spike shows plain PPO unlocking some but not all achievements at the familiar area and degrading (not collapsing) at the larger areas.

## Pokémon Showdown
**Question it would answer:** does uncertainty-driven exploration help when nonstationarity comes from an opponent, as in the changing bandit, rather than from the environment dynamics?
- **Environment (verify):** the Pokémon Showdown battle simulator (a Node.js server) driven from Python through a wrapper such as `poke-env`, which provides Gym-style single-agent environments against scripted opponents; random-battle formats avoid team building; partial observability (hidden opponent information); reward is a win/loss signal, optionally shaped.
- **Agents:** Kairi and Kaiser need new definitions here: a PPO-style policy over structured features is plausible, with Kaiser adding an RND-style bonus on the state features. This changes Kaiser's implementation more than Crafter does, so the "Kaiser = Kairi plus an exploration term" principle must be kept strict.
- **Candidate shift knob:** opponent identity or strength: tune against one scripted opponent, evaluate against a sweep ordered by strength or style (random, max-damage heuristic, a stronger heuristic). Metric: win rate.
- **Protocol mapping:** battle seeds disjoint across tune and eval pools; opponent as the knob; paired by seed where the simulator allows seeding (verify, since battles may not be fully seedable).
- **Cost (rough estimate):** simulation is cheap per step, so the sweep may be feasible on the laptop after a long integration effort (server, wrapper, opponents, observation encoding). The risk is engineering time, not compute.
- **Risks:** exploration may play a small role (the large action space and opponent modelling may dominate), reproducibility of seeded battles, observation design choices that confound the two agents, and a server dependency that makes CI and Colab runs awkward.
- **Go/no-go for a full phase:** a spike shows plain PPO beating the weakest opponent, losing to the strongest, with a graded range in between.

## Recommendation if only one is ever built
Crafter, with the world-area knob, because it tests the open question in the cleanest way (same agents, exploration-heavy) and reuses the protocol and agent definitions, at the cost of GPU time and a network upgrade. Showdown is the more novel test of opponent-driven nonstationarity but needs a new agent design and a heavy integration; do it only if the bandit-style "frozen tuning breaks under shift" finding is the question of interest. Either way: run the spike first, write the spec, and update `CLAUDE.md` and the design spec's non-goals before any code.

## Protocol-legal confirmation of the MultiRoom RND result (discussed 2026-10-24; scope NOT yet decided, nothing built)
Context: the 5M-step probe (`docs/FINDINGS.md`, `docs/spikes/multiroom_probe.csv`) found RND at `rnd_coef` 0.1 reaching 95% training success in about half the steps of plain PPO at 4 MultiRoom rooms (max room size 4), faster on 5/5 seeds, mean 1.17M fewer steps [0.53M, 1.75M]. It is post-hoc.

**What a protocol-legal version must fix**
1. Selection asymmetry: RND had three coefficients to choose from and plain PPO none; the best RND value was reported. Both agents must tune on tune seeds only, then freeze.
2. Grid edge: 0.1 was the top of the range tried, so the RND grid must reach above it (otherwise the run is provisional).
3. Disjoint eval seeds: the probe used tune-pool seeds 0 to 4; the confirmation needs fresh eval seeds, 10 or more.
4. Pre-declared budget S and tuning objective: the spec's rule sets S from default-entropy Kairi on the tune seeds (not from RND results). At 5M steps nearly every setting reaches about 100% success, so the usual objective (success at S) saturates and ties; a speed-based objective (for example training AUC or steps to target) would have to be pre-declared before any run.
5. Entropy matched by construction: in the probe both agents used `ent_coef` 0.01, but the protocol-legal version should tune Kairi's entropy and give Kaiser the same value.

**Code that would change:** `xp/minigrid/env.py` builds DoorKey only (`make_env`, `VecEnv`, `evaluate_policy`); a MultiRoom variant (`MultiRoomEnv` with `minNumRooms=maxNumRooms=N`, `maxRoomSize=4`) means generalising the factory and the knob; `configs/` gets a new config; `xp/minigrid/phase.py` needs Kaiser's fixed entropy taken from Kairi's tuned value (see `docs/superpowers/plans/2026-10-24-phase2-step-a.md`, item 1, for the shelved design of this change); the spec gets an amendment (level, budget rule, tuning objective, tie-break). TDD applies.

**Cost (rough, extrapolated from the probe's 30 to 40 minute runs, not measured):** tuning both agents on a 6-value grid with 3 tune seeds is 36 runs, plus about 20 eval runs: roughly 8 to 10 hours on the laptop; the calibrated budget S is likely below 5M, which lowers it.

**Options discussed (user has not chosen)**
1. *Single-level confirmation, fully protocol-legal (recommended):* familiar level = 4 rooms, size <= 4, no shift sweep; Kairi tunes `ent_coef`, Kaiser tunes `rnd_coef` with a grid reaching above 0.1 and entropy matched; tune seeds only, then 10+ disjoint eval seeds; metrics steps to a success target plus success at the calibrated budget, paired CIs. About 8 to 10 hours.
2. *Pre-declared replication, no tuning:* freeze RND at 0.1 and plain PPO at entropy 0.01, 10 fresh seeds each; about 4 to 7 hours; fixes the seed-selection issue but not the unequal tuning or the grid edge.
3. *Full phase with a shift knob* (for example zero-shot across room counts 2 to 5): answers where RND breaks too, but is the most expensive and zero-shot transfer across room counts is likely near 0% or 100% everywhere.

**Decisions still open if this is picked up:** which option; the tuning objective and tie-break (pre-declare); the RND grid values above 0.1 (for example up to 0.3 or 1.0, with equal-sized grids for both agents); number of tune seeds; the budget rule (spec's calibration vs a fixed S). Next step is the brainstorming process: scope decision, then a spec amendment, then a plan.

## Showdown: tempo and format-specific features (raised 2026-10-29; nothing decided, nothing built)
Context: the Showdown run ended in a tie, and a look at the observation (`xp/showdown/encode.py`) showed what the agents are not told. The matchup-features build (spec `docs/superpowers/specs/2026-10-29-showdown-matchup-features-design.md`, plan `docs/superpowers/plans/2026-10-29-showdown-matchup-features.md`) covers type multipliers and damage estimates only. It does nothing for the items below.

**What the agents do not see today**
- Speed: only base Speed (scaled by 255) and boosts. No actual Speed stats, no "who moves first" comparison, no Choice Scarf or paralysis effects.
- Priority: one scaled number per move; no priority-bracket ordering. Trick Room is only recognizable by its -7 priority; Tailwind looks like any other Flying status move, because move identity is not encoded (only power, accuracy, priority, type, category, Protect and spread flags).
- Trick Room, Tailwind, weather, terrain and screens: on/off flags only, with no remaining duration. The only clock is the turn counter, capped at 40.
- IVs, EVs and natures: not in the observation for either side, including the agent's own Pokemon (real stat spreads are not seen).
- The foe's Open Team Sheet: poke-env does not expose it (probe: `docs/spikes/showdown_matchup_probe.md`); foe stats are `None` and abilities and items are mostly unknown, so ability immunities (Levitate, Flash Fire) are mostly missed.

**What the opponent does:** `DamageHeuristicPlayer` only picks damaging moves, so it never sets Trick Room or Tailwind. Training against it gives almost no signal for field and tempo management; a win rate here says little about it.

**Candidate follow-ups (each a spec amendment plus a full run, about 12 to 15 hours)**
1. A switchable "tempo" block: speed-order comparison per pair of actives (including Trick Room, Tailwind and priority), effect durations, move-identity flags (sets Trick Room, sets Tailwind, and similar).
2. Fill each foe's ability and item from the team pool the env itself draws (exact, and what an Open Team Sheet would show); keep foe moves hidden until revealed.
3. A scripted opponent that uses Trick Room and Tailwind, so field management is tested and not just available.
4. Probe the saved policies for type-chart sensitivity (vary only the target's type and compare the policy's preference with the real multiplier); the replay recorder saves one `policy.pt` per agent.
Learning aids for the network (pretraining on the type table, auxiliary losses) were left out of scope of the matchup spec; revisit only if the feature result is interesting.

## Also still open (smaller)
- **Matchup control run and a finer Kairi `alpha` sweep (added 2026-10-30; not approved):** specified, with costs, in `docs/REMAINING_WORK.md` sections 2 and 3, which is the index of open work.
- The protocol-legal confirmation of the 5M-step MultiRoom probe: see the section above (options and costs written down, scope undecided).
- A deep-RL retune-at-the-shift check (size 8 DoorKey); judged low value, config-only if wanted.
