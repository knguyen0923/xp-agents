# XP — future-work sketch: Crafter and Pokémon Showdown

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

## Also still open (smaller)
- A protocol-legal confirmation of the 5M-step MultiRoom probe (RND at coefficient 0.1 reached 95% in about half the steps of plain PPO, post-hoc): tune the RND coefficient on tune seeds over a grid reaching above 0.1, evaluate on 10+ disjoint eval seeds, matched entropy by construction. Not run; see `docs/FINDINGS.md`.
- A deep-RL retune-at-the-shift check (size 8 DoorKey); judged low value, config-only if wanted.
