# Showdown budget gate (spec 5.6 amendment)

Status: DONE 2026-10-26. **The budget cap S and the familiar opponent are the user's decisions; this page only measures and recommends.** No RND was run and no eval seed or eval team was touched.

Setup: plain PPO (Kairi, `ent_coef` 0.01, untuned), 8 env processes against one local Showdown server, tune seed 0, tune teams only, `gen9championsvgc2026regmc`, masked actions, sparse +1/-1 reward. Probe budget 122,880 agent decisions per rung (120 PPO updates of 1,024 decisions), which is a probe, not a proposed S. Code: `xp/showdown/gate.py`, plan `docs/superpowers/plans/2026-10-26-showdown-learner-and-gate.md`. Raw JSON stayed in the gitignored `results/showdown_gate/`.

## Result per rung

Win rate is over the last 100 finished battles at the given decision count, so each cell has a standard error of about 0.05 (one seed).

| Rung | at 25.6k | at 51.2k | at 87k | final (122.9k) | in 20-80%? | decisions to 80% | decisions/s |
|---|---|---|---|---|---|---|---|
| random | 0.91 | 0.86 | 0.88 | **0.91** | no (too easy) | 15,360 | 283 |
| maxbp (MaxBasePower) | 0.51 | 0.55 | 0.65 | **0.70** | yes | not reached | 268 |
| damage (DamageHeuristic, with Mega) | 0.26 | 0.36 | 0.30 | **0.36** | yes | not reached | 265 |

Reading: Random is solved within about 15k decisions and cannot show anything. MaxBasePower is still rising at the end of the probe (0.51 to 0.70 between 25k and 123k), so at a larger budget it may leave the usable range from above. DamageHeuristic is flat at about 0.3 to 0.37 after roughly 50k decisions, so it stays inside the range with room on both sides.

## Throughput and the machine limit

- One run at a time: 265 to 283 decisions per second including the PPO updates (the lower figure belongs to the harder rungs).
- Two runs at once on the same server (MaxBasePower, 40,960 decisions each): **166 and 165 per second, 331 combined**, a gain of 1.25x over one run. Parallel runs do not scale linearly here, and more than two concurrent runs were not tried.
- The server was not the limit in the spike (about 18% of one core); the Python processes and the machine are.

## Cost of a full phase (56 runs = 36 tuning + 20 eval, one trained agent each)

Hours = 56 x S / rate / 3600, with the measured single rate 265/s and the two-at-a-time aggregate 331/s. Kaiser's RND adds a forward and an update per batch; its cost is not measured and is not in these figures.

| S (decisions per run) | total decisions | one run at a time | two at a time |
|---|---|---|---|
| 122,880 (the probe) | 6.9 M | 7.2 h | 5.8 h |
| 250,000 | 14.0 M | 14.7 h | 11.8 h |
| 500,000 | 28.0 M | 29.4 h | 23.5 h |
| 1,000,000 | 56.0 M | 58.7 h | 47.0 h |

For scale: the closed Crafter phase was estimated at about 20 h; MiniGrid MultiRoom at roughly 11 to 13 h. Runs go on this laptop with the local server, not Colab (the server would have to be installed there too).

## Recommendation (the user decides)

1. **Familiar opponent: DamageHeuristic.** It is the only rung that stays well inside 20-80% (0.36) with headroom on both sides; MaxBasePower is climbing toward the upper edge and may exceed 80% at a larger S, which would push the whole phase out of the usable range.
2. **Cap S = 250,000 decisions per run.** Both middle rungs have mostly plateaued by 100k to 125k, so 250k is about twice the probe and leaves room for a tuned agent to improve, at about 12 to 15 h for the whole phase. The probe budget (123k, about 6 to 7 h) is the cheaper alternative if time is tight; 500k (about 24 to 29 h) only makes sense if early evidence shows the curves still moving.
3. The spec's rule for S (smallest count where default-entropy Kairi reaches a threshold, capped) cannot pick a threshold the DamageHeuristic rung never reaches, so S would simply be the cap. The threshold for "steps to target" metrics also needs choosing: 0.30 would be reached against DamageHeuristic, 0.80 would not.

## Limits and things not understood

- One seed, one tune-team pool of three teams, untuned PPO, 120 updates. The rung order and the 20-80% placement are indicative; a different seed could move a rung by roughly 0.1.
- In the two-at-a-time probe the logs showed two server messages from the opponent's side (`[Invalid choice] There's nothing to choose` and `Can't do anything: The game is over`). They did not appear in the three single-run logs. The cause was not investigated; their effect on opponent strength is unknown but they occurred twice in about 82,000 decisions.
- Games are short (a few turns) and the scripted opponents never Protect or switch deliberately, so the win-rate scale says little about play against humans.
- Team preview is random for both sides; the learner does not choose its four.
- Plain PPO only: nothing here says anything about Kairi versus Kaiser.

## Next (after the user's decisions)

Plan C: RND and the Kaiser learner, matched entropy, equal 6-value tuning grids, `configs/showdown.yaml` with the chosen opponent and S, `xp/showdown/phase.py`, `xp.run` wiring with per-job caching, and the report. No RND result is read before S is fixed.
