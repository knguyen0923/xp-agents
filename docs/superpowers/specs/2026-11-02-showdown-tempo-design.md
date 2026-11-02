# Showdown tempo opponent, tempo features and friendly-fire penalty (spec 5.6 amendment)

Status: draft for the user's review (2026-11-02). Nothing here is built or approved to run. Design choices were made with the user in a brainstorm on 2026-11-02 (recorded under "Decisions").

## Why

Four Showdown runs ended in ties (original, matchup features, control; see `docs/FINDINGS.md`, Phase 4). The scripted opponent (`DamageHeuristicPlayer`) only picks damaging moves, so it never sets Trick Room, Tailwind or a screen, and a win rate against it says nothing about field and tempo management. The agents also cannot see what they would need to manage tempo: only base Speed and boosts, one scaled priority number, and on/off field flags with no remaining duration (`docs/FUTURE_WORK.md`, "Showdown: tempo and format-specific features", candidate follow-ups 1 and 3). Separately, nothing stops an agent from hitting its own partner. The question: against an opponent that uses tempo, do tempo features change the Kairi-vs-Kaiser picture?

## Decisions (user, 2026-11-02)

1. Scope: follow-up 3 (scripted tempo opponent) and follow-up 1 (tempo block) together, plus a friendly-fire penalty. Follow-up 2 (foe ability and item from the team pool) stays deferred.
2. Friendly fire is a reward penalty, not an action mask. The agents learn the avoidance.
3. The penalty is estimated at choice time (not measured from the battle log), fixed, and not tuned.
4. Teams: revise the existing teams so every team carries a setter (not hand-written new teams). Amended during planning (see Change 4): the revised copies live in a new directory, `configs/showdown_teams_tempo/`, because editing `configs/showdown_teams/` in place would invalidate the pinned digests of the four shipped Showdown configs and a test that checks them.
5. Two runs (below). The old `damage` opponent runs stay as history.

## Change 1: friendly-fire penalty

- Config key `env.friendly_fire: <C>` (absent or 0 means off). With it off, rewards, config hashes, results and tests of existing runs are unchanged; a test pins this.
- When a slot's action is chosen, a damaging move that would hit the ally is charged `-C * f`, where `f` is the estimated damage as a fraction of the ally's current HP, `min(1, expected damage / current HP)`, from the existing estimator (`xp/showdown/estimate.py`, mean of the calculator's min and max roll).
  - Hits the ally: the move targets an ally (showdown targets -1 or -2, taken from the action index as in `masks.py`), or its target class is `allAdjacent` (Earthquake, Surf) with the partner alive.
  - Not charged: `allAdjacentFoes` moves (Heat Wave, Rock Slide), `all`-target and status moves, zero-power moves, a fainted or absent partner. An immune partner (Flying or Levitate against Earthquake) has `f = 0` from the estimator, so it is not charged.
  - Both slots can hit the ally in one turn; the penalties add, so one decision's penalty is at least `-2 * C`.
- `C = 1.0` (the size of a win), pinned in the config, identical for both agents, never tuned, so it adds no tuning budget.
- Where it lives: `calc_reward(battle)` does not see the chosen action, so the penalty is computed in the env worker (`xp/showdown/vec_env.py`), from `env.battle1` before `wrapped.step(action)`, and added to the returned reward. The sparse win/loss reward is otherwise unchanged. The penalty is added to the extrinsic reward, so Kaiser's RND bonus sits on top as before.
- Counters: total penalty charged and number of friendly-fire actions per run are logged to stderr like the matchup estimator counters, so a run reports how often the agents hit their partner.
- Tests (no server): ally target on a damaging move; `allAdjacent` with the partner alive; `allAdjacentFoes` is free; immune partner is free; fainted partner is free; status move is free; two charged slots sum; flag off gives zero.

## Change 2: tempo block (switchable)

Same pattern as the matchup block: appended to the observation only when `env.tempo: true` (absent means false); with it off the observation, `OBS_DIM`, hashes and tests are unchanged and a test pins this. `obs_dim(matchup, tempo)` replaces `obs_dim(matchup)`; both flags stack.

Floats added with the flag on:

- **Speed order:** for each of my 2 actives against each of the 2 opposing actives, one float: +1 if mine moves first, -1 if theirs does, 0 if tied or either is absent (2 x 2 = 4). It accounts for Speed stats and boosts, Tailwind on either side, Trick Room (which reverses the order), paralysis and Choice Scarf where known. Move priority is not folded in here; the per-move priority is already in the move features, and a per-move "moves first" number would add 2 x 4 x 2 more floats, which this spec leaves out (YAGNI).
- **Foe Speed stats:** unknown (`None`) in poke-env, so they use the matchup spec's fallback (level 50, 0 EVs, 31 IVs, neutral nature, from current `base_stats`). Own stats are real. Bias as in the matchup spec: real foes invest, so foe speed is underestimated.
- **Remaining turns:** Trick Room (1 field) and, for each side, Tailwind and each of Reflect, Light Screen, Aurora Veil: 1 + 2 x 4 = 9 floats, each `turns_left / full_duration`, 0 when not active. Turns left = start turn + duration - current turn, using the start turns poke-env records (`battle.fields[field]` and `side_conditions[condition]` hold the turn the condition started; verified against `AbstractBattle._side_start`). Durations: Trick Room 5, Tailwind 4, screens 5. Light Clay would make a screen 8 turns; the agent's own Light Clay is known and used, the foe's is not (the foe item is hidden), so a foe screen is assumed 5 turns. Counters never go below 0. A plan test checks the off-by-one against a real server log before any run.
- **Move flags:** `sets_trick_room`, `sets_tailwind`, `sets_screen` (Reflect, Light Screen, Aurora Veil) and `hits_ally` (a damaging move whose target class is `allAdjacent`, so it hits the partner whatever target is chosen; ordinary single-target attacks are not flagged, since the agent picks their target and the penalty teaches that choice). Added per move slot, 4 x 4 = 16 floats for each of my 2 actives and each of the opposing 2, so 4 x 16 = 64 floats. Foe slots stay zero until revealed, as for the rest of the move features.
- Both agents use the identical block, frozen before tuning (spec 5.6).
- Tests (no server): block shape; flag off gives a vector identical to the current encoder; a faster mon reads +1, Trick Room flips it, Tailwind flips a slower mon; paralysis changes the order; remaining turns for each condition at turn t and at expiry; move flags for known moves (Trick Room, Tailwind, Reflect, Earthquake, Heat Wave, Protect).

## Change 3: `TempoPlayer` (new opponent value `tempo`)

`TempoPlayer` extends `DamageHeuristicPlayer` and registers in `make_opponent` as `tempo` (the ladder keeps `random`, `maxbp`, `damage`). Before the damage logic, each active slot may use a setter move, at most one setter per slot per turn, in this order:

1. **Trick Room:** if the move is available, Trick Room is not up, and this side is slower than the foes (its mean active Speed, under the same fallback stats, is below the foes' mean).
2. **Tailwind:** if available, not already up on this side, and Trick Room is not up (Tailwind is wasted under Trick Room).
3. **Screens:** Reflect when most of the foes' known damaging moves are physical and Reflect is not up; Light Screen when most are special and it is not up; Aurora Veil only in snow and not up. A screen is never recast while up. With no foe move revealed, Reflect and Light Screen are not used until one is.
4. Otherwise the damage heuristic, unchanged.

Both slots must not set the same condition on one turn (the second slot falls back to the damage heuristic). It never targets its own partner. It never switches voluntarily, as today. `TempoPlayer` uses the same mega rule as the parent.

Tests (no server, constructed battles): each rule fires and does not fire in the stated cases; no recast of a live condition; no double-set; falls back to damage heuristic when no setter is available; never targets an ally; returns a legal order in every case (checked with the same legality rule as `masks._legal`).

## Change 4: team pools

- Today 4 of the 6 teams carry a setter (Trick Room on `tune1` and `tune2`, Tailwind on `tune3` and `eval3`, Reflect and Light Screen on `eval3`); `eval1` and `eval2` have none, and no eval Pokemon that is legal in the format can learn Trick Room except Indeedee-F, which is not on an eval team.
- New directory `configs/showdown_teams_tempo/{tune,eval}`: copies of the six teams with small revisions so that every team carries at least one setter and each pool has Trick Room, Tailwind and a screen (the plan lists the exact edits; each was checked with Showdown's validator). `configs/showdown_teams/` is not touched, so the four shipped configs, their pinned digests and the finished runs stay valid. Both sides draw from the same pools, so the agents get setters too.
- Every team is checked with `node pokemon-showdown validate-team gen9championsvgc2026regmc < team.txt` (rejected teams make poke-env hang silently). The new configs pin the new pools' digests (`tune_teams_sha`, `eval_teams_sha`). `assert_disjoint` still applies: no shared team or file name across tune and eval.
- A guard test (no server) requires every team to carry at least one setter move and each pool to cover Trick Room, Tailwind and a screen, so a later edit cannot quietly remove them.

## Experiment

Opponent `tempo`, familiar level only (single familiar level, no shift knob, as in the earlier Showdown runs), the revised pools, budget cap S and threshold unchanged unless the plan's gate says otherwise.

| Run | `matchup` | `tempo` | `friendly_fire` | Config |
|---|---|---|---|---|
| 1 | true | false | 1.0 | `configs/showdown-tempo-base.yaml` |
| 2 | true | true | 1.0 | `configs/showdown-tempo.yaml` |

- Each is the usual pipeline (tune both agents on equal-sized grids using tune seeds, freeze, evaluate on eval seeds). About 12 to 15 hours each on 1 worker, about 25 to 30 hours in all. Fresh output folders (`results/showdown-tempo-base`, `results/showdown-tempo`).
- Grids: the 8-value grids of the matchup and control runs (Kairi `ent_coef` [0.001, 0.002, 0.003, 0.006, 0.01, 0.02, 0.05, 0.1]; Kaiser `rnd_coef` [0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0]). A tuned optimum on a grid edge makes the run provisional and stops it (`ProtocolError`); then ask the user and widen both grids equally in a fresh folder.
- Comparisons: Kairi vs Kaiser within each run (training AUC, steps to the 0.30 win-rate threshold, 95% bootstrap CIs); run 2 against run 1 for each agent, paired by eval seed. Because friendly-fire is on in both runs, the run 2 versus run 1 difference is the tempo block alone (the opponent and teams are also identical across the two runs).
- Not comparable with the old `damage` runs: the opponent, the teams and the reward differ. The write-up says so and does not pool them.
- Budget cap S: the user's approval is needed before any RND result is read (CLAUDE.md). The plan's first step re-runs the usable-range gate (`xp/showdown/gate.py`) for plain PPO against `tempo` on the revised pools, reports the number, and asks.
- Counters reported per run: friendly-fire actions and penalty per battle, per agent (so the write-up can say whether either agent learned to avoid its partner).

## Out of scope

Follow-up 2 (foe ability and item from the team pool); learning aids (pretraining, auxiliary losses); switching strategy for either opponent; any change to `xp.protocol`.

## Risks

- The opponent's tempo play may be rarely triggered even with revised teams (a setter needs to survive and be in the 4 picked at team preview); the plan measures the fraction of battles where `tempo` sets anything and reports it.
- The penalty may teach both agents to avoid spread moves altogether; the counters show it.
- The remaining-turn counters could be off by one; the plan checks them against a server log.
- Another tie is a valid result. It would close the Showdown line with five sound ties.

## Files touched (expected)

`xp/showdown/encode.py` (tempo block, `obs_dim`), a new `xp/showdown/tempo.py` (speed order, durations, move flags), `xp/showdown/opponents.py` (`TempoPlayer`), `xp/showdown/friendly_fire.py` (penalty) wired from `xp/showdown/vec_env.py` and `env.py`, `xp/showdown/phase.py` and configs (new keys, hashing), `configs/showdown_teams_tempo/**` (new), new configs, `xp/showdown/gate.py` (opponent and pool options), tests under `tests/`, and the write-ups after the runs.
