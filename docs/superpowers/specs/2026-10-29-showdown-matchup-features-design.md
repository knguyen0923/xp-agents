# Showdown matchup features (spec 5.6 amendment)

Status: draft for the user's review (2026-10-29), amended the same day after the user's modelling assumption (below). Nothing here is built or approved to run.

## Why

The Showdown full run ended in a tie (training AUC Kairi 0.406, Kaiser 0.403). The agents' observation
(`xp/showdown/encode.py`) carries move type, power and category and each Pokemon's types, but no type
effectiveness and no damage estimate, so a network has to learn the type chart and the damage math from
sparse win/loss reward alone. The scripted opponent (`DamageHeuristicPlayer`) gets both directly. A replay
also showed an agent skipping a move that would have knocked out a foe (a 4,096-step smoke policy, so weak
evidence). The question: does handing both agents the matchup information change the picture?

## Change

A switchable observation block, off by default.

- Config key `env.matchup: true|false` (absent means false). With it off, the observation, `OBS_DIM`, config
  hashes, results and tests of the existing run are unchanged; a test pins this.
- Modelling assumption (user, 2026-10-29): the agent is taken to know what every move and ability does. The only
  hidden information is the opponent's moveset, which is unknown until each move is revealed. The features
  below are computed in code from that knowledge; the network sees only the resulting numbers.
- With it on, add two blocks of 2 floats per (attacker, move, target) triple:
  - **Outgoing:** each of my 2 actives, each of its 4 moves, each of the 2 opposing actives: 2 x 4 x 2 x 2 = 32 floats.
  - **Incoming:** each of the 2 opposing actives, each of its 4 move slots, each of my 2 actives, 32 floats, plus
    one `revealed` flag per opposing move slot (2 x 4 = 8). Unrevealed slots are all zeros with `revealed` 0, so
    a revealed immune move (value 0, `revealed` 1) differs from an unknown move. Unrevealed moves are never
    guessed. In total 72 more floats with the flag on.
  - The two numbers per triple:
    1. type multiplier, taken from the damage calculation so ability effects (Levitate, Flash Fire, Sap Sipper,
       Wonder Guard, ...) show up; scaled into the observation range.
    2. estimated damage as a fraction of the target's current HP, `min(1, expected damage / current HP)`. A value
       of 1 means a likely KO.
  - Status and zero-power moves, empty slots, and fainted or absent targets give 0.
- Both agents use the identical block (spec 5.6: the encoding is the same for both and frozen before tuning).

### Damage estimate

- Use `poke_env.calc.damage_calc_gen9.calculate_damage` (min and max roll; use the mean). It models many
  abilities, items, weather and boosts, and its header lists gaps (multi-hit moves averaged, some auras and
  special moves).
- It asserts that both Pokemon have known `stats`. Where the opponent's stats are unknown, fall back to
  estimates from base stats at level 50 with a fixed, stated spread (decided in the plan and written here
  before any run). If the calculator raises for a move, that entry is 0 and a counter records it.
- Abilities and items (probe result, `docs/spikes/showdown_matchup_probe.md`): poke-env does NOT expose the Open
  Team Sheet. After `reset()` foe items are `unknown_item`, foe abilities are only set for single-ability species
  (and one value looked wrong, Gardevoir `pickpocket`), foe stats are all `None`, foe `moves` is empty and only
  the two leads are in `opponent_team`. So the estimate uses own abilities and items, and foe abilities only where
  poke-env has them. Foe ability immunities (Levitate, Flash Fire, ...) are therefore missed unless revealed or
  single-ability. This falls short of the user's assumption that abilities are known; reading each foe's sheet from
  the team pool the env itself draws (exact, and what OTS would show) is a deferred follow-up, not part of this build.
- Foe stats are `None`, so they are always filled from base stats at the fallback spread; own stats are real.
- Measured cost (probe): a working calculator call is under 1 ms, so about 10 to 20 ms per decision worst case;
  re-measure on the smoke config.

### Code shape

- Make the observation size a value passed from the config instead of an import-time constant. `OBS_DIM` is
  imported by `ppo.py` (4 uses), `rnd.py`, `env.py` and `vec_env.py`. The flag-off value stays equal to today's.
- `xp.protocol` stays phase-agnostic; nothing here touches it.
- Tests (no server): block shape (72 extra floats) and zeros for status moves, empty slots and unrevealed move slots, with the `revealed` flag set only for revealed moves; flag off gives a vector identical
  to the current encoder; a known matchup (for example a Water move into a Fire target gives 2x, into a
  Levitate target a Ground move gives 0) using constructed poke-env objects.

## Experiment

- New `configs/showdown-matchup.yaml`: `configs/showdown.yaml` plus `env.matchup: true`, output to its own
  `results/showdown-matchup`. The original run is not touched or rerun.
- Same protocol: tune on tune seeds, freeze, evaluate on eval seeds, same grids, same 250,000 decision cap S,
  same 30% threshold, equal-sized grids for both agents. Both agents are tuned from scratch under the new
  observation (old tuned values need not carry over).
- Comparisons, with 95% bootstrap CIs over seeds: matchup vs original per agent (does the information help),
  and Kairi vs Kaiser within matchup. A tie everywhere is a valid result and gets written up as such.
- A tuned optimum on a grid edge makes the run provisional, as before.
- Cost: about 12 to 15 hours, like the original. The user approves the run before it starts, and approves the
  budget before any RND result is read.

## Out of scope

- A learning aid for the network (pretraining on the type table, auxiliary losses). Revisit only if the
  feature result is interesting.
- Probing the existing policies for type-chart sensitivity (a separate, cheap script that needs saved
  policies; the replay recorder saves one per agent).
- Changing the opponent, the teams, the reward or the action head.
