# Showdown battle replays

A few readable battles from each agent, for illustration. They show how the agents play. They are not evidence about which agent is better: the Showdown comparisons were ties (see `docs/FINDINGS.md`, Phase 4), and a handful of picked battles cannot show a difference.

## How to open one

Open the `.html` file in a browser. **It needs internet**: the page loads the replay viewer from play.pokemonshowdown.com.

## The battles

| File | Agent | Result | Turns | Lead changes | Why it was picked |
|---|---|---|---|---|---|
| [kairi-1-7668815.html](kairi-1-7668815.html) | Kairi | win, 1 Pokémon left | 8 | 3 | highest interest score of Kairi's wins; the lead changes hands three times; a Mega Evolution |
| [kairi-2-7668805.html](kairi-2-7668805.html) | Kairi | win, 1 Pokémon left | 7 | 2 | second-highest; a close finish after two lead changes; a Mega Evolution |
| [kaiser-1-7698701.html](kaiser-1-7698701.html) | Kaiser | win, 1 Pokémon left | 12 | 3 | highest interest score of Kaiser's wins; three lead changes; a Mega Evolution |
| [kaiser-2-7698722.html](kaiser-2-7698722.html) | Kaiser | win, 1 Pokémon left | 6 | 2 | a short, close win after two lead changes; a Mega Evolution |

All four are wins, so they are flattering by construction. Over the 40 or so battles recorded for each agent, Kairi won 23 of 41 and Kaiser 18 of 40 (one training run each, so that gap is not a finding).

## How they were made

- **Agents:** retrained for this, using the tuned hyperparameter from the finished original Showdown run (Kairi `ent_coef` 0.01, Kaiser `rnd_coef` 0.03), training seed 100, the full 250k-decision budget. They are **not** the agents behind the scored seed-100 results: training is deterministic per config but not byte-identical, and policies were not saved in the main run.
- **Opponent:** the same familiar opponent as the research runs (DamageHeuristic), on fresh team draws from the eval pool.
- **Selection:** 40 battles per agent, ranked by an interest score (a close finish, lead changes, length, Mega or Tera use; `xp/showdown/replays.py`). From that ranking we kept wins of 5 to 12 turns, so each is short enough to read and not a blowout, then chose the top 2 per agent by hand after watching them.
- **Recorder:** `scripts/record_showdown_replays.py --agent <kairi|kaiser> --seed 100 --battles 40 --keep 3 --min-turns 5 --max-turns 12 --wins-only`. Full ranking tables stay in the gitignored `replays/` folder.
