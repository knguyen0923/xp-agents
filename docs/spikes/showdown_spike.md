# Showdown spike (spec 5.6)

Status: DONE 2026-10-26. Recommendation below; **the go/no-go is the user's decision.** Nothing was built in `xp/`, `configs/` or `requirements*.txt`. No PPO or RND was trained.
Plan and full progress log: `docs/superpowers/plans/2026-10-26-showdown-spike.md`. Throwaway scripts: `scripts/spikes/showdown_*.py|js`.

Target: `[Gen 9 Champions] VGC 2026 Reg M-C`, format ID `gen9championsvgc2026regmc` (doubles, team preview, bring 6 and pick 4, level 50, Champions mod, Open Team Sheets). The user supplied six teams: `tune1-3` and `eval1-3`.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| S1 install, format, one full battle on real teams | **Pass** (with a data patch to `poke-env`) | Server and wrapper run locally. Format exists, no custom entry needed. All six teams pass the server validator. 120 of 120 battles finished across all six teams, no hangs. Every Mega stone in the pools produced its Mega forme in battle. |
| S2 graded scripted opponent ladder (each step above 60% over 200+ battles) | **Pass** | 240 battles per pairing, teams swapped between sides: MaxBasePower over Random 0.863 (95% +-0.044); DamageHeuristic over MaxBasePower 0.767 (+-0.054); DamageHeuristic over Random 0.954 (+-0.026). |
| S3 throughput and cost | **Pass** (upper bound) | 400 battles with team preview: 60 battles/s, 260 decisions/s on one client; 80 battles/s, 356 decisions/s with 8 battles in flight. Server uses 18% of one core. The Python client is the limit. |
| S4 disjoint pools | **Pass**; seeding only partly available | 100 battles per list on both sides: every battle used only the assigned list's teams (observed from the battle objects, not just the draw log). Tune and eval lists share no team. The engine replays identically under a fixed seed, but a client cannot set that seed (see below). |

## Versions and install (what worked)

- Python 3.9.6 venv; `poke-env==0.10.0` (gymnasium 1.1.1, pettingzoo 1.26.1, numpy 2.0.2).
- Showdown server 0.11.11 needs Node >= 22.18. Local Node v22.23.4 (arm64 tarball) inside `~/xp-showdown-spike/`, nothing installed globally. Shallow clone, `npm install`, copy the example config, then `node pokemon-showdown start --no-security` (about 25 s to come up).
- The `bindaddress` setting is ignored in this version. The scratch checkout was patched (`server/sockets.ts`, `PSBINDADDR: '127.0.0.1'`) so it listens on localhost only. Never connect to the public ladder.
- `poke-env` 0.10.0 lacks 54 Champions species (mostly new Megas) and 2 moves. A battle touching one raises `KeyError` inside the message handler and the battle hangs. Fixed by merging the server's Champions data into the venv's `gen9pokedex.json` and `gen9moves.json` (`showdown_patch_poke_env.py`, `showdown_export_patch.js`; originals kept as `*.orig`). Re-apply after any reinstall. The patch fixes crashes only; Champions stat changes to existing species are read from live battle messages, not from the file.

## Errors and gotchas hit

1. A rejected team makes `poke-env` hang with no exception; the reason is only a `|popup|` line at debug log level. A full phase needs a validation step before any run. We use `node pokemon-showdown validate-team <format> < team.txt` (checked against a deliberately illegal team: it reports the errors; a valid team prints nothing).
2. Showdown strips underscores and truncates usernames to 18 characters, so long or underscored account names silently break challenges. Use short alphanumeric names, fresh per run (an old hung run can leave stale battles).
3. Random-battle sets are not valid Reg M-C teams (7 Pokemon, Item Clause). Real teams were needed.
4. The first `DamageHeuristicPlayer` never set `mega=True`, so no Mega ever fired (0 of 120 battles). Fixed; 17-20 of 20 battles per pairing now contain a Mega. The earlier proxy-format ladder had no Mega and is superseded by the table above.
5. One user team listed `Swampert-Mega @ Swampertite`; it must be the base forme (now `Swampert`, Torrent) so the Mega is reached in battle. All 12 stones in the six teams (10 distinct) map to their base species in the server data.
6. `SimpleHeuristicsPlayer` falls back to random moves in doubles. The middle and top rungs had to be written (`DamageHeuristicPlayer`, about 60 lines).

## S3 cost

Measured with `DamageHeuristicPlayer` on both sides, tune1 vs eval1, mean 4.3 turns per battle.

| Setting | Battles/s | Decisions/s | Server CPU |
|---|---|---|---|
| One battle at a time | 60.1 | 260 | 18% of one core |
| Eight in flight | 79.9 | 356 | 18% of one core |

Full phase = 36 tuning runs + 20 eval runs = 56 runs. **Assumption (placeholder, not a decision): S = 1M agent decisions per run**, which is 56M decisions, about 230k battles per run at 4.3 turns.
- One client process at 260 decisions/s: about 215,000 s = **about 60 h**. At 356 decisions/s: about 44 h.
- With 4 to 5 client processes (total server load stays under one core): **roughly 10 to 15 h**.
- These are **scripted-player rates**. A learner adds a policy forward pass per decision and PPO updates, which is likely a large multiple. The real cost needs a PPO throughput measurement; none was made.
- For comparison, the closed Crafter phase was estimated at 20 h (area 32, MPS, one run at a time).

## S4 detail: pools and seeding

- **Pools:** `showdown_pools.py` gives each player a team builder that draws from its assigned list. Run 1: A draws tune, B draws eval (100 battles). Run 2: swapped (100 battles). Draw counts were tune 38/27/35 and eval 34/32/34; the teams read back from each player's own battle objects matched the draws exactly, and no team from the other list appeared. The six team files are pairwise distinct.
- **Limit of the pools:** species repeat across the tune and eval lists (Sneasler, Rillaboom, Gholdengo, Garchomp, Pelipper, Archaludon, Charizard and others), so the lists are disjoint as teams, not as species. Whether that is acceptable is a design call. With only 3 teams per list, a tuned setting can overfit three teams.
- **Seeding, engine:** `Battle` takes a PRNG seed. Two simulator runs on real teams with the same seed and the same choices gave identical logs; a different seed did not (`showdown_seedtest.js`).
- **Seeding, via the server:** the `seed` option exists in the server's battle code, but a normal client challenge does not pass one, and the only chat route is the admin command `/editbattle reseed` mid-battle. 40 identical MaxBasePower-vs-MaxBasePower battles on fixed teams gave **12 distinct (winner, turns) outcomes**, so battles are not reproducible from the client. Consequence: no disjoint battle-seed pools; disjointness would rest on team pools only, and the unit of resampling stays the independent training seed, as in the other phases. Exact replays of single battles are not available without patching the server or driving the simulator directly.

## Observation and action sketch (no code kept)

- `DoublesEnv` in `poke-env` exposes `MultiDiscrete([107, 107])` per turn for gen 9 (1 pass + 6 switches + 4 moves x 5 targets x 5 gimmick variants, where the gimmick slots include Mega). A factored head of two 107-way categoricals, masked by `battle.available_moves/switches/force_switch`, is workable.
- **Team preview is not in the env action space.** It goes through `Player.teampreview()` (a `/team` string; the pick of 4 from 6 and the lead pair are set by ordering). A policy needs its own head or a scripted rule.
- Features available: per Pokemon species, HP fraction, status, boosts, item, ability, known moves (base power, type, category), Tera and Mega flags; field weather, terrain and side conditions.
- **Main risk:** one fixed, identical encoding for both agents must cover species and moves the agent has not seen, and the 107 x 107 joint action space is large. Open Team Sheets show the opponent's six, which helps team-preview decisions.

## Risks for a full phase

1. **Cost unknown.** The 10-15 h figure is for scripted players only. A PPO throughput measurement is the first thing to do, and could push the phase well past it.
2. **PPO-in-the-middle check not yet done.** The protocol needs plain PPO to land between 20% and 80% against some opponent. The ladder exists, but nothing was trained.
3. **Few teams.** Three tune and three eval teams, and bring-4-of-6 team preview, make outcomes noisy and tuned choices team-specific.
4. **`poke-env` dependency.** It needs a local data patch and the hang-on-error behaviour; both are manageable but fragile.
5. **No battle seeding** from the client (above).
6. **Scripted opponents are short-lived.** Battles last 4 to 6 turns because no scripted player Protects or switches strategically; a learned agent will see longer games and different dynamics.

## Recommendation

**Go with conditions** (the decision is yours; no-go is a defensible alternative, as with Crafter):
- Every gate passes, and the stack runs end to end on your real teams with Megas working.
- Before any amendment to spec 5.6: (a) measure plain PPO throughput on this env for a few thousand decisions, because that fixes the budget S; (b) decide whether 3 + 3 teams is enough or the pools should grow; (c) accept that battles cannot be seeded, so disjointness rests on team pools.
- If you say go: draft a spec 5.6 amendment for VGC doubles (team pools, no battle seeds, a stated budget S, the encoding and action head, the team-preview rule) for your approval before any build. Showdown remains not approved for building until you say so.
- If you say no-go: close this as a documented spike. The scratch directory `~/xp-showdown-spike/` stays in place until you say to delete it.
