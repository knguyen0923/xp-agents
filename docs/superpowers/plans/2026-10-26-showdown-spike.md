# Showdown Spike Plan

> **STATUS: IN PROGRESS, paused 2026-10-26 waiting on the user's teams.** Task 1 steps 1-4 done, step 6 blocked, Tasks 2-5 not started (see Progress log). Pushes need the user's yes.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Find out, cheaply, whether Pokémon Showdown VGC doubles (Champions-style rules, user-supplied set teams) can serve as the Phase 4 generality test: does the stack run here, is there a graded ladder of scripted doubles opponents, is throughput acceptable, and can tune and eval battles be kept disjoint. The output is a go/no-go write-up, not kept code.

**Architecture:** Everything runs in a throwaway directory **outside the repo** (`~/xp-showdown-spike/`: its own venv, a Showdown server checkout, scratch scripts and team files). The repo gets only the write-up `docs/spikes/showdown_spike.md` and, if useful, the labelled throwaway scripts in `scripts/spikes/showdown_*.py`. No change to `xp/`, `requirements*.txt` or `configs/`. No PPO or RND training in this spike.

**Tech stack:** `poke-env` 0.10.0 (downloads for Python 3.9.6; its own `Requires-Python` and dependencies are still to be confirmed on install), Pokémon Showdown server (Node.js; this machine has Node 16.20.2, probably too old), numpy.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md` section 5.6. It assumes a random-battle format; the user has since chosen **VGC 2026 Regulation M-C (Champions rules, doubles) with user-supplied set teams**. Spec 5.6 gets amended only if the spike passes (and only with the user's approval).

## Global constraints

- Python floor 3.9 for anything that ends up in the repo; every such module starts with `from __future__ import annotations`.
- Nothing is installed globally. A newer Node, if needed, is a local download or `nvm` inside the scratch directory; if that is not possible without touching system config, stop and ask.
- Server runs on `localhost` only. **Never connect to the public Showdown ladder or any live server.**
- Do not change `xp/`, `configs/`, `requirements*.txt` or `CLAUDE.md` during the spike.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Update `STATUS.md` and `RESUME.md` when the spike starts and ends (other files first, then a STATUS-only marker commit).
- Stop and report to the user if two attempts at any single install or API step fail (do not rabbit-hole).

## Gates (proposed; the user may change them before results are read)

- **S1 (runs here):** a documented install of server and wrapper works on this machine, the target format **VGC 2026 Regulation M-C** (doubles, team preview, bring 6 and pick 4, level 50, Champions rules; set by the user 2026-10-26) exists on the local server, and one full battle between two scripted players with user-supplied teams completes. If the server has no Reg M-C format ID (the closest official entry may be a different regulation), record the closest existing ID and what a custom format entry for M-C would need (legal Pokémon list, banlist, species and item clauses); the custom entry is built only if it takes under about an hour, otherwise S1 is reported as partial.
- **S2 (opponent ladder):** at least three scripted doubles opponents of clearly different strength exist or can be written in under about 100 lines each (for example random, a max-damage heuristic, a heuristic with targeting and switching rules), with pairwise win rates that are graded (each stronger one beats the weaker one by more than 60% over at least 200 battles). Plain PPO is not trained in this spike, so S2 is judged on the ladder existing; the PPO-in-the-middle check (20% to 80% win rate against some opponent) becomes the first step of the full phase.
- **S3 (throughput):** measured battles per second and decisions (turns) per second, with team preview. Report the implied cost of a full phase: 36 tuning runs plus 20 eval runs at a candidate budget, with the budget stated as an assumption.
- **S4 (disjoint pools):** a scheme where tune, eval and test battles use disjoint pools of teams (user-supplied) is demonstrated: two lists of teams, the player draws only from its list, and logs show no team from the other list. Also record whether battles can be seeded at all (server PRNG seed), which would allow disjoint seed pools on top.
- If any gate fails, write the spike up as a result and stop (as with Crafter and Step B).

## Facts to verify (do not assume)

- Which `gen9vgc…` formats the local server defines, and whether one corresponds to Reg M-C or needs a custom format entry. The M-C legal list and dates have not been checked by us (sources found in the brainstorm disagreed on M-A and M-B dates); the user's choice of M-C stands, and the exact rules are taken from the official regulation, quoted in the write-up.
- Whether teams must be M-C legal for the server to accept them (validation) and how `poke-env` surfaces a rejected team.
- `poke-env` 0.10.0 support for doubles: the battle class, the order or action type for two simultaneous actions, how team preview selection is expressed, and which built-in players work in doubles.
- Server Node requirement versus Node 16.20.2.
- How teams are supplied (packed or export string, a team builder object, or a rotating list).

---

### Task 1: Scratch environment and S1

**Files (all outside the repo):** `~/xp-showdown-spike/venv`, `~/xp-showdown-spike/pokemon-showdown/`, `~/xp-showdown-spike/notes.md` (running log of commands and errors, copied into the write-up).

- [ ] **Step 1:** Create the directory and a Python 3.9 venv; `pip install poke-env==0.10.0`. Record the resolved dependency versions and any `Requires-Python` complaint.
- [ ] **Step 2:** Clone the Showdown server into the scratch directory (shallow clone), read its `package.json` `engines` field, and compare with Node 16.20.2. If too old, obtain a newer Node locally (tarball or `nvm`, scratch directory only), re-check.
- [ ] **Step 3:** `npm install` in the server checkout, copy the example config, start the server on `localhost:8000` with no authentication (as `poke-env` expects), and confirm it listens.
- [ ] **Step 4:** List the formats the server defines (`config/formats.ts` or the in-server format list); note every VGC doubles entry, its team-preview rule, level, and bring/pick counts. Record whether a custom format entry for Champions-style rules is needed and what it would contain.
- [ ] **Step 5:** Write two small valid Reg M-C teams in export format (6 Pokémon each, no repeated species or held items, legal under M-C) in `~/xp-showdown-spike/teams/`. Ask the user for teams first if they have given any; otherwise use these placeholders and say so in the write-up.
- [ ] **Step 6:** Run one full battle between two scripted players with those teams and record the outcome, wall time and any warnings. Verdict for S1.

### Task 2: Doubles opponent ladder (S2)

**Files:** `~/xp-showdown-spike/ladder.py` (throwaway).

- [ ] **Step 1:** Inspect `poke-env`'s built-in players and mark which handle doubles (read the installed source; do not guess). Record the result.
- [ ] **Step 2:** If fewer than three distinct-strength doubles opponents exist, write the missing ones (about 100 lines each at most): a uniformly random legal-action player, a "highest expected damage into the weakest foe" player, and one that also switches or protects by simple rules. Keep each rule set in a comment so the ladder is explainable.
- [ ] **Step 3:** Run at least 200 battles per pairing (random vs heuristic A, A vs B, random vs B), alternating teams between sides to cancel team strength. Record win rates with 95% intervals.
- [ ] **Step 4:** Verdict for S2: graded or not graded, and the lines of code the ladder needed.

### Task 3: Throughput (S3)

**Files:** `~/xp-showdown-spike/throughput.py` (throwaway).

- [ ] **Step 1:** Time at least 200 battles for one pairing with team preview; compute battles/s and turns/s, with the server and the client each on one core. Note CPU usage of the server process (it is likely the bottleneck).
- [ ] **Step 2:** Repeat with 2 and 4 concurrent server connections (separate player pairs) to see whether throughput scales.
- [ ] **Step 3:** Convert to a full-phase cost table: assumed turns per battle, assumed decisions needed per run (stated budget, for example 1M, 3M and 10M decisions), 56 runs total, hours at the measured rate with one and several parallel servers. Verdict for S3 and the cost the user must accept.

### Task 4: Disjoint team pools and seeding (S4)

**Files:** `~/xp-showdown-spike/teams/tune/`, `teams/eval/`, `pools.py` (throwaway).

- [ ] **Step 1:** Create two non-overlapping lists of at least 3 valid teams each, and a player that draws one team per battle only from its assigned list.
- [ ] **Step 2:** Run 100 battles per list and assert from the battle logs that only that list's teams appeared; assert the two lists share no team.
- [ ] **Step 3:** Check whether the server PRNG can be seeded (a format or startup option, to be read from the server source) and whether two identical seeded battles replay identically. Record the answer either way.
- [ ] **Step 4:** Verdict for S4.

### Task 5: Observation and action sketch (no code kept)

- [ ] **Step 1:** From the `poke-env` doubles battle object, list which fields a structured feature vector could use (own and visible opponent Pokémon: species, HP fraction, status, boosts, known moves, item/ability if visible; field conditions) and the shape of the action space per turn (two orders, each a move with a target or a switch, plus the team-preview pick of 4 from 6 and lead selection). State whether the per-turn action space can be handled as a MultiDiscrete or factored categorical.
- [ ] **Step 2:** Write down the main risk that follows: whether the encoding can be identical for both agents and fixed before tuning, and how large the action space is. This feeds the go/no-go but is not a gate.

### Task 6: Write-up and recommendation

**Files:** Create `docs/spikes/showdown_spike.md`; modify `STATUS.md`, `RESUME.md`; optionally commit throwaway scripts to `scripts/spikes/` (labelled throwaway at the top).

- [ ] **Step 1:** Write the spike report: install steps that worked (exact commands), versions, a gate table (S1 to S4 with measured numbers), the cost table, the observation/action sketch, and all errors hit.
- [ ] **Step 2:** Recommendation: **go** (propose a spec 5.6 amendment for VGC doubles, team pools and the budget), **no-go** (close as a documented spike, as with Crafter), or **go with a simpler variant** (for example singles, or a custom format) if S1 or S2 fail only for doubles. The decision is the user's.
- [ ] **Step 3:** Update `STATUS.md` and `RESUME.md` (Showdown spike done, awaiting the user's go/no-go; scratch directory path), commit with the attribution line, then a STATUS-only marker commit. Do not push without the user's yes. Leave `~/xp-showdown-spike/` in place until the user says to delete it.

## Cost and risk summary

- Compute: minutes to a couple of hours on the laptop, no GPU, no Colab.
- Biggest risks: server Node requirement versus this machine; no ready-made scripted doubles opponents (S2); a Champions regulation that Showdown does not implement; team preview picks (bring 6, choose 4) not being exposed cleanly by `poke-env`; throughput too low for a 56-run phase.
- A tie between Kairi and Kaiser remains the likely outcome of a full phase and is a valid result (spec 5.6).

## Progress log (2026-10-26)

Done (scratch dir `~/xp-showdown-spike/`, nothing in the repo changed):
- `poke-env` 0.10.0 installs in a Python 3.9.6 venv (gymnasium 1.1.1, pettingzoo 1.26.1, numpy 2.0.2).
- Showdown server needs Node >=22.18; local Node v22.23.4 (arm64 tarball) in the scratch dir. Server checkout at `pokemon-showdown/`, started with `node pokemon-showdown start --no-security`.
- The `bindaddress` config setting is ignored by this server version; the scratch checkout was patched (`server/sockets.ts`, `PSBINDADDR: '127.0.0.1'`) so it listens on localhost only. Re-apply after any re-clone.
- **Exact format exists:** `[Gen 9 Champions] VGC 2026 Reg M-C`, format ID `gen9championsvgc2026regmc` (Champions mod, doubles, Flat Rules, VGC Timer, Open Team Sheets). No custom format needed.
- Built-in players: `RandomPlayer` and `MaxBasePowerPlayer` support doubles; `SimpleHeuristicsPlayer` falls back to random moves in doubles. A third, stronger opponent must be written (S2).
- Gotcha: a rejected team makes `poke-env` hang with no exception; the reason appears only as a `|popup|` line at debug log level. The full phase needs a team-validation check before any run.
- Placeholder teams taken from the random-doubles generator are illegal in Reg M-C (7 Pokémon, then Item Clause). Real teams are needed; two failed attempts, so stopped.


Done later the same day, without teams (proxy format `gen9randomdoublesbattle`, because `poke-env` cannot parse the Champions-only forms, see below):
- **S1 finding (important):** `poke-env` 0.10.0's bundled Pokedex lacks 54 species that the server's Champions mod defines (mostly new Mega forms such as `hawluchamega`, `dragonitemega`, `raichumegax`; also 5 non-Mega species) and 2 moves (`nihillight`, `polarflare`). A battle that touches one of them raises `KeyError` inside the message handler, which `poke-env` logs and then **hangs** the battle. Teams using those Megas will not run unless `poke-env`'s data files are patched from the server's data (a modest data-patch task, not yet tried). Teams without them should work.
- **S2 (ladder, proxy format, 300 battles per pairing):** MaxBasePower beats Random 291/300 (0.97); the new damage heuristic (`DamageHeuristicPlayer`, ~50 lines, scripts in `scripts/spikes/showdown_*.py`, throwaway) beats MaxBasePower 225/300 (0.75) and Random 297/300 (0.99). A graded 3-tier ladder exists. Not yet measured on Reg M-C teams.
- **S3 (partial):** about 25-34 battles per second for scripted players on one server process in random doubles (no team preview, short battles). An upper bound for Reg M-C; the real number needs the exact format.
- Gotchas: Showdown strips underscores and truncates usernames to 18 characters, so over-long or underscored account names silently break challenges (battle hangs). Use short alphanumeric names.
- **Task 5 sketch:** `DoublesEnv` exposes `MultiDiscrete([107, 107])` per turn for gen9 (1 pass + 6 switches + 4 moves x 5 targets x 5 gimmick variants; the gimmick slots include Mega, which Champions uses), so the factored action head is two 107-way categoricals with masking by `battle.available_moves/switches/force_switch`. **Team preview is not in the env action space:** it goes through `Player.teampreview()` (a `/team` string; pick 4 of 6 and the lead pair are chosen by ordering), so a policy needs its own head or a scripted rule for it. Features available per Pokemon: species, HP fraction, status, boosts, item, ability, known moves (with base power, type, category), Tera/Mega flags; field: weather, terrain and side conditions in `battle`. Main risk for the full phase: a fixed identical feature encoding must cover Pokemon and moves the agent has not seen, and the 107x107 joint action space is large.

Remaining: S1 step 6 (one full Reg M-C battle on real teams), S2 on real teams, S3 on real teams, S4 (needs two disjoint team lists and a seed check), then Task 6. Blocking: at least 3 tune and 3 eval Reg M-C teams from the user (pokepaste); the data patch for the new Megas is done.

**poke-env data patch DONE (2026-10-26, user asked for the new Megas):** the 54 missing species and 2 missing moves were exported from the server's Champions mod (`Dex.mod('champions')`) and merged into the venv's `gen9pokedex.json` / `gen9moves.json` (originals kept as `*.orig`; scripts `scripts/spikes/showdown_patch_poke_env.py`, `showdown_export_patch.js`). Verified: 300 battles in `gen9championsrandomdoublesbattle` ran with no hangs; Megas appeared on our side in 48 of 60 battles, including formerly crashing forms (`hawluchamega`, `floettemega`, `scraftymega`, `absolmegaz`). Ladder in the Champions mod (100 battles per pairing): MaxBasePower beats Random 0.99, damage heuristic beats MaxBasePower 0.72 and Random 0.99; 22-29 battles/s. The patch only fixes parsing/crashes; it does not change `poke-env` features that read Champions rule changes (e.g. altered stats of existing species are read from the live battle messages, not the patched file). Re-apply the patch after any reinstall of poke-env.
