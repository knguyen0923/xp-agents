# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 222d78b -->

## Current state

The project is essentially finished. 212 fast tests pass; CI is green on Python 3.9 and 3.12 at the last push (319a2bf, which includes the Showdown phase).
Design spec: `docs/superpowers/specs/2026-10-22-xp-design.md` (with amendments). Plans: `docs/superpowers/plans/`.

- **Phase 1 (bandit)**: done, including extra shifts (reversed tuning, sigma, k, drift) and ablations.
  Kaiser is the best single agent nearly everywhere; Kairi wins at k=50 and drift 0.1, but retuning
  at the shifted point reverses both wins (frozen-tuning brittleness). Tag-team variants: `tt_fixed`
  beats Kaiser only at lambda>=0.01.
- **Phase 2 (MiniGrid DoorKey)**: done and closed. Curves do not cross; Kairi >= Kaiser; matched-entropy
  Kaiser ties Kairi at every size.
- **MultiRoom confirmation** (spec 5.4): done. Kaiser (rnd_coef 0.6) reached 95% success in 0.70M vs
  2.23M steps for Kairi (ent_coef 0.01), faster on 10/10 seeds. See `docs/MULTIROOM_EVALUATION.md`.
- **Crafter**: spiked and closed (G1 gate failed, 4.52 / 4.37 < 5); may be reconsidered. Needs a spec
  amendment raising the budget first. `docs/spikes/crafter_spike.md`.
- **Showdown**: spiked, gated, built and RUN (opponent DamageHeuristic, S = 250k, threshold 0.30: all user decisions). The full run finished 2026-10-28 20:19 (20 eval seed jobs, 10 per agent). Result: a tie on the familiar opponent (training AUC Kairi 0.406 vs Kaiser 0.403, CIs overlap; Kairi reaches 30% win rate about 5k steps sooner, CIs touch); tuned `ent_coef` 0.01 and `rnd_coef` 0.03, both interior. Tables and figure are in `figs/showdown` (gitignored). Written into README, FINDINGS and WRITEUP (tables in `docs/spikes/showdown_results/`); `docs/findings.html` not updated. Earlier history: the first run was stopped on a misread of the tuning score (`1 - AUC`, lower is better); the Showdown server swap growth was fixed by a patch in the scratch checkout (`docs/spikes/showdown_memory.md`).
- Write-ups: `README.md`, `docs/FINDINGS.md`, `docs/WRITEUP.md`, `docs/findings.html`
  (built by `scripts/build_findings_html.py`), `docs/FUTURE_WORK.md`.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), branch `main`, all pushed.

## Next

**Update 2026-10-29:** the replay recording is DEFERRED by the user. A smoke replay was checked in Chrome and renders and plays (the viewer handles the Champions format). The user then asked whether the agents understand type effectiveness: they do not get it as an input (`xp/showdown/encode.py` has move type, power, category and target types only, no multiplier or damage estimate; nothing in the repo tests whether they learned the chart). Brainstormed a switchable `env.matchup` observation block (ability-aware type multiplier plus estimated damage fraction of target HP, 32 extra floats, off by default so the existing run is untouched). Spec written and committed (`docs/superpowers/specs/2026-10-29-showdown-matchup-features-design.md`, commit ca7cee5), awaiting the user's review. No plan, no code, no run approved. Open items to verify first: poke-env fills opponent abilities from Open Team Sheets at battle start; calculator speed. The experiment would be about 12 to 15 hours and needs the user's approval before it starts.

**Later 2026-10-29:** the matchup spec now also has an incoming-threat block (foe revealed moves against my actives, with `revealed` flags, 72 extra floats in total) under the user's assumption that the agent knows every move and ability and only the opponent's unrevealed moves are hidden. Recorder names changed: replays now show e.g. `Kaiser3x472` vs `Heuristic3x472` (`account_names` in `xp/showdown/vec_env.py`, `rank_replays(agent_prefix=...)`; fast tests pass, not yet tried against the real server).

**Plan written (2026-10-29):** `docs/superpowers/plans/2026-10-29-showdown-matchup-features.md`, 6 tasks (probe poke-env on a real battle, configurable observation size, matchup block, real estimator, wiring, configs and smoke run). Awaiting the user's choice of execution mode; nothing built yet.

**Executing the plan (2026-10-29, branch `matchup-features`, subagent-driven):** Task 2 (configurable observation size) done and reviewed (a33b06c); Task 3 (matchup block, `xp/showdown/matchup.py`, 4d29670) done and reviewed; Task 1 probe run on the real server (99d8e6a, `docs/spikes/showdown_matchup_probe.md`): poke-env does NOT expose the Open Team Sheet (foe stats None, foe abilities and items mostly unknown, foe moves empty at start), so spec and plan were amended (8907ae1) and foe ability awareness is limited; Task 4 (real estimator `xp/showdown/estimate.py`, 144199f; server test passed) is committed; its review found one Important issue (partial stats overwritten), fixed in 0f475a6, scoped re-review running. Left: T5 (wire the flag), T6 (configs and smoke run, needs the user's yes), final review. Order: T2, T3, then T1 (needs the user's yes for the Showdown server run), then T4, T5, T6. Ledger: `.superpowers/sdd/2026-10-29-showdown-matchup-features/progress.md` (git-ignored). Nothing merged or pushed; main is unchanged.

**Future work noted (2026-10-29):** `docs/FUTURE_WORK.md` has a new section on Showdown tempo and format-specific features (speed order, priority brackets, Trick Room and Tailwind durations, IV/EV, foe sheet fill, an opponent that uses field moves). Nothing built for it.

(Older notes follow.) Nothing running except a local Showdown server started for the replay smoke test (stop with `pkill -f "pokemon-showdown start"`). **Battle replays (user approved building, 2026-10-28):** recorder built and committed (`xp/showdown/replays.py` scorer, `scripts/record_showdown_replays.py`, `tests/test_showdown_replays.py`; `ShowdownVecEnv` has an optional `save_replays`). The user wants the most interesting battles: record 40 per agent, keep the top 5 by the agreed score (closeness, comebacks, length, Mega/Tera use). Smoke test on the real server passed (tiny budget); poke-env pages lack a `|win|` line, so the scorer infers the winner from faints. Still open: confirm a replay page renders in a browser (Chrome extension was not connected; the viewer loads from play.pokemonshowdown.com and may not know the Champions format), then the real recording (about 1 to 2 h: train kairi and kaiser on an eval seed, 40 battles each). `docs/findings.html` still has no Showdown section. Possible longer follow-ups, none approved: other MultiRoom room counts, a bigger budget, more DoorKey seeds, reopening Crafter. Showdown commits are not pushed; pushes need the user's yes.

## Last updated

2026-10-29 (commit 222d78b)
