# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 3c11403 -->

## Current state

The project is essentially finished. The fast suite passes; CI is green on Python 3.9 and 3.12 at the last push (`69050c7`, 2026-10-29, which includes the Showdown phase and the matchup features).
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

**Matchup plan built, MERGED into `main` and PUSHED (2026-10-29, CI green); smoke run passed:** matchup smoke (`configs/showdown-matchup-smoke.yaml`) ran clean in 81 s against 72 s for the original smoke (about 12% slower, one run each, includes noise), with `matchup estimator errors=0` on 96 of 96 worker lines (about 7,000 to 8,000 estimator calls each). The estimator also runs for the opponent's view (harmless; skipping it would be a cheap follow-up if the full run needs the time). Detail of what was built: Tasks 1 to 5 and the Task 6 configs are done and task-reviewed (Task 4 needed one fix round): probe (99d8e6a, `docs/spikes/showdown_matchup_probe.md`: poke-env does not expose the Open Team Sheet, foe stats None, foe abilities and items mostly unknown, so the spec was amended), configurable observation size (a33b06c), matchup block `xp/showdown/matchup.py` (4d29670), real estimator `xp/showdown/estimate.py` (144199f, 0f475a6), flag wiring (39c0c95), configs `configs/showdown-matchup.yaml` and `configs/showdown-matchup-smoke.yaml` (a715f8f). The final whole-branch review found 2 Important issues (foe fallback stats stale after a Mega Evolution; estimator error counter invisible) plus test and doc gaps; all 8 items were fixed in 7b4765d and 26828fc and the scoped re-review passed (the fast suite passes). **The full matchup run is PAUSED (stopped on purpose at 13:51 on 2026-10-29, 12 of 36 tuning jobs cached, all Kairi; no eval jobs yet; estimator errors 0).** Resume steps, swap notes and what to do when it finishes are in `RESUME.md`. Left: resume and finish the run (about 12 to 15 hours in total), the report, the comparison with the original run, write-ups, and a Showdown section in `docs/findings.html`.  Ledger: `.superpowers/sdd/2026-10-29-showdown-matchup-features/progress.md` (git-ignored). The final-review fixes (foe Mega stats, visible estimator error counter, layout and cap tests, doc corrections) are merged; the remaining deferred minors are cosmetic.

**Update, 2026-10-29 afternoon (matchup run 2):** the resumed matchup run finished the 18 Kairi tuning jobs and stopped at the protocol guard: Kairi's tuned `ent_coef` 0.003 sat on the grid's lower edge. With the user's approval both grids were widened to 8 values each (Kairi `ent_coef` 0.001 to 0.1, Kaiser `rnd_coef` 0.01 to 3.0; config commit `5e14970`) and the run restarted in a fresh folder, `results/showdown-matchup2` (the old `results/showdown-matchup` is kept as a record, not mixed). About 14 to 17 hours; details and the resume command are at the top of `RESUME.md`. This supersedes "PAUSED" above. **Evening update:** run 2 was found dead at 18:59 (cause unknown), restarted briefly, then PAUSED on purpose at the user's request with 19 of 48 tuning jobs cached (all Kairi); resume steps are at the top of `RESUME.md`.

**Future work noted (2026-10-29):** `docs/FUTURE_WORK.md` has a new section on Showdown tempo and format-specific features (speed order, priority brackets, Trick Room and Tailwind durations, IV/EV, foe sheet fill, an opponent that uses field moves). Nothing built for it.

(Older notes follow.) Nothing running except a local Showdown server started for the replay smoke test (stop with `pkill -f "pokemon-showdown start"`). **Battle replays (user approved building, 2026-10-28):** recorder built and committed (`xp/showdown/replays.py` scorer, `scripts/record_showdown_replays.py`, `tests/test_showdown_replays.py`; `ShowdownVecEnv` has an optional `save_replays`). The user wants the most interesting battles: record 40 per agent, keep the top 5 by the agreed score (closeness, comebacks, length, Mega/Tera use). Smoke test on the real server passed (tiny budget); poke-env pages lack a `|win|` line, so the scorer infers the winner from faints. Still open: confirm a replay page renders in a browser (Chrome extension was not connected; the viewer loads from play.pokemonshowdown.com and may not know the Champions format), then the real recording (about 1 to 2 h: train kairi and kaiser on an eval seed, 40 battles each). `docs/findings.html` still has no Showdown section. Possible longer follow-ups, none approved: other MultiRoom room counts, a bigger budget, more DoorKey seeds, reopening Crafter. Showdown commits are not pushed; pushes need the user's yes.

## Last updated

2026-10-29 evening (matchup run 2 PAUSED at 19 of 48 tuning jobs, by the user's request; see RESUME.md)
