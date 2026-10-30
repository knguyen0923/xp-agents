# XP — Resume

## Nothing in progress (2026-10-30 evening)

No run, build or review is in flight. The matchup run finished 2026-10-30 17:16 and is written up, committed and pushed. No process of ours is running, and the local Showdown server is stopped. Open work is indexed in `docs/REMAINING_WORK.md`; the project snapshot is `STATUS.md`. Older dated notes were removed in the 2026-10-30 polish pass; they live in the git history of this file.

## Settled facts

- **User decisions (2026-10-26):** familiar opponent = DamageHeuristic, cap S = 250,000 decisions per run (spec 5.6 amendment; `docs/spikes/showdown_gate.md`), threshold 0.30. The user approves each step before it is done (see the memory note). Pushes need the user's yes each time.
- **Crafter** closed as a documented spike (2026-10-26): plain PPO reached 4.52 / 4.37 achievements per episode, below the G1 gate of 5. Reopening needs a spec amendment raising S first. **MiniGrid DoorKey** closed (2026-10-24); **MultiRoom confirmation** complete (2026-10-25). Do not reopen either without the user asking.
- **Stop hook:** `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).
- Run `python -m xp.showdown.patch` after any poke-env install. Smoke run on the real server: `docs/spikes/showdown_smoke.md`.

## Running Showdown (server, runs, resuming)

- **Scratch environment (outside the repo, not in git):** `~/xp-showdown-spike/` (Python 3.9 venv with patched `poke-env` 0.10.0, local Node 22.23.4, Showdown checkout, teams in `teams/`). Keep it until the user says to delete it. Re-apply `scripts/spikes/showdown_patch_poke_env.py` after any poke-env reinstall. Keep the 10 s `TIMEOUT_EMPTY_DEALLOCATE` patch in `~/xp-showdown-spike/pokemon-showdown/server/rooms.ts` (`docs/spikes/showdown_memory.md`).
- **Server:** the launcher `scripts/run_showdown.sh` starts one if it is down. By hand: `export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH; cd ~/xp-showdown-spike/pokemon-showdown && node pokemon-showdown start --no-security` (about 25 s). Stop it by PID (the parent `node pokemon-showdown start` takes its children with it); `pkill -f` patterns can match the calling shell.
- **Running:** `ALLOW_CODE_CHANGE=1 caffeinate -i scripts/run_showdown.sh <config> <out> <workers>`. The flag is needed when docs commits moved git HEAD after the run started; resumes are keyed by config hash, which covers the grids, so a changed grid needs a fresh output folder (never hand-edit the hash). **Use 1 worker.** With 2 workers swap climbed from 0.4 GB to 6 GB within 15 minutes and throughput dropped; 1 worker settled around 3 to 4 GB. Roughly 7 jobs per hour on 1 worker without the matchup block, 6 with it.
- **Before starting:** `sysctl vm.swapusage` and `pgrep -fl "xp.run|multiprocessing.spawn|room-battle"` (only the server's `room-battle` should show). A killed job restarts from zero (about 10 minutes of work).
- **Grid edge:** if a tuned optimum lands on a grid edge the run stops with `ProtocolError`. Ask the user, widen both grids equally, and rerun in a fresh folder.
- **When a run finishes:** `python -m xp.report --phase showdown --config <config> --out <out> --figs figs/<name>`; for a cross-run comparison use `scripts/showdown_matchup_compare.py --orig <run A> --new <run B>`; then write-ups (README, `docs/FINDINGS.md`, `docs/WRITEUP.md`, tables under `docs/spikes/`, rebuild `docs/findings.html` with `scripts/build_findings_html.py`).
- **Control run recipe (not approved):** copy `configs/showdown.yaml`, widen the grids to the matchup run's (Kairi `ent_coef` [0.001, 0.002, 0.003, 0.006, 0.01, 0.02, 0.05, 0.1]; Kaiser `rnd_coef` [0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0]), run in a fresh folder on 1 worker (about 9 to 10 hours), then compare. Details: `docs/REMAINING_WORK.md` section 2.
- **Gotchas:** rejected teams make `poke-env` hang silently (validate with `node pokemon-showdown validate-team gen9championsvgc2026regmc < team.txt`); use short alphanumeric account names. Occasional server message `[Invalid choice] There's nothing to choose` / `Can't do anything: The game is over` (cause unknown, never stopped a run).

## Deferred: battle replays

The user deferred recording battles to watch in the Showdown replay viewer. Recorder built and tested: `xp/showdown/replays.py`, `scripts/record_showdown_replays.py`, plan `docs/superpowers/plans/2026-10-27-showdown-battle-replays.md`. Policies are not saved, so recording retrains each agent (30 to 60 minutes each): `XP_NODE=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin/node python scripts/record_showdown_replays.py --agent kairi --seed 100`, then `--agent kaiser`; the server must be running; output goes to gitignored `replays/`. A smoke replay rendered in Chrome. Ask the user first.

Optional longer follow-ups (hours of compute, none approved): see `docs/FUTURE_WORK.md`.
