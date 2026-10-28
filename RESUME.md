# XP — Resume

## SWAP CAUSE FOUND AND FIXED, 2026-10-28 12:30 (read this first)

The Showdown run was relaunched at 10:50 and paused again at about 11:20 because swap grew from 0 to 4 GB in 30 minutes. Nothing is running. Cause: the Showdown sim process (`room-battle.js`) keeps every finished battle (a full `Battle`, `RoomBattleStream` and `BattleActions`, about 66 KB each) until the room's empty-room timer `TIMEOUT_EMPTY_DEALLOCATE` fires, which is 10 minutes. It reached 3.7 GB RSS in the real run. Heap snapshots: +5,113 of each of those objects between two snapshots 50 seconds apart. Details: `docs/spikes/showdown_memory.md`.

- **Fix (in the scratch Showdown checkout, NOT in this repo):** `~/xp-showdown-spike/pokemon-showdown/server/rooms.ts` line 20, `TIMEOUT_EMPTY_DEALLOCATE = 10 * 1000` (was `10 * 60 * 1000`; original saved as `rooms.ts.orig`). Patch the `.ts`: `pokemon-showdown start` runs `node build`, which overwrites any edit to `dist/`. A fresh `~/xp-showdown-spike` needs this patch re-applied.
- **Evidence:** random-play loop, 8 envs, server heap capped at 1 GB. Unpatched: the sim process hit the cap and crashed at about 3.5 minutes (then `poke-env` hangs forever). Patched: flat at about 250 to 320 MB over 26,000 battles in 6 minutes, no crash. Not yet tested with a real PPO job.
- **Safe to shorten:** the timer only runs once both accounts have left the room, and `poke-env` has recorded the result before it sends `/leave`. A battle-replay feature (deferred) should save the replay before leaving the room.
- **To resume:** as in the section below, but first confirm the patch is in place (`grep TIMEOUT_EMPTY_DEALLOCATE ~/xp-showdown-spike/pokemon-showdown/dist/server/rooms.js` after the server builds; expect `10 * 1e3`). Restart the Mac first if swap is still high (`sysctl vm.swapusage`), and watch swap for the first 30 minutes. If it still climbs, stop and tell the user.
- **Not done:** the two other checks proposed (whether rooms empty, whether `destroy()` frees the battle) were skipped because the patched run already answers them. Reducing `n_envs` was not needed; it would also change the config hash and drop the cached tuning jobs.
- **Pitfall:** `pkill -f` patterns can match the calling shell; the first stop attempt died halfway and left workers behind. Check with `pgrep -fl "xp.run|multiprocessing.spawn|pokemon-showdown|room-battle"` after any stop.

## PAUSED 2026-10-28 ~10:05 (earlier pause, same cause)

The Showdown full run was relaunched at 09:24 into `results/showdown` (seeded from a copy of `results/showdown_grid1_aborted`; grid 3 is in `results/showdown_grid3_aborted`), HEAD `4f3ab0f`, with `ALLOW_CODE_CHANGE=1`. The user paused it on purpose at about 10:05 because macOS swap was growing fast (0 to about 7 GB in 20 minutes; free disk fell 78 to 70 GiB). Nothing is running. Saved: all 18 Kairi tuning jobs (Kairi tuned `ent_coef` 0.01, score 0.610, `boundary=none`) and 3 Kaiser tuning jobs; the job in progress was lost. No eval jobs ran. The cause of the swap growth is not found.

- **To resume:** restart the Mac first (clears swap), keep AC power and no sleep on lid close, then `cd ~/VSC/xp-agents && source .venv/bin/activate && ALLOW_CODE_CHANGE=1 caffeinate -i scripts/run_showdown.sh configs/showdown.yaml results/showdown 2` (the launcher starts the server). Do not commit during the run. Watch swap with `sysctl vm.swapusage` and free disk with `df -h .`; pause if free disk drops under about 10 GiB. The user asked to stop on a swap rise before, so if swap climbs fast again, tell the user rather than carrying on.
- **After a stop, kill orphans:** a stop leaves `multiprocessing.spawn` workers (parent PID 1) holding memory; kill them (`pkill -f "multiprocessing.spawn import spawn_main"`).
- **The 30-minute progress loop** lived in the earlier Claude session and is gone; recreate it with `/loop 30m ...` if wanted.
- **Afterwards:** `python -m xp.report --phase showdown --config configs/showdown.yaml --out results/showdown --figs figs/showdown`.

The notes below describe how we got here.

- **Root cause (found 2026-10-28): we misread the tuning score.** `tune_job` stores `score = 1 - training AUC of the win rate` and the tuner minimizes it: LOWER IS BETTER. The `score` values in `run.log` and `kairi.csv` are not win rates. Higher `ent_coef` is worse for Kairi (grid 3 scores 0.794 at 0.1 rising to 0.947 at 3.2), so the tuner was right, and the two grid widenings went the wrong way. The tuner has no bug.
- **Grid 1 was never on an edge.** `[0.003 .. 0.1]` scored about 0.61 to 0.64 for 0.003..0.02, 0.69 at 0.05, 0.80 at 0.1: the best is near 0.01 (interior). Grid 1 was aborted on the misread; it had 17 of 18 tuning jobs (missing: `ent_coef` 0.01 seed 1). Grid 2 `[0.02 .. 0.8]` picked its low edge 0.02 (0.628), which said "go lower".
- **Fix:** Kairi's grid is back to `[0.003, 0.006, 0.01, 0.02, 0.05, 0.1]` in `configs/showdown.yaml`; its config hash is `fdac28308815`, the same as grid 1's, so the cached grid 1 jobs in `results/showdown_grid1_aborted` match. They were made at commit `e6087e2`; only configs and docs changed since, so reusing them needs `ALLOW_CODE_CHANGE=1` (the manifest records both sessions).
- **Next (needs the user's go, ask right before launching):** fresh out dir (copy `results/showdown_grid1_aborted` into it to reuse the cache, or start empty), then `source .venv/bin/activate && caffeinate -i scripts/run_showdown.sh configs/showdown.yaml results/showdown 2` (with `ALLOW_CODE_CHANGE=1` if reusing). About 8 to 12 hours with 2 workers (less with the cache). Do not commit during the run. Afterwards `python -m xp.report --phase showdown --config configs/showdown.yaml --out results/showdown --figs figs/showdown`.
- **If a run is interrupted (sleep, crash):** rerun the same command with the same commit and config; it continues from saved jobs. Check `results/showdown/run.log` (`grep -v -i warn`, ignore `[Invalid choice]`).
- **Machine:** keep AC power, no sleep on lid close. Pause if free disk drops under about 10 GiB.

## Settled facts

- **User decisions (2026-10-26):** familiar opponent = DamageHeuristic, cap S = 250,000 decisions per run (spec 5.6 amendment; `docs/spikes/showdown_gate.md`), threshold 0.30. Plan C: `docs/superpowers/plans/2026-10-26-showdown-phase-wiring.md`. The user approves each step before it is done (see the memory note) and was asked right before this launch.
- **Showdown work is committed but NOT pushed** (the user pushes on request); everything else is pushed. Pushes need the user's yes each time.
- **Crafter** closed as a documented spike (2026-10-26): plain PPO reached 4.52 / 4.37 achievements per episode, below the G1 gate of 5. Reopening needs a spec amendment raising S first. See `docs/spikes/crafter_spike.md`.
- **MiniGrid DoorKey** closed (2026-10-24); **MultiRoom confirmation** complete (2026-10-25). Do not reopen either without the user asking.
- Smoke run on the real server passed: `docs/spikes/showdown_smoke.md`. Run `python -m xp.showdown.patch` after any poke-env install.
- Open: the occasional server message `[Invalid choice] There's nothing to choose` / `Can't do anything: The game is over` (cause unknown, never stopped a run).
- **Scratch env (outside the repo, not in git):** `~/xp-showdown-spike/` (Python 3.9 venv with patched `poke-env` 0.10.0, local Node 22.23.4, Showdown checkout, teams in `teams/`). Keep it until the user says to delete it. Re-apply `scripts/spikes/showdown_patch_poke_env.py` after any poke-env reinstall.
- **Start the server by hand (the launcher does it for you):** `export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH; cd ~/xp-showdown-spike/pokemon-showdown && node pokemon-showdown start --no-security` (about 25 s; stop with `pkill -f "pokemon-showdown start"`).
- **Gotchas:** rejected teams make `poke-env` hang silently (validate with `node pokemon-showdown validate-team gen9championsvgc2026regmc < team.txt`); use short alphanumeric account names.
- Stop hook: `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).

Deferred by the user (2026-10-27, "later", after the full run): recording battles to watch in the Showdown replay viewer. Plan (untracked): `docs/superpowers/plans/2026-10-27-showdown-battle-replays.md`. Trained policies are not saved today, so it needs retraining.

Optional longer follow-ups (hours of compute, none approved): other MultiRoom room counts, a bigger budget, more DoorKey seeds. See `docs/FUTURE_WORK.md`.
