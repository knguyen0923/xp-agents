# XP — Resume

## NOT RUNNING 2026-10-28 (read this first)

The Showdown full run is NOT running. The 2026-10-27 18:32 relaunch (Kairi grid `[0.1 .. 3.2]`) stopped after Kairi tuning with a grid-edge `ProtocolError`; its output is in `results/showdown` (kept, not in git).

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
