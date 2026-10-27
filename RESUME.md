# XP — Resume

Nothing running, nothing in progress. Showdown spike work is committed but NOT pushed (the user pushes on request); everything else is pushed.

- **Crafter** closed as a documented spike (user decision 2026-10-26): plain PPO reached 4.52 (area 32) and 4.37 (area 64) achievements per episode, below the G1 gate of 5. May be reconsidered; reopening needs a spec amendment raising S (2-3M, about 40-60 h) before any RND result is read. See `docs/spikes/crafter_spike.md`.
- **Showdown** is specified (spec 5.6) but NOT approved for building.
- **MiniGrid DoorKey** is closed (2026-10-24); **MultiRoom confirmation** is complete (2026-10-25). Do not reopen either without the user asking.
- **Showdown: foundations DONE (2026-10-26), build approved.** Spec 5.6 amended; `xp/showdown/{teams,encode,env}.py` + tests (183 default tests pass; 1 server test: `XP_NODE=<node22> pytest -m showdown`, needs the server running). Team files are in `configs/showdown_teams/` (user OK'd the public repo). **Next: write Plan B** (PPO learner on `XPDoublesEnv`, throughput measurement, PPO-in-the-middle check vs the ladder, budget cap S, `configs/showdown.yaml`, `xp/showdown/phase.py`, `xp.run` wiring, report). The user must approve the cap S before any RND result is read. Untested: CI on Python 3.12 with `poke-env` (check the CI run after the next push).
  - **Scratch env (outside the repo, not in git):** `~/xp-showdown-spike/` (Python 3.9 venv with patched `poke-env` 0.10.0, local Node 22.23.4, Showdown checkout, teams in `teams/`). Keep it until the user says to delete it. Re-apply `scripts/spikes/showdown_patch_poke_env.py` after any poke-env reinstall.
  - **Start the server:** `export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH; cd ~/xp-showdown-spike/pokemon-showdown && node pokemon-showdown start --no-security` (about 25 s; stop with `pkill -f "pokemon-showdown start"`). Server is STOPPED.
  - **Gotchas:** rejected teams make `poke-env` hang silently (validate with `node pokemon-showdown validate-team gen9championsvgc2026regmc < team.txt`); use short alphanumeric account names.
- Pushes need the user's yes each time.
- Stop hook: `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).

Optional longer follow-ups (hours of compute, none approved): other MultiRoom room counts, a bigger budget, more DoorKey seeds. See `docs/FUTURE_WORK.md`.
