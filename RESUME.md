# XP — Resume

Nothing running, nothing in progress. All work is committed and pushed.

- **Crafter** closed as a documented spike (user decision 2026-10-26): plain PPO reached 4.52 (area 32) and 4.37 (area 64) achievements per episode, below the G1 gate of 5. May be reconsidered; reopening needs a spec amendment raising S (2-3M, about 40-60 h) before any RND result is read. See `docs/spikes/crafter_spike.md`.
- **Showdown** is specified (spec 5.6) but NOT approved for building.
- **MiniGrid DoorKey** is closed (2026-10-24); **MultiRoom confirmation** is complete (2026-10-25). Do not reopen either without the user asking.
- **Showdown spike IN PROGRESS, paused** (2026-10-26). Plan + full progress log: `docs/superpowers/plans/2026-10-26-showdown-spike.md` (read its "Progress log"). Target: `[Gen 9 Champions] VGC 2026 Reg M-C`, format ID `gen9championsvgc2026regmc` (exists on the server, no custom format needed).
  - **Scratch env (outside the repo, not in git):** `~/xp-showdown-spike/` has a Python 3.9 venv with `poke-env` 0.10.0 (data PATCHED for 54 Champions species + 2 moves; originals kept as `*.orig`; re-apply with `scripts/spikes/showdown_patch_poke_env.py` after any reinstall), a local Node 22.23.4, and the Showdown checkout `pokemon-showdown/` (patched `server/sockets.ts` so it listens on 127.0.0.1 only).
  - **Start the server:** `export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH; cd ~/xp-showdown-spike/pokemon-showdown && node pokemon-showdown start --no-security` (about 25 s to come up; stop with `pkill -f "pokemon-showdown start"`). 
  - **Done:** install; format check; 3-tier doubles ladder (random < MaxBasePower < `DamageHeuristicPlayer`, scripts `scripts/spikes/showdown_opponents.py`, `showdown_ladder.py`) graded in both the standard and the Champions random-doubles formats (22-34 battles/s); Mega parsing verified (48 of 60 battles had a Mega); Task 5 action/observation sketch (in the plan log).
  - **Teams in (2026-10-26):** `~/xp-showdown-spike/teams/{tune1-3,eval1-3}.txt`, all validated; S1 step 6 passed, and `DamageHeuristicPlayer` now Megas (see the plan's last entry). **Not done:** S2 and S3 rerun on real teams (earlier ladder numbers had no Mega), S4 (disjoint team lists + seeding check), Task 6 write-up `docs/spikes/showdown_spike.md` and go/no-go. The server may be running; stop with `pkill -f "pokemon-showdown start"`.
  - **Gotchas:** rejected teams make `poke-env` hang silently (reason only at debug log level, `|popup|`); Showdown strips underscores and truncates usernames to 18 chars, so use short alphanumeric account names; random-battle sets are NOT valid Reg M-C teams (Item Clause); an old hung run can leave stale battles, use fresh account names.
- Pushes need the user's yes each time.
- Stop hook: `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).

Optional longer follow-ups (hours of compute, none approved): other MultiRoom room counts, a bigger budget, more DoorKey seeds. See `docs/FUTURE_WORK.md`.
