# XP — Resume

Nothing running, nothing in progress. All work is committed and pushed.

- **Crafter** closed as a documented spike (user decision 2026-10-26): plain PPO reached 4.52 (area 32) and 4.37 (area 64) achievements per episode, below the G1 gate of 5. May be reconsidered; reopening needs a spec amendment raising S (2-3M, about 40-60 h) before any RND result is read. See `docs/spikes/crafter_spike.md`.
- **Showdown** is specified (spec 5.6) but NOT approved for building.
- **MiniGrid DoorKey** is closed (2026-10-24); **MultiRoom confirmation** is complete (2026-10-25). Do not reopen either without the user asking.
- **Showdown spike IN PROGRESS, paused** (2026-10-26): plan and progress log in `docs/superpowers/plans/2026-10-26-showdown-spike.md`. Exact format `gen9championsvgc2026regmc` exists on the server; scratch env in `~/xp-showdown-spike/` (venv, local Node 22, patched server; start with `node pokemon-showdown start --no-security` using that Node). Done: install, ladder (3 tiers, graded) and sketch on a proxy format. Blocked: needs the user's Reg M-C pokepaste teams (3 tune + 3 eval). `poke-env` data patched for the 54 Champions species and 2 moves (verified, scripts in scripts/spikes/showdown_*; re-apply after a reinstall). Then S1 step 6, S2-S4 on real teams, Task 6 write-up. Server is stopped.
- Pushes need the user's yes each time.
- Stop hook: `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).

Optional longer follow-ups (hours of compute, none approved): other MultiRoom room counts, a bigger budget, more DoorKey seeds. See `docs/FUTURE_WORK.md`.
