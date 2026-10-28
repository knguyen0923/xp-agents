# Showdown sim-process memory growth (2026-10-28)

## Problem

During the Showdown full run, macOS swap grew from 0 to about 4 GB in 30 minutes (twice). The Showdown server's battle process (`room-battle.js`, one process that runs all battles) reached 3.7 GB RSS; the two Python workers held about 2 GB between them.

## Cause

Showdown destroys a battle (and its `RoomBattleStream`) only when its room is destroyed. `poke-env` sends `/leave` when a battle ends, which empties the room but does not destroy it. An empty room is destroyed by a timer, `TIMEOUT_EMPTY_DEALLOCATE`, set to 10 minutes (`server/rooms.ts:20`). Until then every finished battle stays in the sim process.

Heap snapshots of the sim process, 50 seconds apart (random play, 8 envs): heap 231 MB to 570 MB; `Battle`, `RoomBattleStream` and `BattleActions` each +5,113 objects, plus their `Pokemon`, plain objects and battle-log strings; about 66 KB of heap per finished battle.

## Fix and test

Set `TIMEOUT_EMPTY_DEALLOCATE = 10 * 1000` in `server/rooms.ts` of the scratch checkout (`~/xp-showdown-spike/pokemon-showdown`). Patch the `.ts` file: `pokemon-showdown start` rebuilds `dist/` from source, so an edit to `dist/server/rooms.js` is silently overwritten (the first test of the fix was invalid for this reason).

Test: 8 envs, random legal moves against DamageHeuristic, server started with `NODE_OPTIONS=--max-old-space-size=1024`.

| time | battles | sim process, 10 min timer | sim process, 10 s timer |
|---|---|---|---|
| 60 s | 4,400 | 627 MB | 314 MB |
| 120 s | 8,900 | 955 MB | 301 MB |
| 180 s | 13,000 | 1,048 MB | 266 MB |
| about 210 s | 15,000 | out of memory, crashed | 239 MB |
| 360 s | 26,000 | not reached | 260 MB |

A crashed sim process is restarted by the server but forgets its battles, so `poke-env` waits forever and the run hangs.

## Caveats

- Not tested with a real PPO job; the real run plays slower than random play, so it should be easier on the server.
- RSS readings under swap pressure are unreliable (macOS compresses and swaps pages, so RSS drops without a free).
- The patch lives outside the repo; a fresh `~/xp-showdown-spike` needs it re-applied.
- A battle-replay feature should save the replay before leaving the room.
