# Crafter spike (spec 5.5)

Status: DONE 2026-10-26. Plain PPO only; no RND result was read. Script: `scripts/spikes/crafter_spike.py`; plan: `docs/superpowers/plans/2026-10-25-crafter-spike.md`.

## Throughput (40,960-step runs, 8 envs, PPO with a small CNN, includes start-up)
| area | CPU steps/s | MPS steps/s |
|---|---|---|
| 32 | 589 | 781 |
| 64 | 279 | 310 |

## Full-phase cost (spec 5.5: 36 tuning runs + 20 eval runs at 1M steps = 56M steps)
Hours = 56,000,000 / steps-per-second / 3600, using the faster device and one run at a time.
| area | steps/s used | hours (one run at a time) |
|---|---|---|
| 32 | 781 (MPS) | 19.9 |
| 64 | 310 (MPS) | 50.2 |
Concurrent runs would be faster only if cores are free; the spike did not measure that.

## Results (1 seed per area, seed 0, 1,000,000 steps, MPS; one seed, so indicative only)
| area | final `ach_100` | early `ach_100` (first 10%) | achievements unlocked in <5% of last 100 episodes | G1 (final >= 5 and >= 5 rare) | G2 (final > early) |
|---|---|---|---|---|---|
| 32 | 4.52 | 3.02 | 16 | **fail** (4.52 < 5) | pass |
| 64 | 4.37 | 3.12 | 16 | **fail** (4.37 < 5) | pass |

Both curves are still rising at 1M steps (area 32: 3.0 at 10k, 3.9 at 500k, 4.5 at 1M; area 64: 3.1, 3.7, 4.4), so a longer budget might pass G1, but that is untested.

Per-achievement unlock rates over the last 100 episodes (full table: `crafter_spike_rates.csv`):
- Area 32: top `collect_sapling` 0.92, `collect_wood` 0.92, `wake_up` 0.92, `defeat_zombie` 0.74, `eat_cow` 0.52; bottom (all 0.00): `make_iron_pickaxe`, `make_iron_sword`, `place_furnace`, `place_stone`, `place_table`.
- Area 64: top `collect_sapling` 0.98, `wake_up` 0.92, `collect_wood` 0.88, `collect_drink` 0.76, `defeat_zombie` 0.55; bottom (all 0.00): same tool and furnace achievements. No table was placed in the last 100 episodes in either area, so the tech tree beyond wood is essentially unreached.

World seeds (Task 1 tests): the world for an episode is `hash((env_seed, episode)) % (2**31 - 1)`, deterministic per env seed and episode count; tune and eval world-seed pools are disjoint, checked in `tests/test_crafter_spike.py`.

## Cost
Measured: 1M steps at area 32 took 1,245 s on MPS (804 steps/s). Full phase (56M steps) is about 19-20 h at area 32 one run at a time. The area 64 run took 4,905 s but includes a manual pause of unknown length, so its wall time is not a clean measurement; the 40,960-step benchmark (310 steps/s MPS) implies about 50 h for the full phase.

## Decision by the spec's rule
No area passes G1, so no area is chosen and there is no deep achievement set (5-95% unlock rates), because only a handful of achievements sit in that band (6 at area 32, 5 at area 64) and the rest are at 0.

## Recommendation
**Stop (no-go) on the full Crafter phase as specified.** Plain PPO does not reach the pre-registered usable range at 1M steps, and the spec fixed S = 1M. If the user wants to continue, the smallest change is a spec amendment that raises S (for example 2-3M at area 32, about 40-60 h for the full phase) before any RND result is read; this needs the user's decision, and the spike gives no evidence the gate would then pass. One seed per area; no RND result was read.
