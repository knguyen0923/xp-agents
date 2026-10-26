# Crafter spike (spec 5.5)

Status: IN PROGRESS. Plain PPO only; no RND result was read. Script: `scripts/spikes/crafter_spike.py`; plan: `docs/superpowers/plans/2026-10-25-crafter-spike.md`.

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
