# Showdown pipeline smoke run (Plan C, Task 4)

Status: DONE 2026-10-26. This checks the plumbing only. **The numbers are not results**: each agent trained for 3 PPO updates (3,072 decisions), so win rates are near zero.

## What ran

`scripts/run_showdown.sh configs/showdown-smoke.yaml results/showdown-smoke 2` on the real local server, 2 workers (8 env processes each): 8 tuning jobs (2 agents x 2 grid values x 2 tune seeds), Kaiser's entropy matched to Kairi's tuned value (0.01), then 4 eval jobs (2 agents x 2 eval seeds, frozen policy evaluated on 20 fresh battles each). The smoke config declares its 2-value grid ends as natural limits only so the run is not provisional; the real config does not.

| Check | Result |
|---|---|
| Fresh run (12 jobs, 2 workers) | finished in 71 s |
| Resume, same command | finished in 5 s from cache, no job rerun |
| `xp.report` on the smoke results | wrote the tables and `degradation.png` |
| Launcher | started the server (port 8000 was free), stopped it again afterwards |
| Preflight | passed: data patch present, team files match the pinned digests, validator accepted all six teams, server reachable |
| Unexplained opponent-side server message (`[Invalid choice] There's nothing to choose`) | 0 in this run; it appeared 3 times in earlier concurrent runs. Cause not investigated, never stopped a run |

## What this says about the full run

- Per job: the smoke jobs are tiny, so they say nothing about the real job time. The estimate for the real run rests on the gate: about 166 decisions/s per run at 2 workers, so 249,856 decisions take about 25 minutes, 56 runs about 11.7 hours, plus RND overhead (unmeasured). One worker: about 15 hours.
- A resume refuses to continue if git `HEAD` moved since the run started, even for a docs commit (`ALLOW_CODE_CHANGE=1` overrides, for a deliberate code change).
- A 3-update agent loses almost every battle against DamageHeuristic; the real budget is 244 updates.

## Before launching the full run (checklist)

- The user approves the launch right before it starts (standing instruction), including which night(s).
- Laptop on AC power and not set to sleep on lid close (`caffeinate -i` stops idle sleep only; closing the lid sleeps a MacBook unless an external display is attached). About 12 to 15 hours of heavy CPU.
- Do not commit during the run; the working tree and `HEAD` stay as they are.
- Two workers: about 18 Python processes plus the server. Close other heavy programs.
- In the morning: `tail results/showdown/run.log` and `ls results/showdown/showdown/tuning | wc -l`.
- If the run is interrupted, rerun the same command; finished jobs are cached.
- A `provisional` result (a tuned optimum on a grid edge) is widened and rerun, not reported.
