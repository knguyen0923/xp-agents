# Crafter reopened: calibrated budget and a Colab-ready phase (spec 5.5 amendment)

Status: draft for the user's review (2026-11-02). The design was approved by the user in a brainstorm on 2026-11-02 (decisions below). Nothing here is built, and no run is approved. Spec 5.5 stays in force except where this document amends it; CLAUDE.md says reopening Crafter needs an amendment that raises the budget S first, and this is that amendment.

## Why

The Crafter spike (`docs/spikes/crafter_spike.md`, 2026-10-26) closed because plain PPO missed the usable-range gate G1 at the pre-registered budget S = 1M steps: final `ach_100` 4.52 at area 32 and 4.37 at area 64, against a floor of 5. Both curves were still rising (area 32: 3.0 at 10k, 3.9 at 500k, 4.5 at 1M), so a longer budget might pass, but that was untested. The question the phase answers is unchanged: does an RND bonus pay off in a long-horizon, sparse-reward, procedurally generated world where exploration is the bottleneck? Kairi tying or winning is a valid outcome.

## Decisions (user, 2026-11-02)

1. Reopen Crafter for the full phase: single familiar world, no shift knob (spec 5.5).
2. Area 32. Area 64 gave the same plain-PPO result at about 2.5 times the cost per step.
3. The budget S is set by a calibration rule with a cap, not fixed (below). The user approves the calibrated S and the measured cost before any tuning run.
4. One resumable notebook on Google Colab (option A), sequential on one runtime, results on Drive. No job sharding (it can be added later if the benchmark shows the run is too slow).
5. A new `xp/crafter/` package with its own PPO and RND, not a generalisation of the MiniGrid learner (amends the "Code changes" paragraph of 5.5).

## Change 1: budget S by calibration

- Config `budget.cap: 3000000`. The calibration runs default-entropy Kairi (`ent_coef` 0.01, an untuned default used only for calibration; Kaiser's tuned runs use Kairi's tuned `ent_coef`, see Protocol) on each tune seed for the whole cap, with no early stopping, logging `ach_100` every 10,240 steps. The tune seeds are the config's 3 tune seeds (`Phase.tune_seeds`), disjoint from the eval seeds as always.
- Rule (modelled on `xp.minigrid.calibrate.suggest_budget`): S is the first logged step from which the mean over the tune seeds of `ach_100` (NaN before 100 episodes counted as 0) is at least `budget.target_ach` = 5 at that step and at the next 2 logged steps (3 consecutive log points, `budget.sustain: 3`, so one lucky checkpoint cannot set S). A crossing within the last 2 log points of the cap does not qualify. If no logged step up to the cap qualifies, the phase stops: the calibration is written up as a result (G1 failed again), no tuning or RND run starts. The target 5 and the cap 3M are declared here, before any calibration run.
- At S, on the same calibration runs, G1's second clause (at least 5 of the 22 achievements unlocked in under 5% of the last 100 episodes, averaged over the tune seeds) and G2 (mean `ach_100` at S exceeds the mean `ach_100` over the first 10% of the steps up to S) are checked and reported. If either fails the phase stops as above.
- The deep achievement set (achievements plain PPO unlocks in 5% to 95% of episodes at S, averaged over the tune seeds) is fixed from these runs and written to the calibration JSON before any RND run (spec 5.5). If it has fewer than 3 achievements (`budget.min_deep: 3`), the phase stops as above, like a failed G1.
- G3 (cost) is the user's decision: the calibration output prints S, the measured steps per second on the Colab runtime, and the implied hours for the full phase (`(36 + 20) * S / steps_per_s / 3600`), plus the calibration's own hours (3 seeds up to the cap) and a note that RND runs are slower than the plain-PPO throughput measured here, so the estimate is a floor. The user then edits `budget.steps` in `configs/crafter.yaml` to the printed S and the run starts, as in MiniGrid. No tuning job may start while `budget.steps` is unset (guard in code).
- Calibration output: `crafter_calibration.json` (S, `reached`, mean curve, log steps, per-achievement rates, deep set, G1 and G2 verdicts, throughput).

## Change 2: `xp/crafter/` (new package)

Follows the Showdown precedent (own PPO inside the phase package) so MiniGrid code, tests and the config hashes of `configs/minigrid.yaml` and `configs/minigrid-multiroom.yaml` do not change, and `xp.protocol` stays phase-agnostic (never imports `xp.crafter`; `xp.crafter` never imports `xp.minigrid` or `xp.showdown`).

- `env.py`: a vectorised Crafter env over `crafter.Env(area=(32, 32), seed=...)`, 64x64x3 observations, 17 actions (`crafter.constants.actions`); the world seed for an episode is `hash((env_seed, episode)) % (2**31 - 1)` as in the spike (checked against crafter 1.8.3); tune and eval world-seed pools disjoint and guarded (`ProtocolError`), as in `xp/minigrid/env.py`. The guard compares env seeds, not individual worlds (worlds come from the hash, so a world collision between pools is possible in principle but negligible); tuple hashing of ints is stable from Python 3.8, so the 3.9 floor and Colab agree. Reports distinct achievements per finished episode.
- `stats.py`: `EpisodeStats` (sliding window of 100 episodes: `ach_100`, per-achievement rates), lifted from `scripts/spikes/crafter_spike.py`, which stays as the closed spike's record.
- `ppo.py`: PPO with GAE and clipping, the same core as `xp.minigrid.ppo`, on the spike's small CNN encoder; Kaiser adds RND (fixed random target CNN, predictor CNN, same intrinsic-reward normalisation as MiniGrid 5.2). Encoder, optimiser and PPO settings identical for both agents, fixed in the config before tuning. Device from config (`cuda`, `mps` or `cpu`; Colab GPU).
- `calibrate.py`: `python -m xp.crafter.calibrate --config ... --out ... --workers ...`, the rule above (`suggest_budget`, G1 and G2 checks, deep set).
- `phase.py`: `run_phase`, `tune_job`, `seed_job` like `xp/minigrid/phase.py`: tune Kairi, take its tuned entropy for Kaiser (matched entropy), tune Kaiser, freeze, evaluate on 10 eval seeds; per-job cached results keyed by config hash; resumable after a disconnect.
- `xp/run.py`: `PHASES` gains `"crafter"`, with `workers` and `allow_code_change` passed like minigrid and showdown.
- `requirements-crafter.txt`: `-r requirements.txt`, `torch>=2.0`, `crafter`. Python floor 3.9; every module starts with `from __future__ import annotations`.
- `configs/crafter.yaml`: phase `crafter`, familiar area 32, seeds (3 tune, 10 eval), `budget.cap: 3000000`, `budget.target_ach: 5`, `budget.steps` unset until calibrated, the grids and objective below; plus `configs/crafter-smoke.yaml` (tiny budget, 2 grid values, 1 tune seed, 2 eval seeds, a few minutes on Colab CPU or GPU) that exercises every path.

## Protocol (spec 5.5, unchanged)

- Tune on 3 tune seeds only, freeze, evaluate on 10 disjoint eval seeds, paired by seed, paired bootstrap 95% CIs, provisional-on-edge (grid-edge optimum stops the run, ask the user, widen both grids equally in a fresh folder).
- Kairi tuned first, Kaiser's `ent_coef` set to Kairi's tuned value.
- Equal 6-value grids: Kairi `ent_coef` {0.003, 0.006, 0.01, 0.02, 0.05, 0.1}; Kaiser `rnd_coef` {0.01, 0.03, 0.1, 0.3, 0.6, 1.0}. The RND scale on images may differ from MiniGrid; values may be adjusted only before any tuning run, with the reason recorded.
- Objective: training AUC of `ach_100` over the logged curve in [0, S], divided by 22; the tuner minimises `1 - auc`. Tie-break: highest mean final `ach_100`, then grid order.
- Secondary metrics on the eval seeds: the official Crafter score (geometric mean over the 22 per-achievement success rates), per-achievement unlock rates, time to first unlock of the deep set.
- 36 tuning runs plus 20 evaluation runs.

## Colab notebook

`notebooks/run_crafter.ipynb`, modelled on `notebooks/run_minigrid.ipynb`:

1. Mount Drive; clone the repo from GitHub; `pip install -r requirements-crafter.txt -e .`; confirm a GPU runtime and print the device.
2. Benchmark cell: a short plain-PPO run printing steps per second (compare with the spike: area 32 CPU 589, MPS 781).
3. Smoke cell: `configs/crafter-smoke.yaml`, a few minutes, to confirm the whole pipeline before spending hours.
4. Calibration cell, then a markdown cell that tells the user to read S, the verdicts and the hours estimate, and stop here.
5. After the user sets `budget.steps` in `configs/crafter.yaml`, pushes and reclones: the full run cell (resumable: after a disconnect rerun the same cell; at most one in-flight job restarts from zero), then a status cell (jobs done of 56), then the report cell (`xp.report --phase crafter`).
- Results and figures go to Drive (`xp-results/`, `xp-figs-crafter/`). Literal paths, as the MiniGrid notebook notes `$VAR` came out empty.
- Colab clones from GitHub, so the branch must be pushed before the notebook can run; pushes need the user's yes each time (CLAUDE.md).

## Tests

No GPU or network for the default run: world-seed pool disjointness and the guard; `suggest_budget` (first crossing, cap fallback, NaN handling, mismatched log steps); `EpisodeStats` (window, rates); CNN and RND output shapes and that RND's target is frozen; G1 and G2 verdict helpers; the config guards (equal grids, `budget.steps` unset blocks tuning, hash stability); that `configs/minigrid*.yaml` hashes and the MiniGrid tests are unchanged. One slow test that steps a real `crafter.Env` for a few hundred steps and finishes an episode, marked `crafter` and excluded from the default `pytest` run (like `showdown`).

## Out of scope

A world-size shift sweep, other Crafter variants, budgets above the cap, job sharding across Colab sessions, a deep-RL tag-team agent.

## Risks

- Calibration may fail again (plain PPO below 5 achievements by 3M, or fewer than 5 rare achievements): the phase stops and that is the result.
- Cost: up to about 60 hours at the spike's 781 steps/s if S reaches the cap; less if S is lower. Colab runtime speed and quota are unmeasured until the benchmark cell runs; a free-tier runtime may not finish the phase.
- Lost work: a job restarts from zero after a Colab disconnect (accepted, no mid-job checkpoints); at S near the cap that is up to about 64 minutes per interruption at 781 steps/s, and a free runtime's session limit means a long phase spans several sessions.
- Seed variance is high; intervals may be wide even at 10 seeds.
- The achievement count is a proxy for exploration; the network size and RND scale are design choices fixed before tuning.

## Amendment (2026-11-03): 17 actions, not 8

The user approved this fix on 2026-11-03. The 8-action count came from the 2026-10-26 spike and spec 5.5 and was wrong: `crafter.Env` has 17 actions (`crafter.constants.actions`). With 8 actions the policy could not place a table or furnace or make any tool, so the spike's G1 failure (4.52 / 4.37) was measured on a crippled action space and is not evidence about PPO on Crafter. `N_ACTIONS` in `xp/crafter/ppo.py` is now 17 (guarded by a `crafter`-marked test against the real env). The calibration (cap 3M, target 5, sustain 3) is unchanged and now measures the 17-action game; the budget rule, grids and protocol are unchanged.
