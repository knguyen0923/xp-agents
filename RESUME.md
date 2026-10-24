# XP — Resume

Nothing is running. Entropy check DONE (2026-10-24): matched-entropy Kaiser (ent 0.03, rnd 0.01) ties Kairi at every size (n=10); in README as post-hoc.
Step A is **drafted, not approved**: `docs/superpowers/plans/2026-10-24-phase2-step-a.md` + `configs/minigrid-a.yaml` (eval seeds 100-129 only so far).
Next: get the user's decisions on the plan's `?` items (tune seeds 3->5, second familiar size), then amend the spec and TDD the Kaiser-entropy-from-Kairi change.
Nothing from this session is pushed (needs the user's yes).

## Agreed direction (brainstorm, 2026-10-24)
User wants a mix of: (1) polish, (2) a stronger research claim, (3) fixing weaknesses. Compute budget: **overnight is fine**.
Order: entropy check -> A (make Phase 2 conclusive) -> cheap Phase 1 extra shifts -> B spike (world where exploration matters) -> C (polish).
Skip other exploration methods (count-based, ICM): they change Kaiser's definition. Do not build Crafter/Showdown.

## Entropy check (done; design kept for reference)
Why: Kairi tuned `ent_coef` (picked 0.03); Kaiser's `ent_coef` is fixed at 0.01 and only `rnd_coef` is tuned (config: `agents.kaiser.fixed.ent_coef`).
Kairi at 0.01 and Kaiser at `rnd_coef` 0 are the same agent and tied in tuning, so Kaiser's deficit at sizes 8/12 may be an entropy effect, not RND.
Design: copy the config with Kaiser `fixed.ent_coef` = 0.03, re-tune `rnd_coef` on the 6-value grid with tune seeds only (protocol-legal), then evaluate
on the same 10 eval seeds and compare with the main Kairi results, paired by seed. Small script like `scripts/tie_robustness.py`: Kaiser tuning (`_tune_agent`)
plus `seed_job`, output `results/ent-fair/`; reuse Kairi's existing results (do not rerun Kairi). Report in the README as a post-hoc extra check, not a replacement
for the main result. Cost ~2 h (about 50 min tuning, 30-60 min eval); use `caffeinate -i -w <pid>` and a background watcher.
Outcome: tie, so A reruns with matched entropy for both agents.

## Then
- A: more eval seeds (30 instead of 10; new config and full rerun, ~5-6 h, mostly Kaiser), a pre-declared tie-break rule, maybe a second familiar size.
- Cheap Phase 1 shifts (minutes locally): reward noise sigma, arm count k, the slow-drift check, reversed tuning direction.
- B spike: harder-exploration MiniGrid (KeyCorridor/MultiRoom) or CoinRun; DoorKey sizes can sit at 0% for all agents (spec risk table).
- C last: CI on every push (spec requires it; check whether it exists), figures in the README, short write-up.

## Facts to remember
- Phase 2 headline (README): curves do not cross; Kairi >= Kaiser everywhere; Kairi's size 8/12 lead holds only for the AUC-selected Kaiser `rnd_coef` (0.001); at 0.01 they tie. One Kaiser seed fails outright at 0.001 (seed 101) and 0.003 (seed 105).
- Stop hook: `STATUS_COMMIT` must equal the commit whose diff to HEAD touches only `STATUS.md` (put other file changes, e.g. this file, in the earlier commit, then a STATUS-only marker commit).
- Pushes need the user's yes each time.
