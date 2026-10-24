# XP — Resume

Nothing is running. **MiniGrid is closed** (user decision 2026-10-24): entropy check done (matched-entropy Kaiser ties Kairi at every size, n=10, README post-hoc);
step A (30 seeds, matched entropy) is shelved, see `docs/superpowers/plans/2026-10-24-phase2-step-a.md`. Do not reopen without the user asking.
Phase 1 extra shifts DONE (reversed, sigma, k, drift; docs/FINDINGS.md, figs/shifts/): Kaiser robust to reversed tuning and noise; Kairi wins at k=50 and drift 0.1 (crossovers in (20,50) and (0.02,0.1)).
Polish DONE (user chose to go straight to it, 2026-10-24): CI exists and is green on Python 3.9 and 3.12; README has a Summary and figures; short write-up `docs/WRITEUP.md`; MiniGrid robustness figure `figs/minigrid_robustness.png`.
Nothing is in flight. Optional, only if the user asks: step B (exploration-heavy world), ablations of the untested k=50/drift explanations. Known nit: CI actions warn about Node 20 deprecation (bump `actions/checkout`/`setup-python` majors).
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
- A: SHELVED (MiniGrid closed).
- Cheap Phase 1 shifts: DONE.
- B spike: harder-exploration MiniGrid (KeyCorridor/MultiRoom) or CoinRun; DoorKey sizes can sit at 0% for all agents (spec risk table).
- C last: CI on every push (spec requires it; check whether it exists), figures in the README, short write-up.

## Facts to remember
- Phase 2 headline (README): curves do not cross; Kairi >= Kaiser everywhere; Kairi's size 8/12 lead holds only for the AUC-selected Kaiser `rnd_coef` (0.001); at 0.01 they tie. One Kaiser seed fails outright at 0.001 (seed 101) and 0.003 (seed 105).
- Stop hook: `STATUS_COMMIT` must equal the commit whose diff to HEAD touches only `STATUS.md` (put other file changes, e.g. this file, in the earlier commit, then a STATUS-only marker commit).
- Pushes need the user's yes each time.
