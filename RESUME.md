# XP — Resume

Nothing is running. **MiniGrid is closed** (user decision 2026-10-24): entropy check done (matched-entropy Kaiser ties Kairi at every size, n=10, README post-hoc);
step A (30 seeds, matched entropy) is shelved, see `docs/superpowers/plans/2026-10-24-phase2-step-a.md`. Do not reopen without the user asking.
Phase 1 extra shifts DONE (reversed, sigma, k, drift; docs/FINDINGS.md, figs/shifts/): Kaiser robust to reversed tuning and noise; Kairi wins at k=50 and drift 0.1 (crossovers in (20,50) and (0.02,0.1)).
Polish DONE (user chose to go straight to it, 2026-10-24): CI exists and is green on Python 3.9 and 3.12; README has a Summary and figures; short write-up `docs/WRITEUP.md`; MiniGrid robustness figure `figs/minigrid_robustness.png`.
Ablations DONE (2026-10-24, user asked for both optional steps): retuning both agents at k=50 / drift 0.1 reverses both Kairi wins (Kaiser wins by 0.086 and 0.168); regret trace at k=50 refutes the early-exploration-cost explanation. See docs/FINDINGS.md "Ablations", figs/ablations/. Result: the Kairi wins are Kaiser's frozen-tuning brittleness.
Step B CLOSED as a documented spike (user chose option 1 after the 28-run MultiRoom spike): RND never helped (docs/FINDINGS.md "Step B spike", docs/spikes/multiroom_spike.csv, scripts/spikes/). No environment code was added.
Nothing is running. The 5M-step RND probe FINISHED (2026-10-24): at 4 MultiRoom rooms (size <= 4), RND coef 0.1 reached 95% training success in a median 1.44M steps vs 2.66M for plain PPO, faster on 5/5 seeds (mean 1.17M fewer steps, CI [0.53M, 1.75M]); 0.03 slower, 0.01 mixed with one failed seed. This is POST-HOC (best of 3 coefs, top of range, tune-pool seeds); docs/FINDINGS.md, README, WRITEUP updated; summary in docs/spikes/multiroom_probe.csv, scripts/spikes/probe_summary.py.
IN PROGRESS (2026-10-25): the user chose Option 1 (single-level, fully protocol-legal MultiRoom confirmation) and approved the design. Spec amendment 5.4 is WRITTEN and committed (0959b04, `docs/superpowers/specs/2026-10-22-xp-design.md`): 4 rooms, max room size 4; tuning objective = training AUC of success; Kairi ent_coef {0.003,0.006,0.01,0.02,0.05,0.1}, Kaiser rnd_coef {0.01,0.03,0.1,0.3,0.6,1.0} with Kaiser entropy = Kairi's tuned value; 3 tune seeds, 10 eval seeds; S = default-Kairi 95% success on tune seeds, cap 5M. Nothing else built, nothing running, no push.
VERIFIED 2026-10-25: amendment 5.4 was cross-checked against FINDINGS.md, the probe CSV, step-A plan and xp/minigrid + xp/protocol code, and corrected (AUC padded over [0,S] because training stops at 95%; objective is a config switch; held-out test success replaces "training success at S"; probe-seed overlap limitation; code-change list; cost 10-13 h incl. calibration). Do not redo this reading. Still unchecked: how xp.report / xp.stats handle a one-point sweep (plan item).
PLAN WRITTEN (2026-10-25): docs/superpowers/plans/2026-10-25-multiroom-confirmation.md (8 tasks, TDD). Spec 5.4 corrected again: Learner.train has NO early stop (always trains the full S), so AUC needs no padding; cost 11-13 h at S=5M.
EXECUTING (2026-10-25) via subagent-driven development on branch feat/multiroom-confirmation (local only, from main bc78c22). Ledger: .superpowers/sdd/2026-10-25-multiroom-confirmation/progress.md (git-ignored; trust it and `git log` after a compaction). Tasks 1-2 complete (reviews clean). Task 3 implemented (3322be5, about 147 tests green), its task review in flight; Tasks 4-6 not started. Task 7 (calibrate, freeze S, 11-13 h run) needs the user's go-ahead; no push, no merge to main without a yes.
NEXT STEP: read the ledger, finish Task 3's review loop, then continue with Task 4.
All commits are pushed through the probe summary commit (user asked to push when the probe finished).
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
