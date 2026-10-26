# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: b83286a -->

## Current state

- Design spec approved: `docs/superpowers/specs/2026-10-22-xp-design.md` (amended during Phase 1:
  Page-Hinkley trigger, 500 tune seeds, crossover reporting semantics).
- Foundation + Phase 1 (bandit, tag-team) implemented and reviewed: plan
  `docs/superpowers/plans/2026-10-22-xp-foundation-and-bandit.md`, fix-wave report
  `docs/superpowers/plans/2026-10-22-fix-wave-report.md`. 88 fast tests pass. Probe on disjoint
  seeds: no agent flagged provisional with the committed grids.
- **Done:** Phase 1 README results, including the Kairi-vs-Kaiser CIs (Kaiser wins at all 7 jump rates, no crossover).
  First Colab attempt failed because notebook variables (`$PHASE`) were empty after a runtime
  restart; the notebook cells are now self-contained (commit `dd396d0`).
  Colab is on Python 3.13, where pinned `numpy==2.0.2` compiled from source and tripped resolver
  warnings. Fixed by loosening `requirements.txt` to `>=` lower bounds (commit `0734460`, pushed).
  Notebook shell commands now use literal paths (commit `c1b33f4`) because
  `$PHASE` expanded empty in Colab. Phase 1 run and report completed on Colab (Kaiser best single agent everywhere; `tt_fixed`
  beats it only at lambda>=0.01; `tt_meta`/`tt_triggered` never robustly win). README results
  still to write; need `gaps.csv` and `crossovers.csv` CIs.
  Cross-version check done: a local rerun (Python 3.9.6, numpy 2.0.2) matches the Colab tables to within rounding.
- Phase 2 (MiniGrid DoorKey, PPO vs PPO+RND): plan `docs/superpowers/plans/2026-10-23-phase2-minigrid.md` executed through Task 8 and merged into `main` (pushed): `xp/minigrid/` implemented and tested (123 tests), Colab launcher `notebooks/run_minigrid.ipynb`, CLAUDE.md updated. Remaining work = Task 9 (calibrate the budget, full run, report, README results), which the user chose to run locally on the laptop later (go-ahead needed before starting; see RESUME.md). The spike scripts were removed; their numbers are in the plan and the spec amendment.
- Phase 2 run COMPLETE (2026-10-23): calibration hit the 2M cap (80% target not reached), tuned Kairi ent_coef=0.03 / Kaiser rnd_coef=0.001, 20 seed jobs done, report generated in `figs/`. Result: curves do not cross; Kairi >= Kaiser everywhere (wins at sizes 8, 12; ties at 5, 6). Phase 2 README results written. Phase 1 README CIs filled in from the Colab CSVs. Post-hoc tie-robustness check (Kaiser rnd_coef 0.003/0.01, Kairi ent_coef 0.1) done: Kairi's 8/12 advantage holds only for the AUC-selected Kaiser setting; README updated. (`figs/degradation.png` committed and `main` pushed at `bf28aef`; `calibration.json` committed in 6c2e4c2).
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), default branch `main`.
  `origin/main` includes the requirements fix (`0734460`); `c1b33f4` is on `origin/main`; the `feat/phase2-minigrid` branch is local only (push needs the user's yes).

- Conclusions and analysis are written up in `docs/FINDINGS.md` (update it as new results land).
- **MiniGrid is closed** (user decision, 2026-10-24): Kaiser shows no measurable RND benefit on DoorKey; matched-entropy Kaiser ties Kairi at every size (n=10). Step A (30-seed rerun) is shelved.
- Phase 1 extra shifts DONE (reversed tuning, sigma, k, drift; `configs/bandit-*.yaml`, `figs/shifts/`, `docs/FINDINGS.md`): Kaiser robust to reversed tuning and noise; Kairi wins at k=50 and drift 0.1. 136 tests pass.
- Polish DONE: CI green (Python 3.9, 3.12), README Summary and figures, `docs/WRITEUP.md`, `figs/minigrid_robustness.png`.
- Ablations DONE: retuning at the shifted point reverses both bandit Kairi wins (frozen-tuning brittleness, not exploration cost); `docs/FINDINGS.md`, `figs/ablations/`.
- Step B CLOSED as a documented MultiRoom spike (28 runs, 2 seeds): RND never helped; `docs/FINDINGS.md`, `docs/spikes/multiroom_spike.csv`, `scripts/spikes/`.
- Pushed through 6e34492; CI green on `actions/checkout@v7` / `setup-python@v7` (Node 20 warning gone). The RND probe commit and this refresh are local.
- Deep-RL check done the cheap way: few-shot paired re-analysis across all variants (`docs/FINDINGS.md`, `scripts/fewshot_compare.py`, `docs/spikes/fewshot_pairs.csv`); README few-shot sentence corrected (paired CI at size 12 favours Kairi for the tuned Kaiser, not robust; ties at matched entropy). Full size-8 retune not run.
- Crafter/Showdown future-work sketch written (`docs/FUTURE_WORK.md`, not approved for building; CLAUDE.md scope rule unchanged).
- 5M-step RND probe DONE: at 4 MultiRoom rooms (size <= 4), RND coef 0.1 reached 95% success in about half the steps of plain PPO (5/5 seeds faster); coef 0.01/0.03 no benefit. Post-hoc, not a protocol result; `docs/FINDINGS.md`, `docs/spikes/multiroom_probe.csv`.
- Protocol-legal confirmation of the MultiRoom RND result: discussed and written down in `docs/FUTURE_WORK.md` (options, costs, open decisions); scope undecided, nothing built.
- MultiRoom confirmation: user chose Option 1 (single level, fully protocol-legal) on 2026-10-25; spec amendment 5.4 written and committed (0959b04). Cross-checked against FINDINGS.md, the probe CSV and the code and corrected (AUC padding, config-switch objective, code-change list, cost 10-13 h); no plan, no code, nothing running.
- Plan written: `docs/superpowers/plans/2026-10-25-multiroom-confirmation.md` (8 tasks; no code started).
- Plan Tasks 1-6 DONE on local branch `feat/multiroom-confirmation` (subagent-driven; 152 tests green; smoke run OK). Final whole-branch review: no Critical; fix wave 57643c8 (training curves kept, matched-entropy guards) and 0930c43 (plan hardening notes) committed, scoped re-review in flight. Tasks 7 (calibrate + 11-13 h run) and 8 (write-up) pending.
- Task 7 RUNNING since 2026-10-25: S=2,938,880 frozen (32e41e2); full run in worktree ../xp-agents-run (see RESUME.md). Re-review of fix wave came back clean.
- Interim evaluation doc written: `docs/MULTIROOM_EVALUATION.md` + `scripts/multiroom_eval_figs.py` (figures in figs/multiroom/eval_*.png, force-added). Branch merged to main and pushed (c0fd8fe); doc commit 32bc10b is local, not pushed. Kairi tuned: ent_coef 0.01 (interior); Kaiser tuning running.
- MultiRoom run was paused 16:36 and RESUMED 18:21 (processes continued, nothing lost).
- MultiRoom confirmation COMPLETE (2026-10-25): Kairi ent_coef 0.01, Kaiser rnd_coef 0.6 (both interior); on 10 eval seeds Kaiser reached 95% in 0.70M vs 2.23M steps (paired CI of difference [-2.12M, -0.93M]), faster on 10/10, AUC 0.78 vs 0.26, solved 10/10 vs 5/10. Written up in `docs/MULTIROOM_EVALUATION.md`, FINDINGS, README, WRITEUP (commit 5d442de, pushed).
- Standalone findings page: `docs/findings.html` (built by `scripts/build_findings_html.py` from `scripts/findings_template.html`; d3 vendored; commit local, not pushed). A private claude.ai artifact copy also exists.
- Crafter and Showdown specified as generality tests (spec 5.5, 5.6; Crafter in scope, spike first; Showdown not approved for building). Spike plan written (docs/superpowers/plans/2026-10-25-crafter-spike.md); the script is committed and smoke-tested; execution mode chosen: subagent-driven (see RESUME.md); spec approved by the user.
- Cleanup and push done (2026-10-25). **Next:** nothing in flight; no open research items, report (Task 7 step 6), Task 8 write-up, then finish branch (merge/push need user's yes)

## Last updated

2026-10-25 (commit fc2cd16)

## Last updated 2026-10-26 (commit b83286a)
- Crafter spike in progress (plan `docs/superpowers/plans/2026-10-25-crafter-spike.md`, subagent-driven, ledger in git-ignored `.superpowers/sdd/2026-10-25-crafter-spike/`). Task 1 DONE: `tests/test_crafter_spike.py` (5 tests, 161 total pass). Task 2 DONE (throughput + cost table in `docs/spikes/crafter_spike.md`). Colab notebook `notebooks/run_crafter_spike.ipynb` written (user chose Colab for Task 4; script upload cell because unpushed). Next: Task 3 asks the user's go-ahead before the two 1M-step runs; then Task 5 write-up. No push without the user's yes; Showdown not approved.
- Crafter spike DONE: G1 fails at both areas (4.52 / 4.37 < 5), recommendation stop; awaiting the user's go/no-go. See RESUME.md.
- Paused 2026-10-26: Crafter go/no-go deferred by the user.
- Polish pass 2026-10-26: README, WRITEUP, FINDINGS, FUTURE_WORK now include the Crafter spike.
- Crafter CLOSED as a documented spike (user decision 2026-10-26). Nothing in progress.
- Cleanup 2026-10-26: unused import removed, README repository map added.
- Crafter closed for time; may be reconsidered later (docs say so).
- findings.html has a Crafter chapter (chart of the two spike curves).
- findings.html charts: uniform height, axis labels on every chart, 13px text, fixed small-panel sizing bug.
- findings.html last section split into shows / does not show.
- Wall-clock caveat refined from logged probe times (not controlled); longer follow-ups (other room counts, bigger budget, more DoorKey seeds) left for later.
