# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 5aa72c7 -->

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
- Executing the plan on local branch `feat/multiroom-confirmation` (subagent-driven): Tasks 1 (level-aware env, f91df44) and 2 (AUC objective, 2eb2057) complete; Task 3 (matched entropy, 3322be5) implemented, review in flight; Tasks 4-8 pending.
- **Next:** finish Task 3 review, then Task 4; the long run (Task 7) needs the user's go-ahead (details in RESUME.md). Last push was b139f20; later commits are local.

## Last updated

2026-10-25 (commit 5aa72c7)
