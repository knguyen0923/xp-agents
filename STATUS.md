# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first. Older dated notes were removed in the 2026-10-30 polish pass; they live in the git history of this file.

<!-- STATUS_COMMIT: 546b744 -->

## Current state

**The project was done on 2026-11-01; on 2026-11-02 the user reopened two future-work items (see Next).** Every phase has a result and a write-up, `v1.0` is tagged and pushed, and CI is green on the latest push. The fast suite passed on 2026-11-01 with no failures. **Matchup control run finished 2026-10-31** (see below). Added 2026-10-31: MIT `LICENSE`, `CITATION.cff`, `ruff` in CI (green on GitHub), the finer Kairi `alpha` sweep (Kairi 0.2236 vs 0.2379 on the old grid, Kaiser 0.1724; no change to the conclusion).
Design spec: `docs/superpowers/specs/2026-10-22-xp-design.md` (with amendments). Plans: `docs/superpowers/plans/`.

- **Phase 1 (bandit)**: done, including extra shifts (reversed tuning, sigma, k, drift) and ablations.
  Kaiser is the best single agent nearly everywhere; Kairi wins at k=50 and drift 0.1, but retuning
  at the shifted point reverses both wins (frozen-tuning brittleness). Tag-team variants: `tt_fixed`
  beats Kaiser only at lambda>=0.01. Tuning-grid caveats (Kairi's grid is effectively 1-D; Kaiser's drift optimum sits at `gamma=1.0`) are in `docs/WRITEUP.md` Limitations.
- **Phase 2 (MiniGrid DoorKey)**: done and closed. Curves do not cross; Kairi >= Kaiser; matched-entropy
  Kaiser ties Kairi at every size.
- **MultiRoom confirmation** (spec 5.4): done. Kaiser (rnd_coef 0.6) reached 95% success in 0.70M vs
  2.23M steps for Kairi (ent_coef 0.01), faster on 10/10 seeds. See `docs/MULTIROOM_EVALUATION.md`.
- **Crafter**: spiked and closed (G1 gate failed, 4.52 / 4.37 < 5); may be reconsidered. Needs a spec
  amendment raising the budget first. `docs/spikes/crafter_spike.md`.
- **Showdown** (opponent DamageHeuristic, S = 250k decisions, threshold 0.30: user decisions), two runs, both a tie:
  - *Original run* (finished 2026-10-28): training AUC Kairi 0.406 vs Kaiser 0.403, CIs overlap; Kairi reaches a 30% win rate about 5k decisions sooner but the interval includes 0. Tuned `ent_coef` 0.01, `rnd_coef` 0.03. Tables: `docs/spikes/showdown_results/`.
  - *Matchup-features run* (type-effectiveness inputs added; finished 2026-10-30): AUC Kairi 0.416 vs Kaiser 0.424, CIs overlap. Against the original run (paired by eval seed) only Kaiser's AUC rises clearly (+0.022 [+0.012, +0.031], 9 of 10 seeds), but the inputs are confounded with wider grids and retuning (Kairi `ent_coef` 0.006, Kaiser `rnd_coef` 1.0). Tables and `cross_run_paired.csv`: `docs/spikes/showdown_matchup_results/` (script `scripts/showdown_matchup_compare.py`).
  - *Control run* (original inputs on the matchup run's 8-value grids; finished 2026-10-31): AUC Kairi 0.409 vs Kaiser 0.410, a tie (tuned Kairi `ent_coef` 0.002, Kaiser `rnd_coef` 0.03). Kaiser's gain over the original run splits into grids and retuning +0.008 [−0.004, +0.018] and inputs +0.014 [+0.008, +0.020], 9 of 10 seeds; Kairi's steps both include 0. Tables: `docs/spikes/showdown_control_results/`.
- Write-ups: `README.md`, `docs/FINDINGS.md`, `docs/WRITEUP.md`, `docs/findings.html`
  (built by `scripts/build_findings_html.py`; Chapter 7 covers both Showdown runs; built and syntax-checked but never viewed in a browser), `docs/FUTURE_WORK.md`.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), branch `main`. Pushes need the user's yes each time.

## Next

**Reopened 2026-11-02 (user decision; spec stage, nothing built or running):** (1) Showdown tempo opponent, tempo feature block and friendly-fire reward penalty, two runs (matchup+penalty, then +tempo): spec `docs/superpowers/specs/2026-11-02-showdown-tempo-design.md`, awaiting the user's review, then a plan; the budget cap S needs the user's approval before any RND result is read. (2) Crafter spec amendment and a Colab-ready notebook: not started, brainstorm questions pending. Smaller items stay ignored. Details in `RESUME.md`.

Nothing is in progress. Tag `v1.0` is done: annotated, points at `27dd4af`, and is on `origin` (verified 2026-11-01). Nothing is required. The user opened `docs/findings.html` Chapter 7 and `figs/showdown-matchup/degradation.png` in a browser on 2026-11-01 and reported no problems. Pushes need the user's yes. Optional work was closed by decision on 2026-11-01 (`docs/REMAINING_WORK.md` section 7). Battle replays: recorded 2026-11-01, four wins kept in `docs/replays/` (index `docs/replays/INDEX.md`; full output in the gitignored `replays/`).

**Close-out brainstorm (2026-11-01, plan "close by decision") finished:** doc cleanup, bandit drift check (optimum confirmed) and the Showdown type-chart probe (no usable preference) are done; the size-8 DoorKey retune was timed once (543 s, training success stayed at the floor) and closed. Showdown follow-ups 1 to 3, MultiRoom other room counts, more DoorKey seeds, a deep-RL tag-team agent and Crafter are closed with written reasons in `docs/REMAINING_WORK.md` section 7. `~/xp-showdown-spike/` is kept. No open work remains.

## Last updated

2026-11-02 (future work reopened: Showdown tempo spec committed 254ad83, Crafter pending; nothing running)