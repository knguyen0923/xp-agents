# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first. Older dated notes were removed in the 2026-10-30 polish pass; they live in the git history of this file.

<!-- STATUS_COMMIT: 954747a -->

## Current state

The project is essentially finished: every phase has a result and a write-up. The fast suite passes (246 tests). **In flight: the matchup control run** (started 2026-10-31 morning, 1 worker, about 9 to 10 hours; see `RESUME.md`). Added 2026-10-31: MIT `LICENSE`, `CITATION.cff`, `ruff` in CI (not yet confirmed green on GitHub), the finer Kairi `alpha` sweep (Kairi 0.2236 vs 0.2379 on the old grid, Kaiser 0.1724; no change to the conclusion).
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
- Write-ups: `README.md`, `docs/FINDINGS.md`, `docs/WRITEUP.md`, `docs/findings.html`
  (built by `scripts/build_findings_html.py`; Chapter 7 covers both Showdown runs; built and syntax-checked but never viewed in a browser), `docs/FUTURE_WORK.md`.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), branch `main`. Pushes need the user's yes each time.

## Next

When the control run finishes: report it (`python -m xp.report --phase showdown --config configs/showdown-control.yaml --out results/showdown-control --figs figs/showdown-control`, then `scripts/showdown_matchup_compare.py`), update the write-ups, then tag `v1.0` (approved; pushing the tag needs the user's yes). The user still has to view `docs/findings.html` Chapter 7 and `figs/showdown-matchup/degradation.png`. **`docs/REMAINING_WORK.md` is the index of open work.** Deferred by the user: battle-replay recording. Longer follow-ups with costs: `docs/FUTURE_WORK.md`.

## Last updated

2026-10-31 (control run started; licence, ruff, alpha sweep done)
