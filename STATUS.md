# XP — Status

Point Claude at this file after `/clear`. It is a snapshot; re-verify against the repo
(`git log`, `git status`, `pytest`). If `RESUME.md` says anything other than "nothing in
progress," read it first.

<!-- STATUS_COMMIT: 16a9d58 -->

## Current state

The project is essentially finished. 212 fast tests pass; CI is green on Python 3.9 and 3.12 at the last push (319a2bf, which includes the Showdown phase).
Design spec: `docs/superpowers/specs/2026-10-22-xp-design.md` (with amendments). Plans: `docs/superpowers/plans/`.

- **Phase 1 (bandit)**: done, including extra shifts (reversed tuning, sigma, k, drift) and ablations.
  Kaiser is the best single agent nearly everywhere; Kairi wins at k=50 and drift 0.1, but retuning
  at the shifted point reverses both wins (frozen-tuning brittleness). Tag-team variants: `tt_fixed`
  beats Kaiser only at lambda>=0.01.
- **Phase 2 (MiniGrid DoorKey)**: done and closed. Curves do not cross; Kairi >= Kaiser; matched-entropy
  Kaiser ties Kairi at every size.
- **MultiRoom confirmation** (spec 5.4): done. Kaiser (rnd_coef 0.6) reached 95% success in 0.70M vs
  2.23M steps for Kairi (ent_coef 0.01), faster on 10/10 seeds. See `docs/MULTIROOM_EVALUATION.md`.
- **Crafter**: spiked and closed (G1 gate failed, 4.52 / 4.37 < 5); may be reconsidered. Needs a spec
  amendment raising the budget first. `docs/spikes/crafter_spike.md`.
- **Showdown**: spiked, gated, built and RUN (opponent DamageHeuristic, S = 250k, threshold 0.30: all user decisions). The full run finished 2026-10-28 20:19 (20 eval seed jobs, 10 per agent). Result: a tie on the familiar opponent (training AUC Kairi 0.406 vs Kaiser 0.403, CIs overlap; Kairi reaches 30% win rate about 5k steps sooner, CIs touch); tuned `ent_coef` 0.01 and `rnd_coef` 0.03, both interior. Tables and figure are in `figs/showdown` (gitignored). Not yet written into FINDINGS/README/WRITEUP. Earlier history: the first run was stopped on a misread of the tuning score (`1 - AUC`, lower is better); the Showdown server swap growth was fixed by a patch in the scratch checkout (`docs/spikes/showdown_memory.md`).
- Write-ups: `README.md`, `docs/FINDINGS.md`, `docs/WRITEUP.md`, `docs/findings.html`
  (built by `scripts/build_findings_html.py`), `docs/FUTURE_WORK.md`.
- GitHub: `origin` = https://github.com/knguyen0923/xp-agents (public), branch `main`, all pushed.

## Next

Nothing running. Remaining: write the Showdown result into the write-ups, then the deferred battle-replay feature (plan `docs/superpowers/plans/2026-10-27-showdown-battle-replays.md`, needs the user's yes). Possible longer follow-ups, none approved: other MultiRoom room counts, a bigger budget, more DoorKey seeds, reopening Crafter. Showdown commits are not pushed; pushes need the user's yes.

## Last updated

2026-10-28 (commit 16a9d58)
