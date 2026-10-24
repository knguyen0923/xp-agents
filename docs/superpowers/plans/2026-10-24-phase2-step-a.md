# Phase 2 step A — make the MiniGrid result conclusive (SHELVED 2026-10-24, nothing run)

**Shelved:** the user chose to close MiniGrid on the existing evidence (tuned run, tie-sensitivity check, matched-entropy check) and move on. Decisions made before
shelving: 5 tune seeds, defer the second familiar size. Kept for reference if MiniGrid is reopened; `configs/minigrid-a.yaml` is an unfinished draft (eval seeds only).

## Why
The matched-entropy check (README, "Matched-entropy check") showed Kaiser at Kairi's entropy ties Kairi at every size (n = 10). The main-run
"Kairi beats Kaiser at sizes 8/12" depends on Kaiser's unmatched `ent_coef` and on a 5-way tuning tie broken by AUC. Step A reruns Phase 2 so the headline
no longer depends on either, with enough seeds to say something about size 8/12.

## Proposed design (decisions marked ?)
1. **Matched entropy by construction.** Kairi tunes `ent_coef` (6 values). Kaiser then uses Kairi's *tuned* `ent_coef` as a fixed value and tunes `rnd_coef`
   (6 values). Grids stay equal-sized. Needs a small code change in `xp/minigrid/phase.py` (`run_phase`/`build_learner`: Kaiser's fixed entropy comes from Kairi's
   tuning result instead of `agents.kaiser.fixed.ent_coef`); the cache key must include it. Tuning order becomes Kairi -> Kaiser.
2. **30 eval seeds** (`seeds.eval: [100, 130]`, drafted in `configs/minigrid-a.yaml`), up from 10. Tune seeds stay disjoint (0-2).
3. **Pre-declared tie-break, written into the spec before the run:** among grid points tied for best tuning score, pick the highest mean training AUC, then the
   smaller coefficient. (Existing code does AUC then grid order, which is the smaller coefficient for these ascending grids; just state it.)
4. ? **More tune seeds (3 -> 5)**: tuning at size 6 saturates (5 of 6 values tie at 0.000), so it barely discriminates. 5 seeds costs ~+65% tuning time (~1.5 h).
   Recommend yes.
5. ? **Second familiar size (8)**: size 6 is at ceiling for both agents; size 8 has headroom (~0.6). Doubles the work, needs a design change (the knob is
   a single familiar value). Recommend deferring; do it only if the 30-seed result at familiar 6 is still a tie.
6. Report as a replacement headline with the old 10-seed run kept as history; paired-by-seed CIs, same tables as now; add the failed-outright seed count per agent.

## Cost (estimates, from the earlier runs; the laptop does it, 3 workers, `caffeinate`)
- Tuning: Kairi 18 jobs + Kaiser 18-30 jobs, ~1.5-2.5 h total.
- Eval: 60 seed jobs (30 per agent); Kaiser jobs ~2.5x Kairi, ~4-5 h. Total ~6-8 h, fine overnight.
- Run is resumable (per-job caches keyed by config hash); a hash mismatch raises.

## Steps
1. Approve design (esp. ? items). Add spec amendment: matched entropy, tie-break rule, seed counts.
2. TDD the `run_phase` change (Kaiser fixed ent from Kairi's tune; hash/cache includes it; equal-grid check still enforced).
3. Finalize `configs/minigrid-a.yaml` (tune seeds, eval seeds); `pytest`.
4. Launch the run to `results/step-a/` in the background; watcher + `caffeinate`.
5. Report with `xp.report`, update the README results and limitations, STATUS/RESUME. Pushing needs a separate yes.

## Risks
- Both agents tie again: that is a valid result ("exploration bonus neither helps nor hurts on DoorKey"); step B (exploration-heavy world) is then the way to
  find where RND pays off.
- Size 12 stays near the floor; size 8 CIs may stay wide even at n = 30.
