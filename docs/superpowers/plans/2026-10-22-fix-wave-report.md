# Fix wave report

Status: DONE_WITH_CONCERNS (all items A1-A7, B1-B7 implemented; A4 probe shows tt_triggered still flagged on window lower edge on 3/3 sets, and tt_meta flagged at a corner with 500 tune seeds).

## Commits
- 32065af Replace CUSUM trigger with Page-Hinkley, shift grids, record spec amendments (A1, A2, A7 spec/CLAUDE.md)
- 94f6dbc Enforce equal grids, write run manifest, validate results in report (A3, A5, A6, B4)
- 99f8f51 Harden stats, tuner, evaluator and result store; strengthen tests (B1, B2, B3, B5, B6, B7)

## Per item
- A1: ChangeTriggered is Page-Hinkley (per-env cumulative mean of |reward-pred|, g=max(0,g+x-m-delta), fire on g>h, remaining=window, g reset, mean kept). Tests rewritten: starts with Kairi, fires on residual jump (after quiet stream), constant residual 0.9 never fires over 500 steps, returns to Kairi after exactly window steps (asserts who(2)==[1], who(3)==[1], who(4)==[0]). Config triggered grid/natural_limits as briefed (delta/h/window, window 2000 = horizon). Spec 5.1 amended. tiny_cfg unchanged (no test needed it).
- A2: kaiser.c and kairi.epsilon grids shifted as briefed; equal-size test still passes.
- A3: run_phase raises ProtocolError on unequal kairi/kaiser grid sizes before any work (test checks nothing is written).
- A4: see probe table below.
- A5: xp/report.py validate_results (phase arg vs cfg, single config hash == phase hash else ConfigMismatchError, every of 5 agents x sweep cell has exactly len(eval_seeds) rows else ProtocolError naming cells), run before require_final and before building anything. Tests: missing cell, short cell, wrong hash, phase mismatch.
- A6: run_phase writes <out>/<phase>/manifest.json (cfg, config_hash, git_commit via git rev-parse HEAD or null, python/numpy/pandas versions); existing manifest with different hash raises ConfigMismatchError. Test added.
- A7: spec section 3 (equal grids, world-vs-agent determinism), section 6 (crossover definition, no multiplicity correction) amended; CLAUDE.md claim corrected.
- B1-B3, B5-B7: done as briefed (shared helper checked_chunk_result in xp/protocol/tune.py used by score_params and evaluate_unit; atomic writes via temp file + os.replace; unit_done false for empty/header-only/short files). B5 test renamed/rewritten.
- B4: Path import removed from xp/protocol/phase.py. `numpy as np` in xp/bandit/phase.py is now USED (grid-size check, np.__version__ in manifest), so it was kept rather than removed.

TDD note: for A1, B2, B3, A3 pieces I wrote tests first and saw failures where run (A1 constant-residual test, A3 and A6 tests, B7 tests failed before implementation). For A5, B1, B2, B3, B6 the tests and implementation were written in the same step and not separately run red; B6 tests exercise already-correct agent code (they pass on the existing implementation, as intended).

## Final pytest
`87 passed, 14 warnings` (all warnings are third-party matplotlib/pyparsing deprecations).

## A4 probe (horizon 2000, K=10, sigma 1, jump_rate 0.001, real grids/natural limits, same code path: score_params + tune; scripts in scratchpad/a4/probe.py)
Tune sets are seeds 1000-1099, 2000-2099, 3000-3099 (outside both pools).

| set | agent | best params | boundary_dims |
|---|---|---|---|
| 1000 | kairi | eps 0.02, alpha 0.4 | none |
| 1000 | kaiser | gamma 0.95, c 0.01 | c (lower edge; noise) |
| 1000 | tt_triggered | delta 0.0, h 0.0, window 10 | window (lower edge, non-natural) |
| 1000 | tt_meta | gamma 0.95, c 0.5 | none |
| 2000 | kairi | eps 0.01, alpha 0.4 | none |
| 2000 | kaiser | gamma 0.98, c 0.1 | none |
| 2000 | tt_triggered | delta 0.4, h 0.0, window 10 | window |
| 2000 | tt_meta | gamma 0.95, c 0.1 | none |
| 3000 | kairi | eps 0.02, alpha 0.4 | none |
| 3000 | kaiser | gamma 0.98, c 0.1 | none |
| 3000 | tt_triggered | delta 0.4, h 0.0, window 10 | window |
| 3000 | tt_meta | gamma 1.0, c 0.2 | none |

Outcome: kairi flagged 0/3, kaiser 1/3 (c=0.01 on set 1000 only), tt_meta 0/3 -- goals met for those. tt_triggered is no longer degenerate (h=0, small delta, beats both singles: 0.159-0.175 vs kaiser 0.172-0.186), but it is flagged on window's LOWER edge (10) on 3/3 sets, so it would still be provisional and block the report.

Extra probe, 500 tune seeds (probe only; sets 0-499 and 1000-1499):
- kairi and kaiser: interior on both (eps 0.02/0.01, alpha 0.4; gamma 0.98, c 0.1), so 500 seeds removes the kaiser c=0.01 noise flag.
- tt_triggered: set 0-499 window 10 flagged; set 1000-1499 interior (delta 0.2, h 0, window 25).
- tt_meta: BOTH 500-seed sets pick gamma 0.8, c 0.05 (grid corner) and are flagged on gamma and c, whereas the 100-seed sets did not flag.

Extra probe, extended triggered window grid [1,2,5,10,25,50,100,2000] (100 seeds, sets 1000/2000/3000): optima window 10 (interior), 5, 2 respectively, all interior but the optimum drifts to very short windows (plateau between 2 and 10), consistent with the real optimum being a short burst of Kaiser control on every above-baseline residual.

## Concerns (brief's design not changed)
1. tt_triggered window lower edge: the briefed window grid [10,...,2000] has min 10 which is a non-natural lower edge; the optimum is at 2-10. Recommended (not applied): extend window grid downward (e.g. [1,2,5,10,25,50,100,2000]) with natural_limits window [1, 2000] (window 1 is the true minimum). Needs controller decision.
2. tt_meta: with 500 tune seeds the optimum is the (gamma 0.8, c 0.05) corner on both sets. If the real run uses 100 tune seeds it was interior on 3/3 here, but this suggests the meta grid should extend toward lower gamma and c (e.g. gamma down to 0.5-0.7, c down to 0.01) or tune seeds be increased; the real result may become provisional.
3. Kaiser c lower edge can flag from noise at 100 seeds (1/3 sets); 500 seeds removes it.

# Part 2 (controller rulings)
Applied: triggered window grid [1,2,5,10,25,100,2000] with natural_limits window [1,2000]; meta grid gamma [0.5,0.65,0.8,0.9,0.95,0.99,1.0], c [0.01,0.02,0.05,0.1,0.2,0.5,1.0]; seeds.tune [0,500]; spec 5.1 ("500 tune seeds") and plan Global Constraints (tune pool [0,500)) updated; CLAUDE.md does not state the pool size. No test depended on the old grids. Full suite: 87 passed (real-config validity and disjoint-pool checks pass).

## Probe with new grids, 500 tune seeds on disjoint sets (1000-1499, 2000-2499, 3000-3499)
| set | agent | best params | boundary_dims |
|---|---|---|---|
| 1000 | kairi | eps 0.01, alpha 0.4 | none |
| 1000 | kaiser | gamma 0.98, c 0.1 | none |
| 1000 | tt_triggered | delta 0.2, h 0, window 25 | none |
| 1000 | tt_meta | gamma 0.65, c 0.01 | c |
| 2000 | kairi | eps 0.02, alpha 0.4 | none |
| 2000 | kaiser | gamma 0.98, c 0.05 | none |
| 2000 | tt_triggered | delta 0.2, h 2.0, window 2000 | none |
| 2000 | tt_meta | gamma 0.65, c 0.01 | c |
| 3000 | kairi | eps 0.02, alpha 0.4 | none |
| 3000 | kaiser | gamma 0.98, c 0.1 | none |
| 3000 | tt_triggered | delta 0.0, h 1.0, window 25 | none |
| 3000 | tt_meta | gamma 0.8, c 0.01 | c |

Outcome: kairi, kaiser, tt_triggered not flagged on any set. tt_meta flagged on 3/3 sets on c's LOWER edge (0.01, non-natural; gamma interior at 0.65-0.8). Per the ruling, not widened further. The meta bandit wants a vanishing exploration bonus (c below 0.01, possibly c=0, i.e. greedy over block means with fast discounting). Decision needed: either accept c=0 as a natural limit (add 0.0 to the grid with natural_limits c [0.0, null]) or leave tt_meta provisional.

# Part 3
Meta grid c now [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] with natural_limits c [0.0, null]. Added test: MetaBandit(c=0.0) tries both agents then picks the higher block-mean agent. Full suite: 88 passed.
Probe (tt_meta only, 500 tune seeds, sets 1000-1499 / 2000-2499 / 3000-3499; same tuned-base-params path):
| set | best params | boundary_dims |
|---|---|---|
| 1000 | gamma 0.65, c 0.01 | none |
| 2000 | gamma 0.65, c 0.01 | none |
| 3000 | gamma 0.8, c 0.01 | none |
c=0.01 is interior on all sets; tt_meta is no longer flagged. With part 2 results, no agent is flagged on any probe set.
