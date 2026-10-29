# Showdown matchup probe (poke-env on a live doubles battle)

Script: `scripts/spikes/showdown_matchup_probe.py`. Run against the local server, format `gen9championsvgc2026regmc`,
tune pool, opponent `damage`, right after `reset()` (before any move). Two runs (different random teams); the second
run added an own-to-own control. Printed evidence is quoted below.

## 1. Opponent abilities and items at battle start

Items: not known. Run 1: `opp sneasler ... item unknown_item`, `opp gardevoir ... item unknown_item`.
Abilities: partly. Run 1 `opp sneasler ability None`, `opp gardevoir ability pickpocket`; run 2
`opp indeedeef ability None`, `opp hydreigon ability levitate`. A species with a single possible ability gets it
filled in (Hydreigon: levitate); species with several get `None` (Sneasler, Indeedee). Gardevoir `pickpocket` is not
one of its own abilities: the pool set has `Ability: Trace`, so this is Trace copying our Weavile's Pickpocket, an
ability revealed in battle (not a parsing quirk). Revealed abilities are real information; the unrevealed rest stays unknown. `moves` is `[]` for both foes. The opponent bench is also unknown:
`opp team keys ['p2: Sneasler', 'p2: Gardevoir']` (only the two leads, not 4 or 6). The team sheet is not exposed
in what I printed; I did not find it. Treat foe abilities and items as unknown unless revealed in battle.

## 2. Opponent stats

`None` for every stat of every foe: `stats {'hp': None, 'atk': None, 'def': None, 'spa': None, 'spd': None, 'spe': None}`,
with `hp 100 100` (percent). Own mons have real numbers: `salamence ... stats {'hp': 171, 'atk': 139, 'def': 100,
'spa': 162, 'spd': 101, 'spe': 167} hp 171 171`. So the base-stat fallback is needed for foes; own mons do not need it.

## 3. `calculate_damage(ident(a), ident(d), move, b)` with keys of `b.team` / `b.opponent_team`

Own -> foe raises, every time: `AssertionError: defender stats not defined` (run 1 `salamence -> sneasler`, all four
moves; run 2 `charizard -> indeedeef`, all four moves). Cause is the foe's `stats` being `None`, not the identifier form.
The identifier form works: own -> own control returned numbers, `CONTROL charizard -> gholdengo heatwave calc (116, 138)`,
`ancientpower (17, 20)`, status moves `(0, 0)`. Foe -> own was not exercised, since foes have no known moves
(`moves []`), so no attacker move object exists; with stats `None` it would also fail on the attacker side. I did not
try filling foe `stats` by hand; that is the obvious next experiment if the calculator is wanted.
Side note: `damage_multiplier(move)` returns a number for foes (type chart works without stats); it is meaningless on status moves.

## 4. Cost per call

Own -> own control: 0.67 ms (first call), 0.22 ms, 0.07 ms, 0.01 ms (status/zero-damage moves). Failing calls
0.02 to 0.11 ms. So a working call is well under 1 ms; 16 + 16 calls per decision is about 10 to 20 ms worst case.

## Conclusion

Update after Task 4: the calculator cannot run on foe stats as they arrive (`None`), but it does run on foes once their
stats are filled from base stats at the fallback spread (level 50, 0 EVs, 31 IVs, neutral nature; recomputed from the
current `base_stats` each call). Task 4 therefore uses the calculator for own -> foe and foe -> own estimates (foe moves
only once revealed). The earlier reading, "own -> own checks only", is out of date.
