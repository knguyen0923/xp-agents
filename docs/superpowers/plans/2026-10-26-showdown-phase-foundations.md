# Showdown Phase Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tested foundation of the Showdown generality phase: the approved spec amendment, disjoint team pools with a validator, a fixed structured feature encoder, and a doubles gymnasium env, all runnable without the server except one marked integration test.

**Architecture:** A new package `xp/showdown/` (never imported from `xp/protocol`) holds three small modules: `teams.py` (pools, disjointness check, Showdown validator call, a seeded per-battle team drawer), `encode.py` (one fixed-length float vector from a `poke-env` `DoubleBattle`) and `env.py` (a `DoublesEnv` subclass with that observation and a sparse win/loss reward). Everything but `env.py`'s battles is unit-tested offline; the single server-backed test carries a `showdown` marker and is excluded from the default `pytest` run.

**Tech Stack:** Python 3.9 (floor), `poke-env==0.10.0` (gymnasium, pettingzoo), numpy, pytest, a local Pokemon Showdown server on Node >= 22 (outside the repo, `~/xp-showdown-spike/`).

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md` section 5.6 (amended in Task 1). Evidence and measured numbers: `docs/spikes/showdown_spike.md`. Spike plan and log: `docs/superpowers/plans/2026-10-26-showdown-spike.md`.

## Global Constraints

- Python floor 3.9; every module starts with `from __future__ import annotations`.
- `xp/protocol/` is phase-agnostic and must never import from a phase package (`xp.bandit`, `xp.minigrid`, `xp.showdown`).
- Tune and eval pools are disjoint; violating this raises `ProtocolError` (`xp.errors`).
- The feature encoding is identical for both agents and fixed before tuning.
- Never connect to the public Showdown ladder or any live server; the server listens on `127.0.0.1` only.
- Tests must not require the server; server-backed tests carry the `showdown` marker and are excluded from the default `pytest` run.
- Do not change `xp/protocol/`, `xp/bandit/`, `xp/minigrid/`, `requirements.txt`, `requirements-minigrid.txt` or any existing config.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Pushes need the user's yes each time. `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md` (commit other files first, then a STATUS-only marker commit).

## What this plan does not cover (Plan B, written after Task 4)

The PPO learner for this env, the throughput measurement, the "plain PPO lands between 20% and 80% against some rung" check, the budget cap S, the tuning grids, `configs/showdown.yaml`, `xp/showdown/phase.py`, the `xp.run` wiring and the report. They are deliberately a second plan: the spec fixes S by measuring default-entropy Kairi on the tune seeds, the learner's action handling (mask or not) depends on how this env behaves under training, and the user must approve the cap before any RND result is read. Plan B also moves `DamageHeuristicPlayer` from `scripts/spikes/showdown_opponents.py` into `xp/showdown/opponents.py` as the ladder's top rung.

## File structure

| Path | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-10-22-xp-design.md` | modified: section 5.6 amendment, status text |
| `CLAUDE.md`, `STATUS.md`, `RESUME.md` | modified: scope, architecture, handoff |
| `pyproject.toml`, `requirements-showdown.txt`, `.github/workflows/ci.yml` | modified or created: dependency, marker, CI install |
| `configs/showdown_teams/tune/`, `configs/showdown_teams/eval/` | created: the six user-supplied teams |
| `xp/showdown/__init__.py`, `teams.py`, `encode.py`, `env.py` | created |
| `tests/test_showdown_teams.py`, `test_showdown_encode.py`, `test_showdown_env.py` | created |

---

### Task 1: Record the approval, amend spec 5.6, update project instructions

**Files:**
- Modify: `docs/superpowers/specs/2026-10-22-xp-design.md` (lines 21, 24, 169 and the end of section 5.6)
- Modify: `CLAUDE.md` (the Scope bullet, Architecture, Commands)
- Modify: `STATUS.md`, `RESUME.md`

**Interfaces:**
- Produces: the written authority and the design decisions that Tasks 2 to 4 and Plan B implement.

- [ ] **Step 1: Update the three status mentions in the spec**

```bash
python3 - <<'EOF'
p = "docs/superpowers/specs/2026-10-22-xp-design.md"
s = open(p).read()
pairs = [
    ("Pokémon Showdown (5.6, specified but not approved for building)", "Pokémon Showdown (5.6, spiked; building approved 2026-10-26)"),
    ("a Showdown implementation until the user approves building it (5.6)", "a Showdown implementation beyond the scope of the 5.6 amendment"),
    ("### 5.6 Phase 4: Pokémon Showdown generality test (specified, not approved for building)", "### 5.6 Phase 4: Pokémon Showdown generality test (spiked; building approved 2026-10-26)"),
]
for old, new in pairs:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w").write(s)
EOF
```

- [ ] **Step 2: Append the amendment to the end of section 5.6** (insert directly before the line `## 6. Statistics and outputs`)

```markdown
*Amendment (2026-10-26; building approved by the user after the spike, `docs/spikes/showdown_spike.md`).* All four spike gates passed. The decisions below replace the (verify) items above; anything not listed keeps the text above.

- **Format and teams.** `[Gen 9 Champions] VGC 2026 Reg M-C`, format ID `gen9championsvgc2026regmc` (doubles, bring 6 pick 4, level 50, Open Team Sheets). The user supplied six teams, stored in `configs/showdown_teams/{tune,eval}/` (3 tune, 3 eval). Every stage draws teams only from its own pool, for both sides of a battle. Tune and eval pools share no team (`ProtocolError` otherwise), but species repeat across pools. Every team is checked with the Showdown validator before any run, because `poke-env` hangs silently on a rejected team.
- **Team preview.** `poke-env`'s default random pick of 4 (and lead pair) for both sides. Learning team preview is out of scope.
- **Observation and reward.** `xp.showdown.encode.encode_battle`: a flat float32 vector of fixed length (own and opponent: 2 actives plus up to 4 others, each with HP fraction, status, boosts, types, base stats, Mega flag and 4 move feature blocks; plus weather, terrain, side conditions and turn). It uses only public enums and numbers, never a species id, so unseen species encode. Identical for both agents and frozen before tuning. Reward is sparse: +1 win, -1 loss, 0 otherwise. Metric: win rate.
- **Actions.** poke-env's `MultiDiscrete([107, 107])` joint action per turn. Whether to mask invalid actions or let poke-env replace them (`strict=False`) is decided in the learner plan from measurements, and fixed identically for both agents before tuning.
- **Seeds.** A client cannot seed server battles (measured: 40 identical battles gave 12 distinct outcomes), so there are no battle-seed pools. Disjointness rests on the team pools. The unit of resampling is the independent training seed, as in 5.2 and 5.4; battles are averaged within a seed.
- **Opponent ladder.** Random, MaxBasePower and `DamageHeuristicPlayer` (with Mega). The familiar opponent is the rung where default-entropy Kairi's win rate after the budget is between 20% and 80%; that check is the first step of the learner plan. If no rung qualifies, the phase stops and is written up.
- **Budget.** S is measured as in 5.4 (default-entropy Kairi on the tune seeds, capped). **The cap is not set here: the user approves it, after a PPO throughput measurement and before any RND result is read.** Scripted-only throughput (260 decisions/s, one client) is an upper bound and implies about 60 h for 56 runs at 1M decisions each.
- **Tests.** Server-backed tests carry the `showdown` marker and are excluded from the default `pytest` run; everything else runs offline in CI.
- **Limitations.** Three teams per pool make outcomes noisy and tuned settings team-specific; no battle seeding; scripted opponents play short games (4 to 6 turns) that a learned agent will not.
```

- [ ] **Step 3: Update `CLAUDE.md`**

```bash
python3 - <<'EOF'
p = "CLAUDE.md"
s = open(p).read()
old = "Showdown is specified (spec 5.6) but not approved for building; do not start it until the user says so."
new = ("Showdown was spiked (`docs/spikes/showdown_spike.md`) and building was approved by the user on 2026-10-26 under the spec 5.6 amendment; "
       "the budget cap S needs the user's approval before any RND result is read.")
assert s.count(old) == 1
s = s.replace(old, new)
old = "- `xp/stats/` and `xp/plots/`"
new = ("- `xp/showdown/` is the Pokemon Showdown phase (`poke-env` doubles env, fixed feature encoder, team pools). It needs a local Showdown server\n"
       "  (outside the repo) only for tests marked `showdown`, which the default `pytest` run excludes. Never import it from `xp.protocol`.\n"
       "- `xp/stats/` and `xp/plots/`")
assert s.count(old) == 1
s = s.replace(old, new.replace("\\n", "\n"))
old = "Full runs go to Colab"
new = ("Showdown tests that need the server: `XP_NODE=<node 22+ binary> pytest -m showdown` (server started as in `RESUME.md`; "
       "`pip install -r requirements-showdown.txt -e .` first).\n\nFull runs go to Colab")
assert s.count(old) == 1
s = s.replace(old, new.replace("\\n", "\n"))
open(p, "w").write(s)
EOF
git diff --stat
```

Expected: `CLAUDE.md` and the spec changed, no assertion error.

- [ ] **Step 4: Update `STATUS.md` and `RESUME.md`**

In `STATUS.md`, replace the Showdown bullet with: `- **Showdown**: spiked and building approved 2026-10-26 (spec 5.6 amendment). Foundations plan: docs/superpowers/plans/2026-10-26-showdown-phase-foundations.md; the learner/phase plan (Plan B) follows it.` In `RESUME.md`, replace the Showdown bullet's first sentence with the same status and add the line `The next step is the first unchecked task in the foundations plan.`

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-10-22-xp-design.md CLAUDE.md STATUS.md RESUME.md
git commit -m "docs: approve Showdown build, amend spec 5.6

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Team pools, validator and packaging

**Files:**
- Create: `xp/showdown/__init__.py` (empty), `xp/showdown/teams.py`, `tests/test_showdown_teams.py`, `requirements-showdown.txt`
- Create: `configs/showdown_teams/tune/tune1.txt`, `tune2.txt`, `tune3.txt`, `configs/showdown_teams/eval/eval1.txt`, `eval2.txt`, `eval3.txt`
- Modify: `pyproject.toml` (optional dependency, pytest marker), `.github/workflows/ci.yml`

**Interfaces:**
- Produces (Task 4 and Plan B rely on these):
  - `load_pool(directory) -> Dict[str, str]`: team name (file stem) to export text, sorted, raises `ProtocolError` if empty.
  - `assert_disjoint(tune: Dict[str, str], evals: Dict[str, str]) -> None`: raises `ProtocolError`.
  - `validate_team(text: str, format_id: str, showdown_dir, node: str = "node") -> None`: raises `ProtocolError` with the validator's message.
  - `PoolTeambuilder(teams: Dict[str, str], seed: int)`: a `poke_env.teambuilder.Teambuilder`; `.drawn: List[str]` logs the team names drawn.

- [ ] **Step 1: Packaging, marker and the team files**

```bash
mkdir -p xp/showdown configs/showdown_teams/tune configs/showdown_teams/eval
touch xp/showdown/__init__.py
for n in 1 2 3; do cp ~/xp-showdown-spike/teams/tune$n.txt configs/showdown_teams/tune/; cp ~/xp-showdown-spike/teams/eval$n.txt configs/showdown_teams/eval/; done
printf -- '-r requirements.txt\npoke-env==0.10.0\n' > requirements-showdown.txt
python3 - <<'EOF'
p = "pyproject.toml"
s = open(p).read()
old = 'minigrid = ["torch>=2.0", "minigrid>=3.0"]'
assert s.count(old) == 1
s = s.replace(old, old + '\nshowdown = ["poke-env==0.10.0"]')
old = 'addopts = "-q"'
assert s.count(old) == 1
s = s.replace(old, 'addopts = "-q -m \'not showdown\'"\nmarkers = ["showdown: needs a running local Pokemon Showdown server (excluded from the default run)"]')
open(p, "w").write(s)
p = ".github/workflows/ci.yml"
s = open(p).read()
old = "      - run: pip install -r requirements-minigrid.txt -e . --extra-index-url https://download.pytorch.org/whl/cpu\n"
assert s.count(old) == 1
s = s.replace(old, old + "      - run: pip install -r requirements-showdown.txt\n")
open(p, "w").write(s)
EOF
source .venv/bin/activate && pip install -r requirements-showdown.txt -e . 2>&1 | tail -2
```

Expected: the install ends without an error. If `poke-env==0.10.0` fails to install, stop and report (two failed attempts at an install step means stop).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_showdown_teams.py`:

```python
from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("poke_env")

from xp.errors import ProtocolError  # noqa: E402
from xp.showdown.teams import PoolTeambuilder, assert_disjoint, load_pool, normalise, validate_team  # noqa: E402

A = "Garchomp @ Choice Scarf\nAbility: Rough Skin\nLevel: 50\nAdamant Nature\n- Earthquake\n- Rock Slide\n"
B = "Rillaboom @ Miracle Seed\nAbility: Grassy Surge\nLevel: 50\nJolly Nature\n- Fake Out\n- Wood Hammer\n"


def test_load_pool_reads_sorted_txt_files(tmp_path):
    (tmp_path / "b.txt").write_text(B)
    (tmp_path / "a.txt").write_text(A)
    (tmp_path / "notes.md").write_text("ignored")
    assert list(load_pool(tmp_path)) == ["a", "b"]


def test_load_pool_empty_directory_raises(tmp_path):
    with pytest.raises(ProtocolError, match="no team files"):
        load_pool(tmp_path)


def test_assert_disjoint_accepts_distinct_pools():
    assert_disjoint({"t1": A}, {"e1": B})


def test_assert_disjoint_rejects_same_text_even_with_whitespace_noise():
    with pytest.raises(ProtocolError, match="also appear"):
        assert_disjoint({"t1": A}, {"e1": A.replace("\n", "  \n") + "\n\n"})


def test_assert_disjoint_rejects_same_name():
    with pytest.raises(ProtocolError, match="share team names"):
        assert_disjoint({"x": A}, {"x": B})


def test_normalise_ignores_trailing_space_and_outer_blank_lines():
    assert normalise("\n" + A.replace("\n", "   \n") + "\n\n") == normalise(A)


def test_pool_teambuilder_draws_only_from_its_pool_and_logs_names():
    tb = PoolTeambuilder({"t1": A, "t2": B}, seed=3)
    for _ in range(30):
        packed = tb.yield_team()
        assert packed
    assert set(tb.drawn) <= {"t1", "t2"} and len(tb.drawn) == 30


def test_pool_teambuilder_is_deterministic_per_seed():
    one = PoolTeambuilder({"t1": A, "t2": B}, seed=5)
    two = PoolTeambuilder({"t1": A, "t2": B}, seed=5)
    for _ in range(10):
        one.yield_team(), two.yield_team()
    assert one.drawn == two.drawn


def test_pool_teambuilder_requires_a_team():
    with pytest.raises(ProtocolError):
        PoolTeambuilder({}, seed=0)


def _fake_run(returncode, stdout="", stderr=""):
    def run(cmd, **kwargs):
        run.cmd, run.kwargs = cmd, kwargs
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return run


def test_validate_team_passes_on_silent_success(monkeypatch, tmp_path):
    run = _fake_run(0)
    monkeypatch.setattr(subprocess, "run", run)
    validate_team(A, "gen9championsvgc2026regmc", tmp_path)
    assert run.cmd[-2:] == ["validate-team", "gen9championsvgc2026regmc"] and run.kwargs["input"] == A


def test_validate_team_raises_with_the_validator_message(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run(1, stdout="Pikachu's move Splash does not exist in Gen 9."))
    with pytest.raises(ProtocolError, match="Splash"):
        validate_team(A, "gen9championsvgc2026regmc", tmp_path)


def test_validate_team_raises_on_output_even_with_exit_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _fake_run(0, stdout="You must bring at least 6 Pokémon"))
    with pytest.raises(ProtocolError, match="at least 6"):
        validate_team(A, "gen9championsvgc2026regmc", tmp_path)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_showdown_teams.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.teams'`.

- [ ] **Step 4: Write the implementation**

Create `xp/showdown/teams.py`:

```python
from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Dict, List

from poke_env.teambuilder import Teambuilder

from xp.errors import ProtocolError


def normalise(text: str) -> str:
    """Team identity ignores trailing spaces and blank-line differences."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def load_pool(directory) -> Dict[str, str]:
    """name (file stem) -> export text, for every *.txt in `directory`, sorted by name."""
    paths = sorted(Path(directory).glob("*.txt"))
    if not paths:
        raise ProtocolError(f"no team files in {directory}")
    return {p.stem: p.read_text() for p in paths}


def assert_disjoint(tune: Dict[str, str], evals: Dict[str, str]) -> None:
    """Tune and eval pools must share no team (same text up to whitespace) and no file name."""
    same_name = set(tune) & set(evals)
    if same_name:
        raise ProtocolError(f"tune and eval pools share team names {sorted(same_name)}")
    seen = {normalise(t) for t in tune.values()}
    dup = [n for n, t in evals.items() if normalise(t) in seen]
    if dup:
        raise ProtocolError(f"eval teams {dup} also appear in the tune pool")


def validate_team(text: str, format_id: str, showdown_dir, node: str = "node") -> None:
    """Run the Showdown validator on one team. `poke-env` hangs silently on a rejected team, so call this first."""
    proc = subprocess.run([node, "pokemon-showdown", "validate-team", format_id], input=text, cwd=str(showdown_dir),
                          capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0 or out:
        raise ProtocolError(f"team rejected for {format_id}: {out or 'exit code ' + str(proc.returncode)}")


class PoolTeambuilder(Teambuilder):
    """Draws one team per battle, only from its own pool, with a seeded RNG. `drawn` logs the names."""

    def __init__(self, teams: Dict[str, str], seed: int):
        if not teams:
            raise ProtocolError("PoolTeambuilder needs at least one team")
        self.teams = dict(teams)
        self.names: List[str] = sorted(self.teams)
        self.drawn: List[str] = []
        self._rng = random.Random(seed)

    def yield_team(self) -> str:
        name = self._rng.choice(self.names)
        self.drawn.append(name)
        return self.join_team(self.parse_showdown_team(self.teams[name]))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_showdown_teams.py -v`
Expected: 12 passed.

- [ ] **Step 6: Check the real pools are disjoint, then run the whole default suite**

```bash
python -c "
from xp.showdown.teams import load_pool, assert_disjoint
t, e = load_pool('configs/showdown_teams/tune'), load_pool('configs/showdown_teams/eval')
assert_disjoint(t, e); print('disjoint', sorted(t), sorted(e))"
pytest
```

Expected: `disjoint ['tune1', 'tune2', 'tune3'] ['eval1', 'eval2', 'eval3']`, then the full suite passes (the 161 existing tests plus 12 new) with no server running.

- [ ] **Step 7: Commit**

```bash
git add xp/showdown/__init__.py xp/showdown/teams.py tests/test_showdown_teams.py configs/showdown_teams requirements-showdown.txt pyproject.toml .github/workflows/ci.yml
git commit -m "feat(showdown): team pools, validator call and per-battle team drawer

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fixed feature encoder

**Files:**
- Create: `xp/showdown/encode.py`, `tests/test_showdown_encode.py`

**Interfaces:**
- Consumes: nothing from Task 2.
- Produces: `OBS_DIM: int` (1811), `MON_DIM`, `MOVE_DIM`, `SLOTS`; `encode_move(move: Optional[Move]) -> np.ndarray`; `encode_pokemon(mon: Optional[Pokemon], can_mega: bool = False) -> np.ndarray`; `encode_battle(battle) -> np.ndarray` (float32, shape `(OBS_DIM,)`, values in [-2, 2]). `battle` needs `team`, `opponent_team` (dicts of `Pokemon`), `active_pokemon`, `opponent_active_pokemon`, `can_mega_evolve` (2-lists), `weather`, `fields`, `side_conditions`, `opponent_side_conditions` (dicts) and `turn`: a real `DoubleBattle` has all of them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_encode.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Weather  # noqa: E402

from xp.showdown import encode as E  # noqa: E402


def mon(species: str, *moves: str) -> Pokemon:
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def battle(own, opp, own_active, opp_active, can_mega=(False, False), weather=None):
    return SimpleNamespace(
        team={m.species + str(i): m for i, m in enumerate(own)},
        opponent_team={m.species + str(i): m for i, m in enumerate(opp)},
        active_pokemon=list(own_active), opponent_active_pokemon=list(opp_active), can_mega_evolve=list(can_mega),
        weather=weather or {}, fields={}, side_conditions={}, opponent_side_conditions={}, turn=3,
    )


def test_dimension_constants_are_consistent():
    assert E.OBS_DIM == 2 * E.SLOTS * E.MON_DIM + E.FIELD_DIM
    assert E.MON_DIM == 3 + len(E.STATUSES) + len(E.BOOSTS) + len(E.TYPES) + len(E.STATS) + 1 + E.N_MOVES * E.MOVE_DIM


def test_missing_pokemon_and_move_are_all_zero():
    assert E.encode_pokemon(None).shape == (E.MON_DIM,) and not E.encode_pokemon(None).any()
    assert E.encode_move(None).shape == (E.MOVE_DIM,) and not E.encode_move(None).any()


def test_hp_fraction_and_boosts_are_encoded_and_scaled():
    p = mon("garchomp")
    p._current_hp, p._max_hp = 50, 100
    p._boosts["atk"] = 6
    v = E.encode_pokemon(p)
    assert v[0] == pytest.approx(0.5)
    boost_start = 3 + len(E.STATUSES)
    assert v[boost_start + E.BOOSTS.index("atk")] == pytest.approx(1.0)


def test_types_are_two_hot_for_dual_types_and_base_stats_are_scaled():
    v = E.encode_pokemon(mon("garchomp"))
    type_start = 3 + len(E.STATUSES) + len(E.BOOSTS)
    assert v[type_start:type_start + len(E.TYPES)].sum() == 2
    stat_start = type_start + len(E.TYPES)
    assert v[stat_start + E.STATS.index("atk")] == pytest.approx(130 / 255)


def test_move_features_flag_protect_and_scale_power():
    protect, quake = E.encode_move(Move("protect", gen=9)), E.encode_move(Move("earthquake", gen=9))
    flags = slice(E.MOVE_DIM - 2, E.MOVE_DIM)
    assert protect[flags].tolist() == [1.0, 0.0]
    assert quake[0] == pytest.approx(100 / 150)
    assert quake[flags][1] == 1.0  # Earthquake hits every adjacent Pokemon


def test_a_species_never_seen_in_training_still_encodes_to_the_same_shape():
    for species in ("rillaboom", "sneasler", "garchomp"):
        assert E.encode_pokemon(mon(species, "protect")).shape == (E.MON_DIM,)


def test_encode_battle_shape_dtype_and_determinism():
    own = [mon("garchomp", "protect", "earthquake"), mon("rillaboom", "fakeout"), mon("sneasler"), mon("excadrill"), mon("gardevoir"), mon("hydreigon")]
    opp = [mon("gholdengo"), mon("pelipper")]
    b = battle(own, opp, own[:2], opp, can_mega=(True, False))
    out = E.encode_battle(b)
    assert out.shape == (E.OBS_DIM,) and out.dtype == np.float32
    assert np.array_equal(out, E.encode_battle(b))
    assert out.min() >= -2.0 and out.max() <= 2.0


def test_first_own_slot_is_the_first_active_with_its_mega_flag():
    own = [mon("garchomp", "protect"), mon("rillaboom")]
    b = battle(own, [mon("gholdengo")], own[:2], [mon("gholdengo")], can_mega=(True, False))
    out = E.encode_battle(b)
    assert np.array_equal(out[:E.MON_DIM], E.encode_pokemon(own[0], True))
    assert np.array_equal(out[E.MON_DIM:2 * E.MON_DIM], E.encode_pokemon(own[1], False))


def test_unrevealed_opponent_slots_are_zero_padded():
    own = [mon("garchomp"), mon("rillaboom")]
    opp = [mon("gholdengo")]
    out = E.encode_battle(battle(own, opp, own, [opp[0], None]))
    opp_side = out[E.SLOTS * E.MON_DIM:2 * E.SLOTS * E.MON_DIM].reshape(E.SLOTS, E.MON_DIM)
    assert opp_side[0].any() and not opp_side[1:].any()


def test_weather_sets_exactly_one_field_bit_in_the_weather_block():
    own = [mon("garchomp"), mon("rillaboom")]
    out = E.encode_battle(battle(own, [mon("gholdengo")], own, [None, None], weather={Weather.SUNNYDAY: 1}))
    field = out[2 * E.SLOTS * E.MON_DIM:]
    assert field[:len(Weather)].sum() == 1 and field[list(Weather).index(Weather.SUNNYDAY)] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_showdown_encode.py -v`
Expected: collection error `ImportError: cannot import name 'encode' from 'xp.showdown'`.

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/encode.py`:

```python
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from poke_env.battle import Field, Move, MoveCategory, Pokemon, PokemonType, SideCondition, Status, Weather

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
STATUSES = [s for s in Status if s is not Status.FNT]
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = ("accuracy", "atk", "def", "evasion", "spa", "spd", "spe")
SPREAD_TARGETS = {"allAdjacent", "allAdjacentFoes", "all"}
SLOTS = 6                      # per side: 2 actives, then up to 4 others
N_MOVES = 4
MOVE_DIM = 3 + len(TYPES) + 3 + 2          # power, accuracy, priority | type | category | is_protect, is_spread
MON_DIM = 3 + len(STATUSES) + len(BOOSTS) + len(TYPES) + len(STATS) + 1 + N_MOVES * MOVE_DIM
FIELD_DIM = len(Weather) + len(Field) + 2 * len(SideCondition) + 1
OBS_DIM = 2 * SLOTS * MON_DIM + FIELD_DIM


def _onehot(item, order: list) -> np.ndarray:
    out = np.zeros(len(order), dtype=np.float32)
    if item in order:
        out[order.index(item)] = 1.0
    return out


def encode_move(move: Optional[Move]) -> np.ndarray:
    if move is None:
        return np.zeros(MOVE_DIM, dtype=np.float32)
    cats = [MoveCategory.PHYSICAL, MoveCategory.SPECIAL, MoveCategory.STATUS]
    return np.concatenate([
        np.array([move.base_power / 150.0, float(move.accuracy), move.priority / 5.0], dtype=np.float32),
        _onehot(move.type, TYPES),
        _onehot(move.category, cats),
        np.array([float(move.id in ("protect", "detect", "spikyshield", "banefulbunker", "kingsshield")),
                  float(move.entry.get("target") in SPREAD_TARGETS)], dtype=np.float32),
    ])


def encode_pokemon(mon: Optional[Pokemon], can_mega: bool = False) -> np.ndarray:
    """Fixed-size features from the public enums and numbers only: no species id, so unseen species still encode."""
    if mon is None:
        return np.zeros(MON_DIM, dtype=np.float32)
    moves = list(mon.moves.values())[:N_MOVES]
    moves += [None] * (N_MOVES - len(moves))
    base = np.array([mon.base_stats.get(k, 0) / 255.0 for k in STATS], dtype=np.float32)
    boosts = np.array([mon.boosts.get(k, 0) / 6.0 for k in BOOSTS], dtype=np.float32)
    types = np.clip(sum((_onehot(t, TYPES) for t in mon.types if t is not None), np.zeros(len(TYPES), dtype=np.float32)), 0, 1)
    return np.concatenate([
        np.array([float(mon.current_hp_fraction), float(mon.fainted), float(mon.active)], dtype=np.float32),
        _onehot(mon.status, STATUSES), boosts, types, base,
        np.array([float(can_mega)], dtype=np.float32),
        *[encode_move(m) for m in moves],
    ])


def _side(team: Iterable[Pokemon], actives: List[Optional[Pokemon]], mega_flags: List[bool]) -> np.ndarray:
    active_ids = {id(m) for m in actives if m is not None}
    others = [m for m in team if id(m) not in active_ids][: SLOTS - 2]
    rows = [encode_pokemon(m, f) for m, f in zip(actives, mega_flags)]
    rows += [encode_pokemon(None)] * (2 - len(rows))
    rows += [encode_pokemon(m) for m in others]
    rows += [encode_pokemon(None)] * (SLOTS - len(rows))
    return np.concatenate(rows)


def _conditions(table: dict, order: list) -> np.ndarray:
    return np.array([float(k in table) for k in order], dtype=np.float32)


def encode_battle(battle) -> np.ndarray:
    """Flat float32 vector of length OBS_DIM for a DoubleBattle (own side, opponent side, field)."""
    own = _side(battle.team.values(), list(battle.active_pokemon), list(battle.can_mega_evolve))
    opp = _side(battle.opponent_team.values(), list(battle.opponent_active_pokemon), [False, False])
    field = np.concatenate([
        _conditions(battle.weather, list(Weather)), _conditions(battle.fields, list(Field)),
        _conditions(battle.side_conditions, list(SideCondition)),
        _conditions(battle.opponent_side_conditions, list(SideCondition)),
        np.array([min(battle.turn, 40) / 40.0], dtype=np.float32),
    ])
    out = np.concatenate([own, opp, field]).astype(np.float32)
    assert out.shape == (OBS_DIM,)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_showdown_encode.py -v`
Expected: 10 passed. (Two bugs were found while preparing this plan and are fixed in the code above: `deduced_target` is an enum, so spread moves are detected from `move.entry["target"]`; and move priority reaches -1.4 after scaling, so the observation bounds in Task 4 are [-2, 2].)

- [ ] **Step 5: Commit**

```bash
git add xp/showdown/encode.py tests/test_showdown_encode.py
git commit -m "feat(showdown): fixed-length battle feature encoder

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Doubles env and the server-backed test

**Files:**
- Create: `xp/showdown/env.py`, `tests/test_showdown_env.py`

**Interfaces:**
- Consumes: `encode_battle`, `OBS_DIM` (Task 3); `PoolTeambuilder`, `load_pool`, `validate_team` (Task 2).
- Produces: `XPDoublesEnv(DoublesEnv)`: construct with `battle_format`, `team=PoolTeambuilder(...)`, `strict=False`, two `AccountConfiguration`s; wrap with `SingleAgentWrapper(env, opponent_player)` for a gymnasium env with `observation_space = Box(-2, 2, (OBS_DIM,))` and `action_space = MultiDiscrete([107, 107])`. Reward +1 / -1 at the end of a battle, else 0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_showdown_env.py`:

```python
"""Needs a running local Showdown server (see docs/spikes/showdown_spike.md). Run with: pytest -m showdown

XP_SHOWDOWN_DIR: the Showdown checkout (default ~/xp-showdown-spike/pokemon-showdown).
XP_NODE: a Node >= 22 binary for the team validator (default `node`; the system Node may be older).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.environment import SingleAgentWrapper  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402

from xp.showdown.encode import OBS_DIM  # noqa: E402
from xp.showdown.env import XPDoublesEnv  # noqa: E402
from xp.showdown.teams import PoolTeambuilder, load_pool, validate_team  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"
TEAMS = Path(__file__).resolve().parent.parent / "configs" / "showdown_teams"
SHOWDOWN_DIR = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))

NODE = os.environ.get("XP_NODE", "node")

pytestmark = pytest.mark.showdown


@pytest.fixture(scope="module")
def tune_pool():
    if not SHOWDOWN_DIR.exists():
        pytest.skip(f"no Showdown checkout at {SHOWDOWN_DIR} (set XP_SHOWDOWN_DIR)")
    pool = load_pool(TEAMS / "tune")
    for text in pool.values():  # a rejected team makes poke-env hang with no error, so validate first
        validate_team(text, FORMAT, SHOWDOWN_DIR, node=NODE)
    return pool


def test_random_play_runs_to_the_end_with_the_xp_observation_and_sparse_reward(tune_pool):
    suffix = str(random.randrange(10**5))
    team = PoolTeambuilder(tune_pool, seed=0)
    env = XPDoublesEnv(
        battle_format=FORMAT, team=team, strict=False, log_level=40,
        account_configuration1=AccountConfiguration("xa" + suffix, None),
        account_configuration2=AccountConfiguration("xb" + suffix, None),
    )
    wrapped = SingleAgentWrapper(env, RandomPlayer(battle_format=FORMAT, start_listening=False))
    try:
        assert wrapped.observation_space.shape == (OBS_DIM,)
        finals = []
        for _ in range(3):
            obs, _ = wrapped.reset()
            assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
            done, reward = False, 0.0
            while not done:
                obs, reward, term, trunc, _ = wrapped.step(wrapped.action_space.sample())
                assert wrapped.observation_space.contains(obs)
                done = term or trunc
            finals.append(reward)
        assert all(r in (1.0, -1.0) for r in finals)
        assert set(team.drawn) <= set(tune_pool)
    finally:
        wrapped.close()
```

- [ ] **Step 2: Start the server and run the test to verify it fails**

```bash
export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH
(cd ~/xp-showdown-spike/pokemon-showdown && nohup node pokemon-showdown start --no-security > ../server.log 2>&1 &)
until lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; do sleep 2; done
XP_NODE=$(which node) pytest -m showdown -v
```

Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.env'`. Account names are short, alphanumeric and random per run (Showdown strips underscores and truncates at 18 characters).

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/env.py`:

```python
from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import DoublesEnv

from xp.showdown.encode import OBS_DIM, encode_battle


class XPDoublesEnv(DoublesEnv):
    """Doubles env with the fixed XP feature vector and a sparse reward: +1 win, -1 loss, 0 otherwise.

    Both sides' teams come from the `team` argument (pass a `PoolTeambuilder`), and team preview uses poke-env's
    default random pick of 4 for both sides. Pair with `SingleAgentWrapper(env, opponent)` for a gymnasium env.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        space = Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)  # move priority reaches -7 / 5 = -1.4
        self.observation_spaces = {agent: space for agent in self.possible_agents}

    def embed_battle(self, battle) -> np.ndarray:
        return encode_battle(battle)

    def calc_reward(self, battle) -> float:
        if battle.won:
            return 1.0
        if battle.lost:
            return -1.0
        return 0.0
```

- [ ] **Step 4: Run the server test and the default suite**

```bash
XP_NODE=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin/node pytest -m showdown -v
pytest
```

Expected: `1 passed, N deselected` for the first command (about 2 s); the second passes with the server test deselected. If the first command hangs, a team was rejected: run `node pokemon-showdown validate-team gen9championsvgc2026regmc < team.txt` in `~/xp-showdown-spike/pokemon-showdown` for each file.

- [ ] **Step 5: Stop the server, update the handoff, commit**

```bash
pkill -f "pokemon-showdown start"
git add xp/showdown/env.py tests/test_showdown_env.py
git commit -m "feat(showdown): doubles env with the fixed observation and a sparse win reward

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

Then update `RESUME.md` and `STATUS.md` (foundations done; next is Plan B, starting with a PPO throughput measurement on this env), commit those, and make the STATUS-only marker commit.

---

## Self-review

**Spec coverage (5.6 and its amendment):** team pools, disjointness, validation and the `showdown` test marker: Task 2. Fixed identical observation, reward and metric definition: Task 3 and Task 4. Approval, format, seeds, team preview, ladder rule, budget rule and limitations: Task 1. Not covered here, by design (Plan B): agents (PPO and PPO plus RND on this env), action-mask decision, ladder rung selection, budget S and its cap, tuning grids, protocol wiring and the report.

**Placeholder scan:** none. The two code blocks marked "Create" are complete files; both were executed against the scratch environment (22 offline tests and the server test passed, the pyproject marker was exercised).

**Type consistency:** `OBS_DIM`, `encode_battle`, `PoolTeambuilder.drawn`, `load_pool`, `validate_team(..., node=...)` have the same names and signatures in the tests, the modules and the Interfaces blocks.

**Known risks:** `poke-env==0.10.0` on the CI Python 3.12 matrix entry is untested (3.9 is); if that job fails, pin or skip it and report. The six team files are user content going into a public repository: the user has not yet confirmed that. Teams using Champions-only Mega forms need the `poke-env` data patch for server battles (`scripts/spikes/showdown_patch_poke_env.py`); the offline tests use base species only, but the Task 4 test does not pass on a fresh `poke-env` install until the patch is applied.
