# Showdown Learner and Budget Gate Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the masked PPO learner (Kairi) and the vectorised Showdown env, then run the budget gate: measured decisions per second and the win-rate curve of plain PPO against each scripted rung, which is what the user needs to approve the budget cap S.

**Architecture:** Four new modules in `xp/showdown/`: `patch.py` (makes `poke-env` know the Champions-only species, without which Mega battles hang), `masks.py` (legal-action masks for the two active slots, built from the battle and checked against `poke-env`'s own decoder), `opponents.py` + `vec_env.py` (the scripted ladder and `n` env processes stepped in parallel with auto-reset) and `ppo.py` (PPO with two masked 107-way heads). `gate.py` ties them together and writes one JSON per rung. No RND yet: Kaiser comes after the cap is approved (Plan C).

**Tech Stack:** Python 3.9, `poke-env==0.10.0`, PyTorch (CPU), numpy, multiprocessing (spawn), pytest; a local Showdown server on Node >= 22 outside the repo.

**Spec:** `docs/superpowers/specs/2026-10-22-xp-design.md` section 5.6 and its 2026-10-26 amendment. Foundations plan: `docs/superpowers/plans/2026-10-26-showdown-phase-foundations.md` (done). Spike: `docs/spikes/showdown_spike.md`.

## Global Constraints

- Python floor 3.9 (no nested same-type quotes in f-strings); every module starts with `from __future__ import annotations` (after any module docstring).
- `xp/protocol/` never imports `xp.showdown`. Do not change `xp/protocol/`, `xp/bandit/`, `xp/minigrid/`, existing configs or requirements files.
- Never connect to the public Showdown ladder; the server listens on `127.0.0.1` only.
- Tests must not require the server; server-backed tests carry the `showdown` marker.
- Tune seeds only here: the gate trains on seed 0 of the tune pool and uses tune teams. Eval seeds and eval teams are not touched.
- Account names for server logins are short, alphanumeric and fresh per run (Showdown strips underscores, truncates at 18 characters, and stale accounts from a hung run block reused names).
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Pushes need the user's yes each time. `STATUS_COMMIT` in `STATUS.md` must equal the commit whose diff to HEAD touches only `STATUS.md`.
- **The budget cap S is the user's decision.** This plan only measures and recommends; it sets no cap in any config.

## Why these pieces, and what was found while preparing this plan

- **Champions data patch (Task 1).** The repo's `poke-env` is unpatched. A Golisopod Mega raised `KeyError: 'golisopodmega'` inside `poke-env`'s message handler and the battle hung. The foundations server test passed only because its random play never Mega Evolved in 3 games; it could not have caught this. The patch data (29 KB, exported from the server's Champions mod) now ships in the repo with an idempotent applier and a guard test.
- **Masks are required (Task 2).** With `strict=False`, `poke-env` turns an invalid action into `DefaultBattleOrder()`, so the executed action differs from the sampled one and the policy gradient is wrong. 60 of the 107 actions (Z-move, Dynamax, Tera) are never legal in this format. The fast mask equals a brute-force decoder reference on 1,857 real battle states (0.26 ms per step; the brute-force version took 5.9 ms, as much as the whole decision budget).
- **Account names (Task 3).** Deriving the suffix from the training seed reused names across runs and hung the next run on stale state; the suffix now comes from the OS random source.
- **Win-rate curve (Task 4).** A first 40-update toy test reached only 0.6 and needed about 100 updates to reach 0.99 on three seeds; the test uses 100.
- **Smoke run.** Plain PPO, 8 env processes, 20,480 decisions against the Random rung: 303 decisions/s including updates, win rate 0.67 -> 0.85. Random is too easy to be the "middle" rung; the gate checks the others.

## Not in this plan (Plan C, after the user approves the cap S)

RND bonus and the Kaiser learner, matched entropy, the tuning grids, `configs/showdown.yaml`, `xp/showdown/phase.py`, the `xp.run` wiring, resumable per-job caching, the report and the bootstrap analysis. They wait for the cap because S fixes every run's length and the spec forbids reading RND results before it is set.

## File structure

| Path | Responsibility |
|---|---|
| `xp/showdown/patch.py`, `champions_patch.json` | make `poke-env` know the Champions-only species and moves |
| `xp/showdown/masks.py` | `action_masks`, `reference_masks`, `restrict_second` |
| `xp/showdown/opponents.py` | `LADDER`, `make_opponent`, `DamageHeuristicPlayer` |
| `xp/showdown/vec_env.py` | `ShowdownVecEnv`: `n` worker processes with auto-reset |
| `xp/showdown/ppo.py` | `ActorCritic`, `masked`, `gae`, `Learner` |
| `xp/showdown/gate.py` | `measure`, CLI `python -m xp.showdown.gate` |
| `tests/test_showdown_{patch,masks,ppo,gate,vec}.py`, `tests/toy_showdown.py` | offline tests, plus the marked server tests in `test_showdown_vec.py` |
| `docs/spikes/showdown_gate.md` | the gate result and the cap recommendation |

---

### Task 1: Champions data patch

**Files:**
- Create: `xp/showdown/patch.py`, `xp/showdown/champions_patch.json`, `tests/test_showdown_patch.py`
- Modify: `.github/workflows/ci.yml`, `CLAUDE.md`, `pyproject.toml` (package data)

**Interfaces:**
- Produces: `patch.PATCH: str` (path to the JSON), `patch.missing() -> Dict[str, List[str]]` (`{"species": [...], "moves": [...]}`, both empty once patched), `patch.apply() -> Dict[str, int]`. CLI: `python -m xp.showdown.patch`.

- [ ] **Step 1: Copy the patch data and write the failing test**

```bash
cp ~/xp-showdown-spike/patch_data.json xp/showdown/champions_patch.json
python3 -c "import json; d = json.load(open('xp/showdown/champions_patch.json')); print({k: len(v) for k, v in d.items()})"
```

Expected: `{'species': 54, 'moves': 2}`. Then create `tests/test_showdown_patch.py`:

```python
from __future__ import annotations

import json

import pytest

pytest.importorskip("poke_env")

from xp.showdown import patch  # noqa: E402


def test_patch_file_has_the_champions_megas_and_moves():
    data = json.load(open(patch.PATCH))
    assert {"golisopodmega", "dragonitemega", "floettemega", "garchompmegaz", "raichumegay"} <= set(data["species"])
    assert len(data["species"]) == 54 and len(data["moves"]) == 2


def test_installed_poke_env_data_has_every_patched_entry():
    """Fails on an unpatched install; fix with `python -m xp.showdown.patch`. Without the patch a Mega battle hangs."""
    assert patch.missing() == {"species": [], "moves": []}, "run: python -m xp.showdown.patch"


def test_apply_is_idempotent():
    patch.apply()
    first = patch.missing()
    patch.apply()
    assert first == patch.missing() == {"species": [], "moves": []}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_showdown_patch.py -v`
Expected: collection error `ImportError: cannot import name 'patch' from 'xp.showdown'`.

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/patch.py`:

```python
"""Merge the Champions-only species and moves into poke-env 0.10.0's bundled gen 9 data.

Why: poke-env's Pokedex lacks 54 species (new Megas and others) and 2 moves that the Showdown Champions mod defines.
A battle that touches one raises KeyError inside poke-env's message handler and the battle HANGS with no error.
The data in `champions_patch.json` was exported from the server's Champions mod (`Dex.mod('champions')`).
Idempotent; originals are kept as `*.orig`. Run: python -m xp.showdown.patch
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List

PATCH = os.path.join(os.path.dirname(__file__), "champions_patch.json")


def _paths() -> Dict[str, str]:
    import poke_env

    root = os.path.join(os.path.dirname(poke_env.__file__), "data", "static")
    return {"species": os.path.join(root, "pokedex", "gen9pokedex.json"), "moves": os.path.join(root, "moves", "gen9moves.json")}


def missing() -> Dict[str, List[str]]:
    """Patch keys not yet present in the installed poke-env data (empty lists once patched)."""
    patch = json.load(open(PATCH))
    out = {}
    for kind, path in _paths().items():
        have = json.load(open(path))
        out[kind] = sorted(set(patch[kind]) - set(have))
    return out


def apply() -> Dict[str, int]:
    """Merge the patch into the installed data files; returns the number of entries added per kind."""
    patch = json.load(open(PATCH))
    added = {}
    for kind, path in _paths().items():
        if not os.path.exists(path + ".orig"):
            shutil.copy(path, path + ".orig")
        data = json.load(open(path + ".orig"))
        before = len(data)
        data.update(patch[kind])
        with open(path, "w") as f:
            json.dump(data, f, sort_keys=True)
        added[kind] = len(data) - before
    return added


if __name__ == "__main__":
    print("added", apply(), "still missing", missing())
```

- [ ] **Step 4: Check the guard test fails on the unpatched install, then apply the patch**

```bash
source .venv/bin/activate
pytest tests/test_showdown_patch.py -v
python -m xp.showdown.patch
pytest tests/test_showdown_patch.py -v
```

Expected: first run `1 failed, 2 passed` (the installed-data test fails with `run: python -m xp.showdown.patch`; the unpatched `.venv` in this repo is the reason this task exists), the patch prints `added {'species': 54, 'moves': 2} still missing {'species': [], 'moves': []}`, second run `3 passed`. If the first run already passes, the install was patched earlier; that is fine.

- [ ] **Step 5: CI, package data and the Commands note**

```bash
python3 - <<'EOF'
p = ".github/workflows/ci.yml"
s = open(p).read()
old = "      - run: pip install -r requirements-showdown.txt\n"
assert s.count(old) == 1
s = s.replace(old, old + "      - run: python -m xp.showdown.patch\n")
open(p, "w").write(s)
p = "pyproject.toml"
s = open(p).read()
old = '[tool.setuptools.packages.find]\ninclude = ["xp*"]\n'
assert s.count(old) == 1
s = s.replace(old, old + '\n[tool.setuptools.package-data]\n"xp.showdown" = ["champions_patch.json"]\n')
open(p, "w").write(s)
p = "CLAUDE.md"
s = open(p).read()
old = "Showdown tests that need the server:"
assert s.count(old) == 1
s = s.replace(old, "After installing `poke-env`, run `python -m xp.showdown.patch` once (it lacks the Champions-only species; without the patch a Mega battle hangs, and a guard test fails).\nShowdown tests that need the server:")
open(p, "w").write(s)
EOF
pytest
```

Expected: the full default suite passes.

- [ ] **Step 6: Commit**

```bash
git add xp/showdown/patch.py xp/showdown/champions_patch.json tests/test_showdown_patch.py .github/workflows/ci.yml pyproject.toml CLAUDE.md
git commit -m "feat(showdown): ship the Champions data patch for poke-env, with a guard test

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Action masks

**Files:**
- Create: `xp/showdown/masks.py`, `tests/test_showdown_masks.py`

**Interfaces:**
- Consumes: a `DoubleBattle` (fields `active_pokemon`, `force_switch`, `available_switches`, `available_moves`, `trapped`, `team`, `can_mega_evolve`, `get_possible_showdown_targets`).
- Produces: `N_ACTIONS = 107`, `PASS = 0`, `MEGA_ACTIONS: frozenset`, `action_masks(battle) -> np.ndarray` (bool, shape `(2, 107)`), `reference_masks(battle)` (same shape, slow, asks `poke-env`'s decoder), `restrict_second(mask1, first_action) -> np.ndarray`, `slot_needs_action(battle, pos) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_masks.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Status  # noqa: E402

from xp.showdown.masks import MEGA_ACTIONS, N_ACTIONS, PASS, action_masks, restrict_second  # noqa: E402


def mon(species, *moves):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def targets(move, user):
    return [0] if move.id == "protect" else [1, 2]  # Earthquake-like spread moves list both foes in this stub


def battle(actives, team, switches=((), ()), force=(False, False), mega=(False, False), trapped=(False, False)):
    return SimpleNamespace(
        active_pokemon=list(actives), team={f"p{i}": m for i, m in enumerate(team)}, force_switch=list(force),
        available_switches=[list(s) for s in switches], trapped=list(trapped), can_mega_evolve=list(mega),
        available_moves=[list(a.moves.values()) if a is not None else [] for a in actives],
        get_possible_showdown_targets=targets,
    )


def legal(mask_row):
    return set(np.flatnonzero(mask_row).tolist())


def test_slot_with_no_active_or_a_fainted_pokemon_can_only_pass():
    a = mon("garchomp", "protect")
    dead = mon("rillaboom", "protect")
    dead._status = Status.FNT
    m = action_masks(battle([a, dead], [a, dead]))
    assert legal(m[1]) == {PASS}
    m = action_masks(battle([a, None], [a]))
    assert legal(m[1]) == {PASS}


def test_moves_map_to_seven_plus_five_k_plus_target_and_mega_adds_twenty():
    a, b = mon("garchomp", "protect", "earthquake"), mon("rillaboom", "protect")
    m = action_masks(battle([a, b], [a, b]))
    assert legal(m[0]) == {9, 15, 16}                      # protect (target 0) and earthquake (targets 1, 2)
    m = action_masks(battle([a, b], [a, b], mega=(True, False)))
    assert legal(m[0]) == {9, 15, 16, 29, 35, 36}
    assert legal(m[1]) == {9}                              # slot 1 cannot Mega


def test_switch_actions_are_one_plus_the_team_position_of_each_available_switch():
    a, b, c, d = mon("garchomp", "protect"), mon("rillaboom", "protect"), mon("sneasler"), mon("excadrill")
    m = action_masks(battle([a, b], [a, b, c, d], switches=([c, d], [c, d])))
    assert {3, 4} <= legal(m[0]) and not ({1, 2} & legal(m[0]))


def test_trapped_slot_cannot_switch():
    a, b, c = mon("garchomp", "protect"), mon("rillaboom", "protect"), mon("sneasler")
    m = action_masks(battle([a, b], [a, b, c], switches=([c], [c]), trapped=(True, False)))
    assert not ({1, 2, 3} & legal(m[0])) and 3 in legal(m[1])


def test_forced_switch_on_one_slot_leaves_the_other_slot_passing():
    a, b, c = mon("garchomp", "protect"), mon("rillaboom", "protect"), mon("sneasler")
    m = action_masks(battle([a, b], [a, b, c], switches=([c], [c]), force=(True, False)))
    assert legal(m[0]) == {3} and legal(m[1]) == {PASS}


def test_restrict_second_removes_a_second_mega_and_a_duplicate_switch():
    row = np.zeros(N_ACTIONS, dtype=bool)
    row[[9, 29, 35, 3, 4]] = True
    after_mega = restrict_second(row, 29)
    assert legal(after_mega) == {9, 3, 4} and not (legal(after_mega) & MEGA_ACTIONS)
    after_switch = restrict_second(row, 3)
    assert legal(after_switch) == {9, 29, 35, 4}
    only_switch = np.zeros(N_ACTIONS, dtype=bool)
    only_switch[3] = True
    assert legal(restrict_second(only_switch, 3)) == {PASS}   # nothing left: pass rather than an empty mask
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_showdown_masks.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.masks'`.

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/masks.py`:

```python
from __future__ import annotations

from typing import List

import numpy as np
from poke_env.environment import DoublesEnv

N_ACTIONS = 107
PASS = 0
MEGA_ACTIONS = frozenset(a for a in range(27, 47))  # "move k and mega evolve", k = 1..4, 5 targets each


def _legal(action: int, battle, pos: int) -> bool:
    """Exactly poke-env's own per-slot validity rule (fake=False raises AssertionError on an invalid action)."""
    try:
        DoublesEnv._action_to_order_individual(np.int64(action), battle, False, pos)
    except Exception:
        return False
    return True


def slot_needs_action(battle, pos: int) -> bool:
    """True when this slot must choose something other than pass."""
    mon = battle.active_pokemon[pos]
    if any(battle.force_switch):
        return bool(battle.force_switch[pos]) and bool(battle.available_switches[pos])
    return mon is not None and not mon.fainted


def reference_masks(battle) -> np.ndarray:
    """Slow reference: ask poke-env's decoder about every action. `action_masks` must equal this on every real state."""
    out = np.zeros((2, N_ACTIONS), dtype=bool)
    for pos in range(2):
        if not slot_needs_action(battle, pos):
            out[pos, PASS] = True
            continue
        for a in range(1, N_ACTIONS):
            out[pos, a] = _legal(a, battle, pos)
        if not out[pos].any():  # nothing legal (for example a struggle turn the decoder rejects): fall back to pass
            out[pos, PASS] = True
    return out


def action_masks(battle) -> np.ndarray:
    """(2, N_ACTIONS) bool: which actions are legal for each slot on its own. Pass (0) is legal only for a slot with nothing to do.

    Built from the battle's own move, target and switch lists (about 100x faster than asking the decoder 212 times).
    """
    out = np.zeros((2, N_ACTIONS), dtype=bool)
    for pos in range(2):
        if not slot_needs_action(battle, pos):
            out[pos, PASS] = True
            continue
        active = battle.active_pokemon[pos]
        if not battle.trapped[pos] and battle.available_switches[pos]:
            ok = {p.base_species for p in battle.available_switches[pos]}
            for i, mon in enumerate(list(battle.team.values())[:6]):
                out[pos, 1 + i] = mon.base_species in ok
        if not battle.force_switch[pos] and active is not None:
            available = {m.id: m for m in battle.available_moves[pos]}
            moves = list(battle.available_moves[pos]) if (
                len(available) == 1 and next(iter(available)) in ("struggle", "recharge")) else list(active.moves.values())
            for k, mv in enumerate(moves[:4]):
                if mv.id not in available:
                    continue
                for target in battle.get_possible_showdown_targets(available[mv.id], active):
                    base = 7 + 5 * k + (target + 2)
                    out[pos, base] = True
                    if battle.can_mega_evolve[pos]:
                        out[pos, base + 20] = True
        if not out[pos].any():
            out[pos, PASS] = True
    return out


def restrict_second(mask1: np.ndarray, first_action: int) -> np.ndarray:
    """Slot 1's mask once slot 0 has chosen: no second Mega, no switching in the same Pokemon (pass stays legal if it was)."""
    m = mask1.copy()
    if first_action in MEGA_ACTIONS:
        for a in MEGA_ACTIONS:
            m[a] = False
    if 1 <= first_action <= 6:
        m[first_action] = False
    if not m.any():
        m[PASS] = True
    return m
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_showdown_masks.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add xp/showdown/masks.py tests/test_showdown_masks.py
git commit -m "feat(showdown): legal-action masks for the two active slots

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

(The check against real battle states is part of Task 3's marked test, once the vector env exists.)

---

### Task 3: Scripted opponents and the vector env

**Files:**
- Create: `xp/showdown/opponents.py`, `xp/showdown/vec_env.py`, `tests/test_showdown_vec.py`

**Interfaces:**
- Consumes: `XPDoublesEnv` (foundations), `PoolTeambuilder`, `action_masks`.
- Produces: `LADDER = ("random", "maxbp", "damage")`, `make_opponent(name, battle_format) -> Player`, `DamageHeuristicPlayer`; `ShowdownVecEnv(n, battle_format, teams, opponent, seed, name="v")` with `.n`, `.obs` `(n, 1811)` float32, `.masks` `(n, 2, 107)` bool, `.step(actions (n, 2)) -> (rewards (n,), dones (n,))`, `.recent_stats() -> (win_rate_100, count)`, `.recent_window_full()`, `.close()`. After `step`, `obs` and `masks` belong to the next decision (finished battles are reset automatically). A worker that hits an error (for example an illegal action) makes `step` raise `RuntimeError("env worker failed: ...")` instead of hanging.

- [ ] **Step 1: Write the failing marked tests**

Create `tests/test_showdown_vec.py`:

```python
"""Needs a running local Showdown server (see test_showdown_env.py for XP_SHOWDOWN_DIR and XP_NODE). Run with: pytest -m showdown"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.environment import DoublesEnv, SingleAgentWrapper  # noqa: E402
from poke_env.player import RandomPlayer  # noqa: E402

from xp.showdown.encode import OBS_DIM  # noqa: E402
from xp.showdown.env import XPDoublesEnv  # noqa: E402
from xp.showdown.masks import N_ACTIONS, action_masks, reference_masks, restrict_second  # noqa: E402
from xp.showdown.teams import PoolTeambuilder, load_pool, validate_team  # noqa: E402
from xp.showdown.vec_env import ShowdownVecEnv  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"
TEAMS = Path(__file__).resolve().parent.parent / "configs" / "showdown_teams"
SHOWDOWN_DIR = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))
NODE = os.environ.get("XP_NODE", "node")

pytestmark = pytest.mark.showdown


@pytest.fixture(scope="module")
def pool():
    if not SHOWDOWN_DIR.exists():
        pytest.skip(f"no Showdown checkout at {SHOWDOWN_DIR} (set XP_SHOWDOWN_DIR)")
    teams = load_pool(TEAMS / "tune")
    for text in teams.values():
        validate_team(text, FORMAT, SHOWDOWN_DIR, node=NODE)
    return teams


def test_fast_masks_equal_the_decoder_reference_and_every_masked_action_is_accepted(pool):
    suffix = str(random.randrange(10**5))
    env = XPDoublesEnv(
        battle_format=FORMAT, team=PoolTeambuilder(pool, 3), strict=True, log_level=40,
        account_configuration1=AccountConfiguration("ta" + suffix, None), account_configuration2=AccountConfiguration("tb" + suffix, None),
    )
    wrapped = SingleAgentWrapper(env, RandomPlayer(battle_format=FORMAT, start_listening=False))
    rng, states = np.random.default_rng(0), 0
    try:
        for _ in range(25):
            wrapped.reset()
            done = False
            while not done:
                m = action_masks(env.battle1)
                assert np.array_equal(m, reference_masks(env.battle1))
                a0 = int(rng.choice(np.flatnonzero(m[0])))
                a1 = int(rng.choice(np.flatnonzero(restrict_second(m[1], a0))))
                DoublesEnv.action_to_order(np.array([a0, a1]), env.battle1, strict=True)  # raises if the pair is illegal
                _, _, term, trunc, _ = wrapped.step(np.array([a0, a1]))
                done, states = term or trunc, states + 1
        assert states > 100
    finally:
        wrapped.close()


def test_vec_env_steps_in_parallel_and_resets_finished_battles(pool):
    venv = ShowdownVecEnv(2, FORMAT, pool, "random", seed=0)
    rng = np.random.default_rng(1)
    try:
        assert venv.obs.shape == (2, OBS_DIM) and venv.masks.shape == (2, 2, N_ACTIONS) and venv.masks.dtype == bool
        finished = 0
        for _ in range(120):
            acts = np.zeros((2, 2), dtype=np.int64)
            for i in range(2):
                a0 = int(rng.choice(np.flatnonzero(venv.masks[i, 0])))
                acts[i] = (a0, int(rng.choice(np.flatnonzero(restrict_second(venv.masks[i, 1], a0)))))
            rewards, dones = venv.step(acts)
            assert set(rewards[dones == 1].tolist()) <= {1.0, -1.0} and (rewards[dones == 0] == 0).all()
            assert venv.obs.shape == (2, OBS_DIM)
            finished += int(dones.sum())
        assert finished >= 4 and venv.recent_stats()[1] == min(finished, 100)
    finally:
        venv.close()


def test_a_worker_reports_an_illegal_action_instead_of_hanging(pool):
    venv = ShowdownVecEnv(1, FORMAT, pool, "random", seed=0)
    try:
        bad = np.array([[N_ACTIONS - 1, N_ACTIONS - 1]])  # a terastallize action: never legal in this format
        with pytest.raises(RuntimeError, match="env worker failed"):
            venv.step(bad)
    finally:
        venv.close()
```

- [ ] **Step 2: Start the server and run them to verify they fail**

```bash
export PATH=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin:$PATH
(cd ~/xp-showdown-spike/pokemon-showdown && nohup node pokemon-showdown start --no-security > ../server.log 2>&1 &)
until lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; do sleep 2; done
XP_NODE=$(which node) pytest -m showdown tests/test_showdown_vec.py -v
```

Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.opponents'` (or `vec_env`).

- [ ] **Step 3: Write the opponents**

Create `xp/showdown/opponents.py` (the same `DamageHeuristicPlayer` as `scripts/spikes/showdown_opponents.py`, which stays as the spike record):

```python
"""Scripted opponents for the Showdown phase: the 3-rung ladder Random < MaxBasePower < DamageHeuristic.

Used through `SingleAgentWrapper`, which only calls `choose_move`, so the players never open a connection.
"""
from __future__ import annotations

import random

from poke_env.battle import DoubleBattle, Move
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder

LADDER = ("random", "maxbp", "damage")


class DamageHeuristicPlayer(Player):
    """Per active mon: best (move, foe target) by base power * accuracy * STAB * type
    effectiveness, weighted up when the target is low on HP; spread moves count both foes.
    Switches randomly only when no move is available. No protect, no switching strategy."""

    def choose_move(self, battle):
        if not isinstance(battle, DoubleBattle) or any(battle.force_switch):
            return self.choose_random_doubles_move(battle)
        foes = battle.opponent_active_pokemon
        orders, switched_in = [], None
        for mon, moves, switches in zip(battle.active_pokemon, battle.available_moves, battle.available_switches):
            switches = [s for s in switches if s != switched_in]
            if not mon or mon.fainted:
                orders.append(None); continue
            if not moves and switches:
                switched_in = random.choice(switches)
                orders.append(SingleBattleOrder(switched_in)); continue
            if not moves:
                orders.append(DefaultBattleOrder()); continue
            best, best_score = None, -1.0
            for move in moves:
                for tgt in battle.get_possible_showdown_targets(move, mon):
                    if tgt not in (1, 2):
                        continue
                    foe = foes[tgt - 1]
                    if foe is None or foe.fainted:
                        continue
                    score = self._score(move, mon, foe)
                    if score > best_score:
                        best, best_score = (move, tgt), score
            if best is None:
                move = random.choice(moves)
                tg = battle.get_possible_showdown_targets(move, mon)
                orders.append(SingleBattleOrder(move, move_target=random.choice(tg)))
            else:
                orders.append(SingleBattleOrder(best[0], move_target=best[1]))
        for slot, order in enumerate(orders):
            # one Mega per side per battle: take the first slot that can
            if isinstance(order, SingleBattleOrder) and isinstance(order.order, Move) and battle.can_mega_evolve[slot]:
                order.mega = True
                break
        if orders[0] or orders[1]:
            return DoubleBattleOrder(orders[0], orders[1])
        return self.choose_random_doubles_move(battle)

    @staticmethod
    def _score(move, user, foe):
        if move.base_power <= 0:
            return 0.0
        s = move.base_power * (move.accuracy if move.accuracy else 1.0)
        if move.type in (user.type_1, user.type_2):
            s *= 1.5
        s *= foe.damage_multiplier(move)
        s *= 2.0 - foe.current_hp_fraction
        return s


def make_opponent(name: str, battle_format: str) -> Player:
    """A scripted opponent that never connects to the server (the env plays its moves)."""
    cls = {"random": RandomPlayer, "maxbp": MaxBasePowerPlayer, "damage": DamageHeuristicPlayer}.get(name)
    if cls is None:
        raise ValueError(f"unknown opponent {name!r}; choose from {LADDER}")
    return cls(battle_format=battle_format, start_listening=False)
```

- [ ] **Step 4: Write the vector env**

Create `xp/showdown/vec_env.py`:

```python
from __future__ import annotations

import multiprocessing
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np


def _worker(conn, cfg: dict) -> None:
    """One process = one env (two accounts on the local server) against one scripted opponent, auto-resetting."""
    from poke_env import AccountConfiguration
    from poke_env.environment import DoublesEnv, SingleAgentWrapper

    from xp.showdown.env import XPDoublesEnv
    from xp.showdown.masks import action_masks
    from xp.showdown.opponents import make_opponent
    from xp.showdown.teams import PoolTeambuilder

    env = XPDoublesEnv(
        battle_format=cfg["battle_format"], team=PoolTeambuilder(cfg["teams"], cfg["team_seed"]), strict=False, log_level=40,
        account_configuration1=AccountConfiguration(cfg["name"] + "a", None),
        account_configuration2=AccountConfiguration(cfg["name"] + "b", None),
    )
    wrapped = SingleAgentWrapper(env, make_opponent(cfg["opponent"], cfg["battle_format"]))
    try:
        obs, _ = wrapped.reset()
        conn.send((obs, action_masks(env.battle1)))
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            DoublesEnv.action_to_order(np.asarray(data, dtype=np.int64), env.battle1, strict=True)  # fail loudly on a mask bug
            obs, reward, term, trunc, _ = wrapped.step(data)
            done = bool(term or trunc)
            if done:
                obs, _ = wrapped.reset()
            conn.send((obs, float(reward), done, action_masks(env.battle1)))
    except BaseException as e:  # report instead of letting the parent wait forever
        conn.send(("error", repr(e)))
    finally:
        wrapped.close()


class ShowdownVecEnv:
    """`n` env processes stepped in parallel. After `step`, `obs` / `masks` belong to the next decision (auto-reset on done).

    Interface used by the learner: `n`, `obs` (n, D) float32, `masks` (n, 2, 107) bool, `step(actions (n, 2)) -> (rewards, dones)`,
    `recent_stats() -> (win_rate over the last 100 finished battles, count)`, `recent_window_full()`, `close()`.
    """

    def __init__(self, n: int, battle_format: str, teams: Dict[str, str], opponent: str, seed: int, name: str = "v"):
        ctx = multiprocessing.get_context("spawn")
        self.n = n
        suffix = "%05d" % random.SystemRandom().randrange(10**5)  # fresh per run, NOT from the seed: stale accounts/battles from an old run block reused names
        self.conns, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            cfg = {"battle_format": battle_format, "teams": teams, "opponent": opponent, "team_seed": seed * 1000 + i,
                   "name": f"{name}{suffix}{i}"[:16]}
            proc = ctx.Process(target=_worker, args=(child, cfg), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        self.results: deque = deque(maxlen=100)
        first = [self._recv(c) for c in self.conns]
        self.obs = np.stack([f[0] for f in first]).astype(np.float32)
        self.masks = np.stack([f[1] for f in first])

    @staticmethod
    def _recv(conn):
        msg = conn.recv()
        if isinstance(msg, tuple) and isinstance(msg[0], str) and msg[0] == "error":
            raise RuntimeError(f"env worker failed: {msg[1]}")
        return msg

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for conn, a in zip(self.conns, actions):
            conn.send(("step", np.asarray(a, dtype=np.int64)))
        rewards, dones, obs, masks = np.zeros(self.n, np.float32), np.zeros(self.n, np.float32), [], []
        for i, conn in enumerate(self.conns):
            o, r, d, m = self._recv(conn)
            rewards[i], dones[i] = r, float(d)
            obs.append(o)
            masks.append(m)
            if d:
                self.results.append(1.0 if r > 0 else 0.0)
        self.obs, self.masks = np.stack(obs).astype(np.float32), np.stack(masks)
        return rewards, dones

    def recent_stats(self) -> Tuple[float, int]:
        return (float(np.mean(self.results)) if self.results else float("nan")), len(self.results)

    def recent_window_full(self) -> bool:
        return len(self.results) == self.results.maxlen

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
```

- [ ] **Step 5: Run the marked tests to verify they pass**

Run: `XP_NODE=$(which node) pytest -m showdown -v`
Expected: 4 passed (the foundations env test plus the three here), about 10 s. If a test hangs, a worker is stuck: stop with Ctrl-C, run `pkill -f "pokemon-showdown start"`, restart the server and rerun (stale accounts from a hung run).

- [ ] **Step 6: Commit**

```bash
git add xp/showdown/opponents.py xp/showdown/vec_env.py tests/test_showdown_vec.py
git commit -m "feat(showdown): scripted opponent ladder and a multi-process vector env

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Masked PPO learner

**Files:**
- Create: `xp/showdown/ppo.py`, `tests/toy_showdown.py`, `tests/test_showdown_ppo.py`

**Interfaces:**
- Consumes: `OBS_DIM`, `N_ACTIONS`, `restrict_second`, and a vec env with the interface from Task 3.
- Produces: `ActorCritic(obs_dim=OBS_DIM, hidden=256)`, `masked(logits, mask) -> Categorical`, `gae(...)`, `Learner(ppo: dict, ent_coef: float, seed: int, device="cpu")` with `.act(obs, masks)` and `.train(env, steps, log_every=10, target=None) -> {"curve": [{"steps", "win_rate_100"}], "steps_to_target", "steps"}`. The `ppo` dict needs `n_steps, epochs, minibatches, lr, gamma, lam, clip, vf_coef` (`n_envs` comes from the env).

- [ ] **Step 1: Write the toy helper and the failing tests**

Create `tests/toy_showdown.py`:

```python
"""Shared toy pieces for the Showdown learner tests: PPO settings and a one-decision battle env."""
from __future__ import annotations

from collections import deque

import numpy as np

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS

PPO = {"n_steps": 16, "epochs": 4, "minibatches": 4, "lr": 0.003, "gamma": 0.99, "lam": 0.95, "clip": 0.2, "vf_coef": 0.5}
GOOD = 3  # the action that wins in the toy env


class ToyVec:
    """One-decision battles: win (+1) when slot 0 plays action GOOD, else lose (-1). Actions 50..106 are never legal."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.results: deque = deque(maxlen=100)
        self._draw()

    def _draw(self):
        self.obs = self.rng.normal(size=(self.n, OBS_DIM)).astype(np.float32)
        self.masks = np.zeros((self.n, 2, N_ACTIONS), dtype=bool)
        self.masks[:, :, :50] = True

    def step(self, actions):
        assert self.masks[np.arange(self.n), 0, actions[:, 0]].all(), "slot 0 played an illegal action"
        rewards = np.where(actions[:, 0] == GOOD, 1.0, -1.0).astype(np.float32)
        self.results.extend((rewards > 0).astype(float).tolist())
        self._draw()
        return rewards, np.ones(self.n, dtype=np.float32)

    def recent_stats(self):
        return (float(np.mean(self.results)) if self.results else float("nan")), len(self.results)

    def recent_window_full(self):
        return len(self.results) == self.results.maxlen
```

Create `tests/test_showdown_ppo.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.toy_showdown import PPO, ToyVec  # noqa: E402
from xp.showdown.masks import N_ACTIONS  # noqa: E402
from xp.showdown.ppo import Learner, masked  # noqa: E402


def test_masked_categorical_gives_illegal_actions_zero_probability():
    logits = torch.zeros(2, N_ACTIONS)
    mask = torch.zeros(2, N_ACTIONS, dtype=torch.bool)
    mask[:, [5, 9]] = True
    probs = masked(logits, mask).probs
    assert probs[:, [5, 9]].sum().item() == pytest.approx(2.0) and probs[~mask].sum().item() == 0.0


def test_act_only_samples_legal_actions_and_conditions_the_second_slot():
    learner = Learner(PPO, 0.01, seed=0)
    env = ToyVec(8)
    env.masks[:, 1, :] = False
    env.masks[:, 1, 4] = env.masks[:, 1, 7] = True
    env.masks[:, 0, :] = False
    env.masks[:, 0, 4] = True  # slot 0 can only switch in "4", so slot 1 must not also pick switch target 4
    for _ in range(20):
        actions, logp, value, cond = learner.act(env.obs, env.masks)
        assert (actions[:, 0] == 4).all() and (actions[:, 1] == 7).all()
        assert logp.shape == (8,) and value.shape == (8,) and cond.shape == (8, 2, N_ACTIONS)


def test_training_is_deterministic_per_seed():
    runs = []
    for _ in range(2):
        learner = Learner(PPO, 0.01, seed=7)
        learner.train(ToyVec(8, seed=1), steps=8 * 16 * 3, log_every=1)
        runs.append([p.detach().numpy().copy() for p in learner.net.parameters()])
    assert all(np.array_equal(a, b) for a, b in zip(*runs))


def test_ppo_learns_the_toy_battle_with_illegal_actions_never_played():
    learner = Learner(PPO, 0.01, seed=0)
    env = ToyVec(16)
    out = learner.train(env, steps=16 * 16 * 100, log_every=5, target=0.9)
    assert out["curve"][-1]["win_rate_100"] > 0.9
    assert out["steps_to_target"] is not None and out["steps_to_target"] <= out["steps"]
    assert out["curve"][0]["win_rate_100"] < out["curve"][-1]["win_rate_100"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_showdown_ppo.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.ppo'`.

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/ppo.py`:

```python
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from xp.showdown.encode import OBS_DIM
from xp.showdown.masks import N_ACTIONS, restrict_second

NEG = -1e9


class ActorCritic(nn.Module):
    """MLP on the fixed feature vector with two 107-way heads (one per active slot) and a value head."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.ModuleList([nn.Linear(hidden, N_ACTIONS), nn.Linear(hidden, N_ACTIONS)])
        self.v = nn.Linear(hidden, 1)

    def heads(self, x: torch.Tensor):
        h = self.body(x)
        return self.pi[0](h), self.pi[1](h), self.v(h).squeeze(-1)


def masked(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Categorical over the legal actions only (illegal logits pushed to -1e9, so their probability is exactly 0)."""
    return Categorical(logits=logits.masked_fill(~mask, NEG))


def gae(rewards, values, next_value, nonterm, gamma: float, lam: float):
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(rewards.shape[0])):
        nv = next_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm[t] - values[t]
        last = delta + gamma * lam * nonterm[t] * last
        adv[t] = last
    return adv


class Learner:
    """Plain PPO with action masks (Kairi). Slot 1's mask is conditioned on slot 0's sampled action (no duplicate switch, one Mega)."""

    def __init__(self, ppo: dict, ent_coef: float, seed: int, device: str = "cpu"):
        self.ppo, self.ent_coef, self.seed = ppo, ent_coef, int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = ActorCritic().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=ppo["lr"], eps=1e-5)
        self.steps_trained = 0

    def act(self, obs: np.ndarray, masks: np.ndarray):
        """Sample one joint action per env. Returns (actions (n, 2), logp (n,), value (n,), conditional masks (n, 2, 107))."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m0 = torch.as_tensor(masks[:, 0], device=self.device)
        with torch.no_grad():
            l0, l1, value = self.net.heads(x)
            d0 = masked(l0, m0)
            a0 = d0.sample()
            m1 = np.stack([restrict_second(masks[i, 1], int(a0[i])) for i in range(len(obs))])
            m1_t = torch.as_tensor(m1, device=self.device)
            d1 = masked(l1, m1_t)
            a1 = d1.sample()
            logp = d0.log_prob(a0) + d1.log_prob(a1)
        cond = np.stack([masks[:, 0], m1], axis=1)
        return torch.stack([a0, a1], dim=1).cpu().numpy(), logp, value, cond

    def train(self, env, steps: int, log_every: int = 10, target: Optional[float] = None) -> dict:
        """Train for about `steps` agent decisions. Returns the win-rate curve and the steps to reach `target` win rate."""
        torch.manual_seed(self.seed + self.steps_trained)
        np.random.seed((self.seed + self.steps_trained) % (2**32))
        c, device = self.ppo, self.device
        n, t_len = env.n, c["n_steps"]
        batch = n * t_len
        updates = max(1, steps // batch)
        obs_buf = np.zeros((t_len, n, OBS_DIM), dtype=np.float32)
        mask_buf = np.zeros((t_len, n, 2, N_ACTIONS), dtype=bool)
        act_buf = torch.zeros((t_len, n, 2), dtype=torch.long, device=device)
        logp_buf = torch.zeros((t_len, n), device=device)
        val_buf = torch.zeros((t_len, n), device=device)
        rew_buf = np.zeros((t_len, n), dtype=np.float32)
        done_buf = np.zeros((t_len, n), dtype=np.float32)
        curve, steps_to_target = [], None
        for u in range(updates):
            for t in range(t_len):
                obs_buf[t] = env.obs
                actions, logp, value, cond = self.act(env.obs, env.masks)
                mask_buf[t], act_buf[t], logp_buf[t], val_buf[t] = cond, torch.as_tensor(actions, device=device), logp, value
                rew_buf[t], done_buf[t] = env.step(actions)
            with torch.no_grad():
                _, _, next_value = self.net.heads(torch.as_tensor(env.obs, dtype=torch.float32, device=device))
            nonterm = 1.0 - torch.as_tensor(done_buf, device=device)
            adv = gae(torch.as_tensor(rew_buf, device=device), val_buf, next_value, nonterm, c["gamma"], c["lam"])
            ret = adv + val_buf
            b_obs = torch.as_tensor(obs_buf.reshape(-1, OBS_DIM), device=device)
            b_mask = torch.as_tensor(mask_buf.reshape(-1, 2, N_ACTIONS), device=device)
            b_act, b_logp = act_buf.reshape(-1, 2), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            for _ in range(c["epochs"]):
                perm = torch.randperm(batch, device=device)
                for s in range(0, batch, batch // c["minibatches"]):
                    idx = perm[s:s + batch // c["minibatches"]]
                    l0, l1, value = self.net.heads(b_obs[idx])
                    d0, d1 = masked(l0, b_mask[idx, 0]), masked(l1, b_mask[idx, 1])
                    new_logp = d0.log_prob(b_act[idx, 0]) + d1.log_prob(b_act[idx, 1])
                    ratio = (new_logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                    vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    ent = (d0.entropy() + d1.entropy()).mean()
                    loss = pg + c["vf_coef"] * vf - self.ent_coef * ent
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()
            self.steps_trained += batch
            if (u + 1) % log_every == 0 or u == updates - 1:
                win, _ = env.recent_stats()
                done_steps = (u + 1) * batch
                curve.append({"steps": done_steps, "win_rate_100": win})
                if target is not None and steps_to_target is None and env.recent_window_full() and win >= target:
                    steps_to_target = done_steps
        return {"curve": curve, "steps_to_target": steps_to_target, "steps": updates * batch}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_showdown_ppo.py -v`
Expected: 4 passed (about 6 s; the learning test needs 100 updates).

- [ ] **Step 5: Commit**

```bash
git add xp/showdown/ppo.py tests/toy_showdown.py tests/test_showdown_ppo.py
git commit -m "feat(showdown): masked PPO learner with two 107-way heads

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The gate: measure, run, report

**Files:**
- Create: `xp/showdown/gate.py`, `tests/test_showdown_gate.py`, `docs/spikes/showdown_gate.md`

**Interfaces:**
- Consumes: `Learner`, `ShowdownVecEnv`, `load_pool`.
- Produces: `measure(learner, env, steps, target=0.8, log_every=5) -> dict` with `steps, wall_s, decisions_per_s, final_win_rate_100, in_usable_range, steps_to_target, target, curve`; CLI `python -m xp.showdown.gate --opponent {random,maxbp,damage} --steps N --out FILE [--n-envs 8 --ent-coef 0.01 --seed 0]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_gate.py`:

```python
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from tests.toy_showdown import PPO, ToyVec  # noqa: E402

from xp.showdown import gate  # noqa: E402
from xp.showdown.ppo import Learner  # noqa: E402


def test_measure_reports_throughput_curve_and_usable_range_flag():
    out = gate.measure(Learner(PPO, 0.01, seed=0), ToyVec(16), steps=16 * 16 * 100, target=0.9, log_every=10)
    assert out["decisions_per_s"] > 0 and out["wall_s"] > 0 and out["steps"] == 16 * 16 * 100
    assert out["curve"][-1]["win_rate_100"] == out["final_win_rate_100"]
    assert out["in_usable_range"] == (0.2 <= out["final_win_rate_100"] <= 0.8)
    assert out["steps_to_target"] is not None and json.dumps(out)  # JSON-serialisable


def test_default_ppo_has_every_key_the_learner_reads():
    assert {"n_envs", "n_steps", "epochs", "minibatches", "lr", "gamma", "lam", "clip", "vf_coef"} <= set(gate.DEFAULT_PPO)
    assert gate.DEFAULT_PPO["n_steps"] * gate.DEFAULT_PPO["n_envs"] % gate.DEFAULT_PPO["minibatches"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_showdown_gate.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'xp.showdown.gate'`.

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/gate.py`:

```python
"""The budget gate: train plain PPO (Kairi) against one scripted rung and report throughput and the win-rate curve.

Answers two questions before any tuning: how many decisions per second a real PPO run gets (this fixes the cost of a
phase and the cap on the budget S), and whether plain PPO lands between 20% and 80% win rate against some rung.
Run: python -m xp.showdown.gate --opponent random --steps 100000 --out gate_random.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from xp.showdown.ppo import Learner

FORMAT = "gen9championsvgc2026regmc"
TEAMS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "showdown_teams"
DEFAULT_PPO = {"n_envs": 8, "n_steps": 128, "epochs": 4, "minibatches": 4, "lr": 0.0003, "gamma": 0.99, "lam": 0.95,
               "clip": 0.2, "vf_coef": 0.5}


def measure(learner: Learner, env, steps: int, target: float = 0.8, log_every: int = 5) -> dict:
    """Train `learner` on `env` for about `steps` decisions; wall-clock throughput includes the PPO updates."""
    t0 = time.perf_counter()
    run = learner.train(env, steps, log_every=log_every, target=target)
    wall = time.perf_counter() - t0
    final = run["curve"][-1]["win_rate_100"]
    return {"steps": run["steps"], "wall_s": wall, "decisions_per_s": run["steps"] / wall, "final_win_rate_100": final,
            "in_usable_range": bool(0.2 <= final <= 0.8), "steps_to_target": run["steps_to_target"], "target": target,
            "curve": run["curve"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train plain PPO against one scripted rung and report throughput and win rate.")
    parser.add_argument("--opponent", required=True, choices=["random", "maxbp", "damage"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_PPO["n_envs"])
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="a tune seed")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from xp.showdown.teams import load_pool
    from xp.showdown.vec_env import ShowdownVecEnv

    ppo = {**DEFAULT_PPO, "n_envs": args.n_envs}
    env = ShowdownVecEnv(args.n_envs, FORMAT, load_pool(TEAMS_DIR / "tune"), args.opponent, seed=args.seed)
    try:
        result = measure(Learner(ppo, args.ent_coef, args.seed), env, args.steps)
    finally:
        env.close()
    result.update({"opponent": args.opponent, "ppo": ppo, "ent_coef": args.ent_coef, "seed": args.seed})
    Path(args.out).write_text(json.dumps(result, indent=1))
    target_pct = f"{result['target']:.0%}"
    print(f"{args.opponent}: {result['steps']} decisions in {result['wall_s']:.0f}s = {result['decisions_per_s']:.0f}/s, "
          f"final win rate {result['final_win_rate_100']:.2f}, steps to {target_pct}: {result['steps_to_target']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests, then the whole default suite**

Run: `pytest tests/test_showdown_gate.py -v && pytest`
Expected: 2 passed, then the full default suite passes with no server needed.

- [ ] **Step 5: Commit the code**

```bash
git add xp/showdown/gate.py tests/test_showdown_gate.py
git commit -m "feat(showdown): budget gate CLI (throughput and win-rate curve per rung)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Run the gate on each rung (about 7 minutes each at 300 decisions/s)**

122,880 decisions is 120 PPO updates of 1,024 decisions. This is a probe budget, not the cap. Server must be running (see Task 3 step 2); close other heavy programs, since throughput is part of the result.

```bash
mkdir -p results/showdown_gate
for r in random maxbp damage; do
  python -m xp.showdown.gate --opponent $r --steps 122880 --out results/showdown_gate/gate_$r.json
done
```

Expected: three lines like `random: 122880 decisions in 400s = 307/s, final win rate 0.9x, steps to 80%: ...`. Check `git check-ignore results` first; `results/` is ignored by git, so these JSON files stay local and the numbers go into the write-up.

- [ ] **Step 7: Optional scaling probe (does a second concurrent run slow the first?)**

```bash
for i in 1 2; do python -m xp.showdown.gate --opponent maxbp --steps 40960 --seed $i --out results/showdown_gate/par_$i.json & done; wait
```

Report the two `decisions_per_s` values next to the single-run value. If each stays near the single-run rate, several runs can share one server; if each halves, the server or the machine is the limit.

- [ ] **Step 8: Write `docs/spikes/showdown_gate.md` and stop the server**

Print the table, then write the report with these sections: (1) per-rung results (final win rate over the last 100 battles, whether it is in 20% to 80%, steps to 80%, decisions per second); (2) which rung, if any, is the "familiar opponent" by the spec's rule (default-entropy Kairi between 20% and 80% after the probe budget; if none qualifies, say so plainly and what that means: the phase stops per the spec, or the user widens the rung set); (3) the cost table for a full phase, `hours = runs x S / decisions_per_s / 3600`, for 56 runs and S in {250k, 500k, 1M}, at the measured single-run rate and (if Step 7 was run) the parallel rate; (4) a recommended cap S, with the reason, labelled as a recommendation; (5) limits: one seed, one team pool, 120 updates, PPO untuned, short games.

```bash
python3 - <<'EOF'
import json
print("| rung | final win rate (last 100) | in 20-80% | steps to 80% | decisions/s |")
print("|---|---|---|---|---|")
for r in ("random", "maxbp", "damage"):
    d = json.load(open(f"results/showdown_gate/gate_{r}.json"))
    print(f"| {r} | {d['final_win_rate_100']:.2f} | {d['in_usable_range']} | {d['steps_to_target']} | {d['decisions_per_s']:.0f} |")
EOF
pkill -f "pokemon-showdown start"
```

- [ ] **Step 9: Update the handoff and commit**

Update `RESUME.md` and `STATUS.md`: Plan B done; the gate numbers; **the cap S is awaiting the user's decision** (state the recommendation); Plan C (RND/Kaiser, config, phase wiring, report) is next and starts only after the cap is approved. Commit the report and handoff files, then make the STATUS-only marker commit. Do not push without the user's yes.

```bash
git add docs/spikes/showdown_gate.md RESUME.md STATUS.md
git commit -m "docs: Showdown budget gate results and cap recommendation

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review

**Coverage of what Plan A deferred:** PPO learner: Task 4. Throughput measurement: Task 5 steps 6 and 7. "Plain PPO between 20% and 80% against some rung": Task 5 step 6 and the report. Mask versus `strict=False` decision: Task 2 (masks, conditioned second slot) and the amendment's open point, now settled by measurement. Moving `DamageHeuristicPlayer` into the package: Task 3. Still deferred (Plan C): RND/Kaiser, tuning grids, config, phase wiring, report, and everything that depends on the approved cap S.

**Placeholder scan:** none. Every "Create" block is a complete file that ran in the scratch tree (37 offline tests, 4 marked tests, the real gate CLI on 20,480 decisions).

**Type consistency:** `ShowdownVecEnv` exposes `n, obs, masks, step, recent_stats, recent_window_full, close`, and the same names are used by `Learner.train`, `ToyVec` and the gate. `Learner(ppo, ent_coef, seed)` is the same signature in `ppo.py`, the tests and `gate.py`. `action_masks`/`reference_masks`/`restrict_second` have the same names in `masks.py`, `ppo.py` and the tests. Mask shape `(2, 107)` and conditional masks `(n, 2, 107)` are consistent.

**Known risks:** (1) The default `pytest` now fails on a fresh install until `python -m xp.showdown.patch` runs (by design: the guard); CI runs the patch. (2) The patch modifies the installed `poke-env` files in the venv (originals kept as `*.orig`); a reinstall needs a re-patch. (3) Throughput of about 300 decisions/s is far below what a 56-run phase at S = 1M wants (about 52 hours at that rate), so the cap recommendation will likely be well under 1M; that is the point of the gate. (4) 120 updates of untuned PPO against three rungs on one seed is a probe, not a result; do not read it as Kairi versus Kaiser.
