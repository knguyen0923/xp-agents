# Showdown Matchup Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-switchable block of matchup features (ability-aware type multiplier and estimated damage fraction, outgoing and incoming, plus `revealed` flags) to the Showdown observation, identical for Kairi and Kaiser, with the flag off leaving everything as it is today.

**Architecture:** The observation size becomes a value computed from `env.matchup` instead of an import-time constant. A pure module (`matchup.py`) assembles the 72 extra floats from an injectable estimator function; a second module (`estimate.py`) is the real poke-env adapter. Unit tests use a stub estimator; the real adapter is checked by a probe and a server-marked test. A new config file runs the experiment under the unchanged protocol.

**Tech Stack:** Python 3.9, NumPy, PyTorch, poke-env 0.10.0 (`poke_env.calc.damage_calc_gen9.calculate_damage`, `Pokemon.damage_multiplier`, `poke_env.stats`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-showdown-matchup-features-design.md`

## Global Constraints

- Python floor 3.9; every module starts with `from __future__ import annotations`.
- `xp.protocol` stays phase-agnostic: nothing here imports from it or changes it. `xp/showdown/` is never imported from `xp.protocol`.
- With `env.matchup` absent or false: the observation vector, `OBS_DIM`, `configs/showdown.yaml`'s config hash, results and all existing tests are unchanged (a test pins this).
- With it on: 72 extra floats (outgoing 32, incoming 32, 8 `revealed` flags), identical features for both agents, frozen before tuning.
- Unrevealed opponent moves are never guessed: their entries are zeros with `revealed` 0.
- Tests must not require the Showdown server unless marked `showdown` (the default `pytest` run excludes that marker).
- Protocol invariants are untouched: same tuning grids, 250,000 decision cap S, 30% threshold, equal-sized grids, disjoint tune and eval seeds.
- The user approves the real run (about 12 to 15 hours) before it starts, and the budget before any RND result is read. This plan builds and smoke-tests only.
- Commit messages end with: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Nothing is pushed without the user's yes.
- After finishing, update `STATUS.md` (and `RESUME.md`); `STATUS_COMMIT` must name the commit whose diff to HEAD touches only `STATUS.md` (commit everything else first, then a STATUS-only marker commit).

## File Structure

- `xp/showdown/encode.py` (modify): add `MATCHUP_DIM`, `obs_dim()`, and the `matchup` / `estimator` arguments of `encode_battle`.
- `xp/showdown/matchup.py` (create): `matchup_block()` and the `Estimator` type. Pure; no poke-env calls beyond reading `Pokemon` attributes.
- `xp/showdown/estimate.py` (create): `PokeEnvEstimator`, the real adapter over `calculate_damage` with the base-stat fallback.
- `xp/showdown/env.py`, `vec_env.py`, `ppo.py`, `rnd.py`, `phase.py` (modify): carry the flag and the observation size through.
- `scripts/record_showdown_replays.py` (modify): the recording env gets the flag too.
- `scripts/spikes/showdown_matchup_probe.py` (create): throwaway probe against the real server.
- `docs/spikes/showdown_matchup_probe.md` (create): the probe's findings.
- `configs/showdown-matchup.yaml`, `configs/showdown-matchup-smoke.yaml` (create).
- Tests: `tests/test_showdown_matchup.py`, `tests/test_showdown_estimate.py` (create); extend `tests/test_showdown_encode.py`, `tests/test_showdown_ppo.py`, `tests/test_showdown_phase.py`.

---

### Task 1: Probe poke-env on a real battle (settles the open items)

This task needs the Showdown server, so ask the user before running it. Its output decides details in Task 4.

**Files:**
- Create: `scripts/spikes/showdown_matchup_probe.py`
- Create: `docs/spikes/showdown_matchup_probe.md`

**Interfaces:**
- Consumes: `xp.showdown.env.XPDoublesEnv`, `xp.showdown.opponents.make_opponent`, `xp.showdown.teams.load_pool` / `PoolTeambuilder` (as `vec_env._worker` uses them).
- Produces: written answers to four questions that Task 4 depends on (see Step 3).

- [ ] **Step 1: Write the probe script**

```python
"""Throwaway probe: what does poke-env know about a real doubles battle, and can calculate_damage run on it?"""
from __future__ import annotations

import time

import numpy as np
from poke_env import AccountConfiguration
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.environment import SingleAgentWrapper

from xp.showdown.env import XPDoublesEnv
from xp.showdown.masks import action_masks
from xp.showdown.opponents import make_opponent
from xp.showdown.teams import PoolTeambuilder, load_pool

FORMAT = "gen9championsvgc2026regmc"


def main() -> None:
    pool = load_pool("configs/showdown_teams/tune")
    env = XPDoublesEnv(battle_format=FORMAT, team=PoolTeambuilder(pool, 1), strict=False, log_level=40,
                       account_configuration1=AccountConfiguration("ProbeAgent1", None),
                       account_configuration2=AccountConfiguration("ProbeOpp1", None))
    wrapped = SingleAgentWrapper(env, make_opponent("damage", FORMAT))
    wrapped.reset()
    b = env.battle1
    print("player_role", b.player_role, "opponent_role", b.opponent_role)
    print("team keys", list(b.team), "opp team keys", list(b.opponent_team))
    for label, mons in (("own", b.active_pokemon), ("opp", b.opponent_active_pokemon)):
        for m in mons:
            if m is None:
                continue
            print(label, m.species, "ability", m.ability, "item", m.item, "stats", m.stats, "hp", m.current_hp, m.max_hp,
                  "moves", list(m.moves))
    # Can calculate_damage run for own -> opp and opp -> own? How long does one call take?
    own = [m for m in b.active_pokemon if m is not None]
    opp = [m for m in b.opponent_active_pokemon if m is not None]

    def ident(mon):
        for team in (b.team, b.opponent_team):
            for key, value in team.items():
                if value is mon:
                    return key
        raise KeyError(mon.species)

    for a, d in ((own[0], opp[0]), (opp[0], own[0])):
        for move in list(a.moves.values())[:4]:
            t0 = time.perf_counter()
            try:
                out = calculate_damage(ident(a), ident(d), move, b)
            except Exception as e:  # report, never hide
                out = f"{type(e).__name__}: {e}"
            print(a.species, "->", d.species, move.id, "calc", out, "mult", d.damage_multiplier(move),
                  f"{(time.perf_counter() - t0) * 1000:.2f} ms")
    wrapped.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (with the user's yes, server running)**

Run: `XP_NODE=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin/node python scripts/spikes/showdown_matchup_probe.py`
Expected: it prints roles, team keys, each active's ability/item/stats/moves, and one line per (attacker, defender, move) with a calc result or an exception.

- [ ] **Step 3: Write the findings**

In `docs/spikes/showdown_matchup_probe.md`, answer these four questions from the output, with the printed evidence:
1. Are the opponent's **abilities and items** filled in at battle start (before any move)? If not, are they in the team sheet data somewhere we can read?
2. Do the opponent's `stats` come back as numbers or `None`? (Decides whether Task 4's base-stat fallback is used for foes, for own mons, or both.)
3. Does `calculate_damage(ident(a), ident(d), move, b)` run with identifiers taken as the keys of `b.team` / `b.opponent_team`? If it raises, record the exact error and the form that works.
4. Cost per call in milliseconds (16 outgoing plus up to 16 incoming calls per decision).

End with one line: which of Task 4's two code paths applies (calculator, or the base-stat formula only).

- [ ] **Step 4: Commit**

```bash
git add scripts/spikes/showdown_matchup_probe.py docs/spikes/showdown_matchup_probe.md
git commit -m "spike(showdown): probe poke-env abilities, stats and calculate_damage on a real battle

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Make the observation size configurable (flag off changes nothing)

**Files:**
- Modify: `xp/showdown/encode.py` (constants near line 15 to 18; `encode_battle` at line 72)
- Modify: `xp/showdown/ppo.py` (lines 11, 21, 64 to 71, 125, 145, 168)
- Modify: `xp/showdown/rnd.py` (nothing if `Learner` passes `obs_dim`; line 8 stays)
- Test: `tests/test_showdown_encode.py`, `tests/test_showdown_ppo.py`

**Interfaces:**
- Produces: `encode.MATCHUP_DIM: int` (72), `encode.obs_dim(matchup: bool = False) -> int`, `Learner(ppo, ent_coef, seed, device="cpu", rnd_coef=None, obs_dim: int = OBS_DIM)` with attribute `learner.obs_dim`.
- Consumed later by: Tasks 3 to 5.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_showdown_encode.py`:

```python
def test_obs_dim_helper_is_unchanged_with_the_flag_off():
    assert E.obs_dim() == E.obs_dim(False) == E.OBS_DIM
    assert E.MATCHUP_DIM == 72 and E.obs_dim(True) == E.OBS_DIM + 72
```

Append to `tests/test_showdown_ppo.py`:

```python
def test_learner_takes_the_observation_size_and_defaults_to_obs_dim():
    from xp.showdown.encode import OBS_DIM

    assert Learner(PPO, 0.01, seed=0).net.body[0].in_features == OBS_DIM
    big = Learner(PPO_N, 0.01, seed=0, rnd_coef=0.1, obs_dim=OBS_DIM + 72)
    assert big.obs_dim == OBS_DIM + 72
    assert big.net.body[0].in_features == OBS_DIM + 72 and big.rnd.obs_dim == OBS_DIM + 72
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_showdown_encode.py::test_obs_dim_helper_is_unchanged_with_the_flag_off tests/test_showdown_ppo.py::test_learner_takes_the_observation_size_and_defaults_to_obs_dim -v`
Expected: FAIL (`AttributeError: ... no attribute 'obs_dim'` / `unexpected keyword argument 'obs_dim'`).

- [ ] **Step 3: Implement**

In `xp/showdown/encode.py`, after the `OBS_DIM = ...` line add:

```python
MATCHUP_DIM = 2 * (2 * N_MOVES * 2) * 2 + 2 * N_MOVES   # outgoing 32 + incoming 32 + 8 revealed flags = 72


def obs_dim(matchup: bool = False) -> int:
    """Length of the observation vector; the matchup block (spec 2026-10-29) is appended only when switched on."""
    return OBS_DIM + (MATCHUP_DIM if matchup else 0)
```

In `xp/showdown/ppo.py`, change `Learner.__init__` and every `OBS_DIM` use inside the class:

```python
    def __init__(self, ppo: dict, ent_coef: float, seed: int, device: str = "cpu", rnd_coef: Optional[float] = None,
                 obs_dim: int = OBS_DIM):
        self.ppo, self.ent_coef, self.seed, self.rnd_coef = ppo, ent_coef, int(seed), rnd_coef
        self.obs_dim = int(obs_dim)
        ...
        self.net = ActorCritic(obs_dim=self.obs_dim, intrinsic=rnd_coef is not None).to(self.device)
        ...
            self.rnd = VectorRND(obs_dim=self.obs_dim).to(self.device)
```

and in `train`: replace `OBS_DIM` at the three places (`obs_buf = np.zeros((t_len, n, OBS_DIM)...)`, `.reshape(-1, OBS_DIM)` twice) with `self.obs_dim`. Keep the module-level `from xp.showdown.encode import OBS_DIM` (still the default).

- [ ] **Step 4: Run the whole Showdown fast suite**

Run: `pytest tests/test_showdown_encode.py tests/test_showdown_ppo.py tests/test_showdown_phase.py -q -p no:warnings`
Expected: all pass (existing Learner calls use the default, so nothing else changes).

- [ ] **Step 5: Commit**

```bash
git add xp/showdown/encode.py xp/showdown/ppo.py tests/test_showdown_encode.py tests/test_showdown_ppo.py
git commit -m "refactor(showdown): observation size is a Learner argument (default unchanged)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The matchup block (pure, stub-tested)

**Files:**
- Create: `xp/showdown/matchup.py`
- Test: `tests/test_showdown_matchup.py`

**Interfaces:**
- Consumes: `encode.MATCHUP_DIM`, `encode.N_MOVES`, poke-env `Pokemon` / `Move`.
- Produces:
  - `Estimator = Callable[[Pokemon, Pokemon, Move], Optional[Tuple[float, float]]]` returning `(multiplier, damage_fraction)` or `None` when the move does no damage (status, zero power, empty).
  - `matchup_block(own_actives, opp_actives, estimate) -> np.ndarray` of shape `(MATCHUP_DIM,)`, float32.
- Layout: `[outgoing 32 | incoming 32 | revealed 8]`. Outgoing index: `((a * 4 + m) * 2 + t) * 2 + k` for own active `a`, move slot `m`, foe `t`, `k` 0 = multiplier, 1 = damage fraction. Incoming uses the same pattern with foes as attackers and my actives as targets. Revealed index: `f * 4 + m` for foe `f`, move slot `m`.
- Scaling: multiplier feature = `multiplier / 4` (0.0625 to 1.0; an immunity and "no damage" both read 0); damage fraction feature = `min(1, fraction)`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon  # noqa: E402

from xp.showdown import encode as E  # noqa: E402
from xp.showdown.matchup import matchup_block  # noqa: E402


def mon(species: str, *moves: str) -> Pokemon:
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def stub(mult=2.0, frac=0.5, damaging=("earthquake", "surf")):
    def estimate(attacker, defender, move):
        return (mult, frac) if move.id in damaging else None
    return estimate


def test_shape_and_dtype():
    out = matchup_block([mon("garchomp", "earthquake")], [mon("rotom")], stub())
    assert out.shape == (E.MATCHUP_DIM,) and out.dtype == np.float32


def test_outgoing_values_and_scaling():
    own = [mon("garchomp", "earthquake", "protect"), None]
    foes = [mon("rotom"), mon("heatran")]
    out = matchup_block(own, foes, stub(mult=2.0, frac=0.5))
    # own active 0, move slot 0 (earthquake): both foes, (mult / 4, fraction)
    assert out[0:4].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    # move slot 1 (protect) is a status move: zeros
    assert not out[4:8].any()
    # own active 1 is empty: zeros
    assert not out[16:32].any()


def test_damage_fraction_is_capped_at_one_and_immunity_reads_zero():
    out = matchup_block([mon("garchomp", "earthquake")], [mon("rotom"), None], stub(mult=0.0, frac=3.0))
    assert out[0] == 0.0 and out[1] == 1.0          # mult 0 (immune), fraction capped
    assert not out[2:4].any()                        # absent target


def test_fainted_target_is_zeros():
    dead = mon("rotom")
    dead._status = None
    dead._current_hp = 0
    out = matchup_block([mon("garchomp", "earthquake")], [dead, None], stub())
    assert not out[0:4].any()


def test_incoming_uses_only_revealed_moves_and_sets_flags():
    foes = [mon("rotom", "surf"), mon("heatran")]          # foe 1 has revealed nothing
    mine = [mon("garchomp"), mon("gyarados")]
    out = matchup_block(mine, foes, stub())
    incoming, flags = out[32:64], out[64:72]
    # foe 0, move slot 0 (surf), my two actives
    assert incoming[0:4].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert flags.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert not incoming[16:32].any()                        # foe 1: nothing revealed


def test_a_revealed_immune_move_differs_from_an_unknown_one():
    foes = [mon("rotom", "surf"), None]
    out = matchup_block([mon("garchomp"), None], foes, stub(mult=0.0, frac=0.0))
    assert out[64] == 1.0 and not out[32:36].any()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_showdown_matchup.py -v -p no:warnings`
Expected: FAIL (`ModuleNotFoundError: xp.showdown.matchup`).

- [ ] **Step 3: Implement `xp/showdown/matchup.py`**

```python
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from poke_env.battle import Move, Pokemon

from xp.showdown.encode import MATCHUP_DIM, N_MOVES

# (attacker, defender, move) -> (type multiplier, expected damage as a fraction of the defender's current HP), or None
# when the move does no damage (status, zero power). Supplied by `xp.showdown.estimate` in a real battle, by a stub in tests.
Estimator = Callable[[Pokemon, Pokemon, Move], Optional[Tuple[float, float]]]

_HALF = 2 * N_MOVES * 2 * 2   # one direction: 2 attackers x 4 moves x 2 targets x 2 numbers = 32


def _live(mon: Optional[Pokemon]) -> bool:
    return mon is not None and not mon.fainted


def _moves(mon: Optional[Pokemon]) -> List[Optional[Move]]:
    """The same move order and cap the main encoder uses; unrevealed foe slots stay None."""
    moves = list(mon.moves.values())[:N_MOVES] if mon is not None else []
    return moves + [None] * (N_MOVES - len(moves))


def _direction(attackers: Sequence[Optional[Pokemon]], targets: Sequence[Optional[Pokemon]], estimate: Estimator) -> np.ndarray:
    out = np.zeros(_HALF, dtype=np.float32)
    for a, attacker in enumerate(attackers[:2]):
        if not _live(attacker):
            continue
        for m, move in enumerate(_moves(attacker)):
            if move is None:
                continue
            for t, target in enumerate(targets[:2]):
                if not _live(target):
                    continue
                est = estimate(attacker, target, move)
                if est is None:
                    continue
                i = ((a * N_MOVES + m) * 2 + t) * 2
                out[i], out[i + 1] = min(est[0] / 4.0, 1.0), min(est[1], 1.0)
    return out


def matchup_block(own_actives: Sequence[Optional[Pokemon]], opp_actives: Sequence[Optional[Pokemon]], estimate: Estimator) -> np.ndarray:
    """Outgoing (my moves into foes), incoming (foes' REVEALED moves into me), then one revealed flag per foe move slot."""
    own = list(own_actives) + [None] * (2 - len(own_actives))
    opp = list(opp_actives) + [None] * (2 - len(opp_actives))
    flags = np.zeros(2 * N_MOVES, dtype=np.float32)
    for f, foe in enumerate(opp[:2]):
        if _live(foe):
            for m, move in enumerate(_moves(foe)):
                flags[f * N_MOVES + m] = float(move is not None)
    out = np.concatenate([_direction(own, opp, estimate), _direction(opp, own, estimate), flags]).astype(np.float32)
    assert out.shape == (MATCHUP_DIM,)
    return out
```

(Importing `encode` from `matchup` is one-way: `encode_battle` imports `matchup` lazily inside the function in Task 5, so there is no import cycle.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_showdown_matchup.py -v -p no:warnings`
Expected: PASS (6 tests). If `test_fainted_target_is_zeros` fails because `mon.fainted` does not follow `_current_hp = 0` on a bare `Pokemon`, set the fainted state the way `tests/test_showdown_encode.py` does for fainted mons (or set `dead._status = Status.FNT`) and keep the assertion.

- [ ] **Step 5: Commit**

```bash
git add xp/showdown/matchup.py tests/test_showdown_matchup.py
git commit -m "feat(showdown): matchup feature block with outgoing, incoming and revealed flags (stub-tested)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The real estimator over poke-env (follows the Task 1 findings)

**Files:**
- Create: `xp/showdown/estimate.py`
- Test: `tests/test_showdown_estimate.py`

**Interfaces:**
- Consumes: `Estimator` from Task 3; the Task 1 findings.
- Produces: `PokeEnvEstimator(battle)` with `__call__(attacker, defender, move) -> Optional[Tuple[float, float]]`, and the constants `FALLBACK_LEVEL = 50`, `FALLBACK_IV = 31`, `FALLBACK_EV = 0`, `FALLBACK_NATURE = "serious"`.

Rules (from the spec): multiplier = `defender.damage_multiplier(move)`, forced to 0 when the calculator says the move deals 0 although it has base power and the type multiplier is positive (ability immunity, Protect). Damage fraction = mean of the calculator's min and max roll divided by the target's current HP in points (`stats["hp"] * current_hp_fraction`). Status and zero-power moves give `None`. If a calculator call raises, return `None` and increment `PokeEnvEstimator.errors`.

- [ ] **Step 1: Write the failing tests (no server; a fake battle and a patched calculator)**

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon  # noqa: E402

from xp.showdown import estimate as ES  # noqa: E402


def mon(species, *moves):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def fake_battle(*mons):
    return SimpleNamespace(team={f"p1: {m.species}": m for m in mons[:1]}, opponent_team={f"p2: {m.species}": m for m in mons[1:]})


def test_status_and_zero_power_moves_give_none(monkeypatch):
    a, d = mon("garchomp", "protect"), mon("rotom")
    est = ES.PokeEnvEstimator(fake_battle(a, d))
    assert est(a, d, a.moves["protect"]) is None


def test_multiplier_comes_from_the_types_and_fraction_from_the_calculator(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")
    d._stats = {"hp": 150, "atk": 80, "def": 90, "spa": 120, "spd": 90, "spe": 70}
    a._stats = dict(d._stats)
    d._current_hp, d._max_hp = 100, 100              # a bare Pokemon may report 0 HP, so set the fraction explicitly
    monkeypatch.setattr(ES, "calculate_damage", lambda *args, **kw: (60, 90))
    mult, frac = ES.PokeEnvEstimator(fake_battle(a, d))(a, d, a.moves["earthquake"])
    assert mult == pytest.approx(d.damage_multiplier(a.moves["earthquake"]))
    assert frac == pytest.approx(75 / 150)


def test_zero_damage_from_a_damaging_move_reads_as_an_immunity(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")
    for p in (a, d):
        p._stats = {"hp": 150, "atk": 80, "def": 90, "spa": 120, "spd": 90, "spe": 70}
    monkeypatch.setattr(ES, "calculate_damage", lambda *args, **kw: (0, 0))
    assert ES.PokeEnvEstimator(fake_battle(a, d))(a, d, a.moves["earthquake"]) == (0.0, 0.0)


def test_a_calculator_error_is_counted_and_returns_none(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")

    def boom(*args, **kw):
        raise AssertionError("stats not defined")

    monkeypatch.setattr(ES, "calculate_damage", boom)
    est = ES.PokeEnvEstimator(fake_battle(a, d))
    assert est(a, d, a.moves["earthquake"]) is None and est.errors == 1


def test_missing_stats_are_filled_from_base_stats_at_the_fallback_spread(monkeypatch):
    a, d = mon("garchomp", "earthquake"), mon("heatran")
    seen = {}

    def spy(att, dfn, move, battle, **kw):
        seen["d_stats"] = battle.opponent_team[dfn].stats
        return (10, 10)

    monkeypatch.setattr(ES, "calculate_damage", spy)
    ES.PokeEnvEstimator(fake_battle(a, d))(a, d, a.moves["earthquake"])
    assert all(isinstance(v, (int, float)) for v in seen["d_stats"].values())
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_showdown_estimate.py -v -p no:warnings`
Expected: FAIL (`ModuleNotFoundError: xp.showdown.estimate`).

- [ ] **Step 3: Implement `xp/showdown/estimate.py`**

```python
from __future__ import annotations

from typing import Optional, Tuple

from poke_env.battle import Move, MoveCategory, Pokemon
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

# Stated assumption (spec): the opponent's real spreads are hidden, so fill missing stats from base stats at level 50
# with no EVs, perfect IVs and a neutral nature. Confirm against docs/spikes/showdown_matchup_probe.md before any run.
FALLBACK_LEVEL = 50
FALLBACK_IV = 31
FALLBACK_EV = 0
FALLBACK_NATURE = "serious"
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


def _fill_stats(mon: Pokemon) -> None:
    """Give `mon.stats` numbers if any are missing, so the calculator's assertions hold. Never overwrites known stats."""
    if all(isinstance(v, (int, float)) for v in mon.stats.values()) and mon.stats:
        return
    raw = compute_raw_stats(mon.species, [FALLBACK_EV] * 6, [FALLBACK_IV] * 6, FALLBACK_LEVEL, FALLBACK_NATURE, GenData.from_gen(9))
    mon.stats = {k: int(v) for k, v in zip(_STAT_ORDER, raw)}


class PokeEnvEstimator:
    """`(attacker, defender, move) -> (type multiplier, damage fraction of the defender's current HP)` for one battle."""

    def __init__(self, battle) -> None:
        self.battle = battle
        self.errors = 0

    def _identifier(self, mon: Pokemon) -> str:
        for team in (self.battle.team, self.battle.opponent_team):
            for key, value in team.items():
                if value is mon:
                    return key
        raise KeyError(mon.species)

    def __call__(self, attacker: Pokemon, defender: Pokemon, move: Move) -> Optional[Tuple[float, float]]:
        if move.category == MoveCategory.STATUS or move.base_power <= 0:
            return None
        multiplier = float(defender.damage_multiplier(move))
        try:
            _fill_stats(attacker)
            _fill_stats(defender)
            lo, hi = calculate_damage(self._identifier(attacker), self._identifier(defender), move, self.battle)
        except Exception:
            self.errors += 1
            return None
        if hi == 0 and multiplier > 0:
            multiplier = 0.0   # ability immunity or Protect: the calculator sees no damage
        hp_points = max(1.0, defender.stats["hp"] * defender.current_hp_fraction)
        return multiplier, min(1.0, ((lo + hi) / 2.0) / hp_points)
```

If Task 1 shows the calculator needs a different identifier form, change only `_identifier` and keep the tests (they use `{role}: {species}` keys; adjust `fake_battle` to match).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_showdown_estimate.py -v -p no:warnings`
Expected: PASS (5 tests).

- [ ] **Step 5: Add a server-marked test of the real calculator**

Append to `tests/test_showdown_estimate.py` (mirroring how `tests/test_showdown_vec.py` marks and starts server tests; copy its fixture and `FORMAT`/pool setup):

```python
@pytest.mark.showdown
def test_real_battle_estimates_are_finite_and_cheap():
    """Needs the local server. After the first decision every estimate is None or in range, and a decision costs under 50 ms."""
    import time

    from tests.test_showdown_vec import real_first_battle   # helper added next to the existing server fixtures

    battle = real_first_battle()
    est = ES.PokeEnvEstimator(battle)
    t0 = time.perf_counter()
    for a in battle.active_pokemon:
        for d in battle.opponent_active_pokemon:
            for move in a.moves.values():
                r = est(a, d, move)
                assert r is None or (0.0 <= r[0] <= 4.0 and 0.0 <= r[1] <= 1.0)
    assert (time.perf_counter() - t0) < 0.05
```

Add `real_first_battle()` to `tests/test_showdown_vec.py` by reusing the setup that file already uses for its server tests (build `XPDoublesEnv` with `PoolTeambuilder`, reset through `SingleAgentWrapper`, return `env.battle1`, close the env). Run it with the server up: `XP_NODE=... pytest -m showdown tests/test_showdown_estimate.py -v`. Expected: PASS. If the 50 ms budget fails, switch both features to the base-stat-only path as the spec allows, and say so in the commit message.

- [ ] **Step 6: Commit**

```bash
git add xp/showdown/estimate.py tests/test_showdown_estimate.py tests/test_showdown_vec.py
git commit -m "feat(showdown): poke-env damage estimator with base-stat fallback

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Wire the flag through encoder, env, vector env, learner and phase

**Files:**
- Modify: `xp/showdown/encode.py:72-84` (`encode_battle`)
- Modify: `xp/showdown/env.py` (`XPDoublesEnv.__init__`, `embed_battle`)
- Modify: `xp/showdown/vec_env.py` (`_worker`, `ShowdownVecEnv.__init__`)
- Modify: `xp/showdown/phase.py` (`make_env`, `build_learner`)
- Modify: `scripts/record_showdown_replays.py` (the recording env)
- Test: `tests/test_showdown_encode.py`, `tests/test_showdown_phase.py`

**Interfaces:**
- Consumes: `matchup_block` (Task 3), `PokeEnvEstimator` (Task 4), `obs_dim` (Task 2).
- Produces:
  - `encode_battle(battle, matchup: bool = False, estimator: Optional[Estimator] = None) -> np.ndarray`. With `matchup=True` and no estimator it builds `PokeEnvEstimator(battle)`.
  - `XPDoublesEnv(matchup: bool = False, **kwargs)`; observation space size `obs_dim(matchup)`.
  - `ShowdownVecEnv(..., matchup: bool = False)`.
  - Config key `env.matchup` (default false) read by `make_env` and `build_learner`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_showdown_encode.py`:

```python
def test_flag_off_vector_is_identical_and_flag_on_appends_the_block():
    own, opp = [mon("garchomp", "earthquake")], [mon("rotom", "surf")]
    b = battle(own, opp, own + [None], opp + [None])
    base = E.encode_battle(b)
    assert E.encode_battle(b, matchup=False).tolist() == base.tolist()
    on = E.encode_battle(b, matchup=True, estimator=lambda a, d, m: (2.0, 0.5) if m.base_power > 0 else None)
    assert on.shape == (E.obs_dim(True),)
    assert on[: E.OBS_DIM].tolist() == base.tolist()
    assert on[E.OBS_DIM] == pytest.approx(0.5)      # first outgoing multiplier: 2 / 4
```

Append to `tests/test_showdown_phase.py` (reuse that file's `cfg` fixture and its toy `fake_make_env` pattern):

```python
def test_matchup_flag_sets_the_learner_observation_size_and_the_hash(tiny_sd_cfg):
    from xp.protocol.phase import config_hash
    from xp.showdown import encode

    off = sp.build_learner(tiny_sd_cfg, "kairi", {"ent_coef": 0.01}, 0)
    assert off.obs_dim == encode.OBS_DIM
    on_cfg = {**tiny_sd_cfg, "env": {**tiny_sd_cfg["env"], "matchup": True}}
    on = sp.build_learner(on_cfg, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.02}, 0)
    assert on.obs_dim == encode.obs_dim(True) and on.rnd.obs_dim == encode.obs_dim(True)
    assert config_hash(tiny_sd_cfg) != config_hash(on_cfg)
```

(Use the fixture name that `tests/test_showdown_phase.py` already defines for its toy config; read the top of that file and match it exactly.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_showdown_encode.py::test_flag_off_vector_is_identical_and_flag_on_appends_the_block tests/test_showdown_phase.py -v -p no:warnings`
Expected: FAIL (`unexpected keyword argument 'matchup'`, `obs_dim` mismatch).

- [ ] **Step 3: Implement**

`encode.py`:

```python
def encode_battle(battle, matchup: bool = False, estimator=None) -> np.ndarray:
    """Flat float32 vector of length obs_dim(matchup) for a DoubleBattle (own side, opponent side, field[, matchup block])."""
    ...existing body producing `out`...
    if matchup:
        from xp.showdown.matchup import matchup_block   # lazy: matchup.py imports this module's constants
        if estimator is None:
            from xp.showdown.estimate import PokeEnvEstimator
            estimator = PokeEnvEstimator(battle)
        out = np.concatenate([out, matchup_block(list(battle.active_pokemon), list(battle.opponent_active_pokemon), estimator)])
    assert out.shape == (obs_dim(matchup),)
    return out.astype(np.float32)
```

(Replace the existing final `assert out.shape == (OBS_DIM,)` / `return out` accordingly; the flag-off path must produce exactly the old bytes.)

`env.py`:

```python
    def __init__(self, matchup: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.matchup = bool(matchup)
        space = Box(-2.0, 2.0, shape=(obs_dim(self.matchup),), dtype=np.float32)
        ...
    def embed_battle(self, battle) -> np.ndarray:
        return encode_battle(battle, matchup=self.matchup)
```

(import `obs_dim` next to `OBS_DIM`.)

`vec_env.py`: add `matchup: bool = False` to `ShowdownVecEnv.__init__`, put `"matchup": matchup` in each worker `cfg`, and in `_worker` pass `matchup=cfg.get("matchup", False)` to `XPDoublesEnv(...)`.

`phase.py`:

```python
def make_env(cfg: dict, stage: str, seed: int):
    ...
    return ShowdownVecEnv(cfg["ppo"]["n_envs"], cfg["env"]["format"], load_pool(team_dir(cfg, stage)), cfg["env"]["opponent"],
                          seed=seed, matchup=bool(cfg["env"].get("matchup", False)))
```

In `build_learner`, compute `dim = obs_dim(bool(cfg["env"].get("matchup", False)))` and pass `obs_dim=dim` to all three `Learner(...)` calls (import `obs_dim` from `xp.showdown.encode`).

`scripts/record_showdown_replays.py`: add `matchup=bool(cfg["env"].get("matchup", False))` to the recording `ShowdownVecEnv(...)` call.

- [ ] **Step 4: Run the fast suite**

Run: `pytest -q -p no:warnings`
Expected: all pass, including the existing Showdown tests unchanged, which proves the flag-off path.

- [ ] **Step 5: Commit**

```bash
git add xp/showdown tests scripts/record_showdown_replays.py
git commit -m "feat(showdown): env.matchup switches the observation block on across encoder, env, learner and phase

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Configs and a real-server smoke run

**Files:**
- Create: `configs/showdown-matchup.yaml`, `configs/showdown-matchup-smoke.yaml`

**Interfaces:**
- Consumes: everything above; `scripts/run_showdown.sh`.

- [ ] **Step 1: Create the configs**

```bash
cp configs/showdown.yaml configs/showdown-matchup.yaml
cp configs/showdown-smoke.yaml configs/showdown-matchup-smoke.yaml
```

In each new file, add `matchup: true` under the `env:` block (after `format:`), and change the first comment line to say it is the matchup-features condition (spec `2026-10-29-showdown-matchup-features-design.md`). In `showdown-matchup.yaml` also change `title:` under `report:` to `Showdown with matchup features, tuned Kairi vs Kaiser`. Change nothing else: grids, seeds, budget and threshold must equal the original.

- [ ] **Step 2: Verify the configs differ only where intended**

Run: `diff configs/showdown.yaml configs/showdown-matchup.yaml; diff configs/showdown-smoke.yaml configs/showdown-matchup-smoke.yaml`
Expected: only the `matchup: true` line, the comment line and (full config) the title.

- [ ] **Step 3: Smoke run on the real server (ask the user first; about 75 seconds)**

Run: `scripts/run_showdown.sh configs/showdown-matchup-smoke.yaml results/showdown-matchup-smoke 2`
Expected: finishes, writes a results table. Then confirm in the log or a one-line check that the learner observation size is `OBS_DIM + 72`, and look at the time per decision against the original smoke run (`configs/showdown-smoke.yaml`) to quantify the slowdown from the calculator. Record both numbers.

- [ ] **Step 4: Update the handoff files and commit**

Add the smoke numbers (observation size, seconds per run versus the original smoke) to `STATUS.md` and `RESUME.md`, and state that the full matchup run (`configs/showdown-matchup.yaml`, about 12 to 15 hours, `ALLOW_CODE_CHANGE=1` not needed on a fresh output directory) is built but awaits the user's approval, including the budget approval before any RND result is read. Commit the configs and docs, then make the STATUS-only marker commit.

```bash
git add configs/showdown-matchup.yaml configs/showdown-matchup-smoke.yaml STATUS.md RESUME.md
git commit -m "feat(showdown): matchup-features configs and smoke result

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review (spec coverage)

- Switchable flag, off by default, old run untouched, pinned by a test: Tasks 2 and 5 (`test_obs_dim_helper...`, `test_flag_off_vector_is_identical...`, whole fast suite).
- 72 floats: outgoing 32, incoming 32, 8 revealed flags, zeros for unrevealed, status, absent and fainted: Task 3.
- Ability-aware multiplier, damage fraction of current HP capped at 1: Task 4 (immunity rule, fraction), Task 3 (scaling).
- Base-stat fallback with a stated spread, calculator errors counted: Task 4 (`FALLBACK_*`, `errors`); the spread is confirmed against the Task 1 findings.
- Open items (opponent abilities at battle start; calculator cost): Task 1 and the server-marked test in Task 4.
- Observation size as a value, not an import-time constant: Task 2 and Task 5.
- Experiment config, same protocol, own output directory: Task 6. The full run and the budget approval stay with the user and are out of this plan.
- Out of scope per the spec: pretraining or auxiliary losses, the policy probe, opponent and teams changes.
- Known simplification to flag to the user: an immunity and a "move not applicable" both read 0 in the multiplier feature, and a Protect-ing foe reads as an immunity because the calculator returns 0 damage.
