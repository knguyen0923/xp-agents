# Showdown tempo opponent, tempo features and friendly-fire penalty: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tempo-aware scripted opponent (`tempo`), a switchable tempo observation block, a friendly-fire reward penalty and tempo-carrying team pools, so two Showdown runs (matchup + penalty, then + tempo block) can be launched.

**Architecture:** Everything lives in `xp/showdown/` and is off by default behind config keys (`env.tempo`, `env.friendly_fire`), so existing observations, rewards, config hashes and the four finished runs are unchanged. New logic goes in two new modules (`friendly_fire.py`, `tempo.py`); `encode.py`, `env.py`, `vec_env.py`, `phase.py`, `opponents.py` and `ppo.py` get small edits. Teams are revised copies in a new directory.

**Tech Stack:** Python 3.9, poke-env 0.10.0 (patched), PyTorch, pytest, a local Pokemon Showdown server (only for tests marked `showdown` and for runs).

**Spec:** `docs/superpowers/specs/2026-11-02-showdown-tempo-design.md`

## Global Constraints

- Every module starts with `from __future__ import annotations`; Python floor 3.9 (no `X | Y` types, no `match`).
- `xp/protocol/` must never import from `xp.showdown`. Nothing in this plan touches `xp/protocol/`.
- With `env.tempo` and `env.friendly_fire` absent, the observation, `OBS_DIM`, rewards, config hashes (`configs/showdown*.yaml` existing files, including `Phase.from_config(...).config_hash == "fdac28308815"` for `configs/showdown.yaml`) and existing tests are unchanged. Do not edit any existing config or anything under `configs/showdown_teams/`.
- Friendly-fire constant `C = 1.0`, pinned in the config, identical for both agents, never tuned. The total penalty charged in one battle is capped at `-1.0` (`FF_BATTLE_CAP`), so a win always returns at least 0 and beats a clean loss (-1); a win through friendly fire counts as a win in every metric (user, 2026-11-02).
- Tempo block size is 77 floats: speed order 4, remaining turns 9, move flags 64. Durations: Trick Room 5, Tailwind 4, screens 5 (8 when the agent's own team holds Light Clay; a foe screen is assumed 5).
- Both agents get equal-sized tuning grids (8 values each: Kairi `ent_coef` [0.001, 0.002, 0.003, 0.006, 0.01, 0.02, 0.05, 0.1]; Kaiser `rnd_coef` [0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0]). A tuned optimum on a grid edge stops the run (`ProtocolError`).
- Budget cap S needs the user's approval before any RND result is read. Runs and pushes need the user's yes each time (project memory: approve each step).
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. The Stop hook needs `STATUS.md`'s `STATUS_COMMIT` to equal the commit whose diff to HEAD touches only `STATUS.md` (commit everything else first, then a STATUS-only commit).
- Run the fast suite with `.venv/bin/pytest -q` (the `showdown` marker is excluded by default); server tests with `XP_NODE=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin/node .venv/bin/pytest -m showdown`.

## File Structure

| File | Responsibility |
|---|---|
| `xp/showdown/friendly_fire.py` (new) | Decode a move action; charge `-C * estimated damage fraction` for hitting the partner. Pure functions. |
| `xp/showdown/tempo.py` (new) | Speed order, remaining turns, move flags; `tempo_block(battle)`. Also `effective_speed`, used by the opponent. |
| `xp/showdown/encode.py` | `TEMPO_DIM`, `obs_dim(matchup, tempo)`, `encode_battle(..., tempo)`. |
| `xp/showdown/env.py` | `XPDoublesEnv(matchup, tempo)`. |
| `xp/showdown/vec_env.py` | Worker computes the penalty before each step; `ShowdownVecEnv` adds `last_won`, `friendly_fire` and `tempo` options. |
| `xp/showdown/ppo.py` | `evaluate` counts wins from `env.last_won`, not from `reward > 0`. |
| `xp/showdown/phase.py` | `make_env` and `build_learner` read the new keys. |
| `xp/showdown/opponents.py` | `TempoPlayer`, registered as `tempo`. |
| `xp/showdown/gate.py` | Gate options for the new opponent, pool and flags. |
| `configs/showdown_teams_tempo/{tune,eval}/*.txt` (new) | Revised copies of the six teams. |
| `configs/showdown-tempo-base.yaml`, `configs/showdown-tempo.yaml`, `configs/showdown-tempo-smoke.yaml` (new) | The two runs and a smoke check. |
| `tests/test_showdown_friendly_fire.py`, `tests/test_showdown_tempo.py`, `tests/test_showdown_tempo_opponent.py`, `tests/test_showdown_tempo_teams.py` (new) | Unit tests, no server (except marked ones). |

---

### Task 1: Friendly-fire penalty function

**Files:**
- Create: `xp/showdown/friendly_fire.py`
- Test: `tests/test_showdown_friendly_fire.py`

**Interfaces:**
- Consumes: the move-action layout from `xp/showdown/masks.py` (action `7 + 5*k + (target + 2)` for move slot `k` 0..3 and showdown target -2..2; Mega variants add 20, covering actions 27..46); an estimator `(attacker, defender, move) -> Optional[(type multiplier, damage fraction)]` such as `xp.showdown.estimate.PokeEnvEstimator`.
- Produces: `decode_move_action(action: int) -> Optional[Tuple[int, int]]` (move index, showdown target, or `None` for a non-move action); `hits_ally(move, target: int) -> bool`; `friendly_fire_penalty(battle, pos: int, action: int, estimator, scale: float) -> float` (zero or negative); `FF_BATTLE_CAP = 1.0`; `capped_penalty(penalty: float, spent: float) -> float` (the part of `penalty` that fits under the cap, given the positive total `spent` already charged this battle).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_friendly_fire.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Move, Pokemon, Status  # noqa: E402

from xp.showdown import friendly_fire as FF  # noqa: E402


def mon(species, *moves):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    return p


def act(k, target):
    return 7 + 5 * k + (target + 2)


def setup(frac=0.4, mult=1.0):
    a = mon("garchomp", "closecombat", "earthquake", "heatwave", "helpinghand")
    b = mon("heatran", "flamethrower")
    calls = []

    def estimator(attacker, defender, move):
        calls.append((attacker, defender, move.id))
        return (mult, frac)

    return SimpleNamespace(active_pokemon=[a, b]), estimator, calls


def test_decode_move_actions_and_mega_variants():
    assert FF.decode_move_action(7) == (0, -2)
    assert FF.decode_move_action(11) == (0, 2)
    assert FF.decode_move_action(12) == (1, -2)
    assert FF.decode_move_action(46) == (3, 2)
    assert FF.decode_move_action(27) == (0, -2)   # the Mega variant of action 7
    assert FF.decode_move_action(0) is None and FF.decode_move_action(3) is None and FF.decode_move_action(47) is None


def test_a_damaging_move_aimed_at_the_partner_is_charged_the_estimated_fraction():
    battle, est, calls = setup(frac=0.4)
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 1.0) == pytest.approx(-0.4)
    assert calls[0][1] is battle.active_pokemon[1]   # the defender is the partner


def test_slot_one_aims_at_target_minus_one():
    battle, est, _ = setup(frac=0.5)
    battle.active_pokemon[1] = mon("garchomp", "closecombat")
    battle.active_pokemon[0] = mon("heatran", "flamethrower")
    assert FF.friendly_fire_penalty(battle, 1, act(0, -1), est, 1.0) == pytest.approx(-0.5)


def test_a_move_aimed_at_a_foe_is_free():
    battle, est, calls = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(0, 1), est, 1.0) == 0.0 and not calls


def test_all_adjacent_moves_hit_the_partner_whatever_the_target():
    battle, est, _ = setup(frac=0.3)
    assert FF.friendly_fire_penalty(battle, 0, act(1, 0), est, 1.0) == pytest.approx(-0.3)   # earthquake


def test_all_adjacent_foes_moves_are_free():
    battle, est, calls = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(2, 0), est, 1.0) == 0.0 and not calls      # heat wave


def test_status_moves_aimed_at_the_partner_are_free():
    battle, est, calls = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(3, -2), est, 1.0) == 0.0 and not calls     # helping hand


def test_the_mega_variant_is_charged_like_the_plain_move():
    battle, est, _ = setup(frac=0.4)
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2) + 20, est, 1.0) == pytest.approx(-0.4)


def test_an_immune_partner_is_free():
    battle, est, _ = setup(frac=0.0, mult=0.0)
    assert FF.friendly_fire_penalty(battle, 0, act(1, 0), est, 1.0) == 0.0


def test_a_fainted_or_absent_partner_is_free():
    battle, est, calls = setup()
    battle.active_pokemon[1]._status = Status.FNT
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 1.0) == 0.0
    battle.active_pokemon[1] = None
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 1.0) == 0.0 and not calls


def test_the_scale_multiplies_and_non_move_actions_are_free():
    battle, est, _ = setup(frac=0.5)
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), est, 2.0) == pytest.approx(-1.0)
    assert FF.friendly_fire_penalty(battle, 0, 3, est, 1.0) == 0.0     # a switch
    assert FF.friendly_fire_penalty(battle, 0, 0, est, 1.0) == 0.0     # pass


def test_an_estimator_that_returns_none_is_free():
    battle, _, _ = setup()
    assert FF.friendly_fire_penalty(battle, 0, act(0, -2), lambda a, d, m: None, 1.0) == 0.0


def test_the_battle_cap_stops_the_running_total_at_minus_one():
    assert FF.FF_BATTLE_CAP == 1.0
    assert FF.capped_penalty(-0.4, 0.0) == pytest.approx(-0.4)
    assert FF.capped_penalty(-0.4, 0.8) == pytest.approx(-0.2)     # only 0.2 of headroom left
    assert FF.capped_penalty(-0.4, 1.0) == 0.0                     # already at the cap
    assert FF.capped_penalty(-2.0, 0.0) == pytest.approx(-1.0)     # one huge decision still stops at the cap
    assert FF.capped_penalty(0.0, 0.5) == 0.0


def test_a_win_with_the_most_friendly_fire_still_returns_more_than_a_clean_loss():
    spent = 0.0
    for raw in (-2.0, -2.0, -2.0):                                 # three decisions of two full-HP hits on the partner
        spent += -FF.capped_penalty(raw, spent)
    assert spent == pytest.approx(1.0) and 1.0 - spent >= 0.0 > -1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_showdown_friendly_fire.py -q`
Expected: collection error / FAIL with `ImportError: cannot import name 'friendly_fire'`.

- [ ] **Step 3: Write the implementation**

Create `xp/showdown/friendly_fire.py`:

```python
"""Friendly-fire reward penalty (spec 2026-11-02): charge an action that would damage the agent's own partner.

Works on the chosen action before the step, from the battle state the agent saw, using the matchup estimator, so an
immune partner (estimated fraction 0) is free and the number does not depend on what the opponent does.
"""
from __future__ import annotations

from typing import Optional, Tuple

from poke_env.battle import MoveCategory

FIRST_MOVE_ACTION = 7      # layout from xp/showdown/masks.py: 7 + 5 * k + (target + 2); Mega variants add 20
LAST_MOVE_ACTION = 46
N_TARGETS = 5
MEGA_OFFSET = 20
FF_BATTLE_CAP = 1.0        # total penalty per battle never exceeds this, so a win (return >= 0) always beats a clean loss (-1)


def capped_penalty(penalty: float, spent: float) -> float:
    """The part of `penalty` (zero or negative) that fits under the per-battle cap, given the positive total `spent` so far."""
    return max(penalty, -(FF_BATTLE_CAP - spent))


def decode_move_action(action: int) -> Optional[Tuple[int, int]]:
    """(move index 0..3, showdown target -2..2) for a move action, or None for pass, a switch or anything else."""
    if not FIRST_MOVE_ACTION <= action <= LAST_MOVE_ACTION:
        return None
    a = action - FIRST_MOVE_ACTION
    if a >= MEGA_OFFSET:
        a -= MEGA_OFFSET
    return a // N_TARGETS, a % N_TARGETS - 2


def hits_ally(move, target: int) -> bool:
    """A damaging move hits the partner when it is aimed at it, or when its target class is allAdjacent (Earthquake, Surf)."""
    if move.category == MoveCategory.STATUS or move.base_power <= 0:
        return False
    return target in (-1, -2) or move.entry.get("target") == "allAdjacent"


def friendly_fire_penalty(battle, pos: int, action: int, estimator, scale: float) -> float:
    """`-scale * estimated damage fraction on the partner` for slot `pos` taking `action`; 0.0 when it does not hurt the partner."""
    decoded = decode_move_action(int(action))
    if decoded is None:
        return 0.0
    active, ally = battle.active_pokemon[pos], battle.active_pokemon[1 - pos]
    if active is None or active.fainted or ally is None or ally.fainted:
        return 0.0
    moves = list(active.moves.values())[:4]
    k, target = decoded
    if k >= len(moves) or not hits_ally(moves[k], target):
        return 0.0
    est = estimator(active, ally, moves[k])
    return 0.0 if est is None else -scale * est[1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_showdown_friendly_fire.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add xp/showdown/friendly_fire.py tests/test_showdown_friendly_fire.py
git commit -m "feat: friendly-fire penalty function" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Wire the penalty into the env, keeping win detection correct

The vec env counts a win as `reward > 0` (`xp/showdown/vec_env.py` and `Learner.evaluate`). A penalty added to the final-step reward could turn a win into a negative number, so wins are tracked separately in `last_won`.

**Files:**
- Modify: `xp/showdown/vec_env.py` (worker, `ShowdownVecEnv.__init__`, `step`)
- Modify: `xp/showdown/ppo.py:97` (`evaluate`)
- Modify: `xp/showdown/phase.py:33-34` (`make_env`)
- Modify: `tests/toy_showdown.py` (`ToyVec` gets `last_won`)
- Test: `tests/test_showdown_ppo.py`, `tests/test_showdown_vec.py`

**Interfaces:**
- Consumes: `friendly_fire_penalty` from Task 1; `PokeEnvEstimator` from `xp.showdown.estimate`.
- Produces: `ShowdownVecEnv(..., friendly_fire: float = 0.0)`; the env attribute `last_won` (float32 array `(n,)`, 1.0 where a battle ended in a win on the last step); `step` returns `rewards` that include the penalty. Worker reply is now `(obs, reward, done, masks, penalty)`. Config key `env.friendly_fire` (float, absent = 0).

- [ ] **Step 1: Write the failing test for `evaluate`**

Append to `tests/test_showdown_ppo.py` (it already imports `Learner`, `PPO`, `OBS_DIM`, `ToyVec`, `GOOD`; if a name is missing add it to the existing import line from `tests.toy_showdown` / `xp.showdown.encode`):

```python
def test_evaluate_counts_wins_even_when_a_penalty_makes_the_reward_negative():
    class Penalised(ToyVec):
        def step(self, actions):
            rewards, dones = super().step(actions)     # sets self.last_won from the sparse win/loss
            return rewards - 3.0, dones                # a friendly-fire penalty on every decision

    env = Penalised(4, seed=1)
    learner = Learner(PPO, 0.01, 0, obs_dim=OBS_DIM)
    learner.act = lambda obs, masks: (np.full((obs.shape[0], 2), GOOD), None, None, None)   # always the winning action
    assert learner.evaluate(env, 40)["win_rate"] == 1.0
```

Make sure `numpy as np` is imported in that test file (add `import numpy as np` if absent).

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_showdown_ppo.py::test_evaluate_counts_wins_even_when_a_penalty_makes_the_reward_negative -q`
Expected: FAIL (`win_rate` 0.0, because every reward is negative).

- [ ] **Step 3: Add `last_won` to `ToyVec`**

In `tests/toy_showdown.py`, in `ToyVec.__init__` add `self.last_won = np.zeros(n, dtype=np.float32)` before `self._draw()`, and in `step` after computing `rewards` add `self.last_won = (rewards > 0).astype(np.float32)`.

- [ ] **Step 4: Change `Learner.evaluate`**

In `xp/showdown/ppo.py`, replace the line

```python
            wins += int(((rewards > 0) & (dones == 1)).sum())
```
with
```python
            wins += int(env.last_won.sum())     # not `rewards > 0`: a friendly-fire penalty can push a win's reward below 0
```

Then find any other fake env used with `evaluate` or `train`: run `grep -rn "def step" tests/ scripts/ xp/showdown/` and add a `last_won` array to each fake that `evaluate` is called on (the toy env in `tests/toy_showdown.py` is the shared one; fix any others the same way).

- [ ] **Step 5: Run the learner tests**

Run: `.venv/bin/pytest tests/test_showdown_ppo.py tests/test_showdown_gate.py tests/test_showdown_replays.py -q`
Expected: all pass, including the new test.

- [ ] **Step 6: Edit the worker and `ShowdownVecEnv`**

In `xp/showdown/vec_env.py`, in `_worker`:

1. After the existing imports inside `_worker`, add:
```python
    from xp.showdown.estimate import PokeEnvEstimator
    from xp.showdown.friendly_fire import capped_penalty, friendly_fire_penalty
```
2. Pass the flag to `XPDoublesEnv` is added in Task 4; for now, before the `try:` add counters and the per-battle running total (also import `capped_penalty` with `friendly_fire_penalty`):
```python
    ff_scale = float(cfg.get("friendly_fire", 0.0))
    ff_actions, ff_total, decisions = 0, 0.0, 0
    ff_spent = 0.0      # penalty charged so far in the current battle (positive number), for the per-battle cap
```
3. In the step loop, replace
```python
            obs, reward, term, trunc, _ = wrapped.step(data)
            done = bool(term or trunc)
            if done:
                obs, _ = wrapped.reset()
            conn.send((obs, float(reward), done, action_masks(env.battle1)))
```
with
```python
            penalty = 0.0
            if ff_scale:     # from the battle state the agent saw, before the step moves the battle on
                est = PokeEnvEstimator(env.battle1)
                for pos in range(2):
                    p = friendly_fire_penalty(env.battle1, pos, int(data[pos]), est, ff_scale)
                    if p < 0.0:
                        ff_actions += 1
                    penalty += p
                penalty = capped_penalty(penalty, ff_spent)      # a win is a win: the battle total never passes -1.0
                ff_spent += -penalty
                ff_total += penalty
                decisions += 1
            obs, reward, term, trunc, _ = wrapped.step(data)
            done = bool(term or trunc)
            if done:
                ff_spent = 0.0                                   # the next battle starts from zero
                obs, _ = wrapped.reset()
            conn.send((obs, float(reward), done, action_masks(env.battle1), float(penalty)))
```
4. In the `finally:` block, after the matchup print, add:
```python
        if ff_scale:
            print(f"friendly fire actions={ff_actions} total_penalty={ff_total:.3f} decisions={decisions} ({cfg['name']})", file=sys.stderr, flush=True)
```

In `ShowdownVecEnv.__init__` add the parameter `friendly_fire: float = 0.0` after `matchup`, put `"friendly_fire": friendly_fire` in the worker `cfg` dict, and add `self.last_won = np.zeros(n, dtype=np.float32)` next to `self.results`.

In `step`, replace the loop body's unpack and bookkeeping with:
```python
        rewards, dones, obs, masks = np.zeros(self.n, np.float32), np.zeros(self.n, np.float32), [], []
        won = np.zeros(self.n, np.float32)
        for i, conn in enumerate(self.conns):
            o, r, d, m, penalty = self._recv(conn)
            rewards[i], dones[i] = r + penalty, float(d)
            obs.append(o)
            masks.append(m)
            if d:
                won[i] = 1.0 if r > 0 else 0.0      # from the sparse win/loss reward, before the penalty
                self.results.append(won[i])
        self.last_won = won
        self.obs, self.masks = np.stack(obs).astype(np.float32), np.stack(masks)
        return rewards, dones
```

- [ ] **Step 7: Pass the key from the config**

In `xp/showdown/phase.py` `make_env`, change the `ShowdownVecEnv(...)` call to add `friendly_fire=float(cfg["env"].get("friendly_fire", 0.0))` after the `matchup=` argument.

- [ ] **Step 8: Add a server test**

Append to `tests/test_showdown_vec.py` (uses the module's `pool` fixture, `FORMAT`, `restrict_second`, `ShowdownVecEnv`, `np`):

```python
def test_friendly_fire_rewards_stay_bounded_and_wins_are_tracked_separately(pool):
    env = ShowdownVecEnv(2, FORMAT, pool, "damage", seed=3, name="ff", friendly_fire=1.0)
    try:
        for _ in range(40):
            a0 = env.masks[:, 0].argmax(axis=1)
            a1 = np.array([restrict_second(env.masks[i, 1], a0[i]).argmax() for i in range(env.n)])
            rewards, dones = env.step(np.stack([a0, a1], axis=1))
            assert (rewards >= -2.0).all() and (rewards <= 1.0).all()     # loss -1 plus at most the -1.0 battle cap
            assert (env.last_won <= dones).all()
    finally:
        env.close()
```

- [ ] **Step 9: Run the fast suite and the server tests**

Run: `.venv/bin/pytest -q` then, with the server started as in `RESUME.md` (or let `scripts/run_showdown.sh` start it), `XP_NODE=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin/node .venv/bin/pytest -m showdown tests/test_showdown_vec.py -q`
Expected: all pass. (If no server is available in the session, say so; the server test then stays unrun and the Task 7 smoke covers it.)

- [ ] **Step 10: Commit**

```bash
git add xp/showdown/vec_env.py xp/showdown/ppo.py xp/showdown/phase.py tests/toy_showdown.py tests/test_showdown_ppo.py tests/test_showdown_vec.py
git commit -m "feat: friendly-fire penalty in the env worker; wins tracked separately from reward" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The tempo block (`tempo.py`)

**Files:**
- Create: `xp/showdown/tempo.py`
- Modify: `xp/showdown/encode.py` (add the constant `TEMPO_DIM = 77` only; wiring is Task 4)
- Test: `tests/test_showdown_tempo.py`

**Interfaces:**
- Consumes: `xp.showdown.estimate._fallback_stats(mon)` (level-50 fallback stats from base stats), `xp.showdown.matchup._moves` and `_live`, `xp.showdown.encode.N_MOVES`.
- Produces: `effective_speed(mon, tailwind: bool, foe: bool) -> float`; `speed_order(own_actives, opp_actives, own_tailwind: bool, opp_tailwind: bool, trick_room: bool) -> np.ndarray` (4 floats); `turns_block(battle) -> np.ndarray` (9 floats); `move_flags(move) -> np.ndarray` (4 floats); `flags_block(own_actives, opp_actives) -> np.ndarray` (64 floats); `tempo_block(battle) -> np.ndarray` (77 floats, order speed, turns, flags); constants `TRICK_ROOM_TURNS=5`, `TAILWIND_TURNS=4`, `SCREEN_TURNS=5`, `LIGHT_CLAY_SCREEN_TURNS=8`, `SETS_TRICK_ROOM`, `SETS_TAILWIND`, `SETS_SCREEN`.

Layout of `turns_block`: index 0 Trick Room; 1 own Tailwind; 2, 3, 4 own Reflect, Light Screen, Aurora Veil; 5 opposing Tailwind; 6, 7, 8 opposing Reflect, Light Screen, Aurora Veil. Each is `turns left / full duration`, 0 when not active. Layout of `flags_block`: my active 0's four move slots, my active 1's, the opposing active 0's, the opposing active 1's; four flags per slot in the order `sets_trick_room, sets_tailwind, sets_screen, hits_ally`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_tempo.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Field, Move, Pokemon, SideCondition, Status  # noqa: E402

from xp.showdown import encode as E  # noqa: E402
from xp.showdown import tempo as T  # noqa: E402
from xp.showdown.estimate import _fallback_stats  # noqa: E402


def mon(species, *moves, spe=None):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    if spe is not None:
        p._stats = {"hp": 150, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": spe}
    return p


def test_dimension_constants_agree():
    assert T.SPEED_DIM + T.TURNS_DIM + T.FLAGS_DIM == E.TEMPO_DIM == 77


def test_faster_own_mon_reads_plus_one_and_trick_room_flips_it():
    own, foe = mon("garchomp", spe=150), mon("heatran")           # the foe's Speed is the level-50 fallback, about 97
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == 1.0
    assert T.speed_order([own, None], [foe, None], False, False, True)[0] == -1.0


def test_tailwind_on_the_foe_side_can_make_it_faster():
    own, foe = mon("garchomp", spe=150), mon("heatran")
    assert T.speed_order([own, None], [foe, None], False, True, False)[0] == -1.0     # about 194 against 150


def test_paralysis_boosts_and_a_scarf_change_the_order():
    own, foe = mon("garchomp", spe=150), mon("heatran")
    own._status = Status.PAR
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == -1.0    # 75 against about 97
    own._status = None
    own._boosts["spe"] = -2
    assert T.effective_speed(own, False, False) == pytest.approx(75.0)                # a -2 stage halves 150
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == -1.0    # 75 against about 97
    own._boosts["spe"] = 0
    own._item = "choicescarf"
    assert T.effective_speed(own, False, False) == pytest.approx(225.0)


def test_equal_speed_and_absent_mons_read_zero():
    foe = mon("heatran")
    own = mon("garchomp", spe=_fallback_stats(foe)["spe"])
    assert T.speed_order([own, None], [foe, None], False, False, False)[0] == 0.0
    assert T.speed_order([None, None], [foe, None], False, False, False).tolist() == [0.0] * 4


def battle_with(fields=None, side=None, opp_side=None, turn=4, items=()):
    team = {}
    for i, item in enumerate(items):
        m = mon("garchomp")
        m._item = item
        team[f"p1: {i}"] = m
    return SimpleNamespace(fields=fields or {}, side_conditions=side or {}, opponent_side_conditions=opp_side or {}, turn=turn, team=team)


def test_turns_left_for_each_condition():
    b = battle_with(fields={Field.TRICK_ROOM: 3}, side={SideCondition.TAILWIND: 3, SideCondition.REFLECT: 3},
                    opp_side={SideCondition.LIGHT_SCREEN: 2})
    out = T.turns_block(b)
    assert out[0] == pytest.approx(4 / 5)            # set on turn 3, lasts through turn 7, 4 left at turn 4
    assert out[1] == pytest.approx(3 / 4)            # tailwind: 3 + 4 - 4
    assert out[2] == pytest.approx(4 / 5)            # reflect: 3 + 5 - 4
    assert out[7] == pytest.approx(3 / 5)            # opposing light screen: 2 + 5 - 4
    assert out[3] == out[4] == out[5] == out[6] == out[8] == 0.0


def test_a_condition_at_or_past_expiry_reads_zero():
    b = battle_with(fields={Field.TRICK_ROOM: 3}, turn=8)
    assert T.turns_block(b)[0] == 0.0


def test_own_light_clay_lengthens_own_screens_only():
    b = battle_with(side={SideCondition.REFLECT: 3}, opp_side={SideCondition.REFLECT: 3}, items=("lightclay",))
    out = T.turns_block(b)
    assert out[2] == pytest.approx(7 / 8)            # 3 + 8 - 4
    assert out[6] == pytest.approx(4 / 5)            # the foe's item is hidden: assumed 5 turns


def test_move_flags_for_known_moves():
    f = lambda name: T.move_flags(Move(name, gen=9)).tolist()   # noqa: E731
    assert f("trickroom") == [1, 0, 0, 0] and f("tailwind") == [0, 1, 0, 0]
    assert f("reflect") == [0, 0, 1, 0] and f("lightscreen") == [0, 0, 1, 0] and f("auroraveil") == [0, 0, 1, 0]
    assert f("earthquake") == [0, 0, 0, 1]                      # hits the partner whatever the target
    assert f("heatwave") == [0, 0, 0, 0] and f("protect") == [0, 0, 0, 0] and f("closecombat") == [0, 0, 0, 0]
    assert T.move_flags(None).tolist() == [0, 0, 0, 0]


def test_flags_block_layout_and_foe_slots():
    own0 = mon("garchomp", "closecombat", "trickroom")
    foe1 = mon("heatran", "flamethrower", "earthquake")
    out = T.flags_block([own0, None], [None, foe1])
    assert out.shape == (T.FLAGS_DIM,)
    assert out[0 * 16 + 1 * 4 + 0] == 1.0                      # my active 0, move slot 1, sets_trick_room
    assert out[3 * 16 + 1 * 4 + 3] == 1.0                      # opposing active 1, move slot 1, hits_ally
    assert out.sum() == 2.0


def test_tempo_block_shape_dtype_and_order():
    own0, foe0 = mon("garchomp", "trickroom", spe=150), mon("heatran")
    b = battle_with(fields={Field.TRICK_ROOM: 3}, turn=4)
    b.active_pokemon, b.opponent_active_pokemon = [own0, None], [foe0, None]
    out = T.tempo_block(b)
    assert out.shape == (E.TEMPO_DIM,) and out.dtype == np.float32
    assert out[0] == -1.0                                      # speed order, flipped by Trick Room
    assert out[T.SPEED_DIM] == pytest.approx(4 / 5)            # remaining turns of Trick Room
    assert out[T.SPEED_DIM + T.TURNS_DIM + 0] == 1.0           # my active 0's first move slot (trickroom) sets_trick_room
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_showdown_tempo.py -q`
Expected: FAIL (`ModuleNotFoundError` / missing attributes).

- [ ] **Step 3: Add the constant to `encode.py`**

In `xp/showdown/encode.py`, after the `MATCHUP_DIM = ...` line add:

```python
TEMPO_DIM = 4 + 9 + 4 * N_MOVES * 4                # speed order 4 + remaining turns 9 + 16 move flags for each of 4 active Pokemon = 77
```

- [ ] **Step 4: Write `tempo.py`**

Create `xp/showdown/tempo.py`:

```python
"""Tempo observation block (spec 2026-11-02): speed order, remaining turns of field effects, move flags.

The agent is taken to know what every move does (matchup spec); the numbers below are computed in code from battle state.
Foe Speed stats are unknown to poke-env, so they use the matchup spec's fallback spread (level 50, 0 EVs, 31 IVs, neutral).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from poke_env.battle import Field, Move, MoveCategory, Pokemon, SideCondition, Status

from xp.showdown.encode import N_MOVES, TEMPO_DIM
from xp.showdown.estimate import _fallback_stats
from xp.showdown.matchup import _live, _moves

TRICK_ROOM_TURNS = 5
TAILWIND_TURNS = 4
SCREEN_TURNS = 5
LIGHT_CLAY_SCREEN_TURNS = 8
SCREENS = (SideCondition.REFLECT, SideCondition.LIGHT_SCREEN, SideCondition.AURORA_VEIL)
SETS_TRICK_ROOM = frozenset({"trickroom"})
SETS_TAILWIND = frozenset({"tailwind"})
SETS_SCREEN = frozenset({"reflect", "lightscreen", "auroraveil"})
N_FLAGS = 4
SPEED_DIM = 4
TURNS_DIM = 9
FLAGS_DIM = 4 * N_MOVES * N_FLAGS


def _boost_multiplier(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def effective_speed(mon: Pokemon, tailwind: bool, foe: bool) -> float:
    """Speed after boosts, paralysis, a known Choice Scarf and Tailwind. Foe (and any mon with unknown stats) uses the fallback spread."""
    known = mon.stats.get("spe") if mon.stats else None
    base = _fallback_stats(mon)["spe"] if (foe or not isinstance(known, (int, float))) else known
    speed = base * _boost_multiplier(mon.boosts.get("spe", 0))
    if mon.status == Status.PAR:
        speed *= 0.5
    if mon.item == "choicescarf":      # foe items are unknown, so this only ever fires for own Pokemon
        speed *= 1.5
    if tailwind:
        speed *= 2.0
    return float(speed)


def speed_order(own_actives: Sequence[Optional[Pokemon]], opp_actives: Sequence[Optional[Pokemon]],
                own_tailwind: bool, opp_tailwind: bool, trick_room: bool) -> np.ndarray:
    """For (my 0, my 1) x (foe 0, foe 1): +1 if mine moves first, -1 if theirs does, 0 for a tie or an absent mon. Trick Room flips it."""
    out = np.zeros(SPEED_DIM, dtype=np.float32)
    for i, mine in enumerate(list(own_actives)[:2]):
        for j, foe in enumerate(list(opp_actives)[:2]):
            if not (_live(mine) and _live(foe)):
                continue
            a, b = effective_speed(mine, own_tailwind, False), effective_speed(foe, opp_tailwind, True)
            if a == b:
                continue
            first = 1.0 if a > b else -1.0
            out[i * 2 + j] = -first if trick_room else first
    return out


def _left(start: int, duration: int, turn: int) -> float:
    return float(np.clip(start + duration - turn, 0, duration)) / duration


def turns_block(battle) -> np.ndarray:
    """Remaining turns / full duration: Trick Room, then own Tailwind and screens, then the foe's. poke-env stores each start turn."""
    out = np.zeros(TURNS_DIM, dtype=np.float32)
    turn = battle.turn
    if Field.TRICK_ROOM in battle.fields:
        out[0] = _left(battle.fields[Field.TRICK_ROOM], TRICK_ROOM_TURNS, turn)
    own_clay = any(m.item == "lightclay" for m in battle.team.values())
    for base, conditions, clay in ((1, battle.side_conditions, own_clay), (5, battle.opponent_side_conditions, False)):
        if SideCondition.TAILWIND in conditions:
            out[base] = _left(conditions[SideCondition.TAILWIND], TAILWIND_TURNS, turn)
        for i, screen in enumerate(SCREENS):
            if screen in conditions:
                out[base + 1 + i] = _left(conditions[screen], LIGHT_CLAY_SCREEN_TURNS if clay else SCREEN_TURNS, turn)
    return out


def move_flags(move: Optional[Move]) -> np.ndarray:
    """sets_trick_room, sets_tailwind, sets_screen, hits_ally (a damaging allAdjacent move hits the partner whatever the target)."""
    if move is None:
        return np.zeros(N_FLAGS, dtype=np.float32)
    damaging = move.category != MoveCategory.STATUS and move.base_power > 0
    return np.array([move.id in SETS_TRICK_ROOM, move.id in SETS_TAILWIND, move.id in SETS_SCREEN,
                     damaging and move.entry.get("target") == "allAdjacent"], dtype=np.float32)


def flags_block(own_actives: Sequence[Optional[Pokemon]], opp_actives: Sequence[Optional[Pokemon]]) -> np.ndarray:
    """Four flags for each of the four move slots of each of my 2 actives and the 2 opposing actives (foe slots stay 0 until revealed)."""
    mons: List[Optional[Pokemon]] = (list(own_actives) + [None] * 2)[:2] + (list(opp_actives) + [None] * 2)[:2]
    rows = []
    for mon in mons:
        rows.extend(move_flags(m) for m in _moves(mon if _live(mon) else None))
    return np.concatenate(rows).astype(np.float32)


def tempo_block(battle) -> np.ndarray:
    own, opp = list(battle.active_pokemon), list(battle.opponent_active_pokemon)
    out = np.concatenate([
        speed_order(own, opp, SideCondition.TAILWIND in battle.side_conditions,
                    SideCondition.TAILWIND in battle.opponent_side_conditions, Field.TRICK_ROOM in battle.fields),
        turns_block(battle),
        flags_block(own, opp),
    ]).astype(np.float32)
    assert out.shape == (TEMPO_DIM,)
    return out
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/test_showdown_tempo.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add xp/showdown/tempo.py xp/showdown/encode.py tests/test_showdown_tempo.py
git commit -m "feat: tempo block (speed order, remaining turns, move flags)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Switch the tempo block on (`obs_dim`, encoder, env, config)

**Files:**
- Modify: `xp/showdown/encode.py` (`obs_dim`, `encode_battle`)
- Modify: `xp/showdown/env.py`
- Modify: `xp/showdown/vec_env.py` (worker and `ShowdownVecEnv`)
- Modify: `xp/showdown/phase.py` (`make_env`, `build_learner`)
- Modify: `scripts/record_showdown_replays.py:71`
- Test: `tests/test_showdown_encode.py`

**Interfaces:**
- Consumes: `tempo_block` from Task 3.
- Produces: `obs_dim(matchup: bool = False, tempo: bool = False) -> int`; `encode_battle(battle, matchup=False, estimator=None, tempo=False)`; `XPDoublesEnv(matchup=False, tempo=False, **kwargs)`; `ShowdownVecEnv(..., tempo: bool = False)`; config key `env.tempo`. The tempo block is appended after the matchup block when both are on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_showdown_encode.py`:

```python
def test_obs_dim_stacks_the_optional_blocks():
    assert E.obs_dim() == E.OBS_DIM and E.obs_dim(False, False) == E.OBS_DIM
    assert E.obs_dim(True) == E.OBS_DIM + E.MATCHUP_DIM
    assert E.obs_dim(False, True) == E.OBS_DIM + E.TEMPO_DIM
    assert E.obs_dim(True, True) == E.OBS_DIM + E.MATCHUP_DIM + E.TEMPO_DIM


def test_tempo_off_is_identical_and_tempo_on_appends_the_block():
    own, foe = mon("garchomp", "trickroom"), mon("heatran", "flamethrower")
    b = battle([own], [foe], [own, None], [foe, None])
    base = E.encode_battle(b)
    assert np.array_equal(E.encode_battle(b, tempo=False), base) and base.shape == (E.OBS_DIM,)
    with_tempo = E.encode_battle(b, tempo=True)
    assert with_tempo.shape == (E.OBS_DIM + E.TEMPO_DIM,)
    assert np.array_equal(with_tempo[:E.OBS_DIM], base)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_showdown_encode.py -q`
Expected: FAIL (`obs_dim() takes ... positional argument` / unexpected keyword `tempo`).

- [ ] **Step 3: Edit `encode.py`**

Replace `obs_dim`:

```python
def obs_dim(matchup: bool = False, tempo: bool = False) -> int:
    """Length of the observation vector; the matchup block (spec 2026-10-29) and the tempo block (spec 2026-11-02) are appended only when switched on."""
    return OBS_DIM + (MATCHUP_DIM if matchup else 0) + (TEMPO_DIM if tempo else 0)
```

Change the `encode_battle` signature and tail:

```python
def encode_battle(battle, matchup: bool = False, estimator=None, tempo: bool = False) -> np.ndarray:
    """Flat float32 vector of length obs_dim(matchup, tempo) for a DoubleBattle (own side, opponent side, field[, matchup block][, tempo block])."""
```
and replace the end of the function (from `if matchup:` on) with:

```python
    if matchup:
        from xp.showdown.matchup import matchup_block  # lazy: matchup.py imports this module's constants
        if estimator is None:
            from xp.showdown.estimate import PokeEnvEstimator
            estimator = PokeEnvEstimator(battle)
        out = np.concatenate([out, matchup_block(list(battle.active_pokemon), list(battle.opponent_active_pokemon), estimator)])
    if tempo:
        from xp.showdown.tempo import tempo_block  # lazy: tempo.py imports this module's constants
        out = np.concatenate([out, tempo_block(battle)])
    assert out.shape == (obs_dim(matchup, tempo),)
    return out.astype(np.float32)
```

- [ ] **Step 4: Run the encode tests**

Run: `.venv/bin/pytest tests/test_showdown_encode.py tests/test_showdown_tempo.py tests/test_showdown_matchup.py -q`
Expected: all pass.

- [ ] **Step 5: Edit `env.py`**

In `xp/showdown/env.py`:

```python
    def __init__(self, matchup: bool = False, tempo: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.matchup = bool(matchup)
        self.tempo = bool(tempo)
```
change the space line to `space = Box(-2.0, 2.0, shape=(obs_dim(self.matchup, self.tempo),), dtype=np.float32)`, and in `embed_battle` replace `return encode_battle(battle)` (the `not self.matchup` branch) with `return encode_battle(battle, tempo=self.tempo)` and the matchup branch call with `encode_battle(battle, matchup=True, estimator=est, tempo=self.tempo)`.

- [ ] **Step 6: Edit `vec_env.py`, `phase.py` and the recorder**

`xp/showdown/vec_env.py`: in `_worker`, add `tempo=cfg.get("tempo", False),` to the `XPDoublesEnv(...)` call (next to `matchup=`). In `ShowdownVecEnv.__init__` add the parameter `tempo: bool = False` after `friendly_fire` and `"tempo": tempo` in the worker `cfg` dict.

`xp/showdown/phase.py`: in `make_env` add `tempo=bool(cfg["env"].get("tempo", False))` after the `friendly_fire=` argument; in `build_learner` replace `dim = obs_dim(bool(cfg["env"].get("matchup", False)))` with
```python
    dim = obs_dim(bool(cfg["env"].get("matchup", False)), bool(cfg["env"].get("tempo", False)))
```

`scripts/record_showdown_replays.py`: in the `ShowdownVecEnv(...)` call at line 71 add `tempo=bool(cfg["env"].get("tempo", False)), friendly_fire=float(cfg["env"].get("friendly_fire", 0.0))` after the `matchup=` argument. Also grep the file for `obs_dim(` or `OBS_DIM` and make any learner construction use `obs_dim(matchup, tempo)` from the config the same way (`grep -n "obs_dim\|OBS_DIM\|Learner(" scripts/record_showdown_replays.py`).

- [ ] **Step 7: Run the fast suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (the showdown-marked tests stay excluded).

- [ ] **Step 8: Commit**

```bash
git add xp/showdown/encode.py xp/showdown/env.py xp/showdown/vec_env.py xp/showdown/phase.py scripts/record_showdown_replays.py tests/test_showdown_encode.py
git commit -m "feat: switchable tempo block in the observation (env.tempo)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `TempoPlayer` (opponent `tempo`)

**Files:**
- Modify: `xp/showdown/opponents.py`
- Test: `tests/test_showdown_tempo_opponent.py`

**Interfaces:**
- Consumes: `xp.showdown.tempo.effective_speed`; `DamageHeuristicPlayer` (existing).
- Produces: `TempoPlayer(DamageHeuristicPlayer)` with `choose_move(battle) -> DoubleBattleOrder`, `_setter(battle, pos, used) -> Optional[Move]`, counters `setter_counts` (a `Counter` of move ids), `battles_seen` and `battles_with_setter` (sets of `battle.battle_tag`); `OPPONENTS = LADDER + ("tempo",)`; `make_opponent("tempo", fmt)` returns a `TempoPlayer`. `LADDER` stays `("random", "maxbp", "damage")`.

Rules (first match per slot, at most one setter per slot per turn, a condition is never set twice on one turn):
1. Trick Room: `trickroom` available, Trick Room not up, and this side's mean live-active Speed is below the foes'.
2. Tailwind: `tailwind` available, not up on this side, Trick Room not up.
3. Screens: `auroraveil` when snow (`Weather.SNOWSCAPE` or `Weather.HAIL`) is up and Aurora Veil is not; else `reflect` when the foes' revealed damaging moves are mostly physical, `lightscreen` when mostly special; never when one is already up; nothing when no foe move is revealed or the counts tie.
4. Otherwise the parent's order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showdown_tempo_opponent.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import Field, Move, Pokemon, SideCondition, Weather  # noqa: E402
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder  # noqa: E402

from xp.showdown import opponents as O  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"


def mon(species, *moves, spe=None):
    p = Pokemon(gen=9, species=species)
    p._moves = {m: Move(m, gen=9) for m in moves}
    if spe is not None:
        p._stats = {"hp": 150, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": spe}
    return p


def make(available0=(), available1=(), mine=(80, 80), foes=("heatran", "heatran"), foe_moves=("flamethrower",), **kw):
    """The opponent's view: its own two actives are slower or faster than the foes' fallback Speed (about 97)."""
    a0, a1 = mon("garchomp", *available0, spe=mine[0]), mon("garchomp", *available1, spe=mine[1])
    f0, f1 = mon(foes[0], *foe_moves), mon(foes[1], *foe_moves)
    return SimpleNamespace(
        battle_tag="b1", active_pokemon=[a0, a1], opponent_active_pokemon=[f0, f1],
        available_moves=[[Move(m, gen=9) for m in available0], [Move(m, gen=9) for m in available1]],
        force_switch=[False, False], fields=kw.get("fields", {}), weather=kw.get("weather", {}),
        side_conditions=kw.get("side", {}), opponent_side_conditions=kw.get("opp_side", {}), team={})


@pytest.fixture
def player():
    return O.make_opponent("tempo", FORMAT)


def test_make_opponent_knows_tempo_and_the_ladder_is_unchanged():
    assert O.LADDER == ("random", "maxbp", "damage") and "tempo" in O.OPPONENTS
    assert isinstance(O.make_opponent("tempo", FORMAT), O.TempoPlayer)
    with pytest.raises(ValueError):
        O.make_opponent("nope", FORMAT)


def test_trick_room_when_slower_and_not_up(player):
    b = make(available0=("trickroom", "closecombat"), mine=(80, 80))
    assert player._setter(b, 0, set()).id == "trickroom"


def test_no_trick_room_when_faster_already_up_or_unavailable(player):
    assert player._setter(make(available0=("trickroom",), mine=(200, 200)), 0, set()) is None
    assert player._setter(make(available0=("trickroom",), mine=(80, 80), fields={Field.TRICK_ROOM: 2}), 0, set()) is None
    assert player._setter(make(available0=("closecombat",), mine=(80, 80)), 0, set()) is None


def test_tailwind_when_not_up_and_no_trick_room(player):
    b = make(available0=("tailwind",))
    assert player._setter(b, 0, set()).id == "tailwind"
    assert player._setter(make(available0=("tailwind",), side={SideCondition.TAILWIND: 2}), 0, set()) is None
    assert player._setter(make(available0=("tailwind",), fields={Field.TRICK_ROOM: 2}), 0, set()) is None


def test_trick_room_takes_precedence_over_tailwind(player):
    b = make(available0=("tailwind", "trickroom"), mine=(80, 80))
    assert player._setter(b, 0, set()).id == "trickroom"


def test_reflect_against_physical_and_light_screen_against_special(player):
    phys = make(available0=("reflect", "lightscreen"), foe_moves=("closecombat", "earthquake"))
    assert player._setter(phys, 0, set()).id == "reflect"
    spec = make(available0=("reflect", "lightscreen"), foe_moves=("flamethrower", "psychic"))
    assert player._setter(spec, 0, set()).id == "lightscreen"


def test_no_screen_with_nothing_revealed_a_tie_or_one_already_up(player):
    assert player._setter(make(available0=("reflect",), foe_moves=()), 0, set()) is None
    assert player._setter(make(available0=("reflect",), foe_moves=("closecombat", "flamethrower")), 0, set()) is None
    up = make(available0=("reflect",), foe_moves=("closecombat",), side={SideCondition.REFLECT: 2})
    assert player._setter(up, 0, set()) is None


def test_aurora_veil_only_in_snow_and_not_recast(player):
    snow = make(available0=("auroraveil",), weather={Weather.SNOWSCAPE: 1})
    assert player._setter(snow, 0, set()).id == "auroraveil"
    assert player._setter(make(available0=("auroraveil",), foe_moves=("closecombat",)), 0, set()) is None
    up = make(available0=("auroraveil",), weather={Weather.HAIL: 1}, side={SideCondition.AURORA_VEIL: 2})
    assert player._setter(up, 0, set()) is None


def test_both_slots_never_set_the_same_condition(player, monkeypatch):
    b = make(available0=("trickroom", "closecombat"), available1=("trickroom", "closecombat"), mine=(80, 80))
    parent = DamageOrders(b)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    order = player.choose_move(b)
    assert order.first_order.order.id == "trickroom"
    assert order.second_order is parent.second_order      # the second slot keeps the parent's order


def DamageOrders(b):
    cc0, cc1 = b.available_moves[0][1], b.available_moves[1][1]
    return DoubleBattleOrder(SingleBattleOrder(cc0, move_target=1), SingleBattleOrder(cc1, move_target=2))


def test_falls_back_to_the_parent_when_no_setter_applies(player, monkeypatch):
    b = make(available0=("closecombat",), available1=("closecombat",))
    parent = DoubleBattleOrder(SingleBattleOrder(b.available_moves[0][0], move_target=1),
                               SingleBattleOrder(b.available_moves[1][0], move_target=2))
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    order = player.choose_move(b)
    assert order.first_order is parent.first_order and order.second_order is parent.second_order


def test_a_setter_keeps_the_mega_flag_and_targets_nobody(player, monkeypatch):
    b = make(available0=("trickroom", "closecombat"), mine=(80, 80))
    parent = DoubleBattleOrder(SingleBattleOrder(b.available_moves[0][1], mega=True, move_target=1), None)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    order = player.choose_move(b)
    assert order.first_order.order.id == "trickroom" and order.first_order.mega and order.first_order.move_target == 0


def test_forced_switches_are_left_to_the_parent(player, monkeypatch):
    b = make(available0=("trickroom",), mine=(80, 80))
    b.force_switch = [True, False]
    parent = DoubleBattleOrder(None, None)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    assert player.choose_move(b) is parent


def test_counters_record_what_was_set(player, monkeypatch):
    b = make(available0=("trickroom", "closecombat"), mine=(80, 80))
    parent = DoubleBattleOrder(SingleBattleOrder(b.available_moves[0][1], move_target=1), None)
    monkeypatch.setattr(O.DamageHeuristicPlayer, "choose_move", lambda self, battle: parent)
    player.choose_move(b)
    assert player.setter_counts["trickroom"] == 1 and player.battles_with_setter == {"b1"} and player.battles_seen == {"b1"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_showdown_tempo_opponent.py -q`
Expected: FAIL (`AttributeError: module ... has no attribute 'OPPONENTS'`).

- [ ] **Step 3: Implement**

In `xp/showdown/opponents.py` change the imports and add the class and registry. Update the import block to:

```python
import random
from collections import Counter

from poke_env.battle import DoubleBattle, Field, Move, MoveCategory, SideCondition, Weather
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder
```
(keeping the existing `from __future__` line and docstring; `DoubleBattle` and `Move` were already imported). Replace the `LADDER` line and `make_opponent` tail as follows, and add the class after `DamageHeuristicPlayer`:

```python
LADDER = ("random", "maxbp", "damage")
OPPONENTS = LADDER + ("tempo",)
```

```python
class TempoPlayer(DamageHeuristicPlayer):
    """The damage heuristic plus field management (spec 2026-11-02): before attacking, a slot may set Trick Room, Tailwind
    or a screen by fixed rules. At most one setter per slot per turn, and a condition is never set twice on one turn.
    Counters (`setter_counts`, `battles_seen`, `battles_with_setter`) let a run report how often it actually sets anything."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setter_counts: Counter = Counter()
        self.battles_seen: set = set()
        self.battles_with_setter: set = set()

    def choose_move(self, battle):
        order = super().choose_move(battle)
        self.battles_seen.add(battle.battle_tag)
        if any(battle.force_switch) or not isinstance(order, DoubleBattleOrder):
            return order
        slots, used = [order.first_order, order.second_order], set()
        for pos in range(2):
            move = self._setter(battle, pos, used)
            if move is None:
                continue
            old = slots[pos]
            slots[pos] = SingleBattleOrder(move, mega=bool(getattr(old, "mega", False)))   # a setter targets nobody (move_target 0)
            self.setter_counts[move.id] += 1
            self.battles_with_setter.add(battle.battle_tag)
        return DoubleBattleOrder(slots[0], slots[1])

    def _setter(self, battle, pos, used):
        """The setter move this slot should use now, or None. Adds the chosen condition to `used`."""
        from xp.showdown.tempo import effective_speed   # lazy: tempo.py imports encode.py

        mon = battle.active_pokemon[pos]
        if mon is None or mon.fainted:
            return None
        avail = {m.id: m for m in battle.available_moves[pos]}
        mine = [m for m in battle.active_pokemon if m is not None and not m.fainted]
        foes = [f for f in battle.opponent_active_pokemon if f is not None and not f.fainted]
        trick_room = Field.TRICK_ROOM in battle.fields

        def pick(move_id):
            used.add(move_id)
            return avail[move_id]

        if "trickroom" in avail and "trickroom" not in used and not trick_room and foes:
            mine_speed = sum(effective_speed(m, SideCondition.TAILWIND in battle.side_conditions, False) for m in mine) / len(mine)
            foe_speed = sum(effective_speed(f, SideCondition.TAILWIND in battle.opponent_side_conditions, True) for f in foes) / len(foes)
            if mine_speed < foe_speed:
                return pick("trickroom")
        if "tailwind" in avail and "tailwind" not in used and SideCondition.TAILWIND not in battle.side_conditions and not trick_room:
            return pick("tailwind")
        if ("auroraveil" in avail and "auroraveil" not in used and SideCondition.AURORA_VEIL not in battle.side_conditions
                and (Weather.SNOWSCAPE in battle.weather or Weather.HAIL in battle.weather)):
            return pick("auroraveil")
        revealed = [mv for f in foes for mv in f.moves.values() if mv.base_power > 0]
        phys = sum(mv.category == MoveCategory.PHYSICAL for mv in revealed)
        spec = sum(mv.category == MoveCategory.SPECIAL for mv in revealed)
        wanted = ("reflect", SideCondition.REFLECT) if phys > spec else ("lightscreen", SideCondition.LIGHT_SCREEN) if spec > phys else None
        if wanted and wanted[0] in avail and wanted[0] not in used and wanted[1] not in battle.side_conditions:
            return pick(wanted[0])
        return None
```

and replace `make_opponent`:

```python
def make_opponent(name: str, battle_format: str) -> Player:
    """A scripted opponent that never connects to the server (the env plays its moves)."""
    cls = {"random": RandomPlayer, "maxbp": MaxBasePowerPlayer, "damage": DamageHeuristicPlayer, "tempo": TempoPlayer}.get(name)
    if cls is None:
        raise ValueError(f"unknown opponent {name!r}; choose from {OPPONENTS}")
    return cls(battle_format=battle_format, start_listening=False)
```

Also update the module docstring's first line to mention the extra rung: "the 3-rung ladder Random < MaxBasePower < DamageHeuristic, plus the tempo-aware TempoPlayer".

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_showdown_tempo_opponent.py -q`
Expected: all pass. If `test_a_setter_keeps_the_mega_flag...` fails because `Player.__init__` rejects extra constructor args, check that `make_opponent` still passes `battle_format` and `start_listening` only, and that `TempoPlayer.__init__` forwards `*args, **kwargs` unchanged.

- [ ] **Step 5: Report the counters from the worker**

In `xp/showdown/vec_env.py` `_worker`, keep a reference to the opponent: replace `wrapped = SingleAgentWrapper(env, make_opponent(cfg["opponent"], cfg["battle_format"]))` with
```python
    opponent = make_opponent(cfg["opponent"], cfg["battle_format"])
    wrapped = SingleAgentWrapper(env, opponent)
```
and in the `finally:` block add
```python
        if cfg["opponent"] == "tempo":
            print(f"tempo opponent battles={len(opponent.battles_seen)} with_setter={len(opponent.battles_with_setter)} "
                  f"setters={dict(opponent.setter_counts)} ({cfg['name']})", file=sys.stderr, flush=True)
```
Also in `xp/showdown/gate.py` allow the new rung: change `choices=["random", "maxbp", "damage"]` to `choices=["random", "maxbp", "damage", "tempo"]`.

- [ ] **Step 6: Run the fast suite and commit**

Run: `.venv/bin/pytest -q`
Expected: all pass.

```bash
git add xp/showdown/opponents.py xp/showdown/vec_env.py xp/showdown/gate.py tests/test_showdown_tempo_opponent.py
git commit -m "feat: TempoPlayer opponent (Trick Room, Tailwind, screens) with usage counters" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Tempo team pools

The edits below were each checked with `node pokemon-showdown validate-team gen9championsvgc2026regmc` while writing this plan (individually and all together). No Pokemon on the eval teams other than Indeedee-F can legally learn Trick Room in this format, so `eval1` swaps Clefable for an Indeedee-F set.

**Files:**
- Create: `configs/showdown_teams_tempo/tune/tune1.txt`, `tune2.txt`, `tune3.txt`; `configs/showdown_teams_tempo/eval/eval1.txt`, `eval2.txt`, `eval3.txt`
- Test: `tests/test_showdown_tempo_teams.py`

**Interfaces:**
- Consumes: `xp.showdown.teams.load_pool`, `assert_disjoint`, `pool_digest`, `validate_team`.
- Produces: two pool directories; guard test; the two digests (printed in Step 4) pinned by Task 7's configs.

- [ ] **Step 1: Write the failing guard test**

Create `tests/test_showdown_tempo_teams.py`:

```python
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from xp.showdown.teams import assert_disjoint, load_pool, validate_team

ROOT = Path(__file__).resolve().parent.parent
TEMPO = ROOT / "configs" / "showdown_teams_tempo"
FORMAT = "gen9championsvgc2026regmc"
KIND = {"Trick Room": "trick_room", "Tailwind": "tailwind", "Reflect": "screen", "Light Screen": "screen", "Aurora Veil": "screen"}


def kinds(text: str) -> set:
    return {KIND[m] for m in re.findall(r"^- (.+?)\s*$", text, re.M) if m in KIND}


@pytest.mark.parametrize("stage", ["tune", "eval"])
def test_every_team_carries_a_setter_and_each_pool_covers_all_three_kinds(stage):
    pool = load_pool(TEMPO / stage)
    assert len(pool) == 3
    for name, text in pool.items():
        assert kinds(text), f"{name} has no Trick Room, Tailwind or screen move"
    assert set().union(*(kinds(t) for t in pool.values())) == {"trick_room", "tailwind", "screen"}


def test_tune_and_eval_pools_are_disjoint():
    assert_disjoint(load_pool(TEMPO / "tune"), load_pool(TEMPO / "eval"))


def test_the_original_pools_are_untouched():
    from xp.showdown.teams import pool_digest

    assert pool_digest(load_pool(ROOT / "configs" / "showdown_teams" / "tune")) == "4ebb9de5ef49"
    assert pool_digest(load_pool(ROOT / "configs" / "showdown_teams" / "eval")) == "bcc8a297c758"


@pytest.mark.showdown
def test_every_tempo_team_validates_on_the_server_checkout():
    showdown_dir = Path(os.environ.get("XP_SHOWDOWN_DIR", Path.home() / "xp-showdown-spike" / "pokemon-showdown"))
    if not showdown_dir.exists():
        pytest.skip(f"no Showdown checkout at {showdown_dir} (set XP_SHOWDOWN_DIR)")
    for stage in ("tune", "eval"):
        for text in load_pool(TEMPO / stage).values():
            validate_team(text, FORMAT, showdown_dir, node=os.environ.get("XP_NODE", "node"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_showdown_tempo_teams.py -q`
Expected: FAIL (`no team files in .../showdown_teams_tempo/tune`).

- [ ] **Step 3: Create the pools**

Copy the originals, then apply the edits with the Edit tool (exact strings; each `- <move>` line occurs once in its file):

```bash
mkdir -p configs/showdown_teams_tempo && cp -R configs/showdown_teams/tune configs/showdown_teams/eval configs/showdown_teams_tempo/
```

| File | Edit |
|---|---|
| `tune/tune1.txt` | none (already carries Trick Room on Indeedee-F) |
| `tune/tune2.txt` | on Pelipper replace the line `- Ice Beam` with `- Tailwind` (the team then has Trick Room and Tailwind) |
| `tune/tune3.txt` | on Weavile replace the line `- Ice Spinner` with `- Reflect` (the team then has Tailwind and Reflect) |
| `eval/eval1.txt` | on Raichu replace `- Zap Cannon` with `- Light Screen`; replace the whole Clefable block (the 8 lines from `Clefable @ Rocky Helmet` to its fourth move) with the Indeedee-F block below |
| `eval/eval2.txt` | on Dragonite replace `- Heat Wave` with `- Tailwind` |
| `eval/eval3.txt` | none (already carries Tailwind, Reflect and Light Screen) |

Indeedee-F block for `eval/eval1.txt` (same position as Clefable, blank line before and after as for the other Pokemon):

```
Indeedee-F @ Sitrus Berry
Ability: Psychic Surge
Level: 50
EVs: 32 HP / 2 SpA / 32 SpD
Calm Nature
- Follow Me
- Helping Hand
- Psychic
- Trick Room
```

Result: tune pool has Trick Room (tune1, tune2), Tailwind (tune2, tune3) and a screen (tune3); eval pool has Trick Room (eval1), Tailwind (eval2, eval3) and screens (eval1, eval3).

- [ ] **Step 4: Run the guard test, validate on the server checkout, print the digests**

Run: `.venv/bin/pytest tests/test_showdown_tempo_teams.py -q`
Expected: all pass (the server-marked test is excluded by default).

Then run the marked validation:
`XP_NODE=$HOME/xp-showdown-spike/node-v22.23.4-darwin-arm64/bin/node .venv/bin/pytest -m showdown tests/test_showdown_tempo_teams.py -q`
Expected: pass (this needs only the Showdown checkout, not a running server).

Then print the digests (write both down for Task 7):
```bash
.venv/bin/python -c "
from xp.showdown.teams import load_pool, pool_digest
for s in ('tune','eval'): print(s, pool_digest(load_pool(f'configs/showdown_teams_tempo/{s}')))"
```

- [ ] **Step 5: Commit**

```bash
git add configs/showdown_teams_tempo tests/test_showdown_tempo_teams.py
git commit -m "feat: tempo team pools (revised copies; every team carries a setter)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Configs, gate options, smoke check

**Files:**
- Create: `configs/showdown-tempo-base.yaml`, `configs/showdown-tempo.yaml`, `configs/showdown-tempo-smoke.yaml`
- Modify: `xp/showdown/gate.py` (pool and flags)
- Test: `tests/test_showdown_phase.py`

**Interfaces:**
- Consumes: the digests from Task 6 (`TUNE_SHA`, `EVAL_SHA` below); `xp.showdown.gate`.
- Produces: configs runnable by `scripts/run_showdown.sh`; gate flags `--teams-dir`, `--matchup`, `--tempo`, `--friendly-fire`.

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_showdown_phase.py` (it already has `ROOT`, `load_config`, `Phase`, `grid_size`, `pool_digest`, `load_pool`, `sp`):

```python
TEMPO_CONFIGS = ("showdown-tempo-base", "showdown-tempo", "showdown-tempo-smoke")


def test_tempo_config_team_hashes_match_the_files_in_the_repo():
    for name in TEMPO_CONFIGS:
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        for stage in ("tune", "eval"):
            assert cfg["env"][f"{stage}_teams"] == f"configs/showdown_teams_tempo/{stage}"
            assert cfg["env"][f"{stage}_teams_sha"] == pool_digest(load_pool(sp.team_dir(cfg, stage)))


def test_tempo_run_configs_differ_only_in_the_tempo_flag_and_are_protocol_legal():
    base = load_config(ROOT / "configs" / "showdown-tempo-base.yaml")
    full = load_config(ROOT / "configs" / "showdown-tempo.yaml")
    for cfg in (base, full):
        assert cfg["env"]["opponent"] == "tempo" and cfg["env"]["matchup"] is True and cfg["env"]["friendly_fire"] == 1.0
        assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 8
        phase = Phase.from_config(cfg)
        assert not set(phase.tune_seeds) & set(phase.eval_seeds)
    assert not base["env"].get("tempo", False) and full["env"]["tempo"] is True
    strip = lambda c: {**c, "env": {k: v for k, v in c["env"].items() if k != "tempo"}, "report": None}   # noqa: E731
    assert strip(base) == strip(full)
    assert Phase.from_config(base).config_hash != Phase.from_config(full).config_hash


def test_old_configs_still_hash_as_before_when_the_new_keys_are_absent():
    cfg = load_config(ROOT / "configs" / "showdown.yaml")
    assert "tempo" not in cfg["env"] and "friendly_fire" not in cfg["env"]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_showdown_phase.py -q`
Expected: FAIL (`FileNotFoundError` for `showdown-tempo-base.yaml`).

- [ ] **Step 3: Create the configs**

```bash
cp configs/showdown-matchup.yaml configs/showdown-tempo-base.yaml
cp configs/showdown-matchup.yaml configs/showdown-tempo.yaml
cp configs/showdown-matchup-smoke.yaml configs/showdown-tempo-smoke.yaml
```

In `configs/showdown-tempo-base.yaml` make these changes (leave every other line, including the grids, as copied):
- Replace the first two comment lines with:
```
# TEMPO BASE condition (spec docs/superpowers/specs/2026-11-02-showdown-tempo-design.md): tempo opponent, matchup block, friendly-fire penalty, NO tempo block. Run 1 of 2.
# Output folder: use `--out results/showdown-tempo-base`. Recorder: `scripts/record_showdown_replays.py --tuned results/showdown-tempo-base/showdown/tuning`.
```
- In the `env:` block, set exactly:
```yaml
env:
  format: gen9championsvgc2026regmc
  matchup: true
  friendly_fire: 1.0            # C: penalty = -C * estimated damage fraction on the partner (spec 2026-11-02); fixed, not tuned
  opponent: tempo               # new scripted opponent (spec 2026-11-02): damage heuristic plus Trick Room, Tailwind and screens
  tune_teams: configs/showdown_teams_tempo/tune
  eval_teams: configs/showdown_teams_tempo/eval
  tune_teams_sha: <TUNE_SHA>    # pool_digest of the directory, printed in Task 6 Step 4
  eval_teams_sha: <EVAL_SHA>
```
  where `<TUNE_SHA>` and `<EVAL_SHA>` are the two 12-character digests printed in Task 6 Step 4 (write the digits, not the angle brackets).
- In `budget:`, leave `steps: 250000` and `target_win: 0.30` as copied, but add after `target_win` the comment ` # budget S: confirm after the Task 7 gate and the user's approval before any RND result is read` (comment only; if the user later changes S, edit the number here before the run).
- In `report:` set `xlabel: Showdown VGC doubles vs the tempo opponent (single familiar level)` and `title: Tempo opponent, matchup block and friendly-fire penalty, tuned Kairi vs Kaiser`.

`configs/showdown-tempo.yaml` is the same as the base config with: the comment lines `# TEMPO condition (...): run 2 of 2, base plus the tempo block. Output folder: use `--out results/showdown-tempo`.`, one extra line `  tempo: true               # tempo observation block (spec 2026-11-02)` in `env:` right after `matchup: true`, and `title: Tempo block added, tuned Kairi vs Kaiser`.

`configs/showdown-tempo-smoke.yaml`: start from the copied smoke config, apply the same `env:` block as the base config plus `tempo: true` (so the smoke exercises every new path), and change its comment header to say it is the 75-second-style end-to-end check for the tempo runs (output folder `results/showdown-tempo-smoke`). Keep its small `budget`, `seeds` and grid values as copied.

- [ ] **Step 4: Extend the gate script**

In `xp/showdown/gate.py` change the constants and `main`:

```python
TEAMS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "showdown_teams"
```
stays as the default; add arguments and use them:

```python
    parser.add_argument("--teams-dir", default=str(TEAMS_DIR), help="directory holding tune/ and eval/ pools (use configs/showdown_teams_tempo for the tempo runs)")
    parser.add_argument("--matchup", action="store_true", help="matchup block on")
    parser.add_argument("--tempo", action="store_true", help="tempo block on")
    parser.add_argument("--friendly-fire", type=float, default=0.0, help="friendly-fire penalty constant C")
```
and replace the env and learner construction:

```python
    from xp.showdown.encode import obs_dim
    from xp.showdown.teams import load_pool
    from xp.showdown.vec_env import ShowdownVecEnv

    ppo = {**DEFAULT_PPO, "n_envs": args.n_envs}
    env = ShowdownVecEnv(args.n_envs, FORMAT, load_pool(Path(args.teams_dir) / "tune"), args.opponent, seed=args.seed,
                         matchup=args.matchup, tempo=args.tempo, friendly_fire=args.friendly_fire)
    try:
        result = measure(Learner(ppo, args.ent_coef, args.seed, obs_dim=obs_dim(args.matchup, args.tempo)), env, args.steps)
    finally:
        env.close()
```
(remove the old `from xp.showdown.teams import load_pool` / `ShowdownVecEnv` import lines and the old env/learner lines they replace). Add to `result.update(...)` the keys `"teams_dir": args.teams_dir, "matchup": args.matchup, "tempo": args.tempo, "friendly_fire": args.friendly_fire`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest -q`
Expected: all pass, including the three new config tests and the existing ones (`test_real_config_hash_is_pinned_to_the_finished_run` still `fdac28308815`).

- [ ] **Step 6: Commit**

```bash
git add configs/showdown-tempo-base.yaml configs/showdown-tempo.yaml configs/showdown-tempo-smoke.yaml xp/showdown/gate.py tests/test_showdown_phase.py
git commit -m "feat: tempo run configs, smoke config and gate options" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: End-to-end smoke (needs the user's yes to start the server and run)

This task starts processes on the laptop; ask the user before running it (project memory: approve each step). It is about a 2 to 5 minute check, not a full run.

**Files:** none changed unless a defect is found.

- [ ] **Step 1: Ask the user for a yes to run the smoke**

Say what will run: `scripts/run_showdown.sh configs/showdown-tempo-smoke.yaml results/showdown-tempo-smoke 1` (starts the local Showdown server if it is down and stops it afterwards).

- [ ] **Step 2: Run the smoke**

Run: `scripts/run_showdown.sh configs/showdown-tempo-smoke.yaml results/showdown-tempo-smoke 1`
Expected: the run finishes ("finished N seed jobs"). In `results/showdown-tempo-smoke/run.log` find the per-worker stderr lines:
`friendly fire actions=... total_penalty=... decisions=...`, `matchup estimator errors=... of calls=...` and `tempo opponent battles=... with_setter=... setters={...}`.

- [ ] **Step 3: Check the three things the smoke is for**

1. Every line above appears (new paths all executed) and nothing raised.
2. `with_setter` is above zero. Report `with_setter / battles` to the user as the measured rate at which the tempo opponent sets anything (spec risk 1). If it is zero over the smoke's battles, stop and report; do not start the runs.
3. Off-by-one check for the remaining-turn counters: during one smoke battle that has a Trick Room or Tailwind up, log `battle.turn` and `tempo.turns_block(battle)` for the turns after it was set (add a temporary `print` in a scratch script, not in the repo) and compare with the Showdown log lines `|-fieldstart|move: Trick Room` and `|-fieldend|move: Trick Room`: the value must reach 0 on the turn the field ends and not before. Report the result; if it is off by one, fix the duration constant in `xp/showdown/tempo.py`, add a test for the corrected value, and commit.

- [ ] **Step 4: Run the gate for the budget cap S (needs the user's yes again)**

Run: `python -m xp.showdown.gate --opponent tempo --steps 250000 --matchup --friendly-fire 1.0 --teams-dir configs/showdown_teams_tempo --out gate_tempo.json`
Report to the user: decisions per second, final win rate, whether it is in the 20 to 80% usable range, steps to 30%. Ask the user to approve (or change) the budget cap S and the 0.30 threshold for the two runs. Do not start either full run before the user answers, and read no RND result before then.

---

### Task 9: Docs and handoff

**Files:**
- Modify: `CLAUDE.md` (the scope line), `docs/FUTURE_WORK.md`, `STATUS.md`, `RESUME.md`, `README.md` (Showdown section, one paragraph)
- Modify: `docs/superpowers/specs/2026-10-22-xp-design.md` (spec 5.6: one line pointing to the new amendment)

- [ ] **Step 1: Point the existing docs at the new work**

- `docs/superpowers/specs/2026-10-22-xp-design.md`: in section 5.6 add a sentence that the tempo amendment (`2026-11-02-showdown-tempo-design.md`) adds the `tempo` opponent, the tempo block and the friendly-fire penalty.
- `CLAUDE.md`: in the Showdown clause of the scope paragraph, add "The tempo follow-up (spec `docs/superpowers/specs/2026-11-02-showdown-tempo-design.md`) was approved by the user on 2026-11-02; the budget cap S needs the user's approval before any RND result is read."
- `docs/FUTURE_WORK.md`: mark candidate follow-ups 1 and 3 under "Showdown: tempo and format-specific features" as "being built, 2026-11-02, see the spec and plan", and leave follow-up 2 open.
- `README.md`: one paragraph in the Showdown section naming the two new run configs and output folders (`results/showdown-tempo-base`, `results/showdown-tempo`); write results only after the runs finish.
- `STATUS.md` and `RESUME.md`: state what is built, what is verified (fast suite, smoke numbers), what is waiting (the user's approval of S, then the two runs, about 12 to 15 hours each on 1 worker), and the exact launch commands:
  `ALLOW_CODE_CHANGE=1 caffeinate -i scripts/run_showdown.sh configs/showdown-tempo-base.yaml results/showdown-tempo-base 1`, then the same with `configs/showdown-tempo.yaml` and `results/showdown-tempo`. Check `sysctl vm.swapusage` and `pgrep -fl "xp.run|multiprocessing.spawn|room-battle"` first (see `RESUME.md`).

- [ ] **Step 2: Run the fast suite, then commit docs, then a STATUS-only marker commit**

Run: `.venv/bin/pytest -q` (all pass).

```bash
git add CLAUDE.md README.md RESUME.md docs/FUTURE_WORK.md docs/superpowers/specs/2026-10-22-xp-design.md
git commit -m "docs: tempo work built; runs waiting on the user's approval of S" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
Then edit `STATUS.md` (state, "Last updated" line, and `STATUS_COMMIT` set to `git rev-parse --short HEAD` of the commit just made) and commit only that file:
```bash
git add STATUS.md && git commit -m "docs: STATUS marker" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
bash .claude/hooks/status-stop-check.sh
```
Expected: `{"continue": true}`.

- [ ] **Step 3: Tell the user**

Report: what was built, the smoke numbers (setter rate, penalty counters, off-by-one result), the gate result, and that each full run needs their yes (and the pushes). Say plainly anything not verified (for example if no server was available for the marked tests).

---

## Self-review (against the spec)

- **Change 1 (penalty):** Task 1 (rule, decoding, all test cases from the spec including immune, fainted, foes-only, two slots), Task 2 (placement in the worker, counters, constant `C` in the config via `env.friendly_fire`). Added by planning: `last_won`, because `reward > 0` win detection breaks when a penalty is added.
- **Change 2 (tempo block):** Task 3 (all 77 floats and tests for each part), Task 4 (switch and flag-off identity).
- **Change 3 (`TempoPlayer`):** Task 5 (all rules, no double-set, never targets an ally, fallback, legality by construction because setters come from `available_moves`). The opponent's usage counters are added so the setter rate can be measured (spec risk 1).
- **Change 4 (teams):** Task 6. Deviates from the spec's first wording (revise in place) for a stated reason; the spec was amended to match.
- **Experiment:** Task 7 (two run configs differing only in `tempo`, equal 8-value grids, test that checks this) and Task 8 (gate and the user's approval of S before any RND result).
- **Open items the executor must not skip:** the Task 8 off-by-one check against a real Showdown log; the user's approval of S.
- **Type consistency:** `obs_dim(matchup, tempo)`, `encode_battle(..., tempo=)`, `ShowdownVecEnv(..., friendly_fire=, tempo=)`, `friendly_fire_penalty(battle, pos, action, estimator, scale)`, `tempo_block(battle)`, `TempoPlayer._setter(battle, pos, used)` are named the same wherever used.
