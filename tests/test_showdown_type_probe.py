from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")

_SPEC = importlib.util.spec_from_file_location("showdown_type_probe", Path(__file__).resolve().parent.parent / "scripts" / "showdown_type_probe.py")
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["showdown_type_probe"] = probe
_SPEC.loader.exec_module(probe)

from xp.showdown.encode import MON_DIM, MOVE_DIM, OBS_DIM, TYPES  # noqa: E402


def _blank() -> np.ndarray:
    return np.zeros(OBS_DIM, dtype=np.float32)


def _set_mon(obs: np.ndarray, side_off: int, slot: int, types: list, moves: list) -> None:
    """Write an active, healthy Pokemon row: `moves` is a list of (power, type_index, status?) tuples."""
    base = probe.mon_slice(side_off, slot)
    obs[base + 2] = 1.0
    for t in types:
        obs[base + probe.TYPE_OFF + t] = 1.0
    for k, (power, type_idx, is_status) in enumerate(moves):
        row = base + probe.MOVES_OFF + k * MOVE_DIM
        obs[row] = power
        obs[row + 3 + type_idx] = 1.0
        obs[row + probe.CATEGORY_OFF + (2 if is_status else 1)] = 1.0


def test_layout_constants_fit_the_observation():
    assert probe.FOE_OFF + 2 * MON_DIM <= OBS_DIM
    assert probe.MOVES_OFF + 4 * MOVE_DIM == MON_DIM


def test_damaging_moves_skip_status_and_zero_power():
    obs, fire, water = _blank(), TYPES.index(TYPES[6]), TYPES.index(TYPES[17])
    _set_mon(obs, 0, 0, [fire], [(0.6, fire, False), (0.0, water, True), (0.0, water, False)])
    assert probe.damaging_moves(obs, 0) == [(0, fire)]


def test_with_foe_type_changes_only_the_foe_type_row():
    obs = _blank()
    _set_mon(obs, probe.FOE_OFF, 0, [TYPES.index(TYPES[9])], [])
    out = probe.with_foe_type(obs, 0, 11)
    changed = np.flatnonzero(out != obs)
    start = probe.mon_slice(probe.FOE_OFF, 0) + probe.TYPE_OFF
    assert set(changed) == {start + 9, start + 11}
    assert out[start + 11] == 1.0 and out[start + 9] == 0.0
    assert obs[start + 9] == 1.0  # the input is not modified


def test_action_index_matches_the_mask_encoding():
    assert [probe.action_index(0, 0), probe.action_index(0, 1), probe.action_index(3, 1)] == [10, 11, 26]


def test_real_type_chart():
    table = probe.multiplier_table()
    idx = {t.name: i for i, t in enumerate(TYPES)}
    assert table[idx["FIRE"], idx["GRASS"]] == 2.0
    assert table[idx["GROUND"], idx["FLYING"]] == 0.0
    assert table[idx["WATER"], idx["FIRE"]] == 2.0


def test_spearman_handles_ties_and_constants():
    assert probe.spearman(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == pytest.approx(-1.0)
    assert probe.spearman(np.array([1, 2, 2, 3.0]), np.array([1, 2, 2, 3.0])) == pytest.approx(1.0)
    assert np.isnan(probe.spearman(np.array([1, 2, 3.0]), np.array([5, 5, 5.0])))


def test_probe_decision_sees_a_planted_type_preference():
    """A net whose slot-0 head rewards the foe's Water one-hot must correlate with a Water-weak move's multiplier."""
    import torch
    from xp.showdown.ppo import ActorCritic

    obs, mask = _blank(), np.zeros((2, 107), dtype=bool)
    electric = [i for i, t in enumerate(TYPES) if t.name == "ELECTRIC"][0]
    _set_mon(obs, 0, 0, [electric], [(0.6, electric, False)])
    _set_mon(obs, probe.FOE_OFF, 0, [0], [])
    mask[0, probe.action_index(0, 0)] = True
    net = ActorCritic()
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        # body is zeroed, so route the target feature straight to the logit through a hand-set first layer
        water = [i for i, t in enumerate(TYPES) if t.name == "WATER"][0]
        net.body[0].weight[0, probe.mon_slice(probe.FOE_OFF, 0) + probe.TYPE_OFF + water] = 5.0
        net.body[2].weight[0, 0] = 1.0
        net.pi[0].weight[probe.action_index(0, 0), 0] = 1.0
    (rho, top, _), = probe.probe_decision(net, obs, mask, probe.multiplier_table())
    assert rho > 0 and top == 1.0  # Water and Flying are the 2x targets of Electric; only Water is boosted, and it wins
