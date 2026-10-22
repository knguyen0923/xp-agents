from __future__ import annotations

import numpy as np
import pytest

from xp.errors import ProtocolError
from xp.protocol.phase import Phase
from xp.protocol.tune import boundary_dims, grid_points, score_params, tune


def test_grid_points_is_cartesian_product():
    pts = grid_points({"a": [1, 2], "b": [10, 20, 30]})
    assert len(pts) == 6
    assert pts[0] == {"a": 1, "b": 10} and pts[-1] == {"a": 2, "b": 30}


def test_boundary_detected_on_min_and_max_edges():
    grid = {"a": [1, 2, 3], "b": [10, 20, 30]}
    assert boundary_dims({"a": 2, "b": 20}, grid) == []
    assert boundary_dims({"a": 1, "b": 20}, grid) == ["a"]
    assert boundary_dims({"a": 2, "b": 30}, grid) == ["b"]
    assert boundary_dims({"a": 3, "b": 10}, grid) == ["a", "b"]


def test_natural_limit_is_exempt_but_other_edge_is_not():
    grid = {"alpha": [0.1, 0.5, 1.0]}
    limits = {"alpha": [None, 1.0]}
    assert boundary_dims({"alpha": 1.0}, grid, limits) == []
    assert boundary_dims({"alpha": 0.1}, grid, limits) == ["alpha"]


def test_single_valued_dimension_is_ignored():
    assert boundary_dims({"a": 5}, {"a": [5]}) == []


def test_tune_finds_interior_optimum_without_flag():
    grid = {"eps": [0.01, 0.1, 0.3], "alpha": [0.05, 0.3, 0.9]}
    result = tune(lambda p: (p["eps"] - 0.1) ** 2 + (p["alpha"] - 0.3) ** 2, grid)
    assert result.best_params == {"eps": 0.1, "alpha": 0.3}
    assert not result.provisional
    assert len(result.table) == 9 and "score" in result.table.columns


def test_tune_flags_edge_optimum_as_provisional():
    grid = {"eps": [0.01, 0.1, 0.3]}
    result = tune(lambda p: p["eps"], grid)  # best is the smallest value
    assert result.best_params == {"eps": 0.01}
    assert result.provisional and result.boundary_dims == ["eps"]


def test_tune_best_params_are_plain_python_types():
    result = tune(lambda p: p["x"], {"x": [3, 1, 2]})
    assert type(result.best_params["x"]) is int


def test_score_params_uses_only_tune_seeds_in_chunks(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    calls = []

    def fn(seeds, knob, params, offset):
        calls.append((list(seeds), knob, offset))
        return np.full(len(seeds), 2.0)

    assert score_params(phase, fn, {}) == 2.0
    assert calls == [(list(range(10)), phase.familiar, 0)]


def test_score_params_rejects_per_seed_fn_that_is_handed_wrong_pool(tiny_cfg):
    tiny_cfg["seeds"]["chunk"] = 4
    phase = Phase.from_config(tiny_cfg)
    seen = []

    def fn(seeds, knob, params, offset):
        seen.append(offset)
        return np.zeros(len(seeds))

    score_params(phase, fn, {})
    assert seen == [0, 1, 2]
    with pytest.raises(ProtocolError):
        phase.assert_tune_seeds(np.array(phase.eval_seeds[:2]))
