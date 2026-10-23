from __future__ import annotations

import math

import pytest

from xp.minigrid.calibrate import suggest_budget


def _curve(values):
    return [{"steps": 1000 * (i + 1), "success_100": v} for i, v in enumerate(values)]


def test_budget_is_first_step_where_mean_success_reaches_target():
    curves = [_curve([0.0, 0.5, 0.9, 1.0]), _curve([0.0, 0.2, 0.8, 1.0])]
    out = suggest_budget(curves, target=0.8, cap=10_000)
    assert out["steps"] == 3000 and out["reached"]  # mean at step 3000 is (0.9 + 0.8) / 2 = 0.85


def test_budget_falls_back_to_cap_and_says_so():
    out = suggest_budget([_curve([0.0, 0.1, 0.2])], target=0.8, cap=5000)
    assert out["steps"] == 5000 and not out["reached"]


def test_nan_success_counts_as_zero():
    out = suggest_budget([_curve([math.nan, 0.9])], target=0.8, cap=5000)
    assert out["steps"] == 2000


def test_mismatched_log_steps_are_rejected():
    with pytest.raises(ValueError):
        suggest_budget([_curve([0.0, 1.0]), _curve([0.0, 1.0, 1.0])], target=0.8, cap=5000)
