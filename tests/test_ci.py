from __future__ import annotations

import numpy as np

from xp.stats.ci import bootstrap_ci, paired_diff_ci


def test_ci_contains_mean_and_is_ordered():
    x = np.random.default_rng(0).normal(2.0, 1.0, size=200)
    mean, lo, hi = bootstrap_ci(x, n_boot=1000)
    assert lo < mean < hi
    assert abs(mean - x.mean()) < 1e-12


def test_ci_is_deterministic_given_seed():
    x = np.arange(50, dtype=float)
    assert bootstrap_ci(x, n_boot=500, seed=3) == bootstrap_ci(x, n_boot=500, seed=3)


def test_ci_width_shrinks_with_more_data():
    rng = np.random.default_rng(1)
    _, lo1, hi1 = bootstrap_ci(rng.normal(size=30), n_boot=1000)
    _, lo2, hi2 = bootstrap_ci(rng.normal(size=3000), n_boot=1000)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_ci_coverage_on_synthetic_data():
    rng = np.random.default_rng(42)
    hits = 0
    reps = 200
    for i in range(reps):
        sample = rng.normal(0.0, 1.0, size=40)
        _, lo, hi = bootstrap_ci(sample, n_boot=400, seed=i)
        hits += lo <= 0.0 <= hi
    assert 0.88 <= hits / reps <= 0.99


def test_paired_diff_ci_detects_shift():
    rng = np.random.default_rng(5)
    base = rng.normal(size=300)
    a = base + 0.5
    b = base
    mean, lo, hi = paired_diff_ci(a, b, n_boot=1000)
    assert abs(mean - 0.5) < 1e-9
    assert lo > 0.4 and hi < 0.6  # pairing removes the shared noise
