from __future__ import annotations

import numpy as np

_BATCH = 500


def bootstrap_ci(x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, _BATCH):
        m = min(_BATCH, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        means[start:start + m] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, **kw):
    """CI of the mean of a - b for paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired samples must have the same shape, got {a.shape} and {b.shape}")
    return bootstrap_ci(a - b, **kw)
