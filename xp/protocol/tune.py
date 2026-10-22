from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from xp.errors import ProtocolError
from xp.protocol.phase import Phase
from xp.protocol.results import TuneResult

PerSeedFn = Callable[[np.ndarray, float, dict, int], np.ndarray]


def checked_chunk_result(values, chunk) -> np.ndarray:
    """A per-seed function must return one value per seed in the chunk, as a 1-D array."""
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) != len(chunk):
        raise ProtocolError(f"per-seed function returned shape {arr.shape} for a chunk of {len(chunk)} seeds")
    return arr


def grid_points(grid: Dict[str, list]) -> List[dict]:
    names = list(grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*(grid[n] for n in names))]


def boundary_dims(best: dict, grid: Dict[str, list], natural_limits: Optional[dict] = None) -> List[str]:
    limits = natural_limits or {}
    flagged = []
    for dim, values in grid.items():
        lo, hi = min(values), max(values)
        if lo == hi:
            continue
        nat_lo, nat_hi = (limits.get(dim) or [None, None])
        value = best[dim]
        at_lo_edge = value == lo and nat_lo != lo
        at_hi_edge = value == hi and nat_hi != hi
        if at_lo_edge or at_hi_edge:
            flagged.append(dim)
    return flagged


def tune(score_fn: Callable[[dict], float], grid: Dict[str, list], natural_limits: Optional[dict] = None) -> TuneResult:
    """Exhaustive grid search; lower score is better."""
    points = grid_points(grid)
    rows = [{**p, "score": float(score_fn(p))} for p in points]
    for row, p in zip(rows, points):
        if not np.isfinite(row["score"]):
            raise ProtocolError(f"non-finite tuning score {row['score']} for params {p}")
    table = pd.DataFrame(rows)
    best_idx = int(table["score"].idxmin())
    best_params = dict(points[best_idx])
    return TuneResult(
        best_params=best_params,
        best_score=float(table.loc[best_idx, "score"]),
        boundary_dims=boundary_dims(best_params, grid, natural_limits),
        table=table,
    )


def score_params(phase: Phase, per_seed_fn: PerSeedFn, params: dict) -> float:
    """Mean metric at the familiar knob value over the tune-seed pool only."""
    seeds = np.array(phase.tune_seeds)
    values = []
    for chunk_index, start in enumerate(range(0, len(seeds), phase.chunk)):
        chunk = seeds[start:start + phase.chunk]
        phase.assert_tune_seeds(chunk)
        values.append(checked_chunk_result(per_seed_fn(chunk, phase.familiar, params, chunk_index), chunk))
    return float(np.concatenate(values).mean())
