from __future__ import annotations

import numpy as np
import pandas as pd

from xp.protocol.phase import Phase
from xp.protocol.results import COLUMNS
from xp.protocol.tune import PerSeedFn


def evaluate_unit(
    phase: Phase,
    agent: str,
    params: dict,
    knob: float,
    provisional: bool,
    per_seed_fn: PerSeedFn,
    metric: str = "regret",
) -> pd.DataFrame:
    """Evaluate one frozen agent at one knob value over the whole eval-seed pool."""
    seeds = np.array(phase.eval_seeds)
    values = []
    for chunk_index, start in enumerate(range(0, len(seeds), phase.chunk)):
        chunk = seeds[start:start + phase.chunk]
        phase.assert_eval_seeds(chunk)
        values.append(np.asarray(per_seed_fn(chunk, knob, params, chunk_index)))
    df = pd.DataFrame(
        {
            "phase": phase.name,
            "agent": agent,
            "knob": knob,
            "seed": seeds,
            "metric": metric,
            "value": np.concatenate(values),
            "provisional": bool(provisional),
            "config_hash": phase.config_hash,
        }
    )
    return df[COLUMNS]
