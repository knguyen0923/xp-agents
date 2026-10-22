from __future__ import annotations

import numpy as np

from xp.protocol.evaluate import evaluate_unit
from xp.protocol.phase import Phase
from xp.protocol.results import COLUMNS


def test_evaluate_unit_shape_columns_and_chunking(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    calls = []

    def fn(seeds, knob, params, offset):
        calls.append((len(seeds), knob, offset, params))
        return seeds.astype(float) / 1000.0

    df = evaluate_unit(phase, "kairi", {"epsilon": 0.1}, 0.05, True, fn)
    assert list(df.columns) == COLUMNS
    assert len(df) == len(phase.eval_seeds)
    assert df["seed"].tolist() == list(phase.eval_seeds)
    assert set(df["agent"]) == {"kairi"} and set(df["metric"]) == {"regret"}
    assert df["provisional"].all() and set(df["knob"]) == {0.05}
    assert set(df["config_hash"]) == {phase.config_hash}
    assert calls == [(10, 0.05, 0, {"epsilon": 0.1}), (10, 0.05, 1, {"epsilon": 0.1})]
    assert np.allclose(df["value"], np.array(phase.eval_seeds) / 1000.0)
