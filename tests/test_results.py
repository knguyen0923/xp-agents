from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xp.errors import ConfigMismatchError, ProvisionalResultError
from xp.protocol.phase import Phase
from xp.protocol.results import (
    COLUMNS,
    TuneResult,
    load_results,
    read_tune,
    require_final,
    unit_done,
    write_tune,
    write_unit,
)


def _df(phase, agent="kairi", knob=0.01, provisional=False):
    return pd.DataFrame(
        {
            "phase": phase.name,
            "agent": agent,
            "knob": knob,
            "seed": [100, 101],
            "metric": "regret",
            "value": [0.5, 0.6],
            "provisional": provisional,
            "config_hash": phase.config_hash,
        }
    )


def test_unit_roundtrip_and_done(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    assert not unit_done(tmp_path, phase, "kairi", 0.01)
    write_unit(tmp_path, phase, "kairi", 0.01, _df(phase))
    assert unit_done(tmp_path, phase, "kairi", 0.01)
    out = load_results(tmp_path, "bandit")
    assert list(out.columns) == COLUMNS
    assert len(out) == 2


def test_hash_mismatch_raises(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    write_unit(tmp_path, phase, "kairi", 0.01, _df(phase))
    tiny_cfg["horizon"] = 99
    other = Phase.from_config(tiny_cfg)
    with pytest.raises(ConfigMismatchError):
        unit_done(tmp_path, other, "kairi", 0.01)


def test_write_unit_rejects_wrong_columns(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    with pytest.raises(ValueError):
        write_unit(tmp_path, phase, "kairi", 0.01, _df(phase).drop(columns=["value"]))


def test_tune_roundtrip(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    table = pd.DataFrame({"epsilon": [0.05, 0.2], "score": [0.4, 0.5]})
    result = TuneResult({"epsilon": 0.05}, 0.4, ["epsilon"], table)
    assert result.provisional
    assert read_tune(tmp_path, phase, "kairi") is None
    write_tune(tmp_path, phase, "kairi", result)
    back = read_tune(tmp_path, phase, "kairi")
    assert back.best_params == {"epsilon": 0.05}
    assert back.boundary_dims == ["epsilon"] and back.provisional
    assert len(back.table) == 2


def test_tune_hash_mismatch_raises(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    write_tune(tmp_path, phase, "kairi", TuneResult.fixed({"block": 10}))
    tiny_cfg["horizon"] = 99
    with pytest.raises(ConfigMismatchError):
        read_tune(tmp_path, Phase.from_config(tiny_cfg), "kairi")


def test_fixed_tune_result_is_not_provisional():
    r = TuneResult.fixed({"block": 50})
    assert not r.provisional and np.isnan(r.best_score)


def test_require_final_blocks_provisional(tiny_cfg):
    phase = Phase.from_config(tiny_cfg)
    require_final(_df(phase, provisional=False))
    with pytest.raises(ProvisionalResultError):
        require_final(_df(phase, provisional=True))
    require_final(_df(phase, provisional=True), allow_provisional=True)


def test_numeric_looking_hash_resumes_and_roundtrips(tiny_cfg, tmp_path):
    import dataclasses

    phase = dataclasses.replace(Phase.from_config(tiny_cfg), config_hash="000123456789")
    df = _df(phase)
    write_unit(tmp_path, phase, "kairi", 0.01, df)
    assert unit_done(tmp_path, phase, "kairi", 0.01)
    out = load_results(tmp_path, "bandit")
    assert out["config_hash"].tolist() == ["000123456789", "000123456789"]
    assert all(isinstance(h, str) for h in out["config_hash"])
    assert out["provisional"].dtype == bool


def test_write_tune_accepts_numpy_scalars(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    result = TuneResult(
        {"epsilon": np.float32(0.5), "window": np.int64(10)},
        np.float64(0.4),
        ["epsilon"],
        pd.DataFrame({"epsilon": [0.5]}),
    )
    write_tune(tmp_path, phase, "kairi", result)
    back = read_tune(tmp_path, phase, "kairi")
    assert back.best_params["epsilon"] == 0.5
    assert back.best_params["window"] == 10
