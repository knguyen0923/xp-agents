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


def _full_df(phase, agent="kairi", knob=0.01, provisional=False):
    df = _df(phase, agent, knob, provisional)
    n = len(phase.eval_seeds)
    return pd.DataFrame({**df.iloc[0].to_dict(), "seed": list(phase.eval_seeds), "value": np.linspace(0.1, 0.9, n)})[COLUMNS]


def test_unit_roundtrip_and_done(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    assert not unit_done(tmp_path, phase, "kairi", 0.01)
    df = _full_df(phase)
    write_unit(tmp_path, phase, "kairi", 0.01, df)
    assert unit_done(tmp_path, phase, "kairi", 0.01)
    out = load_results(tmp_path, "bandit")
    assert list(out.columns) == COLUMNS
    assert len(out) == len(phase.eval_seeds)


def test_hash_mismatch_raises(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    write_unit(tmp_path, phase, "kairi", 0.01, _full_df(phase))
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
    df = _full_df(phase)
    write_unit(tmp_path, phase, "kairi", 0.01, df)
    assert unit_done(tmp_path, phase, "kairi", 0.01)
    out = load_results(tmp_path, "bandit")
    assert out["config_hash"].tolist() == ["000123456789"] * len(phase.eval_seeds)
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


def test_unit_done_is_false_for_empty_header_only_or_short_file(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)  # expects 20 eval seeds
    path = tmp_path / "bandit" / "kairi__jump_rate-0.01.csv"
    path.parent.mkdir(parents=True)
    path.write_text("")
    assert not unit_done(tmp_path, phase, "kairi", 0.01)
    path.write_text(",".join(COLUMNS) + "\n")
    assert not unit_done(tmp_path, phase, "kairi", 0.01)
    write_unit(tmp_path, phase, "kairi", 0.01, _df(phase))  # only 2 of 20 rows
    assert not unit_done(tmp_path, phase, "kairi", 0.01)


def test_unit_done_true_for_complete_file(tiny_cfg, tmp_path):
    phase = Phase.from_config(tiny_cfg)
    df = _df(phase)
    df = pd.concat([df] * (len(phase.eval_seeds) // 2), ignore_index=True)
    write_unit(tmp_path, phase, "kairi", 0.01, df)
    assert unit_done(tmp_path, phase, "kairi", 0.01)


def test_writes_are_atomic_and_leave_no_temp_files(tiny_cfg, tmp_path, monkeypatch):
    import os

    phase = Phase.from_config(tiny_cfg)
    replaced = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (replaced.append(str(b)), real(a, b))[1])
    write_unit(tmp_path, phase, "kairi", 0.01, _df(phase))
    write_tune(tmp_path, phase, "kairi", TuneResult({"epsilon": 0.05}, 0.4, [], pd.DataFrame({"epsilon": [0.05]})))
    assert len(replaced) == 3  # unit csv, tune json, tune csv
    assert sorted(p.name for p in (tmp_path / "bandit").rglob("*") if p.is_file()) == [
        "kairi.csv", "kairi.json", "kairi__jump_rate-0.01.csv"
    ]


def test_seed_job_roundtrip_and_hash_guard(tmp_path, tiny_cfg):
    import pandas as pd
    import pytest

    from xp.errors import ConfigMismatchError
    from xp.protocol.phase import Phase
    from xp.protocol.results import COLUMNS, seed_job_done, write_seed_job

    phase = Phase.from_config(tiny_cfg)
    df = pd.DataFrame([["bandit", "kairi", 1.0, 100, "success", 0.5, False, phase.config_hash]], columns=COLUMNS)
    assert not seed_job_done(tmp_path, phase, "kairi", 100)
    write_seed_job(tmp_path, phase, "kairi", 100, df)
    assert seed_job_done(tmp_path, phase, "kairi", 100)
    other = Phase.from_config({**tiny_cfg, "horizon": 51})
    with pytest.raises(ConfigMismatchError):
        seed_job_done(tmp_path, other, "kairi", 100)
    with pytest.raises(ValueError):
        write_seed_job(tmp_path, phase, "kairi", 101, df.rename(columns={"value": "v"}))


def test_tune_job_roundtrip_and_hash_guard(tmp_path, tiny_cfg):
    import pytest

    from xp.errors import ConfigMismatchError
    from xp.protocol.phase import Phase
    from xp.protocol.results import read_tune_job, write_tune_job

    phase = Phase.from_config(tiny_cfg)
    params = {"ent_coef": 0.01, "rnd_coef": 0.0}
    assert read_tune_job(tmp_path, phase, "kaiser", params, 3) is None
    write_tune_job(tmp_path, phase, "kaiser", params, 3, 0.25)
    assert read_tune_job(tmp_path, phase, "kaiser", params, 3) == 0.25
    other = Phase.from_config({**tiny_cfg, "horizon": 51})
    with pytest.raises(ConfigMismatchError):
        read_tune_job(tmp_path, other, "kaiser", params, 3)
