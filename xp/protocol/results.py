from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from xp.errors import ConfigMismatchError, ProvisionalResultError
from xp.protocol.phase import Phase

COLUMNS = ["phase", "agent", "knob", "seed", "metric", "value", "provisional", "config_hash"]


@dataclass
class TuneResult:
    best_params: dict
    best_score: float
    boundary_dims: list
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def provisional(self) -> bool:
        return bool(self.boundary_dims)

    @classmethod
    def fixed(cls, params: dict) -> "TuneResult":
        return cls(dict(params), float("nan"), [], pd.DataFrame())


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _unit_path(out_dir, phase: Phase, agent: str, knob: float) -> Path:
    return Path(out_dir) / phase.name / f"{agent}__{phase.knob_name}-{knob:g}.csv"


def _tune_paths(out_dir, phase: Phase, agent: str):
    base = Path(out_dir) / phase.name / "tuning"
    return base / f"{agent}.json", base / f"{agent}.csv"


def unit_done(out_dir, phase: Phase, agent: str, knob: float) -> bool:
    path = _unit_path(out_dir, phase, agent, knob)
    if not path.exists():
        return False
    hashes = list(pd.read_csv(path, usecols=["config_hash"], dtype={"config_hash": str})["config_hash"].unique())
    if hashes != [phase.config_hash]:
        raise ConfigMismatchError(
            f"{path} was written under config {hashes}, current config is {phase.config_hash}"
        )
    return True


def write_unit(out_dir, phase: Phase, agent: str, knob: float, df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError(f"results columns must be {COLUMNS}, got {list(df.columns)}")
    path = _unit_path(out_dir, phase, agent, knob)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_results(out_dir, phase_name: str) -> pd.DataFrame:
    files = sorted((Path(out_dir) / phase_name).glob("*__*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    dtypes = {"config_hash": str, "agent": str, "metric": str}
    return pd.concat([pd.read_csv(f, dtype=dtypes) for f in files], ignore_index=True)[COLUMNS]


def write_tune(out_dir, phase: Phase, agent: str, result: TuneResult) -> None:
    json_path, csv_path = _tune_paths(out_dir, phase, agent)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "config_hash": phase.config_hash,
        "best_params": result.best_params,
        "best_score": None if pd.isna(result.best_score) else result.best_score,
        "boundary_dims": result.boundary_dims,
    }
    json_path.write_text(json.dumps(record, indent=2, default=_json_default))
    result.table.to_csv(csv_path, index=False)


def read_tune(out_dir, phase: Phase, agent: str):
    json_path, csv_path = _tune_paths(out_dir, phase, agent)
    if not json_path.exists():
        return None
    record = json.loads(json_path.read_text())
    if record["config_hash"] != phase.config_hash:
        raise ConfigMismatchError(
            f"{json_path} was written under config {record['config_hash']}, "
            f"current config is {phase.config_hash}"
        )
    try:
        table = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        table = pd.DataFrame()
    score = float("nan") if record["best_score"] is None else record["best_score"]
    return TuneResult(record["best_params"], score, record["boundary_dims"], table)


def require_final(df: pd.DataFrame, allow_provisional: bool = False) -> None:
    if allow_provisional or df.empty:
        return
    bad = df.loc[df["provisional"].astype(bool), "agent"].unique()
    if len(bad):
        raise ProvisionalResultError(
            f"provisional (grid-boundary) results for agents {sorted(bad)}: "
            "widen their grids and rerun, or pass allow_provisional for a draft"
        )
