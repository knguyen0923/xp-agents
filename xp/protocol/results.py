from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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


def atomic_write(path: Path, writer) -> None:
    """Write via a temp file in the same directory, then os.replace, so a crash never leaves a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def unit_done(out_dir, phase: Phase, agent: str, knob: float) -> bool:
    path = _unit_path(out_dir, phase, agent, knob)
    if not path.exists():
        return False
    try:
        hashes_col = pd.read_csv(path, usecols=["config_hash"], dtype={"config_hash": str})["config_hash"]
    except (pd.errors.EmptyDataError, ValueError):
        return False  # empty or unreadable file: treat as not done
    if hashes_col.empty:
        return False
    hashes = list(hashes_col.unique())
    if hashes != [phase.config_hash]:
        raise ConfigMismatchError(
            f"{path} was written under config {hashes}, current config is {phase.config_hash}"
        )
    return len(hashes_col) == len(phase.eval_seeds)


def write_unit(out_dir, phase: Phase, agent: str, knob: float, df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError(f"results columns must be {COLUMNS}, got {list(df.columns)}")
    path = _unit_path(out_dir, phase, agent, knob)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


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
    atomic_write(json_path, lambda tmp: tmp.write_text(json.dumps(record, indent=2, default=_json_default)))
    atomic_write(csv_path, lambda tmp: result.table.to_csv(tmp, index=False))


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


def _job_path(out_dir, phase: Phase, agent: str, seed: int) -> Path:
    return Path(out_dir) / phase.name / f"{agent}__seed-{seed}.csv"


def seed_job_done(out_dir, phase: Phase, agent: str, seed: int) -> bool:
    """A per-(agent, training seed) job is done when its atomically written file exists under this config."""
    path = _job_path(out_dir, phase, agent, seed)
    if not path.exists():
        return False
    try:
        hashes = list(pd.read_csv(path, usecols=["config_hash"], dtype={"config_hash": str})["config_hash"].unique())
    except (pd.errors.EmptyDataError, ValueError):
        return False
    if hashes != [phase.config_hash]:
        raise ConfigMismatchError(f"{path} was written under config {hashes}, current config is {phase.config_hash}")
    return True


def write_seed_job(out_dir, phase: Phase, agent: str, seed: int, df: pd.DataFrame) -> None:
    if list(df.columns) != COLUMNS:
        raise ValueError(f"results columns must be {COLUMNS}, got {list(df.columns)}")
    path = _job_path(out_dir, phase, agent, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda tmp: df.to_csv(tmp, index=False))


def _tune_job_path(out_dir, phase: Phase, agent: str, params: dict, seed: int) -> Path:
    key = "_".join(f"{k}-{params[k]:g}" for k in sorted(params))
    return Path(out_dir) / phase.name / "tuning" / "jobs" / f"{agent}__{key}__seed-{seed}.json"


def read_tune_job(out_dir, phase: Phase, agent: str, params: dict, seed: int):
    """Cached tuning score for one (agent, params, tune seed), or None."""
    path = _tune_job_path(out_dir, phase, agent, params, seed)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["config_hash"] != phase.config_hash:
        raise ConfigMismatchError(f"{path} was written under config {record['config_hash']}, current config is {phase.config_hash}")
    return float(record["score"])


def read_tune_job_extra(out_dir, phase: Phase, agent: str, params: dict, seed: int) -> Optional[dict]:
    """Extra diagnostics stored with a tuning job (e.g. training AUC), or None."""
    path = _tune_job_path(out_dir, phase, agent, params, seed)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["config_hash"] != phase.config_hash:
        raise ConfigMismatchError(f"{path} was written under config {record['config_hash']}, current config is {phase.config_hash}")
    return record.get("extra")


def write_tune_job(out_dir, phase: Phase, agent: str, params: dict, seed: int, score: float,
                   extra: Optional[dict] = None) -> None:
    path = _tune_job_path(out_dir, phase, agent, params, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"config_hash": phase.config_hash, "params": params, "seed": seed, "score": float(score)}
    if extra is not None:
        record["extra"] = extra
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(record, default=_json_default)))
