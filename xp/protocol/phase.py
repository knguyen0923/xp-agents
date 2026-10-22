from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import yaml

from xp.errors import ProtocolError


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Phase:
    name: str
    knob_name: str
    familiar: float
    sweep: tuple
    tune_seeds: tuple
    eval_seeds: tuple
    chunk: int
    agent_seed: int
    config_hash: str
    cfg: dict

    @classmethod
    def from_config(cls, cfg: dict) -> "Phase":
        knob = cfg["knob"]
        sweep = tuple(float(v) for v in knob["sweep"])
        familiar = float(knob["familiar"])
        if familiar not in sweep:
            raise ProtocolError(f"familiar value {familiar} is not in the sweep {sweep}")
        seeds = cfg["seeds"]
        tune = tuple(range(seeds["tune"][0], seeds["tune"][1]))
        evals = tuple(range(seeds["eval"][0], seeds["eval"][1]))
        overlap = set(tune) & set(evals)
        if overlap:
            raise ProtocolError(f"tune and eval seed pools overlap, e.g. {sorted(overlap)[:3]}")
        return cls(
            name=cfg["phase"],
            knob_name=knob["name"],
            familiar=familiar,
            sweep=sweep,
            tune_seeds=tune,
            eval_seeds=evals,
            chunk=int(seeds["chunk"]),
            agent_seed=int(seeds["agent_seed"]),
            config_hash=config_hash(cfg),
            cfg=cfg,
        )

    def assert_eval_seeds(self, seeds) -> None:
        bad = set(int(s) for s in np.asarray(seeds)) - set(self.eval_seeds)
        if bad:
            raise ProtocolError(f"seeds {sorted(bad)[:3]} are not in the eval pool")

    def assert_tune_seeds(self, seeds) -> None:
        bad = set(int(s) for s in np.asarray(seeds)) - set(self.tune_seeds)
        if bad:
            raise ProtocolError(f"seeds {sorted(bad)[:3]} are not in the tune pool")
