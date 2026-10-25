from __future__ import annotations

import copy

import pytest

_TINY = {
    "phase": "bandit",
    "k": 3,
    "sigma": 1.0,
    "horizon": 50,
    "knob": {"name": "jump_rate", "familiar": 0.01, "sweep": [0.0, 0.01, 0.05]},
    "seeds": {"tune": [0, 10], "eval": [100, 120], "chunk": 10, "agent_seed": 7},
    "agents": {
        "kairi": {
            "grid": {"epsilon": [0.05, 0.2], "alpha": [0.1, 0.5]},
            "natural_limits": {"alpha": [None, 1.0]},
        },
        "kaiser": {
            "grid": {"gamma": [0.9, 0.99], "c": [0.5, 1.0]},
            "natural_limits": {"gamma": [None, 1.0]},
        },
    },
    "tagteam": {
        "fixed": {"block": 10},
        "triggered": {
            "grid": {"delta": [0.8, 1.2], "h": [2.0, 4.0], "window": [5, 10]},
            "natural_limits": {},
        },
        "meta": {
            "block": 10,
            "grid": {"gamma": [0.9, 1.0], "c": [0.5, 1.0]},
            "natural_limits": {"gamma": [None, 1.0]},
        },
    },
}


@pytest.fixture
def tiny_cfg():
    return copy.deepcopy(_TINY)


_TINY_MG = {
    "phase": "minigrid",
    "knob": {"name": "grid_size", "familiar": 5, "sweep": [5, 6]},
    "seeds": {"tune": [0, 2], "eval": [10, 12], "chunk": 1, "agent_seed": 0},
    "budget": {"steps": 256, "target_success": 0.8, "cap": 2000},
    "eval": {"layouts": 3},
    "fewshot": {"sizes": [6], "steps": 64},
    "ppo": {"n_envs": 2, "n_steps": 16, "epochs": 2, "minibatches": 2, "lr": 0.00025, "gamma": 0.99, "lam": 0.95,
            "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 0.0001},
    "agents": {
        "kairi": {"grid": {"ent_coef": [0.0, 0.01]}, "natural_limits": {"ent_coef": [0.0, None]}},
        "kaiser": {"fixed": {"ent_coef": 0.01}, "grid": {"rnd_coef": [0.0, 0.01]},
                   "natural_limits": {"rnd_coef": [0.0, None]}},
    },
    "report": {"metric": "success", "lower_is_better": False, "agents": ["kairi", "kaiser"]},
}


@pytest.fixture
def tiny_mg_cfg():
    return copy.deepcopy(_TINY_MG)


def _tiny_mr():
    cfg = copy.deepcopy(_TINY_MG)
    cfg["knob"] = {"name": "rooms", "familiar": 2, "sweep": [2]}
    cfg["env"] = {"level": "multiroom", "max_room_size": 4}
    cfg["tuning"] = {"objective": "auc"}
    cfg["budget"] = {"steps": 256, "target_success": 0.95, "cap": 512}
    cfg["fewshot"] = {"sizes": [], "steps": 0}
    cfg["agents"] = {
        "kairi": {"grid": {"ent_coef": [0.01]}, "natural_limits": {"ent_coef": [0.0, None]}},
        "kaiser": {"fixed": {"ent_coef": 0.01, "ent_coef_from": "kairi"}, "grid": {"rnd_coef": [0.01]},
                   "natural_limits": {"rnd_coef": [0.0, None]}},
    }
    cfg["report"] = {"metric": "auc", "lower_is_better": False, "agents": ["kairi", "kaiser"],
                     "extra_metrics": {"success": False, "steps_to_95": True}}
    return cfg


@pytest.fixture
def tiny_mr_cfg():
    return _tiny_mr()
