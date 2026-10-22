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
