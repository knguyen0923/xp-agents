from __future__ import annotations

import pytest

from xp.crafter.stats import ACHIEVEMENTS
from xp.errors import ProtocolError
from xp.protocol.jobs import grid_size
from xp.protocol.phase import Phase, load_config

EXISTING_HASHES = {
    "minigrid": "b41211b3b5c6", "minigrid-multiroom": "39ce53712b8a", "minigrid-a": "51260035781b", "minigrid-entfair": "e41af4a9d320",
    "showdown": "fdac28308815", "showdown-control": "028d33113f41", "showdown-matchup": "7ca893d2d935",
    "showdown-tempo-base": "2083c9b9737b", "showdown-tempo": "435175fb4ec3", "bandit": "50ea69897efe",
}


@pytest.mark.parametrize("name,expected", sorted(EXISTING_HASHES.items()))
def test_existing_config_hashes_are_unchanged(name, expected):
    assert Phase.from_config(load_config(f"configs/{name}.yaml")).config_hash == expected


def test_crafter_config_declares_the_protocol():
    cfg = load_config("configs/crafter.yaml")
    phase = Phase.from_config(cfg)
    assert phase.name == "crafter" and phase.familiar == 32.0 and phase.sweep == (32.0,)
    assert phase.tune_seeds == (0, 1, 2) and phase.eval_seeds == tuple(range(100, 110))
    assert cfg["agents"]["kairi"]["grid"]["ent_coef"] == [0.003, 0.006, 0.01, 0.02, 0.05, 0.1]
    assert cfg["agents"]["kaiser"]["grid"]["rnd_coef"] == [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 6
    assert cfg["agents"]["kaiser"]["fixed"]["ent_coef_from"] == "kairi"
    b = cfg["budget"]
    assert (b["cap"], b["target_ach"], b["sustain"], b["min_deep"]) == (3000000, 5, 3, 3)
    assert "steps" not in b and "deep_set" not in b  # unset until the user sets them from the calibration


def test_smoke_config_is_small_and_valid():
    cfg = load_config("configs/crafter-smoke.yaml")
    Phase.from_config(cfg)
    assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 2
    assert cfg["budget"]["steps"] <= 20000 and set(cfg["budget"]["deep_set"]) <= set(ACHIEVEMENTS)


def test_crafter_does_not_leak_into_other_packages():
    import pathlib

    for pkg in ("minigrid", "showdown", "protocol"):
        for path in pathlib.Path("xp", pkg).glob("*.py"):
            assert "xp.crafter" not in path.read_text(), path
    for path in pathlib.Path("xp", "crafter").glob("*.py"):
        text = path.read_text()
        assert "xp.minigrid" not in text and "xp.showdown" not in text, path


def test_require_budget_blocks_until_set():
    from xp.crafter.phase import require_budget

    cfg = load_config("configs/crafter.yaml")
    with pytest.raises(ProtocolError, match="budget.steps"):
        require_budget(cfg)
    cfg["budget"]["steps"] = 100000
    with pytest.raises(ProtocolError, match="deep_set"):
        require_budget(cfg)
    cfg["budget"]["deep_set"] = ["collect_wood", "not_an_achievement"]
    with pytest.raises(ProtocolError, match="not_an_achievement"):
        require_budget(cfg)
    cfg["budget"]["deep_set"] = ["collect_wood", "eat_cow", "collect_drink"]
    assert require_budget(cfg) == 100000
