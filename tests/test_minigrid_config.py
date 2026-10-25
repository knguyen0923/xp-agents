from __future__ import annotations

from pathlib import Path

import numpy as np

from xp.protocol.phase import Phase, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "minigrid.yaml"


def test_committed_minigrid_config_is_a_valid_phase():
    cfg = load_config(CONFIG)
    phase = Phase.from_config(cfg)
    assert phase.name == "minigrid" and phase.knob_name == "grid_size"
    assert phase.familiar == 6.0 and phase.sweep == (5.0, 6.0, 8.0, 12.0)
    assert not set(phase.tune_seeds) & set(phase.eval_seeds)
    assert len(phase.tune_seeds) == 3 and len(phase.eval_seeds) == 10  # spec 5.2: 3 tuning seeds, 10 final seeds


def test_committed_grids_are_equal_size_and_in_the_fewshot_sizes():
    cfg = load_config(CONFIG)
    sizes = {a: int(np.prod([len(v) for v in cfg["agents"][a]["grid"].values()])) for a in ("kairi", "kaiser")}
    assert sizes["kairi"] == sizes["kaiser"]
    assert set(cfg["fewshot"]["sizes"]) <= set(cfg["knob"]["sweep"])  # spec 5.3: few-shot at sizes 8 and 12 only
    assert cfg["eval"]["layouts"] == 200


def test_committed_report_block_lists_return_metrics():
    extra = load_config(CONFIG)["report"]["extra_metrics"]
    assert "return" in extra and "return_fewshot" in extra


MR_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "minigrid-multiroom.yaml"


def test_multiroom_config_matches_spec_5_4():
    cfg = load_config(MR_CONFIG)
    phase = Phase.from_config(cfg)
    assert phase.name == "minigrid" and phase.knob_name == "rooms"  # name must stay "minigrid" for xp.run dispatch
    assert phase.familiar == 4.0 and phase.sweep == (4.0,)
    assert cfg["env"] == {"level": "multiroom", "max_room_size": 4}
    assert cfg["tuning"]["objective"] == "auc"
    assert cfg["budget"]["target_success"] == 0.95 and cfg["budget"]["cap"] == 5_000_000
    assert phase.tune_seeds == (0, 1, 2) and phase.eval_seeds == tuple(range(100, 110))
    assert cfg["agents"]["kairi"]["grid"]["ent_coef"] == [0.003, 0.006, 0.01, 0.02, 0.05, 0.1]
    assert cfg["agents"]["kaiser"]["grid"]["rnd_coef"] == [0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
    assert cfg["agents"]["kaiser"]["fixed"]["ent_coef_from"] == "kairi"
    assert cfg["agents"]["kaiser"]["fixed"]["ent_coef"] == 0.01  # calibration default entropy


def test_multiroom_config_grids_are_equal_size_and_rnd_interior_contains_the_probe_value():
    cfg = load_config(MR_CONFIG)
    sizes = {a: int(np.prod([len(v) for v in cfg["agents"][a]["grid"].values()])) for a in ("kairi", "kaiser")}
    assert sizes["kairi"] == sizes["kaiser"] == 6
    rnd = cfg["agents"]["kaiser"]["grid"]["rnd_coef"]
    assert min(rnd) < 0.1 < max(rnd)
