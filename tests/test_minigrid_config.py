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
