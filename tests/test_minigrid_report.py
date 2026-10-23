from __future__ import annotations

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from xp.minigrid.phase import run_phase  # noqa: E402
from xp.report import main as report_main  # noqa: E402


def test_report_runs_on_a_minigrid_results_table(tiny_mg_cfg, tmp_path):
    tiny_mg_cfg["report"]["extra_metrics"] = {"success_fewshot": False, "steps_to_80": True}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny_mg_cfg))
    run_phase(tiny_mg_cfg, tmp_path / "res")
    report_main(["--phase", "minigrid", "--config", str(cfg_path), "--out", str(tmp_path / "res"),
                 "--figs", str(tmp_path / "figs"), "--allow-provisional"])
    names = {p.name for p in (tmp_path / "figs").iterdir()}
    assert {"curves.csv", "gaps.csv", "crossovers.csv", "degradation.png",
            "curves__success_fewshot.csv", "crossovers__steps_to_80.csv"} <= names
