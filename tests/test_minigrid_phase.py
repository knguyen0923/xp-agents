from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("minigrid")

from xp.errors import ConfigMismatchError, ProtocolError  # noqa: E402
from xp.minigrid.phase import break_ties, build_learner, run_phase, tune_job  # noqa: E402
from xp.protocol.phase import Phase  # noqa: E402
from xp.protocol.results import TuneResult, load_results, read_tune, read_tune_job, read_tune_job_extra  # noqa: E402


def test_kairi_and_kaiser_differ_only_by_the_rnd_path(tiny_mg_cfg):
    kairi = build_learner(tiny_mg_cfg, "kairi", {"ent_coef": 0.03}, seed=1)
    kaiser = build_learner(tiny_mg_cfg, "kaiser", {"rnd_coef": 0.03}, seed=1)
    assert kairi.rnd is None and kairi.ent_coef == 0.03
    assert kaiser.rnd is not None and kaiser.rnd_coef == 0.03 and kaiser.ent_coef == 0.01


def test_tune_job_rejects_eval_seeds_and_caches(tiny_mg_cfg, tmp_path):
    phase = Phase.from_config(tiny_mg_cfg)
    with pytest.raises(ProtocolError):
        tune_job(tiny_mg_cfg, tmp_path, "kairi", {"ent_coef": 0.0}, seed=10)  # 10 is an eval seed
    score = tune_job(tiny_mg_cfg, tmp_path, "kairi", {"ent_coef": 0.0}, seed=0)
    assert 0.0 <= score <= 1.0
    assert read_tune_job(tmp_path, phase, "kairi", {"ent_coef": 0.0}, 0) == score
    extra = read_tune_job_extra(tmp_path, phase, "kairi", {"ent_coef": 0.0}, 0)
    assert 0.0 <= extra["auc"] <= 1.0 and "steps_to_target" in extra


def test_kaiser_with_zero_rnd_coef_is_plain_kairi(tiny_mg_cfg):
    kaiser0 = build_learner(tiny_mg_cfg, "kaiser", {"rnd_coef": 0.0}, seed=1)
    kairi = build_learner(tiny_mg_cfg, "kairi", {"ent_coef": 0.01}, seed=1)
    assert kaiser0.rnd is None and kaiser0.net.v_int is None
    kaiser0.train(5, 64)
    kairi.train(5, 64)
    assert kaiser0.fingerprint() == kairi.fingerprint()


def _tied_result(scores, grid):
    from xp.protocol.tune import grid_points

    rows = [{**p, "score": sc} for p, sc in zip(grid_points(grid), scores)]
    best = min(range(len(rows)), key=lambda i: rows[i]["score"])
    return TuneResult(grid_points(grid)[best], rows[best]["score"], [], pd.DataFrame(rows))


def test_break_ties_picks_highest_auc_and_recomputes_boundary():
    grid = {"ent_coef": [0.0, 0.01, 0.1]}
    limits = {"ent_coef": [0.0, None]}
    res = _tied_result([0.2, 0.2, 0.5], grid)  # first-in-grid wins the plain argmin
    out = break_ties(res, grid, limits, [0.1, 0.4, 0.2])
    assert out.best_params == {"ent_coef": 0.01} and out.best_score == 0.2 and out.boundary_dims == []
    res = _tied_result([0.5, 0.2, 0.2], grid)
    out = break_ties(res, grid, limits, [0.0, 0.1, 0.3])
    assert out.best_params == {"ent_coef": 0.1} and out.boundary_dims == ["ent_coef"]  # top edge, not a natural limit
    assert break_ties(res, grid, limits, [0.0, 0.3, 0.3]).best_params == {"ent_coef": 0.01}  # AUC tie: grid order


def test_break_ties_leaves_unique_best_alone():
    grid = {"ent_coef": [0.0, 0.01, 0.1]}
    res = _tied_result([0.3, 0.2, 0.5], grid)
    assert break_ties(res, grid, None, [0.9, 0.0, 0.9]) is res


def test_run_phase_end_to_end_resumes_and_guards(tiny_mg_cfg, tmp_path):
    run_phase(tiny_mg_cfg, tmp_path)
    phase = Phase.from_config(tiny_mg_cfg)
    df = load_results(tmp_path, "minigrid")
    assert set(df["agent"]) == {"kairi", "kaiser"}
    assert set(df["seed"]) == set(phase.eval_seeds)
    assert set(df["metric"]) == {"steps_to_80", "reached_80", "success", "return", "success_fewshot", "return_fewshot"}
    zero_shot = df[df["metric"] == "success"]
    assert set(zero_shot["knob"]) == {5.0, 6.0} and zero_shot["value"].between(0, 1).all()
    assert set(df[df["metric"] == "success_fewshot"]["knob"]) == {6.0}
    assert (df["config_hash"] == phase.config_hash).all()
    assert read_tune(tmp_path, phase, "kairi") is not None and read_tune(tmp_path, phase, "kaiser") is not None

    for agent in ("kairi", "kaiser"):
        sub = df[df["agent"] == agent]
        assert (sub["provisional"].astype(bool) == read_tune(tmp_path, phase, agent).provisional).all()
    reached = df[df["metric"] == "reached_80"].set_index(["agent", "seed"])["value"]
    steps = df[df["metric"] == "steps_to_80"].set_index(["agent", "seed"])["value"]
    assert (steps[reached == 0] == tiny_mg_cfg["budget"]["steps"]).all()

    stamps = {p: p.stat().st_mtime_ns for p in (tmp_path / "minigrid").glob("*__seed-*.csv")}
    assert len(stamps) == 4  # 2 agents x 2 eval seeds
    run_phase(tiny_mg_cfg, tmp_path)
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps  # resumed: nothing recomputed

    changed = {**tiny_mg_cfg, "budget": {**tiny_mg_cfg["budget"], "steps": 512}}
    with pytest.raises(ConfigMismatchError):
        run_phase(changed, tmp_path)


def test_unequal_grids_are_refused(tiny_mg_cfg, tmp_path):
    tiny_mg_cfg["agents"]["kaiser"]["grid"]["rnd_coef"] = [0.0, 0.01, 0.03]
    with pytest.raises(ProtocolError):
        run_phase(tiny_mg_cfg, tmp_path)


def test_two_workers_match_one_worker(tiny_mg_cfg, tmp_path):
    run_phase(tiny_mg_cfg, tmp_path / "one", workers=1)
    run_phase(tiny_mg_cfg, tmp_path / "two", workers=2)
    cols = ["agent", "knob", "seed", "metric", "value"]
    a = load_results(tmp_path / "one", "minigrid")[cols].sort_values(cols[:4]).reset_index(drop=True)
    b = load_results(tmp_path / "two", "minigrid")[cols].sort_values(cols[:4]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
