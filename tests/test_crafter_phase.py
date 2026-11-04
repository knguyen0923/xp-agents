from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

from tests.test_crafter_ppo import FakeVec  # noqa: E402
from xp.crafter import phase as cp  # noqa: E402
from xp.errors import ConfigMismatchError, ProtocolError  # noqa: E402
from xp.protocol.phase import Phase  # noqa: E402
from xp.protocol.results import load_results, read_tune_job, read_tune_job_extra  # noqa: E402

TINY = {
    "phase": "crafter",
    "knob": {"name": "area", "familiar": 32, "sweep": [32]},
    "env": {"area": 32},
    "seeds": {"tune": [0, 2], "eval": [100, 102], "chunk": 1, "agent_seed": 0},
    "budget": {"steps": 64, "cap": 128, "target_ach": 5, "sustain": 3, "min_deep": 3, "deep_set": ["collect_wood", "eat_cow", "collect_drink"]},
    "eval": {"episodes": 3},
    "ppo": {"device": "cpu", "log_every": 2, "n_envs": 2, "n_steps": 8, "epochs": 1, "minibatches": 2, "lr": 0.00025, "gamma": 0.99,
            "lam": 0.95, "clip": 0.2, "vf_coef": 0.5, "rnd_lr": 0.0001},
    "agents": {
        "kairi": {"grid": {"ent_coef": [0.01, 0.05]}, "natural_limits": {"ent_coef": [0.01, 0.05]}},
        "kaiser": {"fixed": {"ent_coef": 0.01, "ent_coef_from": "kairi"}, "grid": {"rnd_coef": [0.1, 0.3]},
                   "natural_limits": {"rnd_coef": [0.1, 0.3]}},
    },
    "report": {"metric": "auc", "lower_is_better": False, "agents": ["kairi", "kaiser"]},
}


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(cp, "make_env", lambda c, stage, seed: FakeVec(n=c["ppo"]["n_envs"], period=4, seed=seed))
    return copy.deepcopy(TINY)


def test_kairi_and_kaiser_differ_only_by_the_rnd_path(cfg):
    kairi = cp.build_learner(cfg, "kairi", {"ent_coef": 0.03}, seed=1)
    kaiser = cp.build_learner(cfg, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.03}, seed=1)
    assert kairi.rnd is None and kairi.ent_coef == 0.03
    assert kaiser.rnd is not None and kaiser.rnd_coef == 0.3 and kaiser.ent_coef == 0.03
    assert cp.build_learner(cfg, "kaiser", {"rnd_coef": 0.3}, seed=1).ent_coef == 0.01  # falls back to the fixed default


def test_tune_job_guards_seeds_budget_and_caches(cfg, tmp_path):
    phase = Phase.from_config(cfg)
    with pytest.raises(ProtocolError):
        cp.tune_job(cfg, tmp_path, "kairi", {"ent_coef": 0.01}, seed=100)  # an eval seed
    unset = copy.deepcopy(cfg)
    del unset["budget"]["steps"]
    with pytest.raises(ProtocolError, match="budget.steps"):
        cp.tune_job(unset, tmp_path, "kairi", {"ent_coef": 0.01}, seed=0)
    score = cp.tune_job(cfg, tmp_path, "kairi", {"ent_coef": 0.01}, seed=0)
    assert 0.0 <= score <= 1.0 and read_tune_job(tmp_path, phase, "kairi", {"ent_coef": 0.01}, 0) == score
    extra = read_tune_job_extra(tmp_path, phase, "kairi", {"ent_coef": 0.01}, 0)
    assert 0.0 <= extra["auc"] <= 1.0 and "final" in extra and extra["curve"]


def test_seed_job_writes_every_metric_and_the_curve(cfg, tmp_path):
    cp.seed_job(cfg, tmp_path, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.01}, False, seed=100)
    df = load_results(tmp_path, "crafter")
    got = set(df["metric"])
    assert {"auc", "ach_final", "crafter_score", "deep_first_unlock", "heldout_ach", "heldout_crafter_score"} <= got
    assert "rate_collect_wood" in got and "first_collect_wood" in got and len(df) == 6 + 44
    assert set(df["seed"]) == {100} and set(df["knob"]) == {32.0}
    first = df[df["metric"] == "first_make_iron_sword"]["value"].item()
    assert first == cfg["budget"]["steps"]  # never unlocked: censored at S
    assert (tmp_path / "crafter" / "curves" / "kaiser__seed-100.json").exists()
    with pytest.raises(ProtocolError):
        cp.seed_job(cfg, tmp_path, "kairi", {"ent_coef": 0.01}, False, seed=0)  # a tune seed


def test_run_phase_end_to_end_and_resume(cfg, tmp_path):
    cp.run_phase(cfg, tmp_path)
    df = load_results(tmp_path, "crafter")
    auc = df[df["metric"] == "auc"]
    assert len(auc) == 4 and set(auc["agent"]) == {"kairi", "kaiser"} and set(auc["seed"]) == {100, 101}
    assert not auc["provisional"].any()  # grid ends are declared natural limits
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "crafter").rglob("*.csv")}
    cp.run_phase(cfg, tmp_path)  # resume: every job cached
    assert before == {p: p.stat().st_mtime_ns for p in (tmp_path / "crafter").rglob("*.csv")}
    changed = copy.deepcopy(cfg)
    changed["budget"]["steps"] = 128
    with pytest.raises(ConfigMismatchError):
        cp.run_phase(changed, tmp_path)


def test_run_phase_refuses_unequal_grids_and_unset_budget(cfg, tmp_path):
    bad = copy.deepcopy(cfg)
    bad["agents"]["kaiser"]["grid"]["rnd_coef"] = [0.1]
    with pytest.raises(ProtocolError, match="equal size"):
        cp.run_phase(bad, tmp_path)
    unset = copy.deepcopy(cfg)
    del unset["budget"]["steps"]
    with pytest.raises(ProtocolError, match="budget.steps"):
        cp.run_phase(unset, tmp_path)


def test_tie_break_uses_final_ach_when_asked(cfg, tmp_path):
    """Two grid points with identical scores: tune_agent must pick the one with the higher mean `final`."""
    from xp.protocol.jobs import tune_agent
    from xp.protocol.results import write_tune_job

    phase = Phase.from_config(cfg)

    def fake_job(c, out, agent, params, seed):
        final = 3.0 if params["ent_coef"] == 0.05 else 1.0
        write_tune_job(out, phase, agent, params, seed, 0.5, extra={"auc": 0.1, "final": final, "curve": []})
        return 0.5

    res = tune_agent(cfg, phase, tmp_path, "kairi", 1, fake_job, tie_key="final")
    assert res.best_params == {"ent_coef": 0.05}
    res_default = tune_agent(cfg, phase, tmp_path / "b", "kairi", 1, fake_job)  # default tie key `auc` is equal: grid order wins
    assert res_default.best_params == {"ent_coef": 0.01}


def test_kaiser_grid_edge_optimum_stops_the_run_before_any_seed_job(cfg, tmp_path):
    bad = copy.deepcopy(cfg)
    del bad["agents"]["kaiser"]["natural_limits"]  # a 2-value grid: the optimum is necessarily on an edge, with no exemption
    with pytest.raises(ProtocolError, match="rnd_coef"):
        cp.run_phase(bad, tmp_path)
    assert not list((tmp_path / "crafter").glob("*__seed-*.csv"))  # no seed-job file was written
