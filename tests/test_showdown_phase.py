from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("poke_env")

from tests.toy_showdown import ToyVec  # noqa: E402
from xp.errors import ProtocolError  # noqa: E402
from xp.protocol.jobs import grid_size  # noqa: E402
from xp.protocol.phase import Phase, load_config  # noqa: E402
from xp.protocol.results import load_results  # noqa: E402
from xp.showdown import phase as sp  # noqa: E402
from xp.showdown.teams import load_pool, pool_digest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """The smoke config shrunk to toy-env size, with make_env replaced by the toy battle env."""
    cfg = load_config(ROOT / "configs" / "showdown-smoke.yaml")
    cfg["ppo"].update({"n_envs": 16, "n_steps": 16})
    cfg["budget"]["steps"] = 16 * 16 * 3
    cfg["eval"]["battles"] = 20
    made = []

    def fake_make_env(c, stage, seed):
        made.append((stage, seed))
        return ToyVec(c["ppo"]["n_envs"], seed)

    monkeypatch.setattr(sp, "make_env", fake_make_env)
    monkeypatch.setattr(sp, "preflight", lambda c: None)
    return cfg, tmp_path, made


def test_real_config_matches_the_user_decisions_and_the_protocol():
    cfg = load_config(ROOT / "configs" / "showdown.yaml")
    phase = Phase.from_config(cfg)
    assert cfg["env"]["opponent"] == "damage" and cfg["budget"]["steps"] == 250000
    assert grid_size(cfg["agents"]["kairi"]) == grid_size(cfg["agents"]["kaiser"]) == 6
    assert len(phase.tune_seeds) == 3 and len(phase.eval_seeds) == 10 and not set(phase.tune_seeds) & set(phase.eval_seeds)
    tune_jobs = 2 * 6 * len(phase.tune_seeds)
    assert tune_jobs + 2 * len(phase.eval_seeds) == 56
    assert cfg["ppo"]["n_steps"] * cfg["ppo"]["n_envs"] % cfg["ppo"]["minibatches"] == 0


def test_config_team_hashes_match_the_files_in_the_repo():
    for name in ("showdown", "showdown-smoke"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        for stage in ("tune", "eval"):
            assert cfg["env"][f"{stage}_teams_sha"] == pool_digest(load_pool(sp.team_dir(cfg, stage)))


def test_preflight_rejects_unpatched_data_changed_teams_and_a_missing_server(monkeypatch):
    cfg = load_config(ROOT / "configs" / "showdown.yaml")
    sp.preflight(cfg, check_server=False)  # the repo's real state passes (needs `python -m xp.showdown.patch`)
    changed = copy.deepcopy(cfg)
    changed["env"]["tune_teams_sha"] = "000000000000"
    with pytest.raises(ProtocolError, match="team files changed"):
        sp.preflight(changed, check_server=False)
    monkeypatch.setattr(sp.patch, "missing", lambda: {"species": ["golisopodmega"], "moves": []})
    with pytest.raises(ProtocolError, match="xp.showdown.patch"):
        sp.preflight(cfg, check_server=False)
    monkeypatch.undo()

    def refuse(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(sp.socket, "create_connection", refuse)
    with pytest.raises(ProtocolError, match="no Showdown server"):
        sp.preflight(cfg)


def test_build_learner_makes_plain_ppo_for_kairi_and_for_a_zero_rnd_coefficient():
    cfg = load_config(ROOT / "configs" / "showdown-smoke.yaml")
    assert sp.build_learner(cfg, "kairi", {"ent_coef": 0.01}, 0).rnd is None
    assert sp.build_learner(cfg, "kaiser", {"rnd_coef": 0, "ent_coef": 0.02}, 0).rnd is None
    kaiser = sp.build_learner(cfg, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.02}, 0)
    assert kaiser.rnd is not None and kaiser.rnd_coef == 0.3 and kaiser.ent_coef == 0.02
    with pytest.raises(ValueError):
        sp.build_learner(cfg, "tag-team", {}, 0)


def test_tune_job_trains_on_the_tune_pool_scores_one_minus_auc_and_caches(tiny):
    cfg, out, made = tiny
    score = sp.tune_job(cfg, out, "kairi", {"ent_coef": 0.01}, 0)
    assert 0.0 <= score <= 1.0 and made == [("tune", 0)]
    assert sp.tune_job(cfg, out, "kairi", {"ent_coef": 0.01}, 0) == score and len(made) == 1  # cached: no new env
    with pytest.raises(ProtocolError):
        sp.tune_job(cfg, out, "kairi", {"ent_coef": 0.01}, 100)  # an eval seed


def test_seed_job_trains_on_the_eval_pool_writes_every_metric_and_is_idempotent(tiny):
    cfg, out, made = tiny
    sp.seed_job(cfg, out, "kaiser", {"rnd_coef": 0.1, "ent_coef": 0.01}, False, 100)
    assert made == [("eval", 100)]
    df = load_results(out, "showdown")
    assert set(df["metric"]) == {"steps_to_30", "reached_30", "auc", "win_final", "win_eval"}
    assert set(df["agent"]) == {"kaiser"} and set(df["seed"]) == {100} and not df["provisional"].any()
    assert (out / "showdown" / "curves" / "kaiser__seed-100.json").exists()
    sp.seed_job(cfg, out, "kaiser", {"rnd_coef": 0.1, "ent_coef": 0.01}, False, 100)
    assert len(made) == 1
    with pytest.raises(ProtocolError):
        sp.seed_job(cfg, out, "kairi", {"ent_coef": 0.01}, False, 0)  # a tune seed


def test_run_phase_end_to_end_on_the_toy_env(tiny):
    cfg, out, made = tiny
    sp.run_phase(cfg, out)
    df = load_results(out, "showdown")
    phase = Phase.from_config(cfg)
    for agent in ("kairi", "kaiser"):
        sub = df[(df["agent"] == agent) & (df["metric"] == "auc")]
        assert sorted(sub["seed"]) == list(phase.eval_seeds)
    stages = {s for s, _ in made}
    assert stages == {"tune", "eval"}
    tune_seeds = {seed for stage, seed in made if stage == "tune"}
    eval_seeds = {seed for stage, seed in made if stage == "eval"}
    assert tune_seeds <= set(phase.tune_seeds) and eval_seeds <= set(phase.eval_seeds)
    before = len(made)
    sp.run_phase(cfg, out)  # resumable: everything is cached
    assert len(made) == before


def test_run_phase_rejects_unequal_tuning_grids(tiny):
    cfg, out, _ = tiny
    cfg["agents"]["kaiser"]["grid"]["rnd_coef"] = [0.1]
    with pytest.raises(ProtocolError, match="equal size"):
        sp.run_phase(cfg, out)


def test_matchup_flag_sets_the_learner_observation_size_and_the_hash(tiny):
    from xp.protocol.phase import config_hash
    from xp.showdown import encode

    tiny_sd_cfg = tiny[0]
    off = sp.build_learner(tiny_sd_cfg, "kairi", {"ent_coef": 0.01}, 0)
    assert off.obs_dim == encode.OBS_DIM
    on_cfg = {**tiny_sd_cfg, "env": {**tiny_sd_cfg["env"], "matchup": True}}
    on = sp.build_learner(on_cfg, "kaiser", {"rnd_coef": 0.3, "ent_coef": 0.02}, 0)
    assert on.obs_dim == encode.obs_dim(True) and on.rnd.obs_dim == encode.obs_dim(True)
    assert config_hash(tiny_sd_cfg) != config_hash(on_cfg)
