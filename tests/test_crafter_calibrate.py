# tests/test_crafter_calibrate.py
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch")

from xp.crafter.calibrate import gate_checks, implied_hours, phase_runs, suggest_budget  # noqa: E402
from xp.crafter.stats import ACHIEVEMENTS  # noqa: E402

RARE = {a: 0.0 for a in ACHIEVEMENTS}


def curve(achs, rates=None, step=10240):
    return [{"steps": step * (i + 1), "ach_100": a, "rates": dict(rates or RARE)} for i, a in enumerate(achs)]


def test_suggest_budget_needs_three_consecutive_points():
    c = curve([float("nan"), 3.0, 5.2, 4.0, 5.1, 5.3, 5.4, 6.0])
    s = suggest_budget([c, c], target=5.0, cap=10240 * 8, sustain=3)
    assert s["reached"] and s["steps"] == 10240 * 5 and s["index"] == 4  # the lone 5.2 at index 2 does not count


def test_suggest_budget_ignores_a_crossing_in_the_last_two_points_and_falls_back_to_cap():
    c = curve([1.0, 2.0, 3.0, 4.0, 4.5, 5.5, 5.5])
    s = suggest_budget([c], target=5.0, cap=10240 * 7, sustain=3)
    assert not s["reached"] and s["steps"] == 10240 * 7 and s["index"] == 6


def test_suggest_budget_averages_over_seeds_and_counts_nan_as_zero():
    a = curve([float("nan"), 6.0, 6.0, 6.0])
    b = curve([float("nan"), 4.0, 4.0, 4.0])
    s = suggest_budget([a, b], target=5.0, cap=10240 * 4, sustain=2)
    assert s["mean_ach"][0] == 0.0 and s["reached"] and s["steps"] == 10240 * 2


def test_suggest_budget_rejects_mismatched_log_steps():
    a, b = curve([1.0, 2.0]), curve([1.0, 2.0], step=5000)
    with pytest.raises(ValueError):
        suggest_budget([a, b], 5.0, 20480, 1)


def _rates(rare_count, deep_count):
    r = {a: 0.99 for a in ACHIEVEMENTS}
    names = list(ACHIEVEMENTS)
    for a in names[:rare_count]:
        r[a] = 0.0
    for a in names[rare_count:rare_count + deep_count]:
        r[a] = 0.5
    return r


def test_gate_checks_pass_and_set_the_deep_set():
    c = curve([2.0, 3.0, 5.0, 5.5, 6.0], rates=_rates(rare_count=6, deep_count=4))
    s = suggest_budget([c], 5.0, 10240 * 5, 3)
    g = gate_checks(s, [c])
    assert g["go"] and g["rare_ok"] and g["g2"] and g["deep_ok"] and len(g["deep_set"]) == 4 and g["reasons"] == []


def test_gate_checks_stop_on_too_few_rare_deep_or_no_growth():
    flat = curve([6.0] * 6, rates=_rates(rare_count=2, deep_count=1))
    g = gate_checks(suggest_budget([flat], 5.0, 10240 * 6, 3), [flat])
    assert not g["go"] and not g["rare_ok"] and not g["deep_ok"] and not g["g2"]
    assert len(g["reasons"]) == 3


def test_gate_checks_stop_when_the_target_is_not_reached():
    c = curve([1.0, 2.0, 3.0, 3.5], rates=_rates(6, 4))
    g = gate_checks(suggest_budget([c], 5.0, 10240 * 4, 3), [c])
    assert not g["go"] and any("target" in r for r in g["reasons"])


def test_cost_arithmetic():
    cfg = {"seeds": {"tune": [0, 3], "eval": [100, 110]}, "budget": {"cap": 3000000},
           "agents": {"kairi": {"grid": {"ent_coef": [1] * 6}}, "kaiser": {"grid": {"rnd_coef": [1] * 6}}}}
    assert phase_runs(cfg) == 56
    h = implied_hours(1000000, 500.0, cfg)
    assert h["full_phase_hours"] == pytest.approx(56 * 1000000 / 500.0 / 3600)
    assert h["calibration_hours"] == pytest.approx(3 * 3000000 / 500.0 / 3600)


def test_calibrate_cli_writes_the_json(tmp_path, monkeypatch):
    from tests.test_crafter_ppo import FakeVec
    from xp.crafter import calibrate as cal
    from xp.crafter import phase as cp
    from tests.test_crafter_phase import TINY

    cfg = copy.deepcopy(TINY)
    cfg["budget"].update({"cap": 64, "target_ach": 0.5, "sustain": 1})
    del cfg["budget"]["steps"]  # calibration must run with the budget unset
    monkeypatch.setattr(cp, "make_env", lambda c, stage, seed: FakeVec(n=2, period=2, seed=seed))
    import yaml

    cfg_path, out = tmp_path / "c.yaml", tmp_path / "cal.json"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cal.main(["--config", str(cfg_path), "--out", str(out)])
    data = json.loads(out.read_text())
    assert set(data) >= {"suggestion", "gates", "throughput", "hours", "curves", "go"}
    assert len(data["curves"]) == 2 and data["throughput"]["steps_per_s"] > 0


def _cli_setup(tmp_path, monkeypatch):
    import yaml

    from tests.test_crafter_phase import TINY
    from tests.test_crafter_ppo import FakeVec
    from xp.crafter import phase as cp

    cfg = copy.deepcopy(TINY)
    cfg["budget"].update({"cap": 64, "target_ach": 0.5, "sustain": 1})
    del cfg["budget"]["steps"]
    monkeypatch.setattr(cp, "make_env", lambda c, stage, seed: FakeVec(n=2, period=2, seed=seed))
    cfg_path, out = tmp_path / "c.yaml", tmp_path / "cal.json"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg, cfg_path, out


def test_calibration_resumes_from_per_seed_cache(tmp_path, monkeypatch):
    from xp.crafter import calibrate as cal

    cfg, cfg_path, out = _cli_setup(tmp_path, monkeypatch)
    cal.main(["--config", str(cfg_path), "--out", str(out)])
    first = json.loads(out.read_text())
    assert sorted(p.name for p in tmp_path.glob("cal.seed-*.json")) == ["cal.seed-0.json", "cal.seed-1.json"]
    out.unlink()

    def boom(*a, **k):
        raise AssertionError("calibration_run must not be called when every seed is cached")

    monkeypatch.setattr(cal, "calibration_run", boom)
    cal.main(["--config", str(cfg_path), "--out", str(out)])
    again = json.loads(out.read_text())
    assert again["curves"] == first["curves"] and again["suggestion"]["steps"] == first["suggestion"]["steps"]


def test_calibration_cache_hash_mismatch_raises(tmp_path, monkeypatch):
    import yaml

    from xp.crafter import calibrate as cal
    from xp.errors import ProtocolError

    cfg, cfg_path, out = _cli_setup(tmp_path, monkeypatch)
    cal.main(["--config", str(cfg_path), "--out", str(out)])
    cfg["budget"]["cap"] = 128
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ProtocolError, match="cache"):
        cal.main(["--config", str(cfg_path), "--out", str(out)])


def test_calibration_hash_ignores_budget_steps_and_deep_set():
    from xp.crafter.calibrate import calibration_hash
    from tests.test_crafter_phase import TINY

    a, b = copy.deepcopy(TINY), copy.deepcopy(TINY)
    b["budget"]["steps"], b["budget"]["deep_set"] = 999, ["eat_cow"]
    assert calibration_hash(a) == calibration_hash(b)
    b["budget"]["cap"] = 1
    assert calibration_hash(a) != calibration_hash(b)
