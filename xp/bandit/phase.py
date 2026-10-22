from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.tagteam import ChangeTriggered, FixedSchedule, MetaBandit, TagTeamAgent
from xp.bandit.world import make_world, run_agent
from xp.errors import ConfigMismatchError, ProtocolError
from xp.protocol.evaluate import evaluate_unit
from xp.protocol.phase import Phase
from xp.protocol.results import atomic_write, TuneResult, read_tune, unit_done, write_tune, write_unit
from xp.protocol.tune import PerSeedFn, score_params, tune

AgentBuilder = Callable[[dict, int], object]


def build_factories(cfg: dict) -> Dict[str, AgentBuilder]:
    k = cfg["k"]
    return {
        "kairi": lambda p, s: KairiAgent(k, p["epsilon"], p["alpha"], s),
        "kaiser": lambda p, s: KaiserAgent(k, p["gamma"], p["c"], s),
    }


def make_per_seed_fn(cfg: dict, build_agent: AgentBuilder) -> PerSeedFn:
    def fn(seeds, knob, params, chunk_index):
        world = make_world(seeds, knob, cfg["horizon"], cfg["k"], cfg["sigma"])
        agent = build_agent(params, cfg["seeds"]["agent_seed"] + chunk_index)
        return run_agent(agent, world)

    return fn


def build_tagteam(cfg: dict, rule: str, base: dict) -> AgentBuilder:
    k = cfg["k"]

    def build(params: dict, seed: int) -> TagTeamAgent:
        kairi = KairiAgent(k, base["kairi"]["epsilon"], base["kairi"]["alpha"], seed)
        kaiser = KaiserAgent(k, base["kaiser"]["gamma"], base["kaiser"]["c"], seed)
        if rule == "fixed":
            controller = FixedSchedule(params["block"])
        elif rule == "triggered":
            controller = ChangeTriggered(params["delta"], params["h"], params["window"])
        elif rule == "meta":
            controller = MetaBandit(params["gamma"], params["c"], cfg["tagteam"]["meta"]["block"])
        else:
            raise ValueError(f"unknown tag-team rule {rule!r}")
        return TagTeamAgent(kairi, kaiser, controller)

    return build


def _tune_or_load(phase, out_dir, name, per_seed_fn, spec) -> TuneResult:
    cached = read_tune(out_dir, phase, name)
    if cached is not None:
        return cached
    result = tune(lambda p: score_params(phase, per_seed_fn, p), spec["grid"], spec.get("natural_limits"))
    write_tune(out_dir, phase, name, result)
    return result


def _grid_size(spec: dict) -> int:
    return int(np.prod([len(v) for v in spec["grid"].values()]))


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(cfg: dict, phase: Phase, out_dir) -> None:
    path = Path(out_dir) / phase.name / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text()).get("config_hash")
        if old != phase.config_hash:
            raise ConfigMismatchError(f"{path} was written under config {old}, current config is {phase.config_hash}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cfg": cfg,
        "config_hash": phase.config_hash,
        "git_commit": _git_commit(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2, default=str)))


def run_phase(cfg: dict, out_dir) -> None:
    sizes = {name: _grid_size(cfg["agents"][name]) for name in ("kairi", "kaiser")}
    if sizes["kairi"] != sizes["kaiser"]:
        raise ProtocolError(f"kairi and kaiser tuning grids must have equal size, got {sizes}")
    phase = Phase.from_config(cfg)
    write_manifest(cfg, phase, out_dir)
    factories = build_factories(cfg)

    tuned: Dict[str, TuneResult] = {}
    for name in ("kairi", "kaiser"):
        fn = make_per_seed_fn(cfg, factories[name])
        tuned[name] = _tune_or_load(phase, out_dir, name, fn, cfg["agents"][name])
    base = {name: tuned[name].best_params for name in tuned}
    base_provisional = any(tuned[name].provisional for name in tuned)

    tt = cfg["tagteam"]
    tag_results: Dict[str, TuneResult] = {"tt_fixed": TuneResult.fixed(tt["fixed"])}
    for rule in ("triggered", "meta"):
        name = f"tt_{rule}"
        fn = make_per_seed_fn(cfg, build_tagteam(cfg, rule, base))
        tag_results[name] = _tune_or_load(phase, out_dir, name, fn, tt[rule])
    write_tune(out_dir, phase, "tt_fixed", tag_results["tt_fixed"])

    plan = [(name, make_per_seed_fn(cfg, factories[name]), tuned[name]) for name in ("kairi", "kaiser")]
    for rule in ("fixed", "triggered", "meta"):
        name = f"tt_{rule}"
        plan.append((name, make_per_seed_fn(cfg, build_tagteam(cfg, rule, base)), tag_results[name]))

    for name, fn, result in plan:
        is_tag_team = name.startswith("tt_")
        provisional = result.provisional or (is_tag_team and base_provisional)
        for knob in phase.sweep:
            if unit_done(out_dir, phase, name, knob):
                continue
            df = evaluate_unit(phase, name, result.best_params, knob, provisional, fn)
            write_unit(out_dir, phase, name, knob, df)
            print(f"[{phase.name}] {name} {phase.knob_name}={knob:g} mean regret {df['value'].mean():.4f}", flush=True)
