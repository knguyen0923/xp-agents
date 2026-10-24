from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

from xp.bandit.agents import KaiserAgent, KairiAgent
from xp.bandit.tagteam import ChangeTriggered, FixedSchedule, MetaBandit, TagTeamAgent
from xp.bandit.world import make_world, run_agent
from xp.errors import ProtocolError
from xp.protocol.evaluate import evaluate_unit
from xp.protocol.manifest import write_manifest  # noqa: F401  (re-exported; tests and callers import it from here)
from xp.protocol.phase import Phase
from xp.protocol.results import TuneResult, read_tune, unit_done, write_tune, write_unit
from xp.protocol.tune import PerSeedFn, score_params, tune

AgentBuilder = Callable[..., object]  # (params, seed, k=None) -> agent; k defaults to cfg["k"]

KNOBS = ("jump_rate", "sigma", "k", "drift")


def world_params(cfg: dict, knob_value: float) -> dict:
    """World parameters at one knob value: everything comes from the config except the swept knob.

    `jump_rate` defaults to the knob only when it is the swept knob; otherwise it is the config's top-level `jump_rate`
    (default 0). `drift` is the per-step random-walk std of the arm means (default 0).
    """
    name = cfg["knob"]["name"]
    if name not in KNOBS:
        raise ProtocolError(f"unknown bandit knob {name!r}; expected one of {KNOBS}")
    params = {"lam": float(cfg.get("jump_rate", 0.0)), "k": int(cfg["k"]), "sigma": float(cfg["sigma"]),
              "drift": float(cfg.get("drift", 0.0))}
    if name == "jump_rate":
        params["lam"] = float(knob_value)
    elif name == "k":
        params["k"] = int(round(knob_value))
    elif name == "sigma":
        params["sigma"] = float(knob_value)
    else:
        params["drift"] = float(knob_value)
    return params


def build_factories(cfg: dict) -> Dict[str, AgentBuilder]:
    return {
        "kairi": lambda p, s, k=None: KairiAgent(cfg["k"] if k is None else k, p["epsilon"], p["alpha"], s),
        "kaiser": lambda p, s, k=None: KaiserAgent(cfg["k"] if k is None else k, p["gamma"], p["c"], s),
    }


def make_per_seed_fn(cfg: dict, build_agent: AgentBuilder) -> PerSeedFn:
    def fn(seeds, knob, params, chunk_index):
        w = world_params(cfg, knob)
        world = make_world(seeds, w["lam"], cfg["horizon"], w["k"], w["sigma"], w["drift"])
        agent = build_agent(params, cfg["seeds"]["agent_seed"] + chunk_index, w["k"])
        return run_agent(agent, world)

    return fn


def build_tagteam(cfg: dict, rule: str, base: dict) -> AgentBuilder:
    def build(params: dict, seed: int, k=None) -> TagTeamAgent:
        k = cfg["k"] if k is None else k
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
