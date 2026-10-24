"""Post-hoc sensitivity check (outside the protocol): rerun an agent's eval at another tuning-tied setting.

Kaiser's tuning had a 3-way tie at score 0.0 (rnd_coef 0.001, 0.003, 0.01) and Kairi a 2-way tie (ent_coef 0.03, 0.1); AUC
broke each tie. This trains and evaluates the agent with the given value on the same eval seeds, into a separate out dir
(<out-root>/<rnd|ent>-<value>), so the main results are untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from xp.minigrid.phase import _run_jobs, seed_job
from xp.protocol.phase import Phase, load_config

PARAM = {"kairi": ("ent_coef", "ent"), "kaiser": ("rnd_coef", "rnd")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--agent", required=True, choices=sorted(PARAM))
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    name, short = PARAM[args.agent]
    jobs = [(cfg, Path(args.out_root) / f"{short}-{v:g}", args.agent, {name: v}, False, s)
            for v in args.values for s in phase.eval_seeds]
    _run_jobs(seed_job, jobs, args.workers)
    print(f"finished {len(jobs)} {args.agent} seed jobs", flush=True)


if __name__ == "__main__":
    main()
