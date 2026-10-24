"""Post-hoc check (outside the protocol): Kaiser at Kairi's tuned entropy coefficient.

Kaiser's `ent_coef` is fixed at 0.01 while Kairi tuned 0.03, so Kaiser's deficit could be an entropy effect rather than an RND effect.
The config (configs/minigrid-entfair.yaml) differs only in Kaiser's fixed `ent_coef`. This re-tunes Kaiser's `rnd_coef` with tune
seeds only, then evaluates it on the same eval seeds into a separate out dir, so the main results are untouched. Kairi is not rerun;
compare against the main Kairi CSVs, paired by seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from xp.minigrid.phase import _run_jobs, _tune_agent, seed_job
from xp.protocol.phase import Phase, load_config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    phase = Phase.from_config(cfg)
    tuned = _tune_agent(cfg, phase, Path(args.out), "kaiser", args.workers)
    jobs = [(cfg, Path(args.out), "kaiser", tuned.best_params, tuned.provisional, s) for s in phase.eval_seeds]
    _run_jobs(seed_job, jobs, args.workers)
    print(f"finished {len(jobs)} kaiser seed jobs at {tuned.best_params} (provisional={tuned.provisional})", flush=True)


if __name__ == "__main__":
    main()
