"""Train one frozen Showdown agent, play battles with it, and keep the most interesting replays.

python scripts/record_showdown_replays.py --agent kairi --seed 100 --battles 40 --keep 5 --out replays

Reads the tuned hyperparameter from the finished run (results/showdown/showdown/tuning/<agent>.json); never writes into results/.
Needs the local Showdown server (see RESUME.md) and the torch + poke-env installs. Plan: docs/superpowers/plans/2026-10-27-showdown-battle-replays.md
Output in <out>/<agent>_seed<S>/: policy.pt (frozen weights), all/ (every replay page of the agent's own account),
top/ (the `--keep` best by interest score, rank-prefixed), ranking.csv (all scored battles, best first).
Replay pages load the viewer from play.pokemonshowdown.com, so opening them needs internet.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from xp.protocol.phase import Phase, load_config
from xp.showdown import phase as sp
from xp.showdown.replays import rank_replays
from xp.showdown.teams import load_pool
from xp.showdown.vec_env import ShowdownVecEnv


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", required=True, choices=["kairi", "kaiser"])
    ap.add_argument("--seed", type=int, default=100, help="an eval-pool training seed (default: the first)")
    ap.add_argument("--battles", type=int, default=40, help="battles to play with the frozen policy")
    ap.add_argument("--keep", type=int, default=5, help="how many of the highest-scoring replays to keep in top/")
    ap.add_argument("--config", default="configs/showdown.yaml")
    ap.add_argument("--tuned", default="results/showdown/showdown/tuning", help="folder holding <agent>.json from the finished run")
    ap.add_argument("--out", default="replays")
    ap.add_argument("--steps", type=int, default=None, help="override the training budget (smoke tests only; the real budget is the config's)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    Phase.from_config(cfg).assert_eval_seeds([args.seed])  # eval-pool seeds and teams only
    sp.preflight(cfg)
    params = json.loads((Path(args.tuned) / f"{args.agent}.json").read_text())["best_params"]
    out = Path(args.out) / f"{args.agent}_seed{args.seed}"
    all_dir, top_dir = out / "all", out / "top"
    for d in (all_dir, top_dir):
        d.mkdir(parents=True, exist_ok=True)

    steps = args.steps or cfg["budget"]["steps"]
    learner = sp.build_learner(cfg, args.agent, params, args.seed)
    env = sp.make_env(cfg, "eval", args.seed)  # training env: no replays (it plays thousands of battles)
    try:
        learner.train(env, steps, target=cfg["budget"]["target_win"])
    finally:
        env.close()
    frozen = learner.fingerprint()
    state = {"net": learner.net.state_dict(), "params": params, "agent": args.agent, "seed": args.seed, "steps": steps}
    if learner.rnd is not None:
        state["rnd"] = learner.rnd.state_dict()
    torch.save(state, out / "policy.pt")

    rec = ShowdownVecEnv(cfg["ppo"]["n_envs"], cfg["env"]["format"], load_pool(sp.team_dir(cfg, "eval")), cfg["env"]["opponent"],
                         seed=args.seed + 1000, save_replays=str(all_dir))  # new team draws; recording env
    try:
        result = learner.evaluate(rec, args.battles)
    finally:
        rec.close()
    if learner.fingerprint() != frozen:
        raise RuntimeError("the frozen agent changed while recording")

    ranking = rank_replays(all_dir)
    for i, row in enumerate(ranking[: args.keep], 1):
        shutil.copy(all_dir / row["file"], top_dir / f"{i:02d}_{row['file']}")
    if ranking:
        with open(out / "ranking.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ranking[0]))
            writer.writeheader()
            writer.writerows(ranking)
    print(f"[replays] {args.agent} seed {args.seed}: {result['battles']} battles, win rate {result['win_rate']:.2f}, "
          f"{len(ranking)} scored, kept {min(args.keep, len(ranking))} in {top_dir}", flush=True)


if __name__ == "__main__":
    main()
