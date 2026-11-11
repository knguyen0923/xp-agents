"""The budget gate: train plain PPO (Kairi) against one scripted rung and report throughput and the win-rate curve.

Answers two questions before any tuning: how many decisions per second a real PPO run gets (this fixes the cost of a
phase and the cap on the budget S), and whether plain PPO lands between 20% and 80% win rate against some rung.
Run: python -m xp.showdown.gate --opponent random --steps 100000 --out gate_random.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from xp.showdown.ppo import Learner

FORMAT = "gen9championsvgc2026regmc"
TEAMS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "showdown_teams"
DEFAULT_PPO = {"n_envs": 8, "n_steps": 128, "epochs": 4, "minibatches": 4, "lr": 0.0003, "gamma": 0.99, "lam": 0.95,
               "clip": 0.2, "vf_coef": 0.5}


def measure(learner: Learner, env, steps: int, target: float = 0.8, log_every: int = 5) -> dict:
    """Train `learner` on `env` for about `steps` decisions; wall-clock throughput includes the PPO updates."""
    t0 = time.perf_counter()
    run = learner.train(env, steps, log_every=log_every, target=target)
    wall = time.perf_counter() - t0
    final = run["curve"][-1]["win_rate_100"]
    return {"steps": run["steps"], "wall_s": wall, "decisions_per_s": run["steps"] / wall, "final_win_rate_100": final,
            "in_usable_range": bool(0.2 <= final <= 0.8), "steps_to_target": run["steps_to_target"], "target": target,
            "curve": run["curve"]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train plain PPO against one scripted rung and report throughput and win rate.")
    parser.add_argument("--opponent", required=True, choices=["random", "maxbp", "damage", "tempo"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_PPO["n_envs"])
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="a tune seed")
    parser.add_argument("--teams-dir", default=str(TEAMS_DIR), help="directory holding tune/ and eval/ pools (use configs/showdown_teams_tempo for the tempo runs)")
    parser.add_argument("--matchup", action="store_true", help="matchup block on")
    parser.add_argument("--tempo", action="store_true", help="tempo block on")
    parser.add_argument("--friendly-fire", type=float, default=0.0, help="friendly-fire penalty constant C")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from xp.showdown.encode import obs_dim
    from xp.showdown.teams import load_pool
    from xp.showdown.vec_env import ShowdownVecEnv

    ppo = {**DEFAULT_PPO, "n_envs": args.n_envs}
    env = ShowdownVecEnv(args.n_envs, FORMAT, load_pool(Path(args.teams_dir) / "tune"), args.opponent, seed=args.seed,
                         matchup=args.matchup, tempo=args.tempo, friendly_fire=args.friendly_fire)
    try:
        result = measure(Learner(ppo, args.ent_coef, args.seed, obs_dim=obs_dim(args.matchup, args.tempo)), env, args.steps)
    finally:
        env.close()
    result.update({"opponent": args.opponent, "ppo": ppo, "ent_coef": args.ent_coef, "seed": args.seed,
                   "teams_dir": args.teams_dir, "matchup": args.matchup, "tempo": args.tempo, "friendly_fire": args.friendly_fire})
    Path(args.out).write_text(json.dumps(result, indent=1))
    target_pct = f"{result['target']:.0%}"
    print(f"{args.opponent}: {result['steps']} decisions in {result['wall_s']:.0f}s = {result['decisions_per_s']:.0f}/s, "
          f"final win rate {result['final_win_rate_100']:.2f}, steps to {target_pct}: {result['steps_to_target']}")


if __name__ == "__main__":
    main()
