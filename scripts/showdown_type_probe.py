"""Probe saved Showdown policies for type-chart sensitivity (docs/FUTURE_WORK.md, follow-up 4).

python scripts/showdown_type_probe.py record --agent kairi --seed 100 [--decisions 1500]   # needs the Showdown server
python scripts/showdown_type_probe.py analyze --agent kairi --seed 100                      # offline, from the recorded states

`record` loads replays/<agent>_seed<S>/policy.pt, plays the frozen policy against the configured opponent on the eval team pool, and saves the
observations and masks it saw (replays/<agent>_seed<S>/probe_states.npz, gitignored). It never trains or updates anything.
`analyze` takes every recorded decision where an own active Pokemon has a legal damaging move aimed at a foe active, overwrites ONLY that foe's
type one-hots with each of the 18 single types, and reads the policy's logit for that exact action. Per decision it reports the Spearman correlation
between the logit and the real type multiplier of the move against that type (ties by average rank), and whether the highest-logit type is
super-effective. Both are averaged over decisions and bootstrapped over decisions (the unit of resampling). Original-run agents (no matchup block).
One seed per agent: exploratory, not a Kairi-vs-Kaiser comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xp.protocol.phase import Phase, load_config
from xp.showdown import phase as sp
from xp.showdown.encode import BOOSTS, MON_DIM, MOVE_DIM, N_MOVES, SLOTS, STATS, STATUSES, TYPES
from xp.stats.ci import bootstrap_ci

TYPE_OFF = 3 + len(STATUSES) + len(BOOSTS)          # start of the type one-hots inside one Pokemon row
MOVES_OFF = TYPE_OFF + len(TYPES) + len(STATS) + 1  # start of the first move row inside one Pokemon row
FOE_OFF = SLOTS * MON_DIM                            # the opponent side starts after the own side; its first two rows are the actives
CATEGORY_OFF = 3 + len(TYPES)                        # category one-hots inside a move row: physical, special, status


def mon_slice(side_off: int, slot: int) -> int:
    return side_off + slot * MON_DIM


def damaging_moves(obs: np.ndarray, own_slot: int):
    """Indices k of the own active's moves that are damaging (power > 0 and not a status move) with their type index."""
    base = mon_slice(0, own_slot)
    out = []
    for k in range(N_MOVES):
        row = obs[base + MOVES_OFF + k * MOVE_DIM: base + MOVES_OFF + (k + 1) * MOVE_DIM]
        if row[0] > 0 and row[CATEGORY_OFF + 2] == 0:
            out.append((k, int(np.argmax(row[3:3 + len(TYPES)]))))
    return out


def foe_present(obs: np.ndarray, foe_slot: int) -> bool:
    base = mon_slice(FOE_OFF, foe_slot)
    return obs[base + 2] == 1 and obs[base + 1] == 0   # active flag set, not fainted


def own_present(obs: np.ndarray, own_slot: int) -> bool:
    base = mon_slice(0, own_slot)
    return obs[base + 2] == 1 and obs[base + 1] == 0


def action_index(k: int, foe_slot: int) -> int:
    """Move k (0-based) aimed at foe slot 0 or 1: poke-env's `7 + 5k + (target + 2)` with Showdown targets 1 and 2."""
    return 7 + 5 * k + (foe_slot + 1) + 2


def with_foe_type(obs: np.ndarray, foe_slot: int, type_idx: int) -> np.ndarray:
    out = obs.copy()
    start = mon_slice(FOE_OFF, foe_slot) + TYPE_OFF
    out[start:start + len(TYPES)] = 0.0
    out[start + type_idx] = 1.0
    return out


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        idx = x == v
        r[idx] = r[idx].mean()
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation; NaN when either side is constant."""
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def multiplier_table() -> np.ndarray:
    """(18 move types, 18 defending single types) real multipliers from poke-env's gen 9 chart."""
    from poke_env.data import GenData

    chart = GenData.from_gen(9).type_chart
    return np.array([[float(a.damage_multiplier(d, type_chart=chart)) for d in TYPES] for a in TYPES])


def probe_decision(net, obs: np.ndarray, mask: np.ndarray, table: np.ndarray):
    """Per (own slot, move, foe slot) in one decision: (spearman, top-type-is-super-effective, baseline share of super-effective types)."""
    results = []
    for own_slot in (0, 1):
        if not own_present(obs, own_slot):
            continue
        for k, move_type in damaging_moves(obs, own_slot):
            for foe_slot in (0, 1):
                a = action_index(k, foe_slot)
                if not foe_present(obs, foe_slot) or not mask[own_slot, a]:
                    continue
                batch = np.stack([with_foe_type(obs, foe_slot, t) for t in range(len(TYPES))])
                with torch.no_grad():
                    logits = net.heads(torch.as_tensor(batch, dtype=torch.float32))[own_slot][:, a].numpy()
                mult = table[move_type]
                best = np.flatnonzero(logits == logits.max())
                results.append((spearman(logits, mult), float(np.mean(mult[best] >= 2)), float(np.mean(mult >= 2))))
    return results


def paths(agent: str, seed: int, root: str = "replays"):
    folder = Path(root) / f"{agent}_seed{seed}"
    return folder, folder / "policy.pt", folder / "probe_states.npz"


def load_net(policy_path: Path):
    from xp.showdown.encode import OBS_DIM
    from xp.showdown.ppo import ActorCritic

    state = torch.load(policy_path, map_location="cpu")
    net = ActorCritic(obs_dim=OBS_DIM, intrinsic="v_int.weight" in state["net"])
    net.load_state_dict(state["net"])
    net.eval()
    return net, state


def record(args) -> None:
    cfg = load_config(args.config)
    if cfg["env"].get("matchup", False):
        raise SystemExit("this probe is for the original-run agents (no matchup block)")
    Phase.from_config(cfg).assert_eval_seeds([args.seed])
    sp.preflight(cfg)
    folder, policy_path, out_path = paths(args.agent, args.seed, args.root)
    state = torch.load(policy_path, map_location="cpu")
    learner = sp.build_learner(cfg, args.agent, state["params"], args.seed)
    learner.net.load_state_dict(state["net"])
    frozen = learner.fingerprint()
    from xp.showdown.teams import load_pool
    from xp.showdown.vec_env import ShowdownVecEnv

    env = ShowdownVecEnv(cfg["ppo"]["n_envs"], cfg["env"]["format"], load_pool(sp.team_dir(cfg, "eval")), cfg["env"]["opponent"],
                         seed=args.seed + 2000)   # fresh team draws; not the replay recorder's seed
    obs_list, mask_list = [], []
    try:
        while len(obs_list) * env.n < args.decisions:
            obs_list.append(env.obs.copy())
            mask_list.append(env.masks.copy())
            actions, _, _, _ = learner.act(env.obs, env.masks)
            env.step(actions)
    finally:
        env.close()
    if learner.fingerprint() != frozen:
        raise RuntimeError("the frozen agent changed while recording")
    obs = np.concatenate(obs_list)
    masks = np.concatenate(mask_list)
    np.savez_compressed(out_path, obs=obs, masks=masks)
    print(f"[probe] {args.agent} seed {args.seed}: saved {len(obs)} decisions to {out_path}", flush=True)


def analyze(args) -> None:
    _, policy_path, states_path = paths(args.agent, args.seed, args.root)
    net, _ = load_net(policy_path)
    data = np.load(states_path)
    table = multiplier_table()
    rows = []
    for obs, mask in zip(data["obs"], data["masks"]):
        res = probe_decision(net, obs, mask, table)
        if res:
            arr = np.array(res)
            rows.append(np.nanmean(arr, axis=0) if not np.isnan(arr[:, 0]).all() else np.array([np.nan, arr[:, 1].mean(), arr[:, 2].mean()]))
    rows = np.array(rows)
    usable = rows[~np.isnan(rows[:, 0])]
    rho, rlo, rhi = bootstrap_ci(usable[:, 0])
    top, tlo, thi = bootstrap_ci(usable[:, 1])
    base = float(usable[:, 2].mean())
    summary = {"agent": args.agent, "seed": args.seed, "recorded_decisions": int(len(data["obs"])), "decisions_probed": int(len(rows)),
               "decisions_with_variation": int(len(usable)), "spearman": [rho, rlo, rhi], "top_is_super_effective": [top, tlo, thi],
               "chance_super_effective": base}
    out = Path(args.out) / f"{args.agent}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["record", "analyze"])
    ap.add_argument("--agent", required=True, choices=["kairi", "kaiser"])
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--decisions", type=int, default=1500)
    ap.add_argument("--config", default="configs/showdown.yaml")
    ap.add_argument("--root", default="replays")
    ap.add_argument("--out", default="docs/spikes/showdown_type_probe")
    args = ap.parse_args(argv)
    (record if args.mode == "record" else analyze)(args)


if __name__ == "__main__":
    main()
