"""Progress view for a running Showdown phase: jobs per agent and grid value, rate, ETA, swap.

python scripts/showdown_progress.py [--config configs/showdown-control.yaml] [--out results/showdown-control] [--watch 30]
Counts the finished job files only; it never touches the run.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time

import yaml

BAR = 30


def bar(done: int, total: int) -> str:
    n = round(BAR * done / total) if total else 0
    return "[" + "█" * n + "░" * (BAR - n) + f"] {done:>2}/{total}"


def show(cfg: dict, out: str) -> None:
    root = os.path.join(out, cfg["phase"])
    jobs_dir = os.path.join(root, "tuning", "jobs")
    t0, t1 = cfg["seeds"]["tune"]
    e0, e1 = cfg["seeds"]["eval"]
    n_tune, n_eval = t1 - t0, e1 - e0
    files = glob.glob(os.path.join(jobs_dir, "*.json"))
    stamps = sorted(os.path.getmtime(f) for f in files + glob.glob(os.path.join(root, "*__seed-*.csv")))
    total = 0
    done_all = 0
    print(f"{out}")
    for agent, spec in cfg["agents"].items():
        (knob, values), = spec["grid"].items()
        print(f"\n{agent}: tuning {knob} ({n_tune} seeds per value)")
        a_done = 0
        for v in values:
            d = len(glob.glob(os.path.join(jobs_dir, f"{agent}__{knob}-{v:g}__seed-*.json")))
            a_done += d
            print(f"  {knob}={v:<6g} {bar(d, n_tune)}")
        total += len(values) * n_tune
        done_all += a_done
        print(f"  tuning total  {bar(a_done, len(values) * n_tune)}")
    for agent in cfg["agents"]:
        d = len(glob.glob(os.path.join(root, f"{agent}__seed-*.csv")))
        total += n_eval
        done_all += d
        print(f"\n{agent}: final evaluation {bar(d, n_eval)}")
    print(f"\nOVERALL {bar(done_all, total)}  ({100 * done_all / total:.1f}%)")
    if len(stamps) >= 2 and done_all:
        span = stamps[-1] - stamps[0]
        per_job = span / (len(stamps) - 1)
        left = (total - done_all) * per_job
        print(f"pace: {3600 / per_job:.1f} jobs/hour, about {left / 3600:.1f} h left (ETA {time.strftime('%H:%M', time.localtime(time.time() + left))})")
    elif not done_all:
        print("no job has finished yet")
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    print(f"swap: {swap}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/showdown-control.yaml")
    ap.add_argument("--out", default="results/showdown-control")
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        show(cfg, args.out)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
