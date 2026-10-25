"""Figures for docs/MULTIROOM_EVALUATION.md. Reads the run directory; draws whatever exists so far.

python scripts/multiroom_eval_figs.py --run ../xp-agents-run/results/multiroom --out figs/multiroom
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"kairi": "#1f77b4", "kaiser": "#d62728"}


def _curve(c):
    return np.array([np.nan if p[1] is None else p[1] for p in c], dtype=float)


def calibration(cal_path, out):
    d = json.load(open(cal_path))
    s = d["suggestion"]
    x = s["log_steps"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, c in enumerate(d["curves"]):
        ax.plot(x, [q["success_100"] for q in c], lw=0.8, alpha=0.5, label=f"tune seed {i}")
    ax.plot(x, s["mean_success"], color="k", lw=2, label="mean (3 seeds)")
    ax.axhline(0.95, ls="--", color="grey")
    ax.axvline(s["steps"], ls=":", color="k")
    ax.text(s["steps"], 0.05, f"  S = {s['steps']:,}", fontsize=8)
    ax.set(xlabel="steps", ylabel="training success (last 100 episodes)",
           title="Calibration: default-entropy Kairi (0.01), tune seeds")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load_tuning(run):
    G = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(run, "minigrid/tuning/jobs/*.json"))):
        d = json.load(open(f))
        agent = os.path.basename(f).split("__")[0]
        G[(agent, list(d["params"].values())[0])].append(d)
    return G


def tuning(G, out):
    agents = [a for a in ("kairi", "kaiser") if any(k[0] == a for k in G)]
    if not agents:
        return None
    fig, axes = plt.subplots(len(agents), 3, figsize=(15, 4.2 * len(agents)), squeeze=False)
    for r, a in enumerate(agents):
        keys = sorted(k for k in G if k[0] == a)
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
        best = max(keys, key=lambda k: np.mean([j["extra"]["auc"] for j in G[k]]))
        ax = axes[r][0]
        for k, c in zip(keys, cmap):
            cs = np.array([_curve(j["extra"]["curve"]) for j in G[k]])
            x = [p[0] for p in G[k][0]["extra"]["curve"]]
            ax.plot(x, np.nanmean(cs, 0), color=c, lw=2 if k == best else 1, label=str(k[1]))
        knob = "ent_coef" if a == "kairi" else "rnd_coef"
        ax.set(title=f"{a}: mean training success ({knob})", xlabel="steps", ylabel="success")
        ax.legend(fontsize=7, title=knob)
        ax = axes[r][1]
        ax.bar([str(k[1]) for k in keys],
               [np.mean([j["extra"]["auc"] for j in G[k]]) for k in keys],
               color=[COL[a] if k == best else "#bbbbbb" for k in keys])
        for i, k in enumerate(keys):
            ax.scatter([i] * len(G[k]), [j["extra"]["auc"] for j in G[k]], color="k", s=14, zorder=3)
        ax.set(title=f"{a}: tuning AUC per seed (dots) and mean (bars)", xlabel=knob, ylabel="AUC")
        ax = axes[r][2]
        best_curves = [_curve(j["extra"]["curve"]) for j in G[best]]
        x = [p[0] for p in G[best][0]["extra"]["curve"]]
        for j, c in zip(G[best], best_curves):
            ax.plot(x, c, label=f"seed {j['seed']}")
        ax.set(title=f"{a} best ({knob}={best[1]}): each tune seed", xlabel="steps", ylabel="success")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return agents


def probe(path, out):
    rows = list(csv.DictReader(open(path)))
    labels = [("ppo", ""), ("ppo+rnd", "0.01"), ("ppo+rnd", "0.03"), ("ppo+rnd", "0.1")]
    names = ["PPO", "RND 0.01", "RND 0.03", "RND 0.1"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (a, c) in enumerate(labels):
        v = [(int(r["steps_to_95"]) / 1e6, r["reached_95"] == "True") for r in rows
             if r["agent"] == a and r["rnd_coef"] == c]
        ax.scatter([i] * len(v), [s for s, _ in v], c=["k" if ok else "r" for _, ok in v], s=30)
        ax.hlines(np.median([s for s, _ in v]), i - 0.25, i + 0.25, color="C0")
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set(ylabel="steps to 95% (millions); red = never reached (5M cap)",
           title="Earlier post-hoc probe (5 seeds, bar = median)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def evaluation(run, out):
    files = sorted(glob.glob(os.path.join(run, "minigrid/curves/*.json")))
    if not files:
        return False
    C = collections.defaultdict(list)
    for f in files:
        d = json.load(open(f))
        C[os.path.basename(f).split("__")[0]].append(d)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for a, L in C.items():
        cs = np.array([_curve(d) for d in L])
        x = [p[0] for p in L[0]]
        for c in cs:
            ax.plot(x, c, color=COL[a], alpha=0.2, lw=0.8)
        ax.plot(x, np.nanmean(cs, 0), color=COL[a], lw=2.2, label=f"{a} (n={len(L)})")
    ax.axhline(0.95, ls="--", color="grey")
    ax.set(xlabel="steps", ylabel="training success", title="Eval seeds 100-109: success vs steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/multiroom")
    ap.add_argument("--out", default="figs/multiroom")
    ap.add_argument("--calibration", default="calibration-multiroom.json")
    ap.add_argument("--probe", default="docs/spikes/multiroom_probe.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    calibration(a.calibration, f"{a.out}/eval_calibration.png")
    probe(a.probe, f"{a.out}/eval_probe.png")
    ag = tuning(load_tuning(a.run), f"{a.out}/eval_tuning.png")
    ev = evaluation(a.run, f"{a.out}/eval_curves.png")
    print("tuned agents drawn:", ag, "| eval curves drawn:", ev)


if __name__ == "__main__":
    main()
