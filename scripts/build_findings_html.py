"""Build docs/findings.html: one self-contained page (data, d3 and styles inlined) with the project's findings.

python scripts/build_findings_html.py [--run results/multiroom] [--out docs/findings.html]
Needs results/multiroom (git-ignored) for the MultiRoom tuning jobs and curves; everything else is in the repo.
Charts and copy live in scripts/findings_template.html.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read(path):
    with open(os.path.join(ROOT, path)) as fh:
        return list(csv.DictReader(fh))


def datasets(run):
    d = {}
    d["bandit"] = [
        {"world": w, "agent": r["agent"], "knob": r["knob"], "regret": r["mean"], "lo": r["lo"], "hi": r["hi"]}
        for w, p in (("jump rate", "figs/phase1/curves.csv"), ("arms k", "figs/shifts/k/curves.csv"),
                     ("drift", "figs/shifts/drift/curves.csv"), ("noise sigma", "figs/shifts/sigma/curves.csv"),
                     ("reversed tuning", "figs/shifts/reversed/curves.csv"))
        for r in read(p) if r["agent"] in ("kairi", "kaiser")]
    d["retune"] = [
        {"shift": s, "knob": r["knob"], "kairi_minus_kaiser": r["kairi_minus_kaiser"], "lo": r["lo"], "hi": r["hi"],
         "winner": r["winner"]}
        for s, p in (("retuned at k=50", "figs/ablations/k_at50_pair.csv"),
                     ("retuned at drift 0.1", "figs/ablations/drift_at01_pair.csv")) for r in read(p)]
    d["doorkey"] = [{"agent": r["agent"], "size": r["knob"], "success": r["mean"], "lo": r["lo"], "hi": r["hi"]}
                    for r in read("figs/curves.csv") if r["agent"] in ("kairi", "kaiser")]
    d["mr_probe"] = read("docs/spikes/multiroom_probe.csv")
    d["mr_seeds"] = read("docs/spikes/multiroom_confirmation.csv")
    tun = []
    for f in sorted(glob.glob(os.path.join(ROOT, run, "minigrid/tuning/jobs/*.json"))):
        j = json.load(open(f))
        st = j["extra"]["steps_to_target"]
        tun.append({"agent": os.path.basename(f).split("__")[0], "coef": str(list(j["params"].values())[0]),
                    "seed": str(j["seed"]), "auc": str(j["extra"]["auc"]), "steps_to_95": "" if st is None else str(st)})
    d["mr_tuning"] = tun
    cur = []
    for f in sorted(glob.glob(os.path.join(ROOT, run, "minigrid/curves/*.json"))):
        a, s = os.path.basename(f)[:-5].split("__seed-")
        c = json.load(open(f))
        for i, (x, y) in enumerate(c):
            if i % 4 == 3 or i == len(c) - 1:
                cur.append({"agent": a, "seed": s, "steps": str(x), "success": "" if y is None else str(y)})
    d["mr_curves"] = cur
    seeds = sorted({int(r["seed"]) for r in d["mr_seeds"]})
    val = {(r["agent"], int(r["seed"])): r for r in d["mr_seeds"]}
    rng = np.random.default_rng(0)
    labels = {"auc": "Training AUC", "steps_to_95": "Steps to 95%", "reached_95": "Seeds reaching 95%", "success": "Held-out success"}
    paired = []
    for m, lab in labels.items():
        x = np.array([float(val[("kairi", s)][m]) for s in seeds])
        y = np.array([float(val[("kaiser", s)][m]) for s in seeds])
        diff = y - x
        boot = [rng.choice(diff, len(diff)).mean() for _ in range(20000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        better = int((diff < 0).sum() if m == "steps_to_95" else (diff > 0).sum())
        paired.append({"metric": m, "label": lab, "kairi": f"{x.mean():.4g}", "kaiser": f"{y.mean():.4g}",
                       "diff": f"{diff.mean():.4g}", "lo": f"{lo:.4g}", "hi": f"{hi:.4g}", "better_seeds": f"{better}/{len(diff)}"})
    d["mr_paired"] = paired
    d["crafter"] = read("docs/spikes/crafter_spike_curves.csv")
    return d


HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>XP findings</title>
<style>
:root {
  --color-fg:#1d2230; --color-fg-muted:#5b6475; --color-bg:#f7f8fa; --color-panel:#ffffff; --color-border-line:#dfe3ea;
  --c1:#2f6db5; --c2:#c7432f;
  --cds-gap-xs:6px; --cds-gap-sm:12px; --cds-gap-md:18px; --cds-pad-xs:4px; --cds-pad-sm:8px; --cds-pad-md:14px; --cds-pad-lg:18px;
  --cds-radius:10px; --cds-font-size-caption:12px; --cds-font-size-body:15px; --cds-font-size-heading:16px; --cds-font-size-title:1.55rem;
  --cds-font-weight-medium:600;
  --cds-chart-axis:#9aa3b2; --cds-chart-grid:#e6e9ef; --cds-chart-reference:#7b8496; --cds-chart-muted:#aab1bf; --cds-chart-status-critical:#c7432f;
  --cds-surface-popover:#ffffff; --cds-border:#cfd5df; --cds-shadow-popover:0 4px 16px rgba(0,0,0,.14);
  --font-anthropic-sans:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-anthropic-serif:Georgia, "Iowan Old Style", "Times New Roman", serif;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root {
  --color-fg:#e8ebf2; --color-fg-muted:#9aa3b5; --color-bg:#14171f; --color-panel:#1c2029; --color-border-line:#2c323f;
  --c1:#6aa6ec; --c2:#f0866f;
  --cds-chart-axis:#566074; --cds-chart-grid:#2a303c; --cds-chart-reference:#8a93a6; --cds-chart-muted:#566074; --cds-chart-status-critical:#f0866f;
  --cds-surface-popover:#232835; --cds-border:#39414f; --cds-shadow-popover:0 4px 16px rgba(0,0,0,.5);
  color-scheme: dark; } }
html { background: var(--color-bg); }
body { margin:0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans);
  padding: 28px max(16px, env(safe-area-inset-right, 0px)) 60px max(16px, env(safe-area-inset-left, 0px)); }
</style>
</head><body>
"""

SHIM = """<script>
(function () {
  var store = {};
  window.__DATA = %s;
  Object.keys(window.__DATA).forEach(function (k) { store[k] = { status: 'ok', data: window.__DATA[k] }; });
  window.dash = {
    colors: ['var(--c1)', 'var(--c2)'],
    data: function (id) { return store[id] || { status: 'missing', data: [] }; },
    calc: function (id, o) { store[id] = { status: 'ok', data: o.fn.apply(null, o.inputs.map(function (i) { return store[i].data; })) }; },
    onData: function (fn) {
      fn(); var t;
      window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(fn, 120); });
    }
  };
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/multiroom")
    ap.add_argument("--out", default="docs/findings.html")
    a = ap.parse_args()
    d3 = open(os.path.join(HERE, "vendor/d3.v7.9.0.min.js")).read()
    tpl = open(os.path.join(HERE, "findings_template.html")).read()
    data = json.dumps(datasets(a.run), separators=(",", ":"))
    i = tpl.index("<script>")
    page = HEAD + "<script>" + d3 + "</script>\n" + tpl[:i] + SHIM % data + tpl[i:] + "\n</body></html>\n"
    out = os.path.join(ROOT, a.out)
    with open(out, "w") as fh:
        fh.write(page)
    print(f"wrote {a.out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
