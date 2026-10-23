from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataviz reference palette (light mode): categorical slots in fixed order, plus ink/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def plot_degradation(
    curves, familiar: float, out_path, title: str = "Regret vs jump rate",
    xlabel: str = "jump rate λ (dashed line = familiar)", ylabel: str = "mean per-step regret (95% CI)",
) -> None:
    knobs = sorted(curves["knob"].unique())
    pos = {k: i for i, k in enumerate(knobs)}
    agents = sorted(curves["agent"].unique())  # colour follows the entity (sorted name), not its rank
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    direct_label = len(agents) <= 4
    for i, agent in enumerate(agents):
        color = SERIES[i % len(SERIES)]  # slots are never cycled past 8 in practice
        g = curves[curves["agent"] == agent].sort_values("knob")
        x = [pos[k] for k in g["knob"]]
        ax.fill_between(x, g["lo"], g["hi"], color=color, alpha=0.18, linewidth=0)
        ax.plot(
            x, g["mean"], color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=agent,
        )
        if direct_label:
            ax.annotate(
                agent, (x[-1], list(g["mean"])[-1]), xytext=(8, 0), textcoords="offset points",
                color=INK_2, fontsize=9, va="center",
            )
    ax.axvline(pos[familiar], linestyle="--", linewidth=1, color=MUTED)
    ax.set_xticks(range(len(knobs)))
    ax.set_xticklabels([f"{k:g}" for k in knobs])
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.margins(x=0.08)
    legend = ax.legend(frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
