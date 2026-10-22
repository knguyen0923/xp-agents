from __future__ import annotations

import pandas as pd

from xp.plots.curves import plot_degradation


def test_plot_degradation_writes_png(tmp_path):
    curves = pd.DataFrame(
        {
            "agent": ["A", "A", "B", "B"],
            "knob": [0.0, 0.1, 0.0, 0.1],
            "mean": [0.2, 0.3, 0.5, 0.4],
            "lo": [0.19, 0.28, 0.48, 0.38],
            "hi": [0.21, 0.32, 0.52, 0.42],
        }
    )
    out = tmp_path / "plot.png"
    plot_degradation(curves, familiar=0.0, out_path=out, title="t")
    assert out.exists() and out.stat().st_size > 1000
