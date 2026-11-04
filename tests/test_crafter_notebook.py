from __future__ import annotations

import json


def test_notebook_has_the_planned_cells_and_literal_paths():
    nb = json.load(open("notebooks/run_crafter.ipynb"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for needle in ("drive.mount", "git clone", "requirements-crafter.txt", "torch.cuda.is_available", "crafter-smoke.yaml",
                   "xp.crafter.calibrate", "xp.run --phase crafter", "xp.report --phase crafter", "xp-results", "xp-figs-crafter"):
        assert needle in src, needle
    assert "$" not in "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")  # literal paths: `$VAR` came out empty in Colab
    assert all(c.get("outputs") == [] for c in nb["cells"] if c["cell_type"] == "code")  # committed clean
