from __future__ import annotations

import json

import pytest

pytest.importorskip("poke_env")

from xp.showdown import patch  # noqa: E402


def test_patch_file_has_the_champions_megas_and_moves():
    data = json.load(open(patch.PATCH))
    assert {"golisopodmega", "dragonitemega", "floettemega", "garchompmegaz", "raichumegay"} <= set(data["species"])
    assert len(data["species"]) == 54 and len(data["moves"]) == 2


def test_installed_poke_env_data_has_every_patched_entry():
    """Fails on an unpatched install; fix with `python -m xp.showdown.patch`. Without the patch a Mega battle hangs."""
    assert patch.missing() == {"species": [], "moves": []}, "run: python -m xp.showdown.patch"


def test_apply_is_idempotent():
    patch.apply()
    first = patch.missing()
    patch.apply()
    assert first == patch.missing() == {"species": [], "moves": []}
