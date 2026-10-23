from __future__ import annotations

import json

import pytest

from xp.errors import ProtocolError
from xp.protocol import manifest as mf
from xp.protocol.phase import Phase


def _read(tmp_path):
    return json.loads((tmp_path / "bandit" / "manifest.json").read_text())


def test_first_write_and_same_commit_resume(tiny_cfg, tmp_path, monkeypatch):
    phase = Phase.from_config(tiny_cfg)
    monkeypatch.setattr(mf, "_git_commit", lambda: "aaa")
    mf.write_manifest(tiny_cfg, phase, tmp_path, strict_code=True)
    mf.write_manifest(tiny_cfg, phase, tmp_path, strict_code=True)
    m = _read(tmp_path)
    assert m["git_commit"] == "aaa" and [s["git_commit"] for s in m["sessions"]] == ["aaa"]


def test_strict_code_change_raises_unless_allowed(tiny_cfg, tmp_path, monkeypatch):
    phase = Phase.from_config(tiny_cfg)
    monkeypatch.setattr(mf, "_git_commit", lambda: "aaa")
    mf.write_manifest(tiny_cfg, phase, tmp_path, strict_code=True)
    monkeypatch.setattr(mf, "_git_commit", lambda: "bbb")
    with pytest.raises(ProtocolError, match="aaa.*bbb|bbb.*aaa") as e:
        mf.write_manifest(tiny_cfg, phase, tmp_path, strict_code=True)
    assert "--allow-code-change" in str(e.value)
    assert [s["git_commit"] for s in _read(tmp_path)["sessions"]] == ["aaa"]
    mf.write_manifest(tiny_cfg, phase, tmp_path, strict_code=True, allow_code_change=True)
    m = _read(tmp_path)
    assert [s["git_commit"] for s in m["sessions"]] == ["aaa", "bbb"] and m["git_commit"] == "bbb"


def test_non_strict_never_raises(tiny_cfg, tmp_path, monkeypatch):
    phase = Phase.from_config(tiny_cfg)
    monkeypatch.setattr(mf, "_git_commit", lambda: "aaa")
    mf.write_manifest(tiny_cfg, phase, tmp_path)
    monkeypatch.setattr(mf, "_git_commit", lambda: "bbb")
    mf.write_manifest(tiny_cfg, phase, tmp_path)
    assert _read(tmp_path)["git_commit"] == "bbb"
