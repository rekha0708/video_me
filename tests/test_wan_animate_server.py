from pathlib import Path

import pytest

import services.wan_animate_server as server


def test_health_exposes_fa3_and_mode() -> None:
    body = server.health().body.decode()
    assert "flash_attn_3" in body
    assert "model_loaded" in body
    assert "mode" in body


def test_safe_prepared_dir_rejects_outside_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(server, "WAN_ANIMATE_DATA_ROOT", root.resolve())
    with pytest.raises(Exception, match="outside"):
        server._safe_prepared_dir(str(outside))


def test_safe_prepared_dir_checks_required_inputs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "root"
    prepared = root / "job" / "shot"
    prepared.mkdir(parents=True)
    monkeypatch.setattr(server, "WAN_ANIMATE_DATA_ROOT", root.resolve())
    with pytest.raises(Exception, match="missing"):
        server._safe_prepared_dir(str(prepared))
    for name in ("src_ref.png", "src_pose.mp4", "src_face.mp4"):
        (prepared / name).write_bytes(b"x")
    assert server._safe_prepared_dir(str(prepared)) == prepared.resolve()
