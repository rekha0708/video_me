import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import services.wan_animate_server as server


def test_health_exposes_fa3_and_mode() -> None:
    response = server.health()
    body = json.loads(response.body)
    assert "flash_attn_3" in body
    assert "flash_attn_3_kernel_ready" in body
    assert "flash_attn_3_device_capability" in body
    assert "model_loaded" in body
    assert "mode" in body
    assert "offload_model" in body


def test_health_is_down_when_required_fa3_kernel_is_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(server, "WAN_REQUIRE_FLASH_ATTN_3", True)
    monkeypatch.setattr(server, "_pipeline_error", None)
    monkeypatch.setattr(server, "_model_readiness_error", lambda: None)
    monkeypatch.setattr(
        server,
        "_flash_attn_3_readiness",
        lambda: {
            "imported": True,
            "kernel_ready": False,
            "device_capability": [8, 0],
            "error": "Hopper compute capability 9.0 is required",
        },
    )

    body = json.loads(server.health().body)

    assert body["status"] == "down"
    assert body["flash_attn_3"] is False
    assert "Hopper" in body["error"]


def test_negative_seed_is_replaced_with_secure_random_value(monkeypatch) -> None:
    monkeypatch.setattr(server.secrets, "randbelow", lambda upper: upper - 7)
    assert server._normalize_seed(42) == 42
    assert server._normalize_seed(-1) == 2**31 - 7
    assert server._normalize_seed(-99) == 2**31 - 7


@pytest.mark.asyncio
async def test_generate_passes_normalized_seed_and_reports_it(
    tmp_path: Path, monkeypatch
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    output = tmp_path / "generated.mp4"
    output.write_bytes(b"video")
    captured: dict[str, int] = {}

    monkeypatch.setattr(server, "_pipeline", object())
    monkeypatch.setattr(server, "_pipeline_mode", "animate")
    monkeypatch.setattr(server, "_safe_prepared_dir", lambda value: prepared)
    monkeypatch.setattr(server.secrets, "randbelow", lambda upper: 123456)

    def fake_inference(*args):
        captured["seed"] = args[-1]
        return output

    monkeypatch.setattr(server, "_inference", fake_inference)

    response = await server.generate(
        prepared_dir=str(prepared),
        mode="animate",
        fps=30,
        refert_num=1,
        sampling_steps=20,
        seed=-8,
    )

    assert captured["seed"] == 123456
    assert response.headers["x-wan-seed"] == "123456"
    output.unlink(missing_ok=True)


def test_validate_encoded_video_rejects_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "empty.mp4"
    output.touch()
    with pytest.raises(RuntimeError, match="empty MP4"):
        server._validate_encoded_video(output)


def test_validate_encoded_video_runs_ffprobe(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "result.mp4"
    output.write_bytes(b"encoded-video")
    calls: list[list[str]] = []

    monkeypatch.setattr(server.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_name": "h264", "width": 1280, "height": 720}
                    ],
                    "format": {"duration": "2.5"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    server._validate_encoded_video(output)

    assert calls
    assert calls[0][0] == "/usr/bin/ffprobe"
    assert str(output) == calls[0][-1]


def test_model_readiness_checks_core_and_preprocessor_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(server, "WAN_ANIMATE_MODEL_DIR", tmp_path)
    for relative in server._MODEL_REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "config.json":
            path.write_text("{}", encoding="utf-8")
        elif relative == "diffusion_pytorch_model.safetensors.index.json":
            path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer": "diffusion_pytorch_model-00001-of-00004.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )
        else:
            path.write_bytes(b"x")

    assert server._model_readiness_error() is None
    (tmp_path / "process_checkpoint/det/yolov10m.onnx").unlink()
    assert "yolov10m.onnx" in str(server._model_readiness_error())


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
