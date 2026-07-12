from pathlib import Path

from adapters.video_enhance.ai_adapter import AiVideoEnhanceAdapter
from core.models.capabilities import VideoEnhanceRequest


def _write_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40)
    return path


def _request(tmp_path: Path, **kwargs) -> VideoEnhanceRequest:
    return VideoEnhanceRequest(
        video_uri=str(kwargs.get("video_uri") or _write_mp4(tmp_path)),
        duration_sec=kwargs.get("duration_sec", 2.5),
        output_name=kwargs.get("output_name", "enhanced.mp4"),
        target_width=kwargs.get("target_width", 1080),
        target_height=kwargs.get("target_height", 1920),
        target_fps=kwargs.get("target_fps", 48),
        interpolation=kwargs.get("interpolation", "minterpolate"),
        stage=kwargs.get("stage", "clip"),
        has_burned_text=kwargs.get("has_burned_text", False),
        preserve_audio=kwargs.get("preserve_audio", True),
    )


async def test_realesrgan_rife_builds_ai_pipeline_commands(tmp_path: Path) -> None:
    req = _request(tmp_path)
    adapter = AiVideoEnhanceAdapter(
        backend="realesrgan_rife",
        work_dir=tmp_path / "enhanced",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        realesrgan_python="/venv/bin/python",
        realesrgan_dir=tmp_path / "Real-ESRGAN",
        rife_python="/venv/bin/python",
        rife_dir=tmp_path / "RIFE",
        rife_model_dir=tmp_path / "RIFE" / "train_log",
    )
    commands: list[list[str]] = []

    async def fake_run(cmd: list[str], cwd: Path | None = None) -> None:
        commands.append(cmd)
        if "inference_realesrgan_video.py" in cmd[1]:
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "input_realesrgan.mp4").write_bytes(b"sr")
        elif "inference_video.py" in cmd[1]:
            Path(cmd[cmd.index("--output") + 1]).write_bytes(b"rife")
        else:
            Path(cmd[-1]).write_bytes(b"final")

    adapter._run_command = fake_run  # type: ignore[method-assign]

    result = await adapter.run(req)

    assert result.adapter == "realesrgan_rife"
    assert Path(result.video_uri).read_bytes() == b"final"
    assert "inference_realesrgan_video.py" in commands[0][1]
    assert commands[0][commands[0].index("-n") + 1] == "realesr-general-x4v3"
    assert "inference_video.py" in commands[1][1]
    assert commands[1][commands[1].index("--fps") + 1] == "48"
    assert commands[2][:3] == ["ffmpeg", "-y", "-i"]
    assert "-map" in commands[2]
    assert "-shortest" in commands[2]


async def test_film_extracts_frames_then_normalizes_to_target_fps(tmp_path: Path) -> None:
    req = _request(tmp_path, target_fps=48)
    adapter = AiVideoEnhanceAdapter(
        backend="film",
        work_dir=tmp_path / "enhanced",
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        film_python="/film/bin/python",
        film_dir=tmp_path / "frame-interpolation",
        film_model_path=tmp_path / "FILM" / "film_net" / "Style" / "saved_model",
        film_times_to_interpolate=2,
    )
    commands: list[list[str]] = []

    async def fake_run(cmd: list[str], cwd: Path | None = None) -> None:
        commands.append(cmd)
        if cmd[:2] == ["ffmpeg", "-y"] and cmd[-1].endswith("%08d.png"):
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
        elif cmd[:3] == ["/film/bin/python", "-m", "eval.interpolator_cli"]:
            frames_dir = Path(cmd[cmd.index("--pattern") + 1])
            (frames_dir / "interpolated.mp4").write_bytes(b"film")
        else:
            Path(cmd[-1]).write_bytes(b"final")

    async def fake_probe(path: Path) -> dict[str, float]:
        return {"fps": 16.0, "duration": 2.5, "width": 480, "height": 864}

    adapter._run_command = fake_run  # type: ignore[method-assign]
    adapter._probe_video = fake_probe  # type: ignore[method-assign]

    await adapter.run(req)

    assert commands[0][0] == "ffmpeg"
    assert commands[1][commands[1].index("--fps") + 1] == "64"
    assert commands[2][0] == "ffmpeg"
    vf = commands[2][commands[2].index("-vf") + 1]
    assert vf.startswith("fps=48,")
