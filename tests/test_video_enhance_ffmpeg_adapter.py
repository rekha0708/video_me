from pathlib import Path

from adapters.video_enhance.ffmpeg_adapter import FfmpegVideoEnhanceAdapter
from core.models.capabilities import VideoEnhanceRequest, VideoEnhanceResult


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


def _adapter(tmp_path: Path) -> FfmpegVideoEnhanceAdapter:
    return FfmpegVideoEnhanceAdapter(work_dir=tmp_path / "enhanced")


async def test_run_writes_enhanced_result(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    commands = []

    async def fake_run(cmd: list[str]) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"enhanced")

    adapter._run_ffmpeg = fake_run  # type: ignore[method-assign]

    result = await adapter.run(req)

    assert isinstance(result, VideoEnhanceResult)
    assert result.video_uri.endswith("enhanced.mp4")
    assert Path(result.video_uri).read_bytes() == b"enhanced"
    assert result.width == 1080
    assert result.height == 1920
    assert result.fps == 48.0
    assert result.adapter == "ffmpeg"
    assert commands


def test_filter_uses_minterpolate_and_lanczos(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    vf = adapter._filter(_request(tmp_path, target_fps=48))

    assert "minterpolate=fps=48" in vf
    assert "scale=1080:1920:flags=lanczos" in vf
    assert "setsar=1" in vf


def test_filter_can_use_simple_fps_mode(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    vf = adapter._filter(_request(tmp_path, interpolation="fps", target_fps=32))

    assert vf.startswith("fps=32,")
    assert "minterpolate" not in vf


async def test_run_can_drop_audio(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, preserve_audio=False)
    commands = []

    async def fake_run(cmd: list[str]) -> None:
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"enhanced")

    adapter._run_ffmpeg = fake_run  # type: ignore[method-assign]

    await adapter.run(req)

    assert "-an" in commands[0]
    assert "-c:a" not in commands[0]


async def test_run_raises_for_missing_input(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, video_uri=tmp_path / "missing.mp4")

    try:
        await adapter.run(req)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "Video to enhance not found" in str(exc)
