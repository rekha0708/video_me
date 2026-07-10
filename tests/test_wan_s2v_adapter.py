import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.generate_video.wan_s2v_adapter import WanS2VAdapter
from core.models.capabilities import VideoClip, VideoRequest

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40


def _write_png(tmp_path: Path) -> Path:
    path = tmp_path / "render.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    return path


def _write_wav(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 40)
    return path


def _request(tmp_path: Path, **kwargs) -> VideoRequest:
    return VideoRequest(
        image_uri=str(kwargs.get("image_uri") or _write_png(tmp_path)),
        audio_uri=str(kwargs.get("audio_uri") or _write_wav(tmp_path)),
        action=kwargs.get("action", "sings into a microphone"),
        duration_sec=kwargs.get("duration_sec", 4.0),
        shot_id=kwargs.get("shot_id", "s01"),
        setting=kwargs.get("setting", "warm concert stage"),
        style_suffix=kwargs.get("style_suffix", ""),
    )


def _adapter(tmp_path: Path) -> WanS2VAdapter:
    return WanS2VAdapter(work_dir=tmp_path / "video" / "wan_s2v")


def _mock_httpx(mp4_bytes: bytes = _FAKE_MP4, *, post_error: Exception | None = None):
    mock_get_resp = MagicMock()
    mock_get_resp.raise_for_status = MagicMock()

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.raise_for_status = MagicMock()
    mock_post_resp.content = mp4_bytes

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(return_value=mock_get_resp)
    mock_client.post = AsyncMock(
        side_effect=post_error if post_error else None,
        return_value=mock_post_resp,
    )

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return fake_httpx, mock_client


def test_wan_s2v_declares_native_lipsync_and_voice_unload() -> None:
    assert WanS2VAdapter.native_lipsync is True
    assert WanS2VAdapter.requires_voice_unloaded is True


async def test_run_requires_audio_before_video(tmp_path: Path) -> None:
    req = _request(tmp_path, audio_uri=tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError, match="requires synthesize_voice"):
        await _adapter(tmp_path).run(req)


async def test_run_posts_image_and_audio_to_generate(tmp_path: Path) -> None:
    req = _request(tmp_path, style_suffix="bright animated kids music video")
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await _adapter(tmp_path).run(req)

    assert isinstance(clip, VideoClip)
    assert clip.uri.endswith("clip.mp4")
    call = mock_client.post.call_args
    assert "/generate" in call.args[0]
    assert call.kwargs["data"]["shot_id"] == "s01"
    assert "bright animated kids music video" in call.kwargs["data"]["prompt"]
    assert "audio-synchronized singing" in call.kwargs["data"]["prompt"]
    assert "image" in call.kwargs["files"]
    assert "audio" in call.kwargs["files"]


async def test_run_sends_fps_and_infer_frames(tmp_path: Path) -> None:
    adapter = WanS2VAdapter(work_dir=tmp_path / "video" / "wan_s2v", fps=16)
    req = _request(tmp_path, duration_sec=5.0)
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(req)

    data = mock_client.post.call_args.kwargs["data"]
    assert data["fps"] == "16"
    assert data["infer_frames"] == "81"


async def test_health_down_when_service_unreachable(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "down"
