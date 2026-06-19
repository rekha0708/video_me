from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from adapters.fetch_media.ytdlp_adapter import YtDlpAdapter
from core.models.capabilities import FetchMediaRequest


def _adapter(tmp_path: Path) -> YtDlpAdapter:
    return YtDlpAdapter(work_dir=tmp_path / "media")


# ------------------------------------------------------------------ health

async def test_health_ok(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch("adapters.fetch_media.ytdlp_adapter.shutil.which", return_value="/usr/bin/tool"):
        health = await adapter.health()
    assert health.status == "ok"


async def test_health_down_when_tool_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch(
        "adapters.fetch_media.ytdlp_adapter.shutil.which",
        side_effect=lambda t: None if t == "yt-dlp" else "/usr/bin/tool",
    ):
        health = await adapter.health()
    assert health.status == "down"
    assert "yt-dlp" in (health.reason or "")


async def test_health_down_when_all_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch("adapters.fetch_media.ytdlp_adapter.shutil.which", return_value=None):
        health = await adapter.health()
    assert health.status == "down"


# ------------------------------------------------------------------ run (mocked helpers)

async def test_run_returns_result(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video_file = tmp_path / "media" / "video.mp4"
    audio_file = tmp_path / "media" / "audio.wav"

    adapter._get_info = AsyncMock(return_value={"duration": 42.5})
    adapter._download = AsyncMock(return_value=video_file)
    adapter._extract_audio = AsyncMock(return_value=audio_file)

    result = await adapter.run(FetchMediaRequest(source_url="https://example.com/video"))

    assert result.video_uri == str(video_file)
    assert result.audio_uri == str(audio_file)
    assert result.duration_sec == 42.5
    assert result.source_url == "https://example.com/video"


async def test_run_handles_missing_duration(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video_file = tmp_path / "media" / "video.mp4"
    audio_file = tmp_path / "media" / "audio.wav"

    adapter._get_info = AsyncMock(return_value={})  # no "duration" key
    adapter._download = AsyncMock(return_value=video_file)
    adapter._extract_audio = AsyncMock(return_value=audio_file)

    result = await adapter.run(FetchMediaRequest(source_url="https://example.com/video"))

    assert result.duration_sec == 0.0


async def test_run_propagates_download_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._get_info = AsyncMock(return_value={"duration": 10.0})
    adapter._download = AsyncMock(side_effect=RuntimeError("yt-dlp download failed"))

    with pytest.raises(RuntimeError, match="yt-dlp download failed"):
        await adapter.run(FetchMediaRequest(source_url="https://example.com/video"))


async def test_run_propagates_audio_extraction_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video_file = tmp_path / "media" / "video.mp4"

    adapter._get_info = AsyncMock(return_value={"duration": 10.0})
    adapter._download = AsyncMock(return_value=video_file)
    adapter._extract_audio = AsyncMock(side_effect=RuntimeError("ffmpeg audio extraction failed"))

    with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
        await adapter.run(FetchMediaRequest(source_url="https://example.com/video"))


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    cost = await adapter.estimate_cost(FetchMediaRequest(source_url="https://example.com/video"))
    assert cost.amount == 0.0
