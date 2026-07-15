import asyncio
import io
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.lip_sync.latentsync_adapter import LatentSyncAdapter
from core.models.capabilities import LipSyncRequest, VideoClip

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40


def _make_wav(duration_sec: float = 1.25, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * duration_sec))
    return buf.getvalue()


def _write_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(_FAKE_MP4)
    return path


def _write_wav(tmp_path: Path) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(_make_wav())
    return path


def _request(tmp_path: Path) -> LipSyncRequest:
    return LipSyncRequest(
        video_uri=str(_write_mp4(tmp_path)),
        audio_uri=str(_write_wav(tmp_path)),
        shot_id="s01",
    )


def _adapter(tmp_path: Path, **kwargs) -> LatentSyncAdapter:
    return LatentSyncAdapter(work_dir=tmp_path / "synced", **kwargs)


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


def test_latentsync_declares_voice_unload_requirement() -> None:
    assert LatentSyncAdapter.requires_voice_unloaded is True


async def test_run_posts_guidance_and_steps(tmp_path: Path) -> None:
    adapter = _adapter(
        tmp_path,
        inference_steps=24,
        guidance_scale=1.8,
        job_id="job-token",
    )
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(_request(tmp_path))

    assert isinstance(clip, VideoClip)
    assert clip.uri.endswith("synced.mp4")
    data = mock_client.post.call_args.kwargs["data"]
    assert data["shot_id"] == "s01"
    assert data["job_id"] == "job-token"
    assert data["inference_steps"] == "24"
    assert data["guidance_scale"] == "1.8"


async def test_run_raises_when_video_missing(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path)
    req = LipSyncRequest(video_uri=str(tmp_path / "missing.mp4"), audio_uri=str(audio), shot_id="s01")
    with pytest.raises(FileNotFoundError, match="generate_video"):
        await _adapter(tmp_path).run(req)


async def test_health_down_when_service_unreachable(tmp_path: Path) -> None:
    fake_httpx, mock_client = _mock_httpx()
    mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "down"


async def test_run_cancellation_requests_remote_job_cancel(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, job_id="job-cancel")
    started = asyncio.Event()
    never = asyncio.Event()

    async def hanging_call(*_args):
        started.set()
        await never.wait()

    with (
        patch.object(adapter, "_call_latentsync", side_effect=hanging_call),
        patch.object(
            adapter,
            "_cancel_remote_job",
            new_callable=AsyncMock,
            return_value=True,
        ) as cancel_remote,
    ):
        task = asyncio.create_task(adapter.run(_request(tmp_path)))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    cancel_remote.assert_awaited_once_with()


async def test_remote_cancel_posts_job_token(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, job_id="job-token")
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        assert await adapter._cancel_remote_job() is True

    call = mock_client.post.call_args
    assert call.args[0].endswith("/cancel")
    assert call.kwargs["data"] == {"job_id": "job-token"}
