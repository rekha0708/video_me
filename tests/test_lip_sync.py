import io
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.lip_sync.lip_sync_adapter import LipSyncAdapter
from core.models.capabilities import LipSyncRequest, VideoClip


# ------------------------------------------------------------------ helpers

_FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40


def _make_wav(duration_sec: float = 1.5, sample_rate: int = 22050) -> bytes:
    num_frames = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def _write_wav(tmp_path: Path, duration_sec: float = 1.5) -> Path:
    path = tmp_path / "audio.wav"
    path.write_bytes(_make_wav(duration_sec))
    return path


def _write_mp4(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(_FAKE_MP4)
    return path


def _request(tmp_path: Path, **kwargs) -> LipSyncRequest:
    video = kwargs.get("video_uri") or str(_write_mp4(tmp_path))
    audio = kwargs.get("audio_uri") or str(_write_wav(tmp_path))
    return LipSyncRequest(
        video_uri=video,
        audio_uri=audio,
        shot_id=kwargs.get("shot_id", "s01"),
    )


def _adapter(tmp_path: Path, **kwargs) -> LipSyncAdapter:
    return LipSyncAdapter(work_dir=tmp_path / "synced", **kwargs)


def _mock_httpx(
    mp4_bytes: bytes = _FAKE_MP4,
    *,
    get_error: Exception | None = None,
    post_error: Exception | None = None,
):
    mock_get_resp = MagicMock()
    mock_get_resp.raise_for_status = MagicMock()

    mock_post_resp = MagicMock()
    mock_post_resp.raise_for_status = MagicMock()
    mock_post_resp.content = mp4_bytes

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = AsyncMock(
        side_effect=get_error if get_error else None,
        return_value=mock_get_resp,
    )
    mock_client.post = AsyncMock(
        side_effect=post_error if post_error else None,
        return_value=mock_post_resp,
    )

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return fake_httpx, mock_client


# ------------------------------------------------------------------ _check_inputs

def test_check_inputs_passes_when_both_exist(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video = _write_mp4(tmp_path)
    audio = _write_wav(tmp_path)
    adapter._check_inputs(video, audio, "s01")  # should not raise


def test_check_inputs_raises_when_video_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    audio = _write_wav(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        adapter._check_inputs(tmp_path / "missing.mp4", audio, "s01")
    assert "generate_video" in str(exc_info.value)
    assert "s01" in str(exc_info.value)


def test_check_inputs_raises_when_audio_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video = _write_mp4(tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        adapter._check_inputs(video, tmp_path / "missing.wav", "s01")
    assert "synthesize_voice" in str(exc_info.value)
    assert "s01" in str(exc_info.value)


# ------------------------------------------------------------------ _audio_duration

def test_audio_duration_from_real_wav(tmp_path: Path) -> None:
    audio = _write_wav(tmp_path, duration_sec=2.0)
    duration = _adapter(tmp_path)._audio_duration(audio)
    assert abs(duration - 2.0) < 0.01


def test_audio_duration_returns_zero_for_non_wav(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    duration = _adapter(tmp_path)._audio_duration(bad)
    assert duration == 0.0


def test_audio_duration_returns_zero_for_missing_file(tmp_path: Path) -> None:
    duration = _adapter(tmp_path)._audio_duration(tmp_path / "nonexistent.wav")
    assert duration == 0.0


# ------------------------------------------------------------------ _save_clip

def test_save_clip_writes_synced_mp4(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = adapter._save_clip(_FAKE_MP4, out_dir)
    assert path.name == "synced.mp4"
    assert path.exists()
    assert path.read_bytes() == _FAKE_MP4


# ------------------------------------------------------------------ health

async def test_health_ok_when_service_reachable(tmp_path: Path) -> None:
    fake_httpx, _ = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "ok"


async def test_health_down_when_package_missing(tmp_path: Path) -> None:
    with patch.dict(sys.modules, {"httpx": None}):
        health = await _adapter(tmp_path).health()
    assert health.status == "down"
    assert "httpx" in (health.reason or "")


async def test_health_down_when_service_unreachable(tmp_path: Path) -> None:
    fake_httpx, _ = _mock_httpx(get_error=ConnectionError("refused"))
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await _adapter(tmp_path).health()
    assert health.status == "down"
    assert "unreachable" in (health.reason or "").lower()


# ------------------------------------------------------------------ run

async def test_run_returns_video_clip(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, shot_id="s02")
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(req)

    assert isinstance(clip, VideoClip)
    assert clip.shot_id == "s02"
    assert clip.duration_sec > 0
    assert Path(clip.uri).exists()


async def test_run_creates_shot_subdir(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(_request(tmp_path, shot_id="s04"))

    assert (tmp_path / "synced" / "s04").is_dir()


async def test_run_output_is_synced_mp4(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(_request(tmp_path))

    assert clip.uri.endswith("synced.mp4")


async def test_run_duration_comes_from_audio(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    audio = tmp_path / "precise.wav"
    audio.write_bytes(_make_wav(duration_sec=2.5))
    video = _write_mp4(tmp_path)
    req = LipSyncRequest(video_uri=str(video), audio_uri=str(audio), shot_id="s01")
    fake_httpx, _ = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        clip = await adapter.run(req)

    assert abs(clip.duration_sec - 2.5) < 0.01


async def test_run_raises_when_video_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    audio = _write_wav(tmp_path)
    req = LipSyncRequest(
        video_uri=str(tmp_path / "missing.mp4"),
        audio_uri=str(audio),
        shot_id="s01",
    )
    with pytest.raises(FileNotFoundError, match="generate_video"):
        await adapter.run(req)


async def test_run_raises_when_audio_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    video = _write_mp4(tmp_path)
    req = LipSyncRequest(
        video_uri=str(video),
        audio_uri=str(tmp_path / "missing.wav"),
        shot_id="s01",
    )
    with pytest.raises(FileNotFoundError, match="synthesize_voice"):
        await adapter.run(req)


async def test_run_posts_to_lipsync_endpoint(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(req)

    mock_client.post.assert_called_once()
    assert "/lipsync" in mock_client.post.call_args.args[0]


async def test_run_sends_shot_id_in_payload(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, shot_id="s09")
    fake_httpx, mock_client = _mock_httpx()

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(req)

    data = mock_client.post.call_args.kwargs.get("data") or {}
    assert data.get("shot_id") == "s09"


async def test_run_propagates_api_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    fake_httpx, _ = _mock_httpx(post_error=RuntimeError("lipsync timeout"))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="lipsync timeout"):
            await adapter.run(req)


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    cost = await _adapter(tmp_path).estimate_cost(_request(tmp_path))
    assert cost.amount == 0.0
