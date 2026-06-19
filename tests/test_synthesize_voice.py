import io
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.synthesize_voice.tts_adapter import (
    TtsAdapter,
    _EXPRESSION_EXAGGERATION,
    _FALLBACK_WPS,
    _VOICE_EXTENSIONS,
)
from core.models.capabilities import AudioTrack, VoiceRequest


# ------------------------------------------------------------------ helpers

def _make_wav(duration_sec: float = 1.0, sample_rate: int = 22050) -> bytes:
    """Create a minimal real WAV file in-memory."""
    num_frames = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def _make_ref_wav(voice_dir: Path, member_id: str = "pig_kids_placeholder/c1") -> Path:
    """Write a reference WAV to voice_dir and return its path."""
    ref_path = voice_dir / member_id
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path = ref_path.with_suffix(".wav")
    ref_path.write_bytes(_make_wav(0.5))
    return ref_path


def _request(**kwargs) -> VoiceRequest:
    return VoiceRequest(
        text=kwargs.get("text", "How many apples do you see?"),
        voice_profile_ref=kwargs.get("voice_profile_ref", "voices/pig_kids_placeholder/c1"),
        speaker_id=kwargs.get("speaker_id", "c1"),
        expression=kwargs.get("expression", None),
    )


def _adapter(tmp_path: Path, **kwargs) -> TtsAdapter:
    return TtsAdapter(
        work_dir=tmp_path / "audio",
        voice_dir=kwargs.get("voice_dir", tmp_path / "voices"),
        **{k: v for k, v in kwargs.items() if k != "voice_dir"},
    )


def _mock_httpx(
    wav_bytes: bytes | None = None,
    *,
    get_error: Exception | None = None,
    post_error: Exception | None = None,
):
    mock_get_resp = MagicMock()
    mock_get_resp.raise_for_status = MagicMock()
    mock_get_resp.json.return_value = {"status": "ok"}

    mock_post_resp = MagicMock()
    mock_post_resp.raise_for_status = MagicMock()
    mock_post_resp.content = wav_bytes or _make_wav(1.0)

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


# ------------------------------------------------------------------ voice_name

def test_voice_name_strips_voices_prefix(tmp_path: Path) -> None:
    assert _adapter(tmp_path).voice_name("voices/pig_kids_placeholder/c1") == "pig_kids_placeholder/c1"


def test_voice_name_no_prefix(tmp_path: Path) -> None:
    assert _adapter(tmp_path).voice_name("pig_kids_placeholder/c1") == "pig_kids_placeholder/c1"


def test_voice_name_single_segment(tmp_path: Path) -> None:
    assert _adapter(tmp_path).voice_name("voices/mycast") == "mycast"


# ------------------------------------------------------------------ _check_voice

def test_check_voice_finds_wav(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    ref = _make_ref_wav(voice_dir)
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    path = adapter._check_voice("voices/pig_kids_placeholder/c1")
    assert path == ref


def test_check_voice_finds_mp3(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    mp3_path = voice_dir / "pig_kids_placeholder" / "c1.mp3"
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.write_bytes(b"fake mp3")
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    path = adapter._check_voice("voices/pig_kids_placeholder/c1")
    assert path.suffix == ".mp3"


def test_check_voice_raises_with_track_b_message(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)  # empty voice_dir
    with pytest.raises(RuntimeError) as exc_info:
        adapter._check_voice("voices/pig_kids_placeholder/c1")
    msg = str(exc_info.value)
    assert "Track B" in msg
    assert "pig_kids_placeholder/c1.wav" in msg


# ------------------------------------------------------------------ _exaggeration_for

def test_exaggeration_returns_base_when_no_expression(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, exaggeration=0.5)
    assert adapter._exaggeration_for(None) == 0.5


def test_exaggeration_returns_override_for_known_expression(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, exaggeration=0.5)
    assert adapter._exaggeration_for("excited") == _EXPRESSION_EXAGGERATION["excited"]


def test_exaggeration_is_case_insensitive(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, exaggeration=0.5)
    assert adapter._exaggeration_for("Excited") == _EXPRESSION_EXAGGERATION["excited"]


def test_exaggeration_returns_base_for_unknown_expression(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, exaggeration=0.5)
    assert adapter._exaggeration_for("big grin") == 0.5


# ------------------------------------------------------------------ _wav_duration

def test_wav_duration_real_file(tmp_path: Path) -> None:
    wav = _make_wav(2.0, sample_rate=22050)
    path = tmp_path / "test.wav"
    path.write_bytes(wav)
    duration = _adapter(tmp_path)._wav_duration(path)
    assert abs(duration - 2.0) < 0.01


def test_wav_duration_falls_back_for_non_wav(tmp_path: Path) -> None:
    path = tmp_path / "test.wav"
    path.write_bytes(b"not a wav file")
    # 3 words → 3/3 = 1.0s
    duration = _adapter(tmp_path)._wav_duration(path, text="one two three")
    assert duration == pytest.approx(1.0, abs=0.01)


def test_wav_duration_fallback_minimum_is_one_second(tmp_path: Path) -> None:
    path = tmp_path / "test.wav"
    path.write_bytes(b"bad")
    duration = _adapter(tmp_path)._wav_duration(path, text="")
    assert duration >= 1.0


# ------------------------------------------------------------------ _save_audio

def test_save_audio_writes_file(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    wav = _make_wav(1.0)
    path, _ = adapter._save_audio(wav, out_dir, "hello world")
    assert path.exists()
    assert path.suffix == ".wav"


def test_save_audio_same_text_same_filename(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    wav = _make_wav(1.0)
    path1, _ = adapter._save_audio(wav, out_dir, "same text")
    path2, _ = adapter._save_audio(wav, out_dir, "same text")
    assert path1 == path2


def test_save_audio_different_text_different_filename(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    wav = _make_wav(1.0)
    path1, _ = adapter._save_audio(wav, out_dir, "text one")
    path2, _ = adapter._save_audio(wav, out_dir, "text two")
    assert path1 != path2


def test_save_audio_returns_correct_duration(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    wav = _make_wav(3.0)
    _, duration = adapter._save_audio(wav, out_dir, "something")
    assert abs(duration - 3.0) < 0.05


# ------------------------------------------------------------------ health

async def test_health_ok_when_service_reachable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await adapter.health()
    assert health.status == "ok"


async def test_health_down_when_package_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch.dict(sys.modules, {"httpx": None}):
        health = await adapter.health()
    assert health.status == "down"
    assert "httpx" in (health.reason or "")


async def test_health_down_when_service_unreachable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    fake_httpx, _ = _mock_httpx(get_error=ConnectionError("refused"))
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        health = await adapter.health()
    assert health.status == "down"
    assert "unreachable" in (health.reason or "").lower()


# ------------------------------------------------------------------ run

async def test_run_returns_audio_track(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    _make_ref_wav(voice_dir)
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    fake_httpx, _ = _mock_httpx(_make_wav(1.5))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.run(_request())

    assert isinstance(result, AudioTrack)
    assert result.speaker_id == "c1"
    assert result.duration_sec > 0
    assert Path(result.uri).exists()


async def test_run_creates_speaker_subdir(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    _make_ref_wav(voice_dir)
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    fake_httpx, _ = _mock_httpx(_make_wav(1.0))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(_request())

    assert (tmp_path / "audio" / "c1").is_dir()


async def test_run_raises_when_voice_profile_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)  # empty voice_dir
    with pytest.raises(RuntimeError, match="Track B"):
        await adapter.run(_request())


async def test_run_posts_to_synthesize_endpoint(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    _make_ref_wav(voice_dir)
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    fake_httpx, mock_client = _mock_httpx(_make_wav(1.0))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(_request())

    mock_client.post.assert_called_once()
    assert "/synthesize" in mock_client.post.call_args.args[0]


async def test_run_passes_expression_exaggeration(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    _make_ref_wav(voice_dir)
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    fake_httpx, mock_client = _mock_httpx(_make_wav(1.0))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await adapter.run(_request(expression="excited"))

    data = mock_client.post.call_args.kwargs.get("data") or {}
    assert float(data.get("exaggeration", 0.5)) == _EXPRESSION_EXAGGERATION["excited"]


async def test_run_propagates_api_error(tmp_path: Path) -> None:
    voice_dir = tmp_path / "voices"
    _make_ref_wav(voice_dir)
    adapter = _adapter(tmp_path, voice_dir=voice_dir)
    fake_httpx, _ = _mock_httpx(post_error=RuntimeError("TTS error"))

    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        with pytest.raises(RuntimeError, match="TTS error"):
            await adapter.run(_request())


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    cost = await _adapter(tmp_path).estimate_cost(_request())
    assert cost.amount == 0.0
