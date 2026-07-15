import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from adapters.transcribe.whisper_adapter import WhisperAdapter
from core.models.capabilities import TranscribeRequest, TranscribeResult


def _adapter(**kwargs) -> WhisperAdapter:
    return WhisperAdapter(**kwargs)


def _make_word(word: str, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(word=word, start=start, end=end)


def _make_segment(text: str, start: float, end: float, words=None) -> SimpleNamespace:
    return SimpleNamespace(text=text, start=start, end=end, words=words or [])


def _make_info(language: str = "en") -> SimpleNamespace:
    return SimpleNamespace(language=language)


# ------------------------------------------------------------------ health

async def test_health_ok_when_package_importable() -> None:
    adapter = _adapter()
    fake_module = MagicMock()
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        health = await adapter.health()
    assert health.status == "ok"


async def test_health_down_when_package_missing() -> None:
    adapter = _adapter()
    with patch.dict("sys.modules", {"faster_whisper": None}):
        health = await adapter.health()
    assert health.status == "down"
    assert "faster-whisper" in (health.reason or "")


# ------------------------------------------------------------------ _transcribe (unit)

def _make_model_mock(segments, language="en") -> MagicMock:
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(segments), _make_info(language))
    return mock_model


def test_transcribe_builds_segments_and_full_text() -> None:
    adapter = _adapter()
    segs = [
        _make_segment(
            "One two three.",
            start=0.0,
            end=2.5,
            words=[
                _make_word("One", 0.0, 0.5),
                _make_word("two", 0.6, 1.0),
                _make_word("three.", 1.1, 2.5),
            ],
        ),
        _make_segment(
            "Four five.",
            start=2.6,
            end=4.0,
            words=[
                _make_word("Four", 2.6, 3.0),
                _make_word("five.", 3.1, 4.0),
            ],
        ),
    ]
    adapter._model = _make_model_mock(segs)

    result = adapter._transcribe("audio.wav")

    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0].text == "One two three."
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 2.5
    assert len(result.segments[0].words) == 3
    assert result.segments[0].words[0].word == "One"
    assert result.segments[0].words[0].start == 0.0
    assert result.full_text == "One two three. Four five."


def test_transcribe_passes_configured_vad_filter() -> None:
    adapter = _adapter(vad_filter=False)
    adapter._model = _make_model_mock([])

    adapter._transcribe("song.wav")

    kwargs = adapter._model.transcribe.call_args.kwargs
    assert kwargs["vad_filter"] is False


def test_transcribe_auto_detects_language_by_default() -> None:
    adapter = _adapter()
    adapter._model = _make_model_mock([])

    adapter._transcribe("song.wav")

    kwargs = adapter._model.transcribe.call_args.kwargs
    assert "language" not in kwargs


def test_transcribe_auto_language_value_auto_detects() -> None:
    adapter = _adapter(language="auto")
    adapter._model = _make_model_mock([])

    adapter._transcribe("song.wav")

    kwargs = adapter._model.transcribe.call_args.kwargs
    assert "language" not in kwargs


def test_transcribe_passes_configured_language() -> None:
    adapter = _adapter(language="en")
    adapter._model = _make_model_mock([])

    adapter._transcribe("song.wav")

    kwargs = adapter._model.transcribe.call_args.kwargs
    assert kwargs["language"] == "en"


def test_transcribe_filters_blank_words() -> None:
    adapter = _adapter()
    segs = [
        _make_segment(
            "Hello world.",
            start=0.0,
            end=1.5,
            words=[
                _make_word("Hello", 0.0, 0.5),
                _make_word("  ", 0.5, 0.6),  # blank — should be dropped
                _make_word("world.", 0.7, 1.5),
            ],
        ),
    ]
    adapter._model = _make_model_mock(segs)

    result = adapter._transcribe("audio.wav")

    assert len(result.segments[0].words) == 2
    assert result.segments[0].words[1].word == "world."


def test_transcribe_handles_no_words() -> None:
    """Segments with no word timestamps (some models omit them) don't crash."""
    adapter = _adapter()
    segs = [_make_segment("Hello.", start=0.0, end=1.0, words=None)]
    adapter._model = _make_model_mock(segs)

    result = adapter._transcribe("audio.wav")

    assert len(result.segments) == 1
    assert result.segments[0].words == []


def test_transcribe_empty_audio_returns_empty() -> None:
    adapter = _adapter()
    adapter._model = _make_model_mock([])

    result = adapter._transcribe("audio.wav")

    assert result.segments == []
    assert result.full_text == ""
    assert result.language == "en"


def test_transcribe_preserves_non_english_language() -> None:
    adapter = _adapter()
    segs = [_make_segment("Hola mundo.", 0.0, 1.0)]
    adapter._model = _make_model_mock(segs, language="es")

    result = adapter._transcribe("audio.wav")

    assert result.language == "es"


# ------------------------------------------------------------------ vocal isolation

def test_transcribe_skips_isolation_by_default() -> None:
    adapter = _adapter()
    adapter._model = _make_model_mock([])

    with patch.object(adapter, "_isolate_vocals") as mock_iso:
        adapter._transcribe("audio.wav")

    mock_iso.assert_not_called()
    assert adapter._model.transcribe.call_args.args[0] == "audio.wav"


def test_transcribe_uses_isolated_path_when_requested() -> None:
    adapter = _adapter()
    adapter._model = _make_model_mock([])

    with patch.object(adapter, "_isolate_vocals", return_value="/tmp/vocals.wav") as mock_iso:
        adapter._transcribe("song.wav", isolate_vocals=True)

    mock_iso.assert_called_once_with("song.wav")
    assert adapter._model.transcribe.call_args.args[0] == "/tmp/vocals.wav"


async def test_run_forwards_isolate_vocals_flag_to_transcribe() -> None:
    adapter = _adapter()
    adapter._model = _make_model_mock([])

    with patch.object(adapter, "_transcribe", wraps=adapter._transcribe) as spy:
        await adapter.run(TranscribeRequest(audio_uri="audio.wav", isolate_vocals=True))

    spy.assert_called_once_with("audio.wav", True)


def test_isolate_vocals_returns_original_when_demucs_venv_missing(tmp_path) -> None:
    adapter = _adapter()
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    with patch("adapters.transcribe.whisper_adapter._DEMUCS_PYTHON") as mock_python:
        mock_python.exists.return_value = False
        result = adapter._isolate_vocals(str(audio_path))

    assert result == str(audio_path)


def test_isolate_vocals_runs_demucs_and_returns_vocals_path(tmp_path) -> None:
    adapter = _adapter()
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"fake")

    def fake_run(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1])
        vocals = out_dir / "htdemucs" / "song" / "vocals.wav"
        vocals.parent.mkdir(parents=True, exist_ok=True)
        vocals.write_bytes(b"vox")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("adapters.transcribe.whisper_adapter._DEMUCS_PYTHON") as mock_python, \
         patch("adapters.transcribe.whisper_adapter.subprocess.run", side_effect=fake_run) as mock_run:
        mock_python.exists.return_value = True
        mock_python.__str__.return_value = "/workspace/.venv_demucs/bin/python"
        result = adapter._isolate_vocals(str(audio_path))

    assert result.endswith("vocals.wav")
    assert Path(result).exists()
    mock_run.assert_called_once()


def test_isolate_vocals_falls_back_on_nonzero_exit(tmp_path) -> None:
    adapter = _adapter()
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"fake")

    with patch("adapters.transcribe.whisper_adapter._DEMUCS_PYTHON") as mock_python, \
         patch(
             "adapters.transcribe.whisper_adapter.subprocess.run",
             return_value=SimpleNamespace(returncode=1, stdout="", stderr="boom"),
         ):
        mock_python.exists.return_value = True
        result = adapter._isolate_vocals(str(audio_path))

    assert result == str(audio_path)


def test_isolate_vocals_falls_back_on_timeout(tmp_path) -> None:
    adapter = _adapter()
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"fake")

    with patch("adapters.transcribe.whisper_adapter._DEMUCS_PYTHON") as mock_python, \
         patch(
             "adapters.transcribe.whisper_adapter.subprocess.run",
             side_effect=subprocess.TimeoutExpired(cmd="demucs", timeout=600),
         ):
        mock_python.exists.return_value = True
        result = adapter._isolate_vocals(str(audio_path))

    assert result == str(audio_path)


# ------------------------------------------------------------------ run (async dispatch)

async def test_run_dispatches_to_transcribe() -> None:
    adapter = _adapter()
    segs = [_make_segment("Test.", 0.0, 1.0)]
    adapter._model = _make_model_mock(segs)

    result = await adapter.run(TranscribeRequest(audio_uri="audio.wav"))

    assert result.full_text == "Test."
    assert result.language == "en"


async def test_run_cancellation_waits_for_executor_work() -> None:
    adapter = _adapter()
    loop = asyncio.get_running_loop()
    executor_work: asyncio.Future[TranscribeResult] = loop.create_future()
    result = TranscribeResult(segments=[], language="en", full_text="")

    with patch.object(loop, "run_in_executor", return_value=executor_work):
        task = asyncio.create_task(
            adapter.run(TranscribeRequest(audio_uri="audio.wav"))
        )
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        # The caller's cancellation is deliberately held until the native
        # executor work finishes, so another GPU job cannot overlap it.
        assert not task.done()

        executor_work.set_result(result)
        with pytest.raises(asyncio.CancelledError):
            await task


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero() -> None:
    adapter = _adapter()
    cost = await adapter.estimate_cost(TranscribeRequest(audio_uri="audio.wav"))
    assert cost.amount == 0.0


# ------------------------------------------------------------------ lazy model loading

def test_ensure_model_lazy_loads() -> None:
    adapter = _adapter(model_size="tiny")
    assert adapter._model is None

    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
        model = adapter._ensure_model()

    mock_cls.assert_called_once_with(
        "tiny",
        device="cpu",
        compute_type="int8",
        download_root=None,
        local_files_only=True,
        revision=None,
    )
    assert model is mock_instance
    assert adapter._model is mock_instance


def test_ensure_model_uses_configured_download_root_and_revision() -> None:
    adapter = _adapter(
        model_size="large-v3",
        download_root="/workspace/.cache/huggingface/hub",
        revision="abc123",
    )

    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
        model = adapter._ensure_model()

    mock_cls.assert_called_once_with(
        "large-v3",
        device="cpu",
        compute_type="int8",
        download_root="/workspace/.cache/huggingface/hub",
        local_files_only=True,
        revision="abc123",
    )
    assert model is mock_instance


def test_ensure_model_can_allow_runtime_downloads_when_configured() -> None:
    adapter = _adapter(local_files_only=False)

    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
        adapter._ensure_model()

    kwargs = mock_cls.call_args.kwargs
    assert kwargs["local_files_only"] is False


def test_ensure_model_does_not_reload() -> None:
    adapter = _adapter()
    existing = MagicMock()
    adapter._model = existing

    result = adapter._ensure_model()

    assert result is existing


async def test_unload_drops_loaded_model() -> None:
    adapter = _adapter()
    adapter._model = MagicMock()

    result = await adapter.unload()

    assert result is True
    assert adapter._model is None


async def test_unload_is_noop_when_model_is_not_loaded() -> None:
    adapter = _adapter()

    result = await adapter.unload()

    assert result is True
    assert adapter._model is None


# ------------------------------------------------------------------ stage_hook (dashboard events)

def test_ensure_model_notifies_stage_hook_with_duration() -> None:
    stage_hook = MagicMock()
    adapter = _adapter(model_size="large-v3", device="cuda", compute_type="float16", stage_hook=stage_hook)

    mock_cls = MagicMock()
    with patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=mock_cls)}):
        adapter._ensure_model()

    calls = stage_hook.call_args_list
    assert calls[0].args == ("whisper_model_load", "stage_started")
    assert "large-v3" in calls[0].kwargs["message"]
    assert calls[1].args == ("whisper_model_load", "stage_completed")
    assert "large-v3" in calls[1].kwargs["message"]
    assert "took" in calls[1].kwargs["message"]


def test_ensure_model_does_not_renotify_when_cached() -> None:
    stage_hook = MagicMock()
    adapter = _adapter(stage_hook=stage_hook)
    adapter._model = MagicMock()

    adapter._ensure_model()

    stage_hook.assert_not_called()


async def test_unload_notifies_stage_hook_with_duration() -> None:
    stage_hook = MagicMock()
    adapter = _adapter(model_size="large-v3", stage_hook=stage_hook)
    adapter._model = MagicMock()

    await adapter.unload()

    calls = stage_hook.call_args_list
    assert calls[0].args == ("whisper_model_unload", "stage_started")
    assert calls[1].args == ("whisper_model_unload", "stage_completed")
    assert "took" in calls[1].kwargs["message"]


async def test_unload_does_not_notify_when_nothing_loaded() -> None:
    stage_hook = MagicMock()
    adapter = _adapter(stage_hook=stage_hook)

    await adapter.unload()

    stage_hook.assert_not_called()


def test_isolate_vocals_notifies_stage_hook_on_success(tmp_path) -> None:
    stage_hook = MagicMock()
    adapter = _adapter(stage_hook=stage_hook)
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"fake")

    def fake_run(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1])
        vocals = out_dir / "htdemucs" / "song" / "vocals.wav"
        vocals.parent.mkdir(parents=True, exist_ok=True)
        vocals.write_bytes(b"vox")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("adapters.transcribe.whisper_adapter._DEMUCS_PYTHON") as mock_python, \
         patch("adapters.transcribe.whisper_adapter.subprocess.run", side_effect=fake_run):
        mock_python.exists.return_value = True
        mock_python.__str__.return_value = "/workspace/.venv_demucs/bin/python"
        adapter._isolate_vocals(str(audio_path))

    calls = stage_hook.call_args_list
    assert calls[0].args == ("vocal_isolation", "stage_started")
    assert calls[1].args == ("vocal_isolation", "stage_completed")
    assert "isolated" in calls[1].kwargs["message"].lower()


def test_isolate_vocals_notifies_stage_hook_on_fallback(tmp_path) -> None:
    stage_hook = MagicMock()
    adapter = _adapter(stage_hook=stage_hook)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    with patch("adapters.transcribe.whisper_adapter._DEMUCS_PYTHON") as mock_python:
        mock_python.exists.return_value = False
        adapter._isolate_vocals(str(audio_path))

    stage_hook.assert_called_once()
    assert stage_hook.call_args.args == ("vocal_isolation", "stage_completed")
    assert "skipped" in stage_hook.call_args.kwargs["message"].lower()
