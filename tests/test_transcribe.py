from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from adapters.transcribe.whisper_adapter import WhisperAdapter
from core.models.capabilities import TranscribeRequest


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


# ------------------------------------------------------------------ run (async dispatch)

async def test_run_dispatches_to_transcribe() -> None:
    adapter = _adapter()
    segs = [_make_segment("Test.", 0.0, 1.0)]
    adapter._model = _make_model_mock(segs)

    result = await adapter.run(TranscribeRequest(audio_uri="audio.wav"))

    assert result.full_text == "Test."
    assert result.language == "en"


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

    mock_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")
    assert model is mock_instance
    assert adapter._model is mock_instance


def test_ensure_model_does_not_reload() -> None:
    adapter = _adapter()
    existing = MagicMock()
    adapter._model = existing

    result = adapter._ensure_model()

    assert result is existing
