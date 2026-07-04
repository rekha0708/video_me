"""Tests for adapters/story_ingest (structured parser, heuristic, LLM segmenter)."""
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from adapters.story_ingest.llm_adapter import LlmStorySegmentAdapter
from adapters.story_ingest.parser import heuristic_segments, parse_structured_story
from core.models.capabilities import TranscribeResult

_STRUCTURED = """
0-4.5: Max finds a red leaf in the garden.
4.5-9: Zoe wonders why leaves change color.

9-14: They ask the wise old oak tree.
"""


# ------------------------------------------------------------------ parse_structured_story


def test_structured_story_parses_segments() -> None:
    result = parse_structured_story(_STRUCTURED)
    assert isinstance(result, TranscribeResult)
    assert len(result.segments) == 3
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 4.5
    assert result.segments[1].text == "Zoe wonders why leaves change color."
    assert "wise old oak" in result.full_text
    assert result.language == "en"


def test_structured_story_language_passthrough() -> None:
    result = parse_structured_story("0-5: नमस्ते!", language="hi")
    assert result is not None
    assert result.language == "hi"


def test_free_text_returns_none() -> None:
    assert parse_structured_story("Once upon a time there was a leaf.") is None


def test_mixed_format_returns_none() -> None:
    text = "0-5: Max finds a leaf.\nAnd then something happens without a timestamp."
    assert parse_structured_story(text) is None


def test_non_monotonic_times_return_none() -> None:
    assert parse_structured_story("0-5: First.\n3-8: Overlapping.") is None


def test_zero_length_span_returns_none() -> None:
    assert parse_structured_story("5-5: Instant.") is None


def test_empty_text_returns_none() -> None:
    assert parse_structured_story("   \n  ") is None


# ------------------------------------------------------------------ heuristic_segments


def test_heuristic_splits_sentences_with_durations() -> None:
    result = heuristic_segments("Max finds a leaf. Zoe smiles at him and waves hello!")
    assert len(result.segments) == 2
    for seg in result.segments:
        assert 5.0 <= (seg.end - seg.start) <= 8.0
    # Monotonic, gap-free timeline
    assert result.segments[1].start == result.segments[0].end


def test_heuristic_caps_long_sentences_at_8s() -> None:
    long_sentence = "word " * 60  # 60 words / 2 wps = 30s uncapped
    result = heuristic_segments(long_sentence.strip() + ".")
    assert result.segments[0].end - result.segments[0].start == 8.0


# ------------------------------------------------------------------ LlmStorySegmentAdapter


def _mock_httpx(content: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=resp)
    fake = MagicMock()
    fake.AsyncClient = MagicMock(return_value=client)
    return fake, client


async def test_llm_segmentation_happy_path() -> None:
    fake_httpx, client = _mock_httpx(
        '{"segments": [{"text": "Max finds a leaf.", "start": 0, "end": 5},'
        ' {"text": "Zoe waves.", "start": 5, "end": 10}]}'
    )
    adapter = LlmStorySegmentAdapter("http://llm.test/v1", "test-model")
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.segment("Max finds a leaf. Zoe waves.")
    assert len(result.segments) == 2
    assert result.segments[1].start == 5.0
    assert "/chat/completions" in client.post.call_args.args[0]


async def test_llm_invalid_json_falls_back_to_heuristic() -> None:
    fake_httpx, _ = _mock_httpx("I cannot produce JSON, sorry!")
    adapter = LlmStorySegmentAdapter("http://llm.test/v1", "test-model")
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.segment("Max finds a leaf. Zoe waves.")
    # Heuristic fallback still yields a valid timed transcript
    assert len(result.segments) == 2
    assert result.segments[0].end >= 5.0


async def test_llm_bad_segment_shape_falls_back() -> None:
    fake_httpx, _ = _mock_httpx('{"segments": [{"text": "", "start": 0, "end": 5}]}')
    adapter = LlmStorySegmentAdapter("http://llm.test/v1", "test-model")
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.segment("Max finds a leaf.")
    assert result.segments  # heuristic output, never empty


async def test_llm_http_error_falls_back() -> None:
    fake_httpx, client = _mock_httpx("{}")
    client.post = AsyncMock(side_effect=RuntimeError("connection refused"))
    adapter = LlmStorySegmentAdapter("http://llm.test/v1", "test-model")
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.segment("Max finds a leaf.")
    assert result.segments


async def test_llm_repairable_json_is_repaired() -> None:
    # Trailing comma + missing closing brace — json_repair territory
    fake_httpx, _ = _mock_httpx(
        '{"segments": [{"text": "Max finds a leaf.", "start": 0, "end": 5},]'
    )
    adapter = LlmStorySegmentAdapter("http://llm.test/v1", "test-model")
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await adapter.segment("Max finds a leaf.")
    assert result.segments[0].text == "Max finds a leaf."
