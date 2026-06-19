import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.analyze_content.llm_adapter import LlmAnalyzeAdapter, _strip_markdown_fence
from core.models.capabilities import AnalyzeRequest, TranscribeResult, TranscriptSegment
from core.models.profile import ChannelProfile


# ------------------------------------------------------------------ fixtures

def _profile() -> ChannelProfile:
    return ChannelProfile(
        id="education_kids",
        genre_content="educational_kids",
        tone="warm, playful",
        format="animated_character",
        made_for_kids=True,
        target_audience={"age_range": "3-6", "reading": "pre_reader"},
        pedagogy={"one_concept_per_video": True, "repetition": True},
    )


def _transcript(text: str = "One two three four five.", end: float = 30.0) -> TranscribeResult:
    return TranscribeResult(
        segments=[TranscriptSegment(text=text, start=0.0, end=end)],
        language="en",
        full_text=text,
    )


def _request(**kwargs) -> AnalyzeRequest:
    return AnalyzeRequest(
        transcript=kwargs.get("transcript", _transcript()),
        channel_profile=kwargs.get("channel_profile", _profile()),
    )


def _adapter(**kwargs) -> LlmAnalyzeAdapter:
    return LlmAnalyzeAdapter(**kwargs)


def _good_llm_payload() -> dict:
    return {
        "content_genre": "educational_kids",
        "topic": "counting to five",
        "tone": "warm and playful",
        "hook": "Can you count to five?",
        "structure": ["intro", "counting demonstration", "practice", "outro"],
        "pacing": "slow",
        "call_to_action": "Count along with us!",
        "music_genre": None,
        "visual_style": None,
        "learning_objective": {
            "concept": "counting to five",
            "age_range": "3-6",
            "success_phrase": "I can count to five!",
            "key_vocabulary": ["one", "two", "three", "four", "five"],
            "reinforcement_count": 3,
        },
    }


def _mock_openai(json_payload: dict | None = None, *, api_error: Exception | None = None):
    """Return a fake openai module wired to return json_payload or raise api_error."""
    raw = json.dumps(json_payload or _good_llm_payload())
    mock_choice = MagicMock()
    mock_choice.message.content = raw
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_completions = MagicMock()
    if api_error:
        mock_completions.create = AsyncMock(side_effect=api_error)
    else:
        mock_completions.create = AsyncMock(return_value=mock_response)

    mock_models = MagicMock()
    mock_models.list = AsyncMock(return_value=MagicMock())

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions = mock_completions
    mock_client_instance.models = mock_models

    mock_AsyncOpenAI = MagicMock(return_value=mock_client_instance)
    fake_openai = MagicMock()
    fake_openai.AsyncOpenAI = mock_AsyncOpenAI

    return fake_openai, mock_client_instance


# ------------------------------------------------------------------ strip_markdown_fence

def test_strip_fence_removes_json_code_block() -> None:
    text = "```json\n{\"a\": 1}\n```"
    assert _strip_markdown_fence(text) == '{"a": 1}'


def test_strip_fence_removes_plain_code_block() -> None:
    text = "```\n{\"a\": 1}\n```"
    assert _strip_markdown_fence(text) == '{"a": 1}'


def test_strip_fence_passthrough_on_plain_json() -> None:
    text = '{"a": 1}'
    assert _strip_markdown_fence(text) == '{"a": 1}'


# ------------------------------------------------------------------ health

async def test_health_ok_when_api_reachable() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai()
    with patch.dict(sys.modules, {"openai": fake_openai}):
        health = await adapter.health()
    assert health.status == "ok"


async def test_health_down_when_package_missing() -> None:
    adapter = _adapter()
    with patch.dict(sys.modules, {"openai": None}):
        health = await adapter.health()
    assert health.status == "down"
    assert "openai" in (health.reason or "").lower()


async def test_health_down_when_api_unreachable() -> None:
    adapter = _adapter()
    fake_openai, mock_client = _mock_openai()
    mock_client.models.list = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.dict(sys.modules, {"openai": fake_openai}):
        health = await adapter.health()
    assert health.status == "down"
    assert "unreachable" in (health.reason or "").lower()


# ------------------------------------------------------------------ _build_messages

def test_build_messages_includes_transcript_text() -> None:
    adapter = _adapter()
    req = _request(transcript=_transcript("Hello world."))
    messages = adapter._build_messages(req)
    user_msg = messages[1]["content"]
    assert "Hello world." in user_msg


def test_build_messages_truncates_long_transcript() -> None:
    adapter = _adapter()
    long_text = "x" * 7_000
    req = _request(transcript=_transcript(long_text))
    messages = adapter._build_messages(req)
    user_msg = messages[1]["content"]
    assert "[truncated]" in user_msg
    assert len(user_msg) < 10_000


def test_build_messages_includes_channel_genre() -> None:
    adapter = _adapter()
    req = _request()
    messages = adapter._build_messages(req)
    assert "educational_kids" in messages[1]["content"]


# ------------------------------------------------------------------ _parse_response

def test_parse_response_returns_content_metadata() -> None:
    adapter = _adapter()
    req = _request()
    raw = json.dumps(_good_llm_payload())
    metadata = adapter._parse_response(raw, req)

    assert metadata.topic == "counting to five"
    assert metadata.hook == "Can you count to five?"
    assert metadata.learning_objective is not None
    assert metadata.learning_objective.concept == "counting to five"
    assert "one" in metadata.learning_objective.key_vocabulary


def test_parse_response_overrides_language_from_transcript() -> None:
    adapter = _adapter()
    transcript = _transcript()
    transcript.language = "es"
    req = AnalyzeRequest(transcript=transcript, channel_profile=_profile())
    payload = {**_good_llm_payload(), "language": "en"}  # LLM says en
    metadata = adapter._parse_response(json.dumps(payload), req)
    assert metadata.language == "es"  # transcript wins


def test_parse_response_overrides_length_from_transcript() -> None:
    adapter = _adapter()
    req = _request(transcript=_transcript(end=42.7))
    payload = {**_good_llm_payload(), "length_sec": 999}  # LLM says 999
    metadata = adapter._parse_response(json.dumps(payload), req)
    assert metadata.length_sec == 42  # int(42.7)


def test_parse_response_handles_markdown_fence() -> None:
    adapter = _adapter()
    req = _request()
    raw = f"```json\n{json.dumps(_good_llm_payload())}\n```"
    metadata = adapter._parse_response(raw, req)
    assert metadata.topic == "counting to five"


def test_parse_response_raises_on_invalid_json() -> None:
    adapter = _adapter()
    req = _request()
    with pytest.raises(RuntimeError, match="invalid JSON"):
        adapter._parse_response("this is not json", req)


# ------------------------------------------------------------------ run (full async)

async def test_run_returns_content_metadata() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai()
    with patch.dict(sys.modules, {"openai": fake_openai}):
        result = await adapter.run(_request())
    assert result.topic == "counting to five"
    assert result.language == "en"
    assert result.learning_objective is not None


async def test_run_passes_correct_params_to_api() -> None:
    adapter = _adapter(model="llama3.1:8b", temperature=0.2)
    fake_openai, mock_client = _mock_openai()
    with patch.dict(sys.modules, {"openai": fake_openai}):
        await adapter.run(_request())
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama3.1:8b"
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["response_format"] == {"type": "json_object"}


async def test_run_propagates_api_error() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai(api_error=RuntimeError("LLM timeout"))
    with patch.dict(sys.modules, {"openai": fake_openai}):
        with pytest.raises(RuntimeError, match="LLM timeout"):
            await adapter.run(_request())


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero() -> None:
    adapter = _adapter()
    cost = await adapter.estimate_cost(_request())
    assert cost.amount == 0.0
