import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.critique.vlm_adapter import (
    VlmCritiqueAdapter,
    _script_line_count,
    _script_summary,
    _strip_markdown_fence,
)
from core.models.capabilities import CritiqueRequest
from core.models.content import LearningObjective, Line, Scene, Script
from core.models.guardrails import SourceRights


# ------------------------------------------------------------------ fixtures

def _script(**kwargs) -> Script:
    return Script(
        mode="transformed",
        learning_objective=LearningObjective(
            concept="counting",
            age_range="3-6",
            success_phrase="Children learn to count.",
        ),
        scenes=kwargs.get(
            "scenes",
            [
                Scene(
                    setting="classroom",
                    lines=[
                        Line(speaker="max", text="Let's count together."),
                        Line(speaker="zoe", text="One, two, three!"),
                    ],
                )
            ],
        ),
        caption_text="Let's count together. One, two, three!",
        source_rights=SourceRights(kind="transformed", rights_cleared=True, notes=""),
    )


def _request(video_uri: str, **kwargs) -> CritiqueRequest:
    return CritiqueRequest(
        video_uri=video_uri,
        script=kwargs.get("script", _script()),
        channel_profile_id=kwargs.get("channel_profile_id", "education_kids"),
    )


def _adapter(**kwargs) -> VlmCritiqueAdapter:
    return VlmCritiqueAdapter(**kwargs)


def _good_payload(verdict: str = "pass") -> dict:
    return {
        "scores": {
            "age_appropriateness": 0.95,
            "learning_clarity": 0.9,
            "caption_readability": 0.85,
            "character_consistency": 0.8,
            "engagement": 0.82,
        },
        "verdict": verdict,
        "reasons": ["Safe and clear for review."],
        "suggested_param_overrides": {},
    }


def _mock_openai(json_payload: dict | None = None, *, api_error: Exception | None = None):
    raw = json.dumps(json_payload or _good_payload())
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

    mock_client = MagicMock()
    mock_client.chat.completions = mock_completions
    mock_client.models = mock_models

    fake_openai = MagicMock()
    fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

    return fake_openai, mock_client


# ------------------------------------------------------------------ helpers

def test_strip_markdown_fence_removes_json_block() -> None:
    assert _strip_markdown_fence("```json\n{\"verdict\": \"pass\"}\n```") == '{"verdict": "pass"}'


def test_script_line_count_counts_all_lines() -> None:
    assert _script_line_count(_script()) == 2


def test_script_summary_includes_caption_and_dialogue() -> None:
    summary = _script_summary(_script())
    assert "Children learn to count." in summary
    assert "max: Let's count together." in summary


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


async def test_health_down_when_api_unreachable() -> None:
    adapter = _adapter()
    fake_openai, mock_client = _mock_openai()
    mock_client.models.list = AsyncMock(side_effect=ConnectionError("refused"))
    with patch.dict(sys.modules, {"openai": fake_openai}):
        health = await adapter.health()
    assert health.status == "down"
    assert "unreachable" in (health.reason or "").lower()


# ------------------------------------------------------------------ preflight

async def test_run_raises_when_video_missing(tmp_path) -> None:
    adapter = _adapter()
    missing = tmp_path / "missing.mp4"
    with pytest.raises(FileNotFoundError, match="assemble_video"):
        await adapter.run(_request(str(missing)))


async def test_run_rejects_empty_script_without_api_call(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake mp4")
    adapter = _adapter()
    fake_openai, mock_client = _mock_openai()

    with patch.dict(sys.modules, {"openai": fake_openai}):
        result = await adapter.run(_request(str(video), script=_script(scenes=[])))

    assert result.verdict == "reject"
    assert "no scenes" in result.reasons[0].lower()
    mock_client.chat.completions.create.assert_not_called()


# ------------------------------------------------------------------ _build_messages

def test_build_messages_includes_video_uri(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    req = _request(str(video))
    messages = _adapter()._build_messages(req)
    assert str(video) in messages[1]["content"]


def test_build_messages_includes_channel_profile() -> None:
    messages = _adapter()._build_messages(_request("s3://bucket/video.mp4"))
    assert "education_kids" in messages[1]["content"]


# ------------------------------------------------------------------ _parse_response

def test_parse_response_returns_critique_result() -> None:
    result = _adapter()._parse_response(json.dumps(_good_payload("regenerate")))
    assert result.verdict == "regenerate"
    assert result.scores["learning_clarity"] == 0.9


def test_parse_response_handles_markdown_fence() -> None:
    raw = f"```json\n{json.dumps(_good_payload())}\n```"
    assert _adapter()._parse_response(raw).verdict == "pass"


def test_parse_response_raises_on_invalid_json() -> None:
    with pytest.raises(RuntimeError, match="invalid JSON"):
        _adapter()._parse_response("not json")


def test_parse_response_raises_on_invalid_verdict() -> None:
    payload = {**_good_payload(), "verdict": "maybe"}
    with pytest.raises(RuntimeError, match="invalid result shape"):
        _adapter()._parse_response(json.dumps(payload))


# ------------------------------------------------------------------ run

async def test_run_returns_model_verdict(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake mp4")
    adapter = _adapter(model="llava:latest")
    fake_openai, mock_client = _mock_openai(_good_payload("pass"))

    with patch.dict(sys.modules, {"openai": fake_openai}):
        result = await adapter.run(_request(str(video)))

    assert result.verdict == "pass"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llava:latest"
    assert call_kwargs["response_format"] == {"type": "json_object"}


async def test_run_propagates_api_error(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake mp4")
    adapter = _adapter()
    fake_openai, _ = _mock_openai(api_error=RuntimeError("VLM timeout"))

    with patch.dict(sys.modules, {"openai": fake_openai}):
        with pytest.raises(RuntimeError, match="VLM timeout"):
            await adapter.run(_request(str(video)))


async def test_estimate_cost_is_zero(tmp_path) -> None:
    cost = await _adapter().estimate_cost(_request(str(tmp_path / "video.mp4")))
    assert cost.amount == 0.0
