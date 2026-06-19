import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.adapt_script.llm_adapter import (
    LlmAdaptScriptAdapter,
    _compute_caption_text,
    _format_cast_block,
)
from core.models.capabilities import AdaptScriptRequest
from core.models.content import ContentMetadata, LearningObjective
from core.models.profile import Cast, CastMember, ChannelProfile


# ------------------------------------------------------------------ fixtures

def _objective() -> LearningObjective:
    return LearningObjective(
        concept="counting to five",
        age_range="3-6",
        success_phrase="I can count to five!",
        key_vocabulary=["one", "two", "three", "four", "five"],
        reinforcement_count=3,
    )


_SENTINEL = object()


def _metadata(objective: LearningObjective | None = _SENTINEL) -> ContentMetadata:  # type: ignore[assignment]
    return ContentMetadata(
        content_genre="educational_kids",
        topic="counting to five",
        tone="warm and playful",
        hook="Can you count to five?",
        pacing="slow",
        length_sec=42,
        learning_objective=_objective() if objective is _SENTINEL else objective,
    )


def _cast() -> Cast:
    return Cast(
        id="pig_kids_placeholder",
        species="pig",
        is_original_synthetic=True,
        members=[
            CastMember(
                id="c1", name="Pippa", gender="girl",
                visual_descriptor="round pig in teal overalls",
                lora_ref="loras/c1", voice_profile_ref="voices/c1",
                personality="curious, asks the questions",
                signature_expressions=["wide-eyed wonder", "big grin"],
            ),
            CastMember(
                id="c2", name="Milo", gender="boy",
                visual_descriptor="small pig in green hoodie",
                lora_ref="loras/c2", voice_profile_ref="voices/c2",
                personality="playful and silly",
                signature_expressions=["giggly smile", "surprised blink"],
            ),
            CastMember(
                id="c3", name="Nia", gender="girl",
                visual_descriptor="pig in purple jumper",
                lora_ref="loras/c3", voice_profile_ref="voices/c3",
                personality="knows the answer and explains gently",
                signature_expressions=["confident smile", "thinking face"],
            ),
            CastMember(
                id="c4", name="Luma", gender="girl",
                visual_descriptor="pig in yellow rain boots",
                lora_ref="loras/c4", voice_profile_ref="voices/c4",
                personality="shy and kind",
                signature_expressions=["soft smile", "happy clap"],
            ),
        ],
    )


def _profile() -> ChannelProfile:
    return ChannelProfile(
        id="education_kids",
        genre_content="educational_kids",
        tone="warm, simple, playful",
        format="animated_character",
        made_for_kids=True,
        target_length_sec=45,
        pedagogy={"one_concept_per_video": True, "repetition": True},
    )


def _request(**kwargs) -> AdaptScriptRequest:
    return AdaptScriptRequest(
        metadata=kwargs.get("metadata", _metadata()),
        cast=kwargs.get("cast", _cast()),
        channel_profile=kwargs.get("channel_profile", _profile()),
    )


def _adapter(**kwargs) -> LlmAdaptScriptAdapter:
    return LlmAdaptScriptAdapter(**kwargs)


def _llm_scenes() -> list[dict]:
    return [
        {
            "setting": "sunny meadow",
            "characters_present": ["c1", "c2"],
            "lines": [
                {"speaker": "c1", "text": "How many apples are there?",
                 "expression": "wide-eyed wonder", "action": None},
                {"speaker": "c2", "text": "Let us count them!",
                 "expression": "giggly smile", "action": None},
            ],
        },
        {
            "setting": "sunny meadow",
            "characters_present": ["c1", "c2", "c3"],
            "lines": [
                {"speaker": "c3", "text": "One, two, three, four, five!",
                 "expression": "confident smile", "action": None},
                {"speaker": "c1", "text": "Five apples!",
                 "expression": "big grin", "action": None},
            ],
        },
        {
            "setting": "sunny meadow",
            "characters_present": ["c1", "c2", "c3", "c4"],
            "lines": [
                {"speaker": "c2", "text": "Count with me! One, two, three!",
                 "expression": "giggly smile", "action": None},
                {"speaker": "c4", "text": "Four and five! You did it!",
                 "expression": "happy clap", "action": None},
            ],
        },
        {
            "setting": "sunny meadow",
            "characters_present": ["c1", "c2", "c3", "c4"],
            "lines": [
                {"speaker": "c4", "text": "I can count to five!",
                 "expression": "soft smile", "action": None},
                {"speaker": "c1", "text": "Me too!",
                 "expression": "big grin", "action": None},
            ],
        },
    ]


def _good_llm_payload() -> dict:
    return {"scenes": _llm_scenes()}


def _mock_openai(json_payload: dict | None = None, *, api_error: Exception | None = None):
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

    mock_client = MagicMock()
    mock_client.chat.completions = mock_completions
    mock_client.models = mock_models

    fake_openai = MagicMock()
    fake_openai.AsyncOpenAI = MagicMock(return_value=mock_client)

    return fake_openai, mock_client


# ------------------------------------------------------------------ helpers

def test_compute_caption_text_joins_all_lines() -> None:
    scenes = [
        {"lines": [{"text": "Hello."}, {"text": "World."}]},
        {"lines": [{"text": "Goodbye."}]},
    ]
    assert _compute_caption_text(scenes) == "Hello. World. Goodbye."


def test_compute_caption_text_skips_blank_lines() -> None:
    scenes = [{"lines": [{"text": "  "}, {"text": "Hi."}]}]
    assert _compute_caption_text(scenes) == "Hi."


def test_format_cast_block_includes_all_member_ids() -> None:
    block = _format_cast_block(_cast())
    for mid in ("c1", "c2", "c3", "c4"):
        assert f'"{mid}"' in block


def test_format_cast_block_includes_names_and_personalities() -> None:
    block = _format_cast_block(_cast())
    assert "Pippa" in block
    assert "curious" in block
    assert "wide-eyed wonder" in block


# ------------------------------------------------------------------ health

async def test_health_ok_when_api_reachable() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": fake_openai}
    ):
        health = await adapter.health()
    assert health.status == "ok"


async def test_health_down_when_package_missing() -> None:
    adapter = _adapter()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": None}
    ):
        health = await adapter.health()
    assert health.status == "down"


# ------------------------------------------------------------------ _build_messages

def test_build_messages_includes_all_cast_ids() -> None:
    adapter = _adapter()
    messages = adapter._build_messages(_request())
    user_msg = messages[1]["content"]
    for mid in ('"c1"', '"c2"', '"c3"', '"c4"'):
        assert mid in user_msg


def test_build_messages_includes_concept_and_vocabulary() -> None:
    adapter = _adapter()
    messages = adapter._build_messages(_request())
    user_msg = messages[1]["content"]
    assert "counting to five" in user_msg
    assert "one" in user_msg
    assert "five" in user_msg


def test_build_messages_includes_success_phrase() -> None:
    adapter = _adapter()
    messages = adapter._build_messages(_request())
    user_msg = messages[1]["content"]
    assert "I can count to five!" in user_msg


# ------------------------------------------------------------------ _parse_response

def test_parse_response_returns_script() -> None:
    adapter = _adapter()
    req = _request()
    script = adapter._parse_response(json.dumps(_good_llm_payload()), req)

    assert script.mode == "transformed"
    assert len(script.scenes) == 4
    assert script.scenes[0].setting == "sunny meadow"
    assert script.scenes[0].lines[0].speaker == "c1"


def test_parse_response_always_sets_transformed_mode() -> None:
    adapter = _adapter()
    payload = {**_good_llm_payload(), "mode": "verbatim"}  # LLM tries verbatim
    script = adapter._parse_response(json.dumps(payload), _request())
    assert script.mode == "transformed"


def test_parse_response_always_sets_rights_cleared() -> None:
    adapter = _adapter()
    # LLM tries to return uncleared rights — adapter must override
    payload = {
        **_good_llm_payload(),
        "source_rights": {"kind": "verbatim", "rights_cleared": False, "notes": "bad"},
    }
    script = adapter._parse_response(json.dumps(payload), _request())
    assert script.source_rights.rights_cleared is True
    assert script.source_rights.kind == "transformed"


def test_parse_response_injects_learning_objective_from_metadata() -> None:
    adapter = _adapter()
    req = _request()
    script = adapter._parse_response(json.dumps(_good_llm_payload()), req)
    assert script.learning_objective.concept == "counting to five"
    assert "one" in script.learning_objective.key_vocabulary


def test_parse_response_computes_caption_text_when_absent() -> None:
    adapter = _adapter()
    script = adapter._parse_response(json.dumps(_good_llm_payload()), _request())
    assert "apples" in script.caption_text
    assert len(script.caption_text) > 10


def test_parse_response_uses_llm_caption_text_when_present() -> None:
    adapter = _adapter()
    payload = {**_good_llm_payload(), "caption_text": "Custom captions here."}
    script = adapter._parse_response(json.dumps(payload), _request())
    assert script.caption_text == "Custom captions here."


def test_parse_response_raises_on_invalid_json() -> None:
    adapter = _adapter()
    with pytest.raises(RuntimeError, match="invalid JSON"):
        adapter._parse_response("not json", _request())


# ------------------------------------------------------------------ run guardrail

async def test_run_raises_when_learning_objective_missing() -> None:
    adapter = _adapter()
    req = _request(metadata=_metadata(objective=None))
    with pytest.raises(ValueError, match="learning_objective"):
        await adapter.run(req)


# ------------------------------------------------------------------ run end-to-end

async def test_run_returns_script() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": fake_openai}
    ):
        script = await adapter.run(_request())

    assert script.mode == "transformed"
    assert script.source_rights.rights_cleared is True
    assert script.learning_objective.concept == "counting to five"
    assert len(script.scenes) == 4


async def test_run_passes_model_and_temperature_to_api() -> None:
    adapter = _adapter(model="llama3.1:8b", temperature=0.5)
    fake_openai, mock_client = _mock_openai()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": fake_openai}
    ):
        await adapter.run(_request())

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama3.1:8b"
    assert call_kwargs["temperature"] == 0.5


async def test_run_propagates_api_error() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai(api_error=RuntimeError("connection refused"))
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": fake_openai}
    ):
        with pytest.raises(RuntimeError, match="connection refused"):
            await adapter.run(_request())


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero() -> None:
    adapter = _adapter()
    cost = await adapter.estimate_cost(_request())
    assert cost.amount == 0.0
