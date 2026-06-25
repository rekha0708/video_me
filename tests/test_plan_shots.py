import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.plan_shots.llm_adapter import (
    LlmPlanShotsAdapter,
    estimate_duration,
    make_line_ref,
    make_scene_ref,
    trim_characters,
)
from core.models.capabilities import PlanShotsRequest
from core.models.content import (
    LearningObjective,
    Line,
    Scene,
    Script,
    Shot,
    Storyboard,
)
from core.models.guardrails import SourceRights
from core.models.profile import Cast, CastMember


# ------------------------------------------------------------------ fixtures

def _cast() -> Cast:
    return Cast(
        id="pig_kids",
        species="pig",
        is_original_synthetic=True,
        members=[
            CastMember(
                id="c1", name="Pippa", gender="girl",
                visual_descriptor="round pig", lora_ref="l/c1",
                voice_profile_ref="v/c1", personality="curious",
                signature_expressions=["wide-eyed wonder"],
            ),
            CastMember(
                id="c2", name="Milo", gender="boy",
                visual_descriptor="small pig", lora_ref="l/c2",
                voice_profile_ref="v/c2", personality="playful",
                signature_expressions=["giggly smile"],
            ),
            CastMember(
                id="c3", name="Nia", gender="girl",
                visual_descriptor="purple pig", lora_ref="l/c3",
                voice_profile_ref="v/c3", personality="explains",
                signature_expressions=["confident smile"],
            ),
            CastMember(
                id="c4", name="Luma", gender="girl",
                visual_descriptor="yellow pig", lora_ref="l/c4",
                voice_profile_ref="v/c4", personality="shy",
                signature_expressions=["soft smile"],
            ),
        ],
    )


def _script() -> Script:
    obj = LearningObjective(
        concept="counting to five",
        age_range="3-6",
        success_phrase="I can count to five!",
    )
    rights = SourceRights(kind="transformed", rights_cleared=True, notes="ok")
    return Script(
        learning_objective=obj,
        scenes=[
            Scene(
                setting="sunny meadow",
                characters_present=["c1", "c2"],
                lines=[
                    Line(speaker="c1", text="How many apples?",
                         expression="wide-eyed wonder"),
                    Line(speaker="c2", text="Let us count them!",
                         expression="giggly smile"),
                ],
            ),
            Scene(
                setting="apple tree",
                characters_present=["c1", "c2", "c3"],
                lines=[
                    Line(speaker="c3", text="One two three four five!",
                         expression="confident smile"),
                    Line(speaker="c1", text="Five apples!",
                         expression="big grin"),
                    Line(speaker="c4", text="I can count to five!",
                         expression="soft smile"),
                ],
            ),
        ],
        caption_text="How many apples? Let us count them! One two three four five!",
        source_rights=rights,
    )


def _request() -> PlanShotsRequest:
    return PlanShotsRequest(script=_script(), cast=_cast())


def _adapter(**kwargs) -> LlmPlanShotsAdapter:
    return LlmPlanShotsAdapter(**kwargs)


def _llm_shots_payload(flat_lines: list) -> dict:
    """Build a minimal valid LLM response for the given flat_lines list."""
    shots = []
    cameras = ["close-up", "medium", "medium", "close-up", "medium"]
    for i, (ref, speaker, _) in enumerate(flat_lines):
        shots.append({
            "ref": ref,
            "camera": cameras[i % len(cameras)],
            "action": "speaks warmly",
            "characters_on_screen": [speaker],
        })
    return {"shots": shots}


def _mock_openai(json_payload: dict | None = None, *, api_error: Exception | None = None):
    raw = json.dumps(json_payload or {})
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


# ------------------------------------------------------------------ pure helpers

def test_make_scene_ref() -> None:
    assert make_scene_ref(0) == "scene-1"
    assert make_scene_ref(2) == "scene-3"


def test_make_line_ref() -> None:
    assert make_line_ref(0, 0) == "scene-1-line-0"
    assert make_line_ref(1, 2) == "scene-2-line-2"


def test_estimate_duration_short_line() -> None:
    # 2 words → 1.0s → clamped to 5.0 (new floor)
    assert estimate_duration("Hi there") == 5.0


def test_estimate_duration_normal_line() -> None:
    # 5 words → 2.5s → clamped to 5.0 (new floor)
    assert estimate_duration("One two three four five") == 5.0


def test_estimate_duration_long_line() -> None:
    # 20 words → 10.0s → clamped to 8.0 (new ceiling)
    long = " ".join(["word"] * 20)
    assert estimate_duration(long) == 8.0


def test_trim_characters_keeps_speaker_first() -> None:
    assert trim_characters(["c3", "c1"], "c1") == ["c1", "c3"]


def test_trim_characters_limits_to_two() -> None:
    result = trim_characters(["c1", "c2", "c3", "c4"], "c1")
    assert result == ["c1", "c2"]
    assert len(result) == 2


def test_trim_characters_adds_speaker_when_missing() -> None:
    result = trim_characters([], "c2")
    assert result == ["c2"]


def test_trim_characters_deduplicates_speaker() -> None:
    result = trim_characters(["c1", "c1", "c2"], "c1")
    assert result == ["c1", "c2"]


# ------------------------------------------------------------------ _build_messages

def test_build_messages_contains_all_line_refs() -> None:
    adapter = _adapter()
    messages, flat_lines = adapter._build_messages(_request())
    user_msg = messages[1]["content"]
    for ref, _, _ in flat_lines:
        assert ref in user_msg


def test_build_messages_flat_lines_count_equals_total_lines() -> None:
    adapter = _adapter()
    _, flat_lines = adapter._build_messages(_request())
    total = sum(len(s.lines) for s in _script().scenes)
    assert len(flat_lines) == total


def test_build_messages_contains_speaker_names() -> None:
    adapter = _adapter()
    messages, _ = adapter._build_messages(_request())
    user_msg = messages[1]["content"]
    assert "Pippa" in user_msg
    assert "Milo" in user_msg


def test_build_messages_contains_camera_options() -> None:
    adapter = _adapter()
    messages, _ = adapter._build_messages(_request())
    assert "close-up" in messages[1]["content"]
    assert "reaction" in messages[1]["content"]


def test_build_messages_contains_settings() -> None:
    adapter = _adapter()
    messages, _ = adapter._build_messages(_request())
    assert "sunny meadow" in messages[1]["content"]
    assert "apple tree" in messages[1]["content"]


# ------------------------------------------------------------------ _parse_response

def test_parse_response_one_shot_per_line() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    payload = _llm_shots_payload(flat_lines)
    storyboard = adapter._parse_response(json.dumps(payload), req, flat_lines)
    assert len(storyboard.shots) == len(flat_lines)


def test_parse_response_sequential_shot_ids() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    storyboard = adapter._parse_response(
        json.dumps(_llm_shots_payload(flat_lines)), req, flat_lines
    )
    assert storyboard.shots[0].shot_id == "s01"
    assert storyboard.shots[1].shot_id == "s02"
    assert storyboard.shots[4].shot_id == "s05"


def test_parse_response_scene_refs_match_script() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    storyboard = adapter._parse_response(
        json.dumps(_llm_shots_payload(flat_lines)), req, flat_lines
    )
    # First 2 shots are from scene 1, next 3 from scene 2
    assert storyboard.shots[0].scene_ref == "scene-1"
    assert storyboard.shots[1].scene_ref == "scene-1"
    assert storyboard.shots[2].scene_ref == "scene-2"


def test_parse_response_settings_match_script() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    storyboard = adapter._parse_response(
        json.dumps(_llm_shots_payload(flat_lines)), req, flat_lines
    )
    assert storyboard.shots[0].setting == "sunny meadow"
    assert storyboard.shots[2].setting == "apple tree"


def test_parse_response_dialogue_line_refs_set() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    storyboard = adapter._parse_response(
        json.dumps(_llm_shots_payload(flat_lines)), req, flat_lines
    )
    assert storyboard.shots[0].dialogue_line_refs == ["scene-1-line-0"]
    assert storyboard.shots[2].dialogue_line_refs == ["scene-2-line-0"]


def test_parse_response_duration_derived_from_word_count() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    # LLM payload has arbitrary durations — adapter ignores them
    storyboard = adapter._parse_response(
        json.dumps(_llm_shots_payload(flat_lines)), req, flat_lines
    )
    for shot in storyboard.shots:
        assert _MIN_SHOT_SEC <= shot.duration_sec <= _MAX_SHOT_SEC


def test_parse_response_enforces_two_character_max() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    # LLM ignores the rule and puts 4 characters in a shot
    payload = _llm_shots_payload(flat_lines)
    payload["shots"][0]["characters_on_screen"] = ["c1", "c2", "c3", "c4"]
    storyboard = adapter._parse_response(json.dumps(payload), req, flat_lines)
    assert len(storyboard.shots[0].characters_on_screen) <= 2


def test_parse_response_speaker_always_first() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    payload = _llm_shots_payload(flat_lines)
    # LLM lists speaker second
    payload["shots"][0]["characters_on_screen"] = ["c2", "c1"]  # c1 is speaker
    storyboard = adapter._parse_response(json.dumps(payload), req, flat_lines)
    assert storyboard.shots[0].characters_on_screen[0] == "c1"


def test_parse_response_uses_default_when_llm_returns_fewer_shots() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    # LLM only returns 2 shots but there are 5 lines
    payload = {"shots": _llm_shots_payload(flat_lines)["shots"][:2]}
    storyboard = adapter._parse_response(json.dumps(payload), req, flat_lines)
    assert len(storyboard.shots) == len(flat_lines)
    # Shots 3-5 get defaults
    assert storyboard.shots[2].camera == "medium"
    assert storyboard.shots[2].action == "speaks"


def test_parse_response_raises_on_invalid_json() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        adapter._parse_response("not json", req, flat_lines)


# ------------------------------------------------------------------ health

async def test_health_ok() -> None:
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


# ------------------------------------------------------------------ run end-to-end

async def test_run_returns_storyboard() -> None:
    adapter = _adapter()
    req = _request()
    _, flat_lines = adapter._build_messages(req)
    fake_openai, _ = _mock_openai(_llm_shots_payload(flat_lines))

    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": fake_openai}
    ):
        storyboard = await adapter.run(req)

    assert isinstance(storyboard, Storyboard)
    assert len(storyboard.shots) == 5
    assert all(isinstance(s, Shot) for s in storyboard.shots)


async def test_run_propagates_api_error() -> None:
    adapter = _adapter()
    fake_openai, _ = _mock_openai(api_error=RuntimeError("timeout"))
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
        sys.modules, {"openai": fake_openai}
    ):
        with pytest.raises(RuntimeError, match="timeout"):
            await adapter.run(_request())


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero() -> None:
    assert (await _adapter().estimate_cost(_request())).amount == 0.0


# ------------------------------------------------------------------ import for clamping check

from adapters.plan_shots.llm_adapter import _MIN_SHOT_SEC, _MAX_SHOT_SEC
