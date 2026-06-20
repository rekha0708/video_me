import pytest
from pydantic import ValidationError

from core.config import load_app_config
from core.models.content import LearningObjective, Line, Scene, Script, Shot
from core.models.guardrails import SourceRights
from core.models.profile import Cast


def test_config_loads_default_profiles() -> None:
    config = load_app_config()

    assert config.channel_profile.id == "education_kids"
    assert config.channel_profile.made_for_kids is True
    assert config.cast.is_original_synthetic is True
    assert len(config.cast.members) == 2


def test_cast_must_be_original_synthetic() -> None:
    with pytest.raises(ValidationError):
        Cast.model_validate(
            {
                "id": "bad",
                "species": "pig",
                "is_original_synthetic": False,
                "members": [],
            }
        )


def test_script_requires_rights_clearance() -> None:
    objective = LearningObjective(
        concept="counting to five",
        age_range="3-6",
        success_phrase="I can count to five.",
    )

    with pytest.raises(ValidationError):
        Script(
            learning_objective=objective,
            scenes=[Scene(setting="classroom", lines=[Line(speaker="c1", text="One!")])],
            caption_text="One!",
            source_rights=SourceRights(kind="transformed", rights_cleared=False, notes="missing approval"),
        )


def test_storyboard_limits_phase1_shots_to_two_characters() -> None:
    with pytest.raises(ValidationError):
        Shot(
            shot_id="s1",
            scene_ref="scene-1",
            characters_on_screen=["c1", "c2", "c3"],
            setting="playroom",
            camera="medium",
            action="count blocks",
            duration_sec=3.0,
        )

