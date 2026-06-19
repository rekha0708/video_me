from typing import Literal

from pydantic import BaseModel, Field, field_validator

from core.models.guardrails import SourceRights


class LearningObjective(BaseModel):
    concept: str
    age_range: str
    success_phrase: str
    key_vocabulary: list[str] = Field(default_factory=list)
    reinforcement_count: int = 2


class ContentMetadata(BaseModel):
    content_genre: str
    music_genre: str | None = None
    topic: str
    tone: str
    hook: str
    structure: list[str] = Field(default_factory=list)
    pacing: str
    visual_style: str | None = None
    length_sec: int
    call_to_action: str | None = None
    language: str = "en"
    learning_objective: LearningObjective | None = None


class Line(BaseModel):
    speaker: str
    text: str
    expression: str | None = None
    action: str | None = None
    start: float | None = None
    end: float | None = None


class Scene(BaseModel):
    setting: str
    characters_present: list[str] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)


class Script(BaseModel):
    mode: Literal["verbatim", "adapted", "transformed"] = "transformed"
    learning_objective: LearningObjective
    scenes: list[Scene]
    caption_text: str
    source_rights: SourceRights

    @field_validator("source_rights")
    @classmethod
    def require_rights_cleared(cls, value: SourceRights) -> SourceRights:
        if not value.rights_cleared:
            raise ValueError("Script requires cleared source rights.")
        return value


class Shot(BaseModel):
    shot_id: str
    scene_ref: str
    characters_on_screen: list[str] = Field(default_factory=list)
    setting: str
    camera: str
    action: str
    dialogue_line_refs: list[str] = Field(default_factory=list)
    duration_sec: float

    @field_validator("characters_on_screen")
    @classmethod
    def prefer_one_or_two_characters(cls, value: list[str]) -> list[str]:
        if len(value) > 2:
            raise ValueError("Phase 1 shots should keep to 1-2 characters on screen.")
        return value


class Storyboard(BaseModel):
    shots: list[Shot]

