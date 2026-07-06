from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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


class ShotOverlay(BaseModel):
    """LLM-authored chart/diagram panel composited over the upper third of the shot.

    Drawn deterministically by the render_overlays stage (matplotlib) — diffusion
    models cannot render legible charts. ``callout`` is big-text only (title +
    optional caption); the chart kinds need 2-6 labels with matching values.
    """
    kind: Literal["bar", "line", "pie", "callout"]
    title: str
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    caption: str = ""                    # small unit/context note under the chart
    duration_sec: float | None = None    # None → visible for the whole shot
    png_uri: str | None = None           # set by render_overlays; rides the plan artifact

    @model_validator(mode="after")
    def check_data_shape(self) -> "ShotOverlay":
        if self.kind in ("bar", "line", "pie"):
            if not (2 <= len(self.labels) <= 6) or len(self.labels) != len(self.values):
                raise ValueError("chart overlays need 2-6 labels with matching values")
            if self.kind == "pie" and any(v < 0 for v in self.values):
                raise ValueError("pie values must be non-negative")
        return self


class Shot(BaseModel):
    shot_id: str
    scene_ref: str
    characters_on_screen: list[str] = Field(default_factory=list)
    setting: str
    camera: str
    action: str
    dialogue_line_refs: list[str] = Field(default_factory=list)
    duration_sec: float
    overlay: ShotOverlay | None = None  # optional chart/diagram panel for this shot

    @field_validator("characters_on_screen")
    @classmethod
    def prefer_one_or_two_characters(cls, value: list[str]) -> list[str]:
        if len(value) > 2:
            raise ValueError("Phase 1 shots should keep to 1-2 characters on screen.")
        return value


class Storyboard(BaseModel):
    shots: list[Shot]

