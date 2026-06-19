from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChannelProfile(BaseModel):
    id: str
    genre_content: str
    target_audience: dict[str, Any] = Field(default_factory=dict)
    tone: str
    format: Literal["talking_head", "animated_character", "dance_lifestyle", "other"]
    aspect_ratio: str = "9:16"
    target_length_sec: int = 45
    language: str = "en"
    made_for_kids: bool = False
    disclosure_label_required: bool = True
    pedagogy: dict[str, Any] = Field(default_factory=dict)


class CastMember(BaseModel):
    id: str
    name: str
    gender: str | None = None
    visual_descriptor: str
    lora_ref: str
    voice_profile_ref: str
    personality: str
    signature_expressions: list[str] = Field(default_factory=list)
    signature_moves: list[str] = Field(default_factory=list)


class Cast(BaseModel):
    id: str
    species: str
    is_original_synthetic: bool
    members: list[CastMember]
    design_constraints: list[str] = Field(default_factory=list)

    @field_validator("is_original_synthetic")
    @classmethod
    def require_original_synthetic(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Cast must be original synthetic.")
        return value

