from typing import Any, Literal

from pydantic import BaseModel, Field

from core.models.content import ContentMetadata, Script, Storyboard
from core.models.profile import Cast, CastMember, ChannelProfile


# ---------- fetch_media ----------

class FetchMediaRequest(BaseModel):
    source_url: str


class FetchMediaResult(BaseModel):
    video_uri: str
    audio_uri: str
    duration_sec: float
    source_url: str


# ---------- separate_audio ----------

class SeparateAudioRequest(BaseModel):
    audio_uri: str


class SeparateAudioResult(BaseModel):
    stems: dict[str, str]  # e.g. {"vocals": "...", "background": "..."}


# ---------- transcribe ----------

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None = None
    words: list[WordTimestamp] = Field(default_factory=list)


class TranscribeRequest(BaseModel):
    audio_uri: str


class TranscribeResult(BaseModel):
    segments: list[TranscriptSegment]
    language: str
    full_text: str


# ---------- analyze_content → ContentMetadata (already defined in content.py) ----------

class AnalyzeRequest(BaseModel):
    transcript: TranscribeResult
    channel_profile: ChannelProfile


# ---------- adapt_script → Script (already defined in content.py) ----------

class AdaptScriptRequest(BaseModel):
    metadata: ContentMetadata
    cast: Cast
    channel_profile: ChannelProfile


# ---------- plan_shots → Storyboard (already defined in content.py) ----------

class PlanShotsRequest(BaseModel):
    script: Script
    cast: Cast
    critique_notes: list[str] = Field(default_factory=list)  # injected on re-plan


# ---------- render_character ----------

class RenderCharacterRequest(BaseModel):
    member: CastMember
    setting: str
    expression: str | None = None


class ImageSet(BaseModel):
    images: list[str]  # URIs
    member_id: str


# ---------- synthesize_voice ----------

class VoiceRequest(BaseModel):
    text: str
    voice_profile_ref: str
    speaker_id: str
    expression: str | None = None


class AudioTrack(BaseModel):
    uri: str
    duration_sec: float
    speaker_id: str | None = None


# ---------- generate_video ----------

class VideoRequest(BaseModel):
    image_uri: str
    action: str
    duration_sec: float
    shot_id: str
    audio_uri: str | None = None  # set when video adapter has native_lipsync=True


class VideoClip(BaseModel):
    uri: str
    duration_sec: float
    shot_id: str | None = None


# ---------- critique_plan ----------

class PlanCritiqueRequest(BaseModel):
    storyboard: Storyboard
    script: Script
    cast: Cast


class PlanCritiqueResult(BaseModel):
    verdict: Literal["pass", "revise"]
    scores: dict[str, float] = Field(default_factory=dict)
    revision_notes: list[str] = Field(default_factory=list)


# ---------- lip_sync → VideoClip ----------

class LipSyncRequest(BaseModel):
    video_uri: str
    audio_uri: str
    shot_id: str


# ---------- mix_audio → AudioTrack ----------

class MixAudioRequest(BaseModel):
    tracks: list[AudioTrack]
    music_uri: str | None = None
    target_loudness_lufs: float = -14.0


# ---------- assemble_video ----------

class AssembleRequest(BaseModel):
    clips: list[VideoClip]
    audio: AudioTrack
    caption_text: str
    aspect_ratio: str = "9:16"
    made_for_kids: bool = True
    disclosure_label_required: bool = True


class FinalVideo(BaseModel):
    uri: str
    duration_sec: float
    sidecar_uri: str | None = None


# ---------- critique ----------

class CritiqueRequest(BaseModel):
    video_uri: str
    script: Script
    channel_profile_id: str


class CritiqueResult(BaseModel):
    scores: dict[str, float]
    verdict: Literal["pass", "regenerate", "reject"]
    reasons: list[str]
    suggested_param_overrides: dict[str, Any] = Field(default_factory=dict)
    sampled_frame_uris: list[str] = Field(default_factory=list)


# ---------- publish ----------

class PublishRequest(BaseModel):
    video: FinalVideo
    rights_cleared: bool
    made_for_kids: bool
    disclosure_label_required: bool
    learning_objective_summary: str


class PublishResult(BaseModel):
    review_path: str
    metadata_path: str
    status: str = "pending_review"
