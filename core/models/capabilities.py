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
    isolate_vocals: bool = False  # run Demucs vocal separation before transcription


class TranscribeResult(BaseModel):
    segments: list[TranscriptSegment]
    language: str
    full_text: str


# ---------- analyze_content → ContentMetadata (already defined in content.py) ----------

class AnalyzeRequest(BaseModel):
    transcript: TranscribeResult
    channel_profile: ChannelProfile


# ---------- analyze_visuals → VisualContext ----------

class VisualSegment(BaseModel):
    start: float
    end: float
    setting: str                                   # observed location/background at this time
    props: list[str] = Field(default_factory=list)  # notable objects/props on screen
    chart: str = ""  # short phrase if a chart/graph/diagram is visible ("" = none)


class VisualContext(BaseModel):
    segments: list[VisualSegment] = Field(default_factory=list)
    summary: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.segments


class AnalyzeVisualsRequest(BaseModel):
    video_uri: str
    segments: list[TranscriptSegment]


# ---------- adapt_script → Script (already defined in content.py) ----------

class AdaptScriptRequest(BaseModel):
    metadata: ContentMetadata
    cast: Cast
    channel_profile: ChannelProfile
    language: str = "en"  # BCP-47 code: "en" | "hi"
    visual_context: VisualContext | None = None  # grounds scene settings in the source video


# ---------- plan_shots → Storyboard (already defined in content.py) ----------

class PlanShotsRequest(BaseModel):
    script: Script
    cast: Cast
    critique_notes: list[str] = Field(default_factory=list)  # injected on re-plan
    visual_context: VisualContext | None = None  # source-video chart hints for overlay authoring


# ---------- render_overlays → per-shot chart panel PNGs ----------

class RenderOverlaysRequest(BaseModel):
    shots: list[Any]  # list[Shot] — Any avoids the circular import (same as ImageApprovalRequest)


class RenderOverlaysResult(BaseModel):
    images: dict[str, str] = Field(default_factory=dict)   # shot_id → PNG path
    skipped: dict[str, str] = Field(default_factory=dict)  # shot_id → reason


class OverlayWindow(BaseModel):
    """A rendered overlay PNG + the absolute time window it is visible in the final video."""
    shot_id: str
    png_uri: str
    start_sec: float
    end_sec: float


# ---------- render_character ----------

class RenderCharacterRequest(BaseModel):
    member: CastMember
    setting: str
    expression: str | None = None
    shot_id: str = ""  # scopes render output per shot so per-shot backgrounds don't collide
    camera: str = ""   # Shot.camera framing (close-up/medium/reaction/wide) → render prompt
    action: str = ""   # Shot.action → render prompt, so each still shows the shot's pose/angle
    other_members: list[CastMember] = Field(default_factory=list)
    # Per-cast overrides from config/casts/<cast>/params.py (None/"" → adapter defaults).
    lora_file: str = ""
    lora_weight: float | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    trigger: str = ""
    # Appended to the render prompt ("" → adapter's cartoon-style default, kept
    # for the original kids_duo cast; photorealistic casts should set this).
    style_suffix: str = ""


class ImageSet(BaseModel):
    images: list[str]  # URIs
    member_id: str


# ---------- synthesize_voice ----------

class VoiceRequest(BaseModel):
    text: str
    voice_profile_ref: str
    speaker_id: str
    expression: str | None = None
    language: str = "en"  # BCP-47 code: "en" | "hi"


class AudioTrack(BaseModel):
    uri: str
    duration_sec: float
    speaker_id: str | None = None


# ---------- generate_video ----------

class VideoDriver(BaseModel):
    uri: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    mode: Literal["animate", "replace"] = "animate"
    prepared_dir: str | None = None


class VideoRequest(BaseModel):
    image_uri: str
    action: str
    duration_sec: float
    shot_id: str
    setting: str = ""             # per-shot scene/environment description for the video prompt
    audio_uri: str | None = None  # set when video adapter has native_lipsync=True
    # Same per-cast style_suffix as RenderCharacterRequest ("" → adapter's cartoon default).
    style_suffix: str = ""
    # Driving-video conditioning used only by Wan2.2 Animate. Keeping this
    # optional preserves the request contract for every existing backend.
    driver: VideoDriver | None = None


class PreparedWanAnimateInput(BaseModel):
    shot_id: str
    prepared_dir: str
    driver_uri: str
    start_sec: float
    end_sec: float
    frame_count: int
    fps: int
    width: int
    height: int
    cache_hit: bool = False


class VideoClip(BaseModel):
    uri: str
    duration_sec: float
    shot_id: str | None = None


# ---------- critique_images ----------

class ImageCandidateScore(BaseModel):
    candidate_index: int
    scores: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


class ImageCritiqueRequest(BaseModel):
    shot_id: str
    shot_prompt: str          # human-readable description of setting + action
    candidate_uris: list[str] # N local file paths (PNG)
    cast_descriptor: str      # visual_descriptor of the primary character
    other_descriptors: list[str] = Field(default_factory=list)
    feedback_examples: list[dict] = Field(default_factory=list)  # few-shot from log


class ImageCritiqueResult(BaseModel):
    winner_index: int         # 0-based into candidate_uris
    winner_uri: str
    candidate_uris: list[str] = Field(default_factory=list)  # so approval gates can serve all candidates
    candidate_scores: list[ImageCandidateScore] = Field(default_factory=list)
    overall_reasoning: str = ""
    # "vlm": winner picked by the VLM critique; "user": synthetic result built
    # from user-provided reference images (story_images mode — no render/VLM ran);
    # "single": only one candidate was rendered, so the VLM call was skipped.
    origin: Literal["vlm", "user", "single"] = "vlm"


class ImageApprovalRequest(BaseModel):
    shots: list[Any]          # list[Shot] — avoids circular import
    critique_results: list[ImageCritiqueResult]
    cast_id: str


class ImageApprovalResult(BaseModel):
    approved_uris: list[str]              # one per shot (may be human-overridden)
    overrides: dict[str, int] = Field(default_factory=dict)  # shot_id → candidate_index


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
    overlays: list[OverlayWindow] = Field(default_factory=list)  # chart panels, time-windowed
    # Per-shot tracks (same order as clips), used for crossfading clip boundaries.
    # Empty ([]) → adapter falls back to plain concat + the pre-combined `audio` above.
    audio_tracks: list[AudioTrack] = Field(default_factory=list)
    # Preserve exact clip/audio timing for source-timed modes. When True, the
    # assembler uses hard concat + the pre-combined audio even if audio_tracks exist.
    preserve_timing: bool = False


class FinalVideo(BaseModel):
    uri: str
    duration_sec: float
    sidecar_uri: str | None = None


# ---------- video_enhance ----------

class VideoEnhanceRequest(BaseModel):
    video_uri: str
    duration_sec: float
    output_name: str = "enhanced.mp4"
    target_width: int = 1080
    target_height: int = 1920
    target_fps: int = 48
    interpolation: Literal["fps", "minterpolate", "rife", "film"] = "minterpolate"
    stage: Literal["clip", "final"] = "clip"
    has_burned_text: bool = False
    preserve_audio: bool = True


class VideoEnhanceResult(BaseModel):
    video_uri: str
    duration_sec: float
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    adapter: str = ""
    notes: list[str] = Field(default_factory=list)


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
    language: str = "en"  # tags the review folder + metadata (distinguishes both-language runs)


class PublishResult(BaseModel):
    review_path: str
    metadata_path: str
    status: str = "pending_review"
