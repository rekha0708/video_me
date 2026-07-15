from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DashboardJobStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PENDING_TRANSCRIPT_REVIEW = "pending_transcript_review"
    PENDING_PLAN_APPROVAL = "pending_plan_approval"
    PENDING_IMAGE_APPROVAL = "pending_image_approval"
    PENDING_FINAL_REVIEW = "pending_final_review"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    STALLED = "stalled"


class DashboardQueueStatus(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DashboardQueueAction(StrEnum):
    START = "start"
    RESUME = "resume"
    RETRY_STAGE = "retry_stage"
    RERUN_SHOT = "rerun_shot"


class DashboardEventLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DashboardApprovalKind(StrEnum):
    TRANSCRIPT = "transcript"
    PLAN = "plan"
    IMAGES = "images"
    FINAL_PUBLISH = "final_publish"


class DashboardApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DashboardArtifactKind(StrEnum):
    JSON = "json"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    LOG = "log"
    SIDECAR = "sidecar"
    DEBUG_BUNDLE = "debug_bundle"


class DashboardAssetKind(StrEnum):
    """Media types accepted by the dashboard's opaque asset layer."""

    VIDEO = "video"
    IMAGE = "image"


class DashboardAssetStatus(StrEnum):
    """Lifecycle of a media asset before and after a job claims it."""

    STAGED = "staged"
    CLAIMED = "claimed"
    EXPIRED = "expired"


class DashboardSource(BaseModel):
    kind: Literal[
        "url",
        "upload",
        "file",
        "story",
        "story_images",
        "lora_training",
        "animate",
    ] = "url"
    url: str = ""

    @model_validator(mode="after")
    def _require_url_for_media_kinds(self) -> "DashboardSource":
        self.url = self.url.strip()
        if self.kind in ("story", "story_images", "lora_training", "animate"):
            # Story / LoRA jobs have no media source; the url column is NOT NULL
            # and shown in the jobs list, so give it a descriptive placeholder.
            if not self.url:
                if self.kind == "lora_training":
                    self.url = "lora-training://dashboard-upload"
                elif self.kind == "animate":
                    self.url = "animate://direct-input"
                else:
                    self.url = "story://direct-input"
        elif not self.url:
            raise ValueError("source url is required")
        return self


class DashboardJobOverrides(BaseModel):
    llm_model: str | None = None
    whisper_device: Literal["cpu", "cuda"] | None = None
    whisper_compute_type: str | None = None
    whisper_language: str | None = None
    whisper_isolate_vocals: bool | None = None
    render_adapter: Literal["a1111", "comfyui_flux", "musubi_flux"] | None = None
    video_adapter: Literal["wan_s2v", "wan", "wan_lightx2v", "wan_animate", "ltx"] | None = None
    lipsync_adapter: Literal["latentsync", "musetalk", "none"] | None = None
    tts_adapter: Literal["chatterbox", "fish_s2"] | None = None
    image_candidates: int | None = Field(default=None, ge=1, le=10)
    max_shot_duration_sec: float | None = Field(default=None, ge=2.0, le=10.0)
    lipsync_failure_policy: Literal["fallback_raw", "fail"] | None = None
    lipsync_max_retries: int | None = Field(default=None, ge=0, le=5)
    av_sync_duration_tolerance_sec: float | None = Field(default=None, ge=0.05, le=2.0)
    av_sync_failure_policy: Literal["warn", "fail"] | None = None
    video_upscale_enabled: bool | None = None
    video_upscale_target_fps: int | None = Field(default=None, ge=16, le=60)
    video_enhance_enabled: bool | None = None
    video_enhance_adapter: Literal[
        "ffmpeg",
        "rife",
        "film",
        "realesrgan_rife",
        "realesrgan_film",
        "latent_rife",
        "latent_film",
    ] | None = None
    video_enhance_target_fps: int | None = Field(default=None, ge=16, le=60)
    auto_approve_plan: bool | None = None
    auto_approve_images: bool | None = None
    auto_approve_transcript: bool | None = None
    wan_animate_mode: Literal["animate", "replace"] | None = None
    wan_animate_driver_source: Literal["job_source", "upload", "local"] | None = None
    wan_animate_driver_uri: str | None = None
    wan_animate_timeline: Literal["source_timestamps", "sequential"] | None = None
    wan_animate_subject_selection: Literal["largest", "center"] | None = None
    wan_animate_resolution_area: Literal["480p", "720p"] | None = None
    wan_animate_retarget_pose: bool | None = None
    wan_animate_use_flux_retarget: bool | None = None
    wan_animate_refert_num: Literal[1, 5] | None = None
    wan_animate_sampling_steps: int | None = Field(default=None, ge=10, le=40)
    wan_animate_mask_iterations: int | None = Field(default=None, ge=0, le=10)
    wan_animate_mask_kernel: int | None = Field(default=None, ge=1, le=31)
    wan_animate_mask_w_len: int | None = Field(default=None, ge=1, le=8)
    wan_animate_mask_h_len: int | None = Field(default=None, ge=1, le=8)


class LoraTrainingRequest(BaseModel):
    cast_member_id: str = ""
    image_paths: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Versioned Wan Animate direct-workflow request
# ---------------------------------------------------------------------------


_ASSET_ID_PATTERN = r"^ast_[A-Za-z0-9_-]{20,64}$"
WAN_ANIMATE_MAX_DRIVER_RANGE_SEC = 30.0


class AnimateDriverInput(BaseModel):
    """The already-ingested driving video and optional selected range."""

    asset_id: str = Field(min_length=24, max_length=68, pattern=_ASSET_ID_PATTERN)
    target_confirmed: Literal[True]
    timeline: Literal["full_driver", "selected_range"] = "full_driver"
    start_sec: float | None = Field(default=None, ge=0.0)
    end_sec: float | None = Field(default=None, gt=0.0)
    subject_selection: Literal["largest", "center"] = "largest"

    @model_validator(mode="after")
    def _validate_range_and_subject(self) -> "AnimateDriverInput":
        if self.timeline == "selected_range":
            if self.start_sec is None or self.end_sec is None:
                raise ValueError("selected_range requires both start_sec and end_sec")
            if self.end_sec <= self.start_sec:
                raise ValueError("end_sec must be greater than start_sec")
        elif self.start_sec is not None or self.end_sec is not None:
            raise ValueError("start_sec/end_sec are only valid for selected_range")

        return self


AnimateStylingTarget = Literal[
    "clothing",
    "jewelry",
    "bags",
    "footwear",
    "makeup",
    "hair",
    "other",
]


class WardrobeSpec(BaseModel):
    """Complete-look controls for a FLUX.2 LoRA render or reference edit.

    The legacy ``wardrobe`` name is retained in the version-1 API, but the
    contract covers clothing, jewelry, bags, footwear, makeup, hair, and any
    other user-directed styling detail.
    """

    change_targets: list[AnimateStylingTarget] = Field(default_factory=list, max_length=7)
    clothing_type: str = Field(default="", max_length=200)
    primary_color: str = Field(default="", max_length=100)
    material_pattern: str = Field(default="", max_length=200)
    jewelry: list[str] = Field(default_factory=list, max_length=12)
    bags: list[str] = Field(default_factory=list, max_length=8)
    footwear: str = Field(default="", max_length=200)
    makeup: str = Field(default="", max_length=400)
    hair: str = Field(default="", max_length=400)
    accessories: list[str] = Field(default_factory=list, max_length=12)
    details: str = Field(default="", max_length=1000)
    negative_constraints: str = Field(default="", max_length=1000)
    garment_asset_ids: list[str] = Field(default_factory=list, max_length=8)
    accessory_asset_ids: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("jewelry", "bags", "accessories")
    @classmethod
    def _clean_styling_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if any(len(value) > 240 for value in cleaned):
            raise ValueError("each styling item must be at most 240 characters")
        return cleaned

    @field_validator("change_targets")
    @classmethod
    def _deduplicate_change_targets(
        cls, values: list[AnimateStylingTarget]
    ) -> list[AnimateStylingTarget]:
        return list(dict.fromkeys(values))

    @field_validator("garment_asset_ids", "accessory_asset_ids")
    @classmethod
    def _validate_asset_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("asset IDs must not contain duplicates")
        # Reuse Pydantic's field-pattern semantics without making callers
        # depend on an internal constrained-string alias.
        import re

        if any(re.fullmatch(_ASSET_ID_PATTERN, value) is None for value in values):
            raise ValueError("invalid dashboard asset ID")
        return values

    @model_validator(mode="after")
    def _validate_reference_scope(self) -> "WardrobeSpec":
        overlap = set(self.garment_asset_ids) & set(self.accessory_asset_ids)
        if overlap:
            raise ValueError(
                "the same reference image cannot be both clothing and styling detail"
            )
        targets = set(self.change_targets)
        if targets:
            field_scopes = {
                "clothing": any(
                    (
                        self.clothing_type.strip(),
                        self.primary_color.strip(),
                        self.material_pattern.strip(),
                    )
                ),
                "jewelry": bool(self.jewelry),
                "bags": bool(self.bags),
                "footwear": bool(self.footwear.strip()),
                "makeup": bool(self.makeup.strip()),
                "hair": bool(self.hair.strip()),
                "other": bool(self.accessories),
            }
            off_scope = [
                target
                for target, has_direction in field_scopes.items()
                if has_direction and target not in targets
            ]
            if off_scope:
                raise ValueError(
                    "styling fields contain directions outside change_targets: "
                    + ", ".join(off_scope)
                )
        if self.garment_asset_ids and targets and "clothing" not in targets:
            raise ValueError(
                "clothing reference images require clothing in change_targets"
            )
        if self.accessory_asset_ids:
            detail_targets = {
                "jewelry",
                "bags",
                "footwear",
                "makeup",
                "hair",
                "other",
            }
            scoped_by_target = bool(targets & detail_targets)
            scoped_by_description = any(
                (
                    self.jewelry,
                    self.bags,
                    self.footwear.strip(),
                    self.makeup.strip(),
                    self.hair.strip(),
                    self.accessories,
                    self.details.strip(),
                )
            )
            if (targets and not scoped_by_target) or (
                not targets and not scoped_by_description
            ):
                raise ValueError(
                    "styling-detail reference images require a jewelry, bag, footwear, "
                    "makeup, hair, other target, or a matching text direction"
                )
        return self

    def has_direction(self) -> bool:
        return any(
            (
                self.change_targets,
                self.clothing_type.strip(),
                self.primary_color.strip(),
                self.material_pattern.strip(),
                self.jewelry,
                self.bags,
                self.footwear.strip(),
                self.makeup.strip(),
                self.hair.strip(),
                self.accessories,
                self.details.strip(),
                self.garment_asset_ids,
                self.accessory_asset_ids,
            )
        )


class AnimateCharacterOptions(BaseModel):
    look_source: Literal["auto_lora", "styled_lora", "exact_image"] = "auto_lora"
    cast_ref: str | None = Field(default=None, max_length=200)
    member_id: str | None = Field(default=None, max_length=200)
    exact_image_asset_id: str | None = Field(
        default=None,
        min_length=24,
        max_length=68,
        pattern=_ASSET_ID_PATTERN,
    )
    wardrobe: WardrobeSpec | None = None
    consistency: Literal["job"] = "job"

    @field_validator("cast_ref", "member_id")
    @classmethod
    def _strip_optional_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_look_source(self) -> "AnimateCharacterOptions":
        if self.look_source in ("auto_lora", "styled_lora"):
            if not self.cast_ref or not self.member_id:
                raise ValueError("generated look modes require cast_ref and member_id")
            if self.exact_image_asset_id is not None:
                raise ValueError("exact_image_asset_id is only valid for exact_image")
        if self.look_source == "auto_lora" and self.wardrobe is not None:
            raise ValueError("wardrobe is only valid for styled_lora")
        if self.look_source == "styled_lora":
            if self.wardrobe is None or not self.wardrobe.has_direction():
                raise ValueError("styled_lora requires a non-empty wardrobe specification")
        if self.look_source == "exact_image":
            if self.exact_image_asset_id is None:
                raise ValueError("exact_image requires exact_image_asset_id")
            if self.wardrobe is not None:
                raise ValueError("exact_image cannot include a wardrobe specification")
        return self


class AnimateAudioOptions(BaseModel):
    mode: Literal["driver", "cast_voice", "none"] = "driver"
    voice_member_id: str | None = Field(default=None, max_length=200)
    script_policy: Literal["verbatim"] = "verbatim"
    timing: Literal["match_driver"] = "match_driver"

    @field_validator("voice_member_id")
    @classmethod
    def _strip_voice_member_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_voice(self) -> "AnimateAudioOptions":
        if self.mode == "cast_voice" and not self.voice_member_id:
            raise ValueError("cast_voice requires voice_member_id")
        if self.mode != "cast_voice" and self.voice_member_id is not None:
            raise ValueError("voice_member_id is only valid for cast_voice")
        return self


class AnimateLipSyncOptions(BaseModel):
    enabled: bool = False
    backend: Literal["latentsync", "musetalk"] = "latentsync"


class AnimateOutputOptions(BaseModel):
    generation_area: Literal["480p", "720p"] = "720p"
    export: Literal["generated", "scale_1080p", "vertical_1080x1920"] = "generated"
    preserve_aspect: Literal[True] = True
    target_fps: Literal["generated", 48] = "generated"


class AnimateAdvancedOptions(BaseModel):
    """Mode-specific Wan controls kept out of the primary dashboard form."""

    retarget_pose: bool = False
    use_flux_retarget: bool = False
    refert_num: Literal[1, 5] = 1
    sampling_steps: int = Field(default=20, ge=10, le=40)
    mask_iterations: int | None = Field(default=None, ge=0, le=10)
    mask_kernel: int | None = Field(default=None, ge=1, le=31)
    mask_w_len: int | None = Field(default=None, ge=1, le=8)
    mask_h_len: int | None = Field(default=None, ge=1, le=8)

    @model_validator(mode="after")
    def _validate_advanced_dependencies(self) -> "AnimateAdvancedOptions":
        if self.use_flux_retarget and not self.retarget_pose:
            raise ValueError("Flux retargeting requires pose retargeting")
        if self.mask_kernel is not None and self.mask_kernel % 2 == 0:
            raise ValueError("mask_kernel must be odd")
        return self

    def has_replacement_controls(self) -> bool:
        return any(
            value is not None
            for value in (
                self.mask_iterations,
                self.mask_kernel,
                self.mask_w_len,
                self.mask_h_len,
            )
        )


class WanAnimateJobOptions(BaseModel):
    schema_version: Literal[1] = 1
    mode: Literal["animate", "replace"] = "animate"
    driver: AnimateDriverInput
    character: AnimateCharacterOptions
    audio: AnimateAudioOptions = Field(default_factory=AnimateAudioOptions)
    lipsync: AnimateLipSyncOptions = Field(default_factory=AnimateLipSyncOptions)
    output: AnimateOutputOptions = Field(default_factory=AnimateOutputOptions)
    advanced: AnimateAdvancedOptions = Field(default_factory=AnimateAdvancedOptions)

    @model_validator(mode="after")
    def _validate_mode_specific_options(self) -> "WanAnimateJobOptions":
        if self.mode == "replace" and (
            self.advanced.retarget_pose or self.advanced.use_flux_retarget
        ):
            raise ValueError("Wan Animate replacement mode does not support pose retargeting")
        if self.mode == "animate" and self.advanced.has_replacement_controls():
            raise ValueError("Wan Animate motion-transfer mode does not support replacement masks")
        if self.audio.mode == "cast_voice":
            if not self.character.cast_ref or not self.character.member_id:
                raise ValueError("cast_voice requires a selected cast and member")
            if self.audio.voice_member_id != self.character.member_id:
                raise ValueError("voice_member_id must match the target character member_id")
        if self.audio.mode == "none" and self.lipsync.enabled:
            raise ValueError("lip-sync requires driver or cast-voice audio")
        return self


class DashboardAssetRecord(BaseModel):
    """Durable asset metadata; the server path is never serialized to clients."""

    asset_id: str = Field(min_length=24, max_length=68, pattern=_ASSET_ID_PATTERN)
    owner_id: str = Field(min_length=1, max_length=200)
    kind: DashboardAssetKind
    status: DashboardAssetStatus = DashboardAssetStatus.STAGED
    original_name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    claimed_job_id: str | None = None
    claimed_at: datetime | None = None
    storage_path: str = Field(exclude=True, repr=False)

    @field_validator("created_at", "expires_at", "claimed_at")
    @classmethod
    def _require_aware_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("asset timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> "DashboardAssetRecord":
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        if self.status == DashboardAssetStatus.CLAIMED:
            if not self.claimed_job_id or self.claimed_at is None:
                raise ValueError("claimed assets require claimed_job_id and claimed_at")
        elif self.claimed_job_id is not None or self.claimed_at is not None:
            raise ValueError("only claimed assets may have claim metadata")
        return self


class CreateDashboardJobRequest(BaseModel):
    workflow_kind: Literal["pipeline", "wan_animate_direct"] = "pipeline"
    source: DashboardSource | None = None
    rights_cleared: bool = False
    target_language: Literal["en", "hi", "both"] = "en"
    mode: Literal["standard", "critique"] = "standard"
    render_mode: Literal["full", "source_audio", "re_voice"] = "full"
    audio_profile: Literal["auto", "single_speaker", "singing", "multi_speaker"] = "auto"
    gpu_price_per_hour: float = Field(default=0.0, ge=0.0)
    phase: Literal[
        "transcribe",
        "script_plan",
        "plan",
        "render",
        "assemble",
        "all",
        "noop",
        "lora_train",
    ] = "all"
    run_critique: bool = False
    overrides: DashboardJobOverrides = Field(default_factory=DashboardJobOverrides)
    idempotency_key: str | None = None
    # Story input modes: the story replaces the transcript (kind="story"/"story_images");
    # character_images maps cast member_id → server-side image path (kind="story_images").
    cast_ref: str | None = None
    story_text: str | None = None
    character_images: dict[str, str] = Field(default_factory=dict)
    lora_training: LoraTrainingRequest | None = None
    animate: WanAnimateJobOptions | None = None

    @model_validator(mode="after")
    def _require_story_fields(self) -> "CreateDashboardJobRequest":
        if self.workflow_kind == "wan_animate_direct":
            if self.animate is None:
                raise ValueError("wan_animate_direct requires animate options")
            # The nested driver asset is the source of truth. Preserve a
            # descriptive source row for the existing jobs list/repository
            # without copying an opaque ID into a URL-shaped legacy field.
            self.source = DashboardSource(kind="animate")
            if self.phase != "all":
                raise ValueError("wan_animate_direct requires phase='all'")
        else:
            if self.source is None:
                raise ValueError("pipeline jobs require source")
            if self.animate is not None:
                raise ValueError("animate options require workflow_kind='wan_animate_direct'")

        # Both branches above guarantee source before the compatibility
        # validators below run. Keeping their behavior unchanged is important
        # for stored pre-Animate queue payloads.
        assert self.source is not None
        if self.source.kind in ("story", "story_images"):
            if not (self.story_text or "").strip():
                raise ValueError("story_text is required for story source kinds")
        if self.source.kind == "story_images" and not self.character_images:
            raise ValueError("story_images requires at least one character image")
        if self.phase == "lora_train":
            if self.source.kind != "lora_training":
                raise ValueError("lora_train phase requires source.kind='lora_training'")
            if self.lora_training is None:
                raise ValueError("lora_training is required for lora_train phase")
            if not self.lora_training.cast_member_id.strip():
                raise ValueError("lora_training.cast_member_id is required")
            if not self.lora_training.image_paths:
                raise ValueError("lora_training.image_paths requires at least one image")
        if self.overrides.video_adapter == "wan_animate":
            source = self.overrides.wan_animate_driver_source or "job_source"
            driver_uri = (self.overrides.wan_animate_driver_uri or "").strip()
            if source in ("upload", "local") and not driver_uri:
                raise ValueError("Wan Animate upload/local driver requires wan_animate_driver_uri")
            if self.source.kind in ("story", "story_images") and source == "job_source":
                raise ValueError("Story jobs using Wan Animate require an uploaded or local driver video")
            if self.overrides.wan_animate_use_flux_retarget and not self.overrides.wan_animate_retarget_pose:
                raise ValueError("Wan Animate Flux retargeting requires pose retargeting")
            if (self.overrides.wan_animate_mode or "animate") == "replace" and (
                self.overrides.wan_animate_retarget_pose
                or self.overrides.wan_animate_use_flux_retarget
            ):
                raise ValueError("Wan Animate replacement mode does not support pose retargeting")
        return self


class DashboardJobRecord(BaseModel):
    job_id: str
    source_url: str
    source_kind: Literal[
        "url",
        "upload",
        "file",
        "story",
        "story_images",
        "lora_training",
        "animate",
    ]
    status: DashboardJobStatus
    phase: str
    target_language: str
    rights_cleared: bool
    current_stage: str | None = None
    current_shot_id: str | None = None
    approval_kind: str | None = None
    completed_phases: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    queued_at: datetime | None = None
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    last_heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    terminal_error: dict[str, Any] | None = None
    request: dict[str, Any] = Field(default_factory=dict)


class DashboardQueueItem(BaseModel):
    queue_id: str
    job_id: str
    action: DashboardQueueAction
    payload: dict[str, Any] = Field(default_factory=dict)
    status: DashboardQueueStatus = DashboardQueueStatus.QUEUED
    priority: int = 100
    created_at: datetime = Field(default_factory=utc_now)
    claimed_at: datetime | None = None
    claimed_by: str | None = None
    completed_at: datetime | None = None
    error: dict[str, Any] | None = None


class DashboardEvent(BaseModel):
    event_id: int
    job_id: str
    event_type: str
    level: DashboardEventLevel = DashboardEventLevel.INFO
    stage_name: str | None = None
    shot_id: str | None = None
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DashboardArtifact(BaseModel):
    artifact_id: str
    job_id: str
    stage_name: str | None = None
    shot_id: str | None = None
    kind: DashboardArtifactKind
    uri: str
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    previewable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class DashboardApprovalRequest(BaseModel):
    approval_id: str
    job_id: str
    kind: DashboardApprovalKind
    status: DashboardApprovalStatus = DashboardApprovalStatus.PENDING
    iteration: int = 1
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    decided_at: datetime | None = None
    reviewer: str | None = None


class WorkerHeartbeat(BaseModel):
    worker_id: str
    hostname: str | None = None
    process_id: int | None = None
    version: str | None = None
    current_job_id: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    last_heartbeat_at: datetime = Field(default_factory=utc_now)


class DashboardJobDetail(BaseModel):
    job: DashboardJobRecord
    queue: list[DashboardQueueItem] = Field(default_factory=list)
    events: list[DashboardEvent] = Field(default_factory=list)
    pending_approval: DashboardApprovalRequest | None = None


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    message_id: str
    job_id: str
    role: ChatRole
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
