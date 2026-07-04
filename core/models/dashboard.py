from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class DashboardSource(BaseModel):
    kind: Literal["url", "upload", "file", "story", "story_images"] = "url"
    url: str = ""

    @model_validator(mode="after")
    def _require_url_for_media_kinds(self) -> "DashboardSource":
        self.url = self.url.strip()
        if self.kind in ("story", "story_images"):
            # Story jobs have no media source; the url column is NOT NULL and
            # shown in the jobs list, so give it a descriptive placeholder.
            if not self.url:
                self.url = "story://direct-input"
        elif not self.url:
            raise ValueError("source url is required")
        return self


class DashboardJobOverrides(BaseModel):
    llm_model: str | None = None
    whisper_device: Literal["cpu", "cuda"] | None = None
    whisper_compute_type: str | None = None
    render_adapter: Literal["a1111", "comfyui_flux", "musubi_flux"] | None = None
    video_adapter: Literal["wan", "ltx"] | None = None
    tts_adapter: Literal["chatterbox", "fish_s2"] | None = None
    image_candidates: int | None = Field(default=None, ge=1, le=10)
    auto_approve_plan: bool | None = None
    auto_approve_images: bool | None = None


class CreateDashboardJobRequest(BaseModel):
    source: DashboardSource
    rights_cleared: bool = False
    target_language: Literal["en", "hi", "both"] = "en"
    mode: Literal["standard", "critique"] = "standard"
    phase: Literal["transcribe", "script_plan", "plan", "render", "assemble", "all", "noop"] = "all"
    run_critique: bool = False
    overrides: DashboardJobOverrides = Field(default_factory=DashboardJobOverrides)
    idempotency_key: str | None = None
    # Story input modes: the story replaces the transcript (kind="story"/"story_images");
    # character_images maps cast member_id → server-side image path (kind="story_images").
    story_text: str | None = None
    character_images: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_story_fields(self) -> "CreateDashboardJobRequest":
        if self.source.kind in ("story", "story_images"):
            if not (self.story_text or "").strip():
                raise ValueError("story_text is required for story source kinds")
        if self.source.kind == "story_images" and not self.character_images:
            raise ValueError("story_images requires at least one character image")
        return self


class DashboardJobRecord(BaseModel):
    job_id: str
    source_url: str
    source_kind: Literal["url", "upload", "file", "story", "story_images"]
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
