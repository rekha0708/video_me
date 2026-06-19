from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from core.models.common import ArtifactRef


class JobStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class StageResult(BaseModel):
    stage_name: str
    status: JobStatus
    artifact: ArtifactRef | None = None
    adapter_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    source_url: str
    channel_profile_ref: str
    cast_ref: str
    script_mode: str = "transformed"
    status: JobStatus = JobStatus.CREATED
    stage_results: dict[str, StageResult] = Field(default_factory=dict)
    rights_cleared: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cost_total: float = 0.0

