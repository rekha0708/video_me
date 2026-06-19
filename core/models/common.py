from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactRef(BaseModel):
    uri: str
    media_type: str = "application/octet-stream"
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    reason: str | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CostEstimate(BaseModel):
    amount: float = 0.0
    currency: str = "USD"
    units: str = "job"
    notes: str | None = None

