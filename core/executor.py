import logging
from typing import Any

from pydantic import BaseModel

from core.models.job import Job, JobStatus
from core.observability import log_event
from core.storage import ArtifactStore, JobRepository, completed_stage

logger = logging.getLogger(__name__)


class StageError(Exception):
    def __init__(self, stage_name: str, reason: str) -> None:
        self.stage_name = stage_name
        self.reason = reason
        super().__init__(f"Stage '{stage_name}' failed: {reason}")


def check_rights(job: Job) -> None:
    """Pipeline gate before adapt_script: block if rights are not cleared."""
    if not job.rights_cleared:
        job.status = JobStatus.BLOCKED
        raise StageError(
            "check_rights",
            "rights_cleared is False — set source rights on the job before adapt_script.",
        )


async def run_stage(
    stage_name: str,
    capability: Any,
    request: BaseModel,
    job: Job,
    artifact_store: ArtifactStore,
    job_store: JobRepository,
) -> Any:
    """
    Run one pipeline stage: health-check → invoke capability → persist artifact → update job.

    Returns the capability result so the caller can thread it into the next stage.
    Raises StageError on adapter health failure; propagates adapter exceptions otherwise.
    """
    adapter_name = getattr(capability, "name", "unknown")
    log_event(logger, "stage_started", job_id=job.job_id, stage=stage_name, adapter=adapter_name)

    health = await capability.health()
    if health.status == "down":
        raise StageError(stage_name, f"Adapter '{adapter_name}' is down: {health.reason}")

    result = await capability.run(request)

    payload = (
        result.model_dump(mode="json")
        if isinstance(result, BaseModel)
        else {"result": str(result)}
    )
    artifact = artifact_store.put_json(job.job_id, stage_name, payload)
    stage_result = completed_stage(stage_name, artifact, adapter_name=adapter_name)
    job.stage_results[stage_name] = stage_result
    job_store.save_stage_result(job.job_id, stage_result)
    job_store.save_job(job)

    log_event(logger, "stage_completed", job_id=job.job_id, stage=stage_name, adapter=adapter_name)
    return result
