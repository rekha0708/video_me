import logging

from core.config import AppConfig, load_app_config
from core.models.job import Job, JobStatus
from core.observability import log_event
from core.storage import completed_stage, create_artifact_store, create_job_store

logger = logging.getLogger(__name__)


NOOP_STAGES = ("create_job", "noop_dag", "record_result")


async def run_noop_job(
    source_url: str = "noop://phase-0",
    app_config: AppConfig | None = None,
) -> Job:
    config = app_config or load_app_config()
    settings = config.settings
    artifacts = create_artifact_store(settings)
    jobs = create_job_store(settings)

    job = Job(
        source_url=source_url,
        channel_profile_ref=config.channel_profile.id,
        cast_ref=config.cast.id,
        rights_cleared=False,
    )
    job.status = JobStatus.RUNNING
    jobs.save_job(job)
    log_event(logger, "job_started", job_id=job.job_id, workflow_engine=settings.workflow_engine)

    for stage_name in NOOP_STAGES:
        log_event(logger, "stage_started", job_id=job.job_id, stage=stage_name, adapter="noop")
        artifact = artifacts.put_json(
            job.job_id,
            stage_name,
            {
                "job_id": job.job_id,
                "stage": stage_name,
                "status": "completed",
                "note": "Phase 0 no-op stage recorded successfully.",
            },
        )
        result = completed_stage(stage_name, artifact)
        job.stage_results[stage_name] = result
        jobs.save_stage_result(job.job_id, result)
        jobs.save_job(job)
        log_event(logger, "stage_completed", job_id=job.job_id, stage=stage_name, adapter="noop")

    job.status = JobStatus.COMPLETED
    jobs.save_job(job)
    log_event(logger, "job_completed", job_id=job.job_id)
    return job
