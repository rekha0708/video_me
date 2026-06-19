import pytest
from pydantic import BaseModel

from core.config import Settings, load_app_config
from core.executor import StageError, check_rights, run_stage
from core.models.common import CostEstimate, HealthStatus
from core.models.job import Job, JobStatus
from core.storage import create_artifact_store, create_job_store


class _SimpleRequest(BaseModel):
    value: str


class _SimpleResult(BaseModel):
    echoed: str


class _OkCapability:
    name = "test_cap"

    async def health(self) -> HealthStatus:
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: _SimpleRequest) -> CostEstimate:
        return CostEstimate()

    async def run(self, req: _SimpleRequest) -> _SimpleResult:
        return _SimpleResult(echoed=req.value)


class _DownCapability(_OkCapability):
    async def health(self) -> HealthStatus:
        return HealthStatus(status="down", reason="service unavailable")


def _make_stores(tmp_path):
    config = load_app_config()
    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
    )
    return (
        create_artifact_store(config.settings),
        create_job_store(config.settings),
        config,
    )


def _make_job(config) -> Job:
    return Job(
        source_url="test://x",
        channel_profile_ref=config.channel_profile.id,
        cast_ref=config.cast.id,
        rights_cleared=True,
    )


@pytest.mark.asyncio
async def test_run_stage_persists_artifact_and_updates_job(tmp_path) -> None:
    artifact_store, job_store, config = _make_stores(tmp_path)
    job = _make_job(config)
    job_store.save_job(job)

    result = await run_stage(
        "test_stage",
        _OkCapability(),
        _SimpleRequest(value="hello"),
        job,
        artifact_store,
        job_store,
    )

    assert result.echoed == "hello"
    assert "test_stage" in job.stage_results
    assert job.stage_results["test_stage"].adapter_name == "test_cap"
    assert job.stage_results["test_stage"].artifact is not None

    # Persisted to store
    saved = job_store.get_job(job.job_id)
    assert saved is not None
    assert "test_stage" in saved.stage_results


@pytest.mark.asyncio
async def test_run_stage_raises_on_down_adapter(tmp_path) -> None:
    artifact_store, job_store, config = _make_stores(tmp_path)
    job = _make_job(config)
    job_store.save_job(job)

    with pytest.raises(StageError) as exc_info:
        await run_stage(
            "test_stage",
            _DownCapability(),
            _SimpleRequest(value="hello"),
            job,
            artifact_store,
            job_store,
        )

    assert "down" in str(exc_info.value).lower()


def test_check_rights_blocks_uncleared_job() -> None:
    job = Job(
        source_url="test://x",
        channel_profile_ref="education_kids",
        cast_ref="pig_kids_placeholder",
        rights_cleared=False,
    )
    with pytest.raises(StageError) as exc_info:
        check_rights(job)

    assert job.status == JobStatus.BLOCKED
    assert "rights_cleared" in str(exc_info.value)


def test_check_rights_passes_cleared_job() -> None:
    job = Job(
        source_url="test://x",
        channel_profile_ref="education_kids",
        cast_ref="pig_kids_placeholder",
        rights_cleared=True,
    )
    check_rights(job)  # must not raise
    assert job.status == JobStatus.CREATED
