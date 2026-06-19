import pytest

from core.config import Settings, load_app_config
from core.workflow import NOOP_STAGES, run_noop_job


@pytest.mark.asyncio
async def test_noop_job_records_all_stages(tmp_path) -> None:
    config = load_app_config()
    config.settings = Settings(
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "video_me.db",
    )

    job = await run_noop_job(app_config=config)

    assert job.status == "completed"
    assert set(job.stage_results) == set(NOOP_STAGES)
    for result in job.stage_results.values():
        assert result.artifact is not None
        assert result.adapter_name == "noop"

