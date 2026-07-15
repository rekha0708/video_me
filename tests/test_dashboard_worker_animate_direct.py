from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.config import load_app_config
from core.models.dashboard import CreateDashboardJobRequest
from services.dashboard_repository import DashboardRepository
from services.dashboard_worker import DashboardWorker


ASSET_ID = "ast_abcdefghijklmnopqrstuvwxyz123456"
IMAGE_ID = "ast_zyxwvutsrqponmlkjihgfedcba654321"


def _request() -> CreateDashboardJobRequest:
    return CreateDashboardJobRequest.model_validate(
        {
            "workflow_kind": "wan_animate_direct",
            "rights_cleared": True,
            "animate": {
                "mode": "replace",
                "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
                "character": {
                    "look_source": "exact_image",
                    "exact_image_asset_id": IMAGE_ID,
                },
                "audio": {"mode": "none"},
            },
        }
    )


@pytest.mark.asyncio
async def test_worker_dispatches_direct_request_without_generic_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_app_config()
    config.settings.sqlite_path = tmp_path / "dashboard.db"
    repo = DashboardRepository(config.settings.sqlite_path)
    worker = DashboardWorker(repo, config, worker_id="animate-test-worker")
    request = _request()
    _, queued = repo.create_queued_job(request)
    action = repo.claim_next_action(worker.worker_id)
    assert action is not None

    direct = AsyncMock()
    generic = AsyncMock()
    monkeypatch.setattr(worker, "_run_wan_animate_direct", direct)
    monkeypatch.setattr(worker, "_run_pipeline", generic)

    await worker._execute_pipeline(action)

    direct.assert_awaited_once()
    assert direct.await_args.args[0].workflow_kind == "wan_animate_direct"
    assert direct.await_args.args[1] == action.job_id
    generic.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_runs_direct_job_with_job_scoped_assets_and_persists_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.animate_workflow as animate_workflow
    import core.storage as storage
    import services.dashboard_assets as dashboard_assets

    config = load_app_config()
    config.settings.sqlite_path = tmp_path / "dashboard.db"
    config.settings.data_dir = tmp_path / "data"
    config.settings.local_video_dir = tmp_path / "local-videos"
    config.settings.local_video_dir.mkdir(parents=True)
    repo = DashboardRepository(config.settings.sqlite_path)
    worker = DashboardWorker(repo, config, worker_id="animate-test-worker")
    request = _request()
    job, _ = repo.create_queued_job(request)

    asset_store = object()
    asset_store_factory = Mock(return_value=asset_store)
    monkeypatch.setattr(dashboard_assets, "DashboardAssetStore", asset_store_factory)

    look_path = tmp_path / "look.png"
    raw_path = tmp_path / "raw.mp4"
    final_path = tmp_path / "final.mp4"
    look_path.write_bytes(b"look")
    raw_path.write_bytes(b"raw")
    final_path.write_bytes(b"final")
    result = SimpleNamespace(
        canonical_look_uri=str(look_path),
        raw_video_uri=str(raw_path),
        final_video_uri=str(final_path),
        audio_uri=None,
        duration_sec=4.0,
        model_dump=lambda **_kwargs: {
            "job_id": job.job_id,
            "final_video_uri": str(final_path),
        },
    )
    run_direct = AsyncMock(return_value=result)
    monkeypatch.setattr(animate_workflow, "run_wan_animate_direct_job", run_direct)

    artifact_store = SimpleNamespace(put_json=Mock())
    monkeypatch.setattr(storage, "create_artifact_store", Mock(return_value=artifact_store))

    await worker._run_wan_animate_direct(request, job.job_id)

    asset_store_factory.assert_called_once_with(
        db_path=config.settings.sqlite_path,
        storage_root=config.settings.data_dir / "dashboard_assets",
        allowed_server_roots=[config.settings.local_video_dir],
        max_total_bytes=config.settings.dashboard_asset_quota_bytes,
    )
    assert run_direct.await_args.kwargs["asset_store"] is asset_store
    assert run_direct.await_args.kwargs["image_approval"] is not None
    artifact_store.put_json.assert_called_once_with(
        job.job_id,
        "wan_animate_direct",
        result.model_dump(mode="json"),
    )
    persisted = repo.get_job(job.job_id)
    assert persisted is not None
    assert persisted.status.value == "completed"
    assert "wan_animate_direct" in persisted.completed_phases
    artifacts = repo.list_artifacts(job.job_id)
    assert {artifact.metadata.get("role") for artifact in artifacts} == {
        "canonical_character_look",
        "raw_wan_animate",
        "final_video",
    }


def test_direct_request_always_selects_wan_animate_cleanup_target(tmp_path: Path) -> None:
    config = load_app_config()
    config.settings.sqlite_path = tmp_path / "dashboard.db"
    repo = DashboardRepository(config.settings.sqlite_path)
    worker = DashboardWorker(repo, config, worker_id="animate-test-worker")

    target = worker._video_cleanup_target(_request())

    assert target == (
        "wan_animate_model_unload",
        "Wan Animate video model",
        config.settings.wan_animate_base_url,
    )
