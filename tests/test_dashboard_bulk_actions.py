"""Tests for POST /api/jobs/bulk-cancel and /api/jobs/bulk-delete."""
from pathlib import Path

import pytest

_has_fastapi = True
try:
    import fastapi  # noqa: F401
except ImportError:
    _has_fastapi = False

pytestmark = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


def _make_client_and_repo(tmp_path: Path):
    from fastapi.testclient import TestClient
    from services.dashboard_api import create_app
    from services.dashboard_repository import DashboardRepository
    from core.config import AppConfig, Settings
    from core.models.profile import ChannelProfile, Cast, CastMember

    settings = Settings(
        data_dir=str(tmp_path / "data"),
        artifact_dir=str(tmp_path / "art"),
        sqlite_path=str(tmp_path / "test.db"),
    )
    cfg = AppConfig(
        settings=settings,
        channel_profile=ChannelProfile(
            id="test", name="test", aspect_ratio="9:16",
            genre_content="education", tone="friendly",
            format="animated_character", made_for_kids=True,
        ),
        cast=Cast(id="kids_duo", species="human", is_original_synthetic=True, members=[
            CastMember(id="max", name="Max", visual_descriptor="boy", lora_ref="loras/max",
                       voice_profile_ref="voices/max", personality="friendly"),
        ]),
    )
    repo = DashboardRepository(Path(settings.sqlite_path))
    app = create_app(config_loader=lambda: cfg, repository=repo)
    return TestClient(app, raise_server_exceptions=False), repo


def _seed_job(repo, *, status: str) -> str:
    from core.models.dashboard import CreateDashboardJobRequest, DashboardJobStatus, DashboardSource

    req = CreateDashboardJobRequest(
        source=DashboardSource(kind="file", url="file:///tmp/video.mp4"),
        rights_cleared=True,
        phase="all",
    )
    job, _queue_item = repo.create_queued_job(req)
    repo.update_job_status(job.job_id, DashboardJobStatus[status.upper()])
    return job.job_id


def test_bulk_cancel_mixed_statuses(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    running_id = _seed_job(repo, status="running")
    completed_id = _seed_job(repo, status="completed")

    resp = client.post(
        "/api/jobs/bulk-cancel",
        json={"job_ids": [running_id, completed_id, "does-not-exist"]},
    )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[running_id] == "cancel_requested"
    assert results[completed_id] == "already_terminal"
    assert results["does-not-exist"] == "not_found"

    assert repo.get_job(running_id).status.value == "cancel_requested"
    assert repo.get_job(completed_id).status.value == "completed"


def test_bulk_delete_only_removes_terminal_jobs(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    running_id = _seed_job(repo, status="running")
    failed_id = _seed_job(repo, status="failed")

    resp = client.post(
        "/api/jobs/bulk-delete",
        json={"job_ids": [running_id, failed_id, "does-not-exist"]},
    )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert results[running_id] == "skipped_active"
    assert results[failed_id] == "deleted"
    assert results["does-not-exist"] == "not_found"

    assert repo.get_job(running_id) is not None  # untouched
    assert repo.get_job(failed_id) is None  # gone


def test_bulk_delete_removes_events_and_queue_rows(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="cancelled")
    repo.record_event(job_id, "job_cancelled", "test event")

    resp = client.post("/api/jobs/bulk-delete", json={"job_ids": [job_id]})

    assert resp.status_code == 200
    assert resp.json()["results"][job_id] == "deleted"
    assert repo.get_job(job_id) is None
    assert repo.list_events(job_id) == []
    assert repo.list_queue(job_id) == []
