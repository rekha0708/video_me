"""Tests for POST /api/jobs/{job_id}/retry — retryable statuses + overrides."""
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


def _seed_job(
    repo,
    *,
    status: str,
    phase: str = "all",
    video_adapter: str | None = None,
    lipsync_adapter: str | None = None,
    render_mode: str = "full",
):
    from core.models.dashboard import (
        CreateDashboardJobRequest, DashboardJobOverrides, DashboardJobStatus, DashboardSource,
    )

    req = CreateDashboardJobRequest(
        source=DashboardSource(kind="file", url="file:///tmp/video.mp4"),
        rights_cleared=True,
        phase=phase,
        render_mode=render_mode,
        overrides=DashboardJobOverrides(
            video_adapter=video_adapter,
            lipsync_adapter=lipsync_adapter,
        ),
    )
    job, _queue_item = repo.create_queued_job(req)
    repo.update_job_status(job.job_id, DashboardJobStatus[status.upper()])
    return job.job_id


def test_retry_allowed_when_failed(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="failed")
    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_retry_allowed_when_completed(tmp_path: Path) -> None:
    """A completed job can be re-run — this is what lets us re-test
    generate_video/assemble_video changes without re-rendering images."""
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed")
    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_retry_rejected_when_running(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="running")
    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NOT_RETRYABLE"


def test_retry_rejected_for_unknown_job(tmp_path: Path) -> None:
    client, _repo = _make_client_and_repo(tmp_path)
    resp = client.post("/api/jobs/does-not-exist/retry")
    assert resp.status_code == 404


def test_retry_without_body_preserves_original_overrides(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed", video_adapter="ltx")
    client.post(f"/api/jobs/{job_id}/retry")
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["overrides"]["video_adapter"] == "ltx"


def test_retry_with_body_overrides_video_adapter(tmp_path: Path) -> None:
    """Lets you compare ltx vs wan against the same cached renders without
    touching the job's original config."""
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed", video_adapter="ltx")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"video_adapter": "wan"})
    assert resp.status_code == 200
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["overrides"]["video_adapter"] == "wan"


def test_retry_with_body_overrides_whisper_language(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="failed")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"whisper_language": "en"})
    assert resp.status_code == 200
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["overrides"]["whisper_language"] == "en"


def test_retry_with_body_overrides_lipsync_adapter(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(
        repo,
        status="completed",
        video_adapter="wan",
        lipsync_adapter="musetalk",
    )
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"lipsync_adapter": "latentsync"})
    assert resp.status_code == 200
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["overrides"]["video_adapter"] == "wan"
    assert latest.payload["overrides"]["lipsync_adapter"] == "latentsync"


def test_retry_with_body_leaves_other_overrides_untouched(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed", video_adapter="ltx")
    client.post(f"/api/jobs/{job_id}/retry", json={"video_adapter": "wan"})
    job = repo.get_job(job_id)
    # original job record's own overrides are unchanged; only the requeue payload differs
    assert job.request["overrides"]["video_adapter"] == "ltx"


def test_retry_resets_status_to_queued(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed")
    client.post(f"/api/jobs/{job_id}/retry")
    job = repo.get_job(job_id)
    from core.models.dashboard import DashboardJobStatus
    assert job.status == DashboardJobStatus.QUEUED


def test_retry_with_phase_override(tmp_path: Path) -> None:
    """Operator can re-run a completed job from a specific phase — e.g.
    'assemble' to redo only the final concat, keeping all cached renders."""
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed", phase="all")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"phase": "assemble"})
    assert resp.status_code == 200
    assert resp.json()["phase"] == "assemble"
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["phase"] == "assemble"


def test_retry_with_phase_and_video_adapter(tmp_path: Path) -> None:
    """Phase + video adapter can be overridden together."""
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed", phase="all", video_adapter="ltx")
    resp = client.post(
        f"/api/jobs/{job_id}/retry",
        json={"phase": "render", "video_adapter": "wan"},
    )
    assert resp.status_code == 200
    assert resp.json()["phase"] == "render"
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["phase"] == "render"
    assert latest.payload["overrides"]["video_adapter"] == "wan"


def test_retry_with_body_overrides_render_mode(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed", render_mode="full")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"render_mode": "source_audio"})
    assert resp.status_code == 200
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["render_mode"] == "source_audio"


def test_retry_with_body_overrides_audio_profile(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="failed")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"audio_profile": "singing"})
    assert resp.status_code == 200
    queue = repo.list_queue(job_id)
    latest = queue[-1]
    assert latest.payload["audio_profile"] == "singing"


def test_retry_rejects_invalid_render_mode(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"render_mode": "bad"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_RENDER_MODE"


def test_retry_rejects_invalid_audio_profile(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"audio_profile": "podcast"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_AUDIO_PROFILE"


def test_retry_rejects_invalid_phase(tmp_path: Path) -> None:
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="completed")
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"phase": "bogus"})
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "INVALID_PHASE"


def test_retry_without_phase_uses_job_phase(tmp_path: Path) -> None:
    """When no phase is supplied the job's own phase is used (backwards compat)."""
    client, repo = _make_client_and_repo(tmp_path)
    job_id = _seed_job(repo, status="failed", phase="render")
    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["phase"] == "render"
