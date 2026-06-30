from pathlib import Path

from core.models.dashboard import (
    CreateDashboardJobRequest,
    DashboardApprovalKind,
    DashboardArtifactKind,
    DashboardEventLevel,
    DashboardQueueStatus,
    DashboardSource,
)
from services.dashboard_repository import DashboardRepository


def _repo(tmp_path: Path) -> DashboardRepository:
    return DashboardRepository(tmp_path / "dashboard.db")


def _request() -> CreateDashboardJobRequest:
    return CreateDashboardJobRequest(
        source=DashboardSource(kind="url", url="https://example.com/video"),
        rights_cleared=True,
        target_language="hi",
        phase="all",
    )


def test_create_queued_job_persists_job_queue_and_event(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    job, queue_item = repo.create_queued_job(_request())

    saved = repo.get_job(job.job_id)
    assert saved is not None
    assert saved.status == "queued"
    assert saved.source_url == "https://example.com/video"
    assert saved.target_language == "hi"
    assert saved.rights_cleared is True

    queue = repo.list_queue(job.job_id)
    assert [item.queue_id for item in queue] == [queue_item.queue_id]
    assert queue[0].status == DashboardQueueStatus.QUEUED

    events = repo.list_events(job.job_id)
    assert len(events) == 1
    assert events[0].event_type == "job_queued"


def test_claim_next_action_marks_queue_item_claimed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    job, queue_item = repo.create_queued_job(_request())

    claimed = repo.claim_next_action("worker-1")

    assert claimed is not None
    assert claimed.queue_id == queue_item.queue_id
    assert claimed.status == DashboardQueueStatus.CLAIMED
    assert claimed.claimed_by == "worker-1"

    saved = repo.list_queue(job.job_id)[0]
    assert saved.status == DashboardQueueStatus.CLAIMED


def test_job_detail_includes_queue_events_and_pending_approval(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    job, _ = repo.create_queued_job(_request())
    approval = repo.create_approval_request(
        job.job_id,
        DashboardApprovalKind.PLAN,
        request={"shots": []},
    )
    repo.record_event(
        job.job_id,
        "approval_requested",
        "Plan approval requested.",
        level=DashboardEventLevel.INFO,
    )

    detail = repo.get_job_detail(job.job_id)

    assert detail is not None
    assert detail.job.job_id == job.job_id
    assert detail.pending_approval is not None
    assert detail.pending_approval.approval_id == approval.approval_id
    assert any(event.event_type == "approval_requested" for event in detail.events)


def test_artifact_and_worker_heartbeat_are_persisted(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    job, _ = repo.create_queued_job(_request())

    artifact = repo.record_artifact(
        job.job_id,
        DashboardArtifactKind.JSON,
        uri=".local/artifacts/example.json",
        stage_name="plan_shots",
        previewable=False,
        metadata={"stage": "plan_shots"},
    )
    heartbeat = repo.heartbeat_worker("worker-1", current_job_id=job.job_id)

    artifacts = repo.list_artifacts(job.job_id)
    assert [item.artifact_id for item in artifacts] == [artifact.artifact_id]
    assert artifacts[0].metadata == {"stage": "plan_shots"}

    latest = repo.latest_worker_heartbeat()
    assert latest is not None
    assert latest.worker_id == heartbeat.worker_id
    assert latest.current_job_id == job.job_id


def test_dashboard_api_module_imports_without_fastapi_dependency() -> None:
    from services import dashboard_api

    assert callable(dashboard_api.create_app)
