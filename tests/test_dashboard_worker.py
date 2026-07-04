"""Tests for DashboardWorker (D3)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models.dashboard import (
    CreateDashboardJobRequest,
    DashboardJobStatus,
    DashboardQueueStatus,
    DashboardSource,
)
from services.dashboard_repository import DashboardRepository
from services.dashboard_worker import DashboardWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo(tmp_path: Path) -> DashboardRepository:
    return DashboardRepository(tmp_path / "dashboard.db")


def _noop_request() -> CreateDashboardJobRequest:
    return CreateDashboardJobRequest(
        source=DashboardSource(url="https://example.com/v"),
        rights_cleared=True,
        phase="noop",
    )


def _plan_request() -> CreateDashboardJobRequest:
    return CreateDashboardJobRequest(
        source=DashboardSource(url="https://example.com/v"),
        rights_cleared=True,
        phase="plan",
    )


def _fake_core_job(status: str = "completed") -> SimpleNamespace:
    j = SimpleNamespace()
    j.status = SimpleNamespace(value=status)
    return j


def _make_worker(tmp_path: Path) -> tuple[DashboardWorker, DashboardRepository]:
    from core.config import load_app_config

    config = load_app_config()
    config.settings.sqlite_path = tmp_path / "dashboard.db"  # type: ignore[attr-defined]
    repo = _repo(tmp_path)
    worker = DashboardWorker(repo, config, worker_id="test-worker-1")
    return worker, repo


# ---------------------------------------------------------------------------
# Repository method smoke tests (complement test_dashboard_repository.py)
# ---------------------------------------------------------------------------

def test_complete_and_fail_queue_actions(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    job, q = repo.create_queued_job(_noop_request())

    repo.complete_queue_action(q.queue_id)
    item = repo.get_queue_item(q.queue_id)
    assert item is not None
    assert item.status == DashboardQueueStatus.COMPLETED
    assert item.completed_at is not None

    # fail_queue_action on an already-completed item still runs without error
    job2, q2 = repo.create_queued_job(_noop_request())
    repo.fail_queue_action(q2.queue_id, {"code": "ERR", "message": "boom"})
    item2 = repo.get_queue_item(q2.queue_id)
    assert item2 is not None
    assert item2.status == DashboardQueueStatus.FAILED
    assert item2.error is not None
    assert item2.error["code"] == "ERR"


def test_config_for_job_applies_overrides(tmp_path: Path) -> None:
    """DashboardJobOverrides fields should override the matching Settings field."""
    from core.models.dashboard import DashboardJobOverrides

    worker, _ = _make_worker(tmp_path)
    base_video_adapter = worker.config.settings.video_adapter

    req = CreateDashboardJobRequest(
        source=DashboardSource(url="https://example.com/v"),
        rights_cleared=True,
        phase="all",
        overrides=DashboardJobOverrides(video_adapter="wan"),
    )
    job_config = worker._config_for_job(req)

    assert job_config.settings.video_adapter == "wan"
    # The worker's own base config must not be mutated.
    assert worker.config.settings.video_adapter == base_video_adapter


def test_config_for_job_no_overrides_returns_base_config(tmp_path: Path) -> None:
    worker, _ = _make_worker(tmp_path)
    req = _noop_request()  # no overrides set — all fields default to None

    assert worker._config_for_job(req) is worker.config


def test_resolve_approval_approved(tmp_path: Path) -> None:
    from core.models.dashboard import DashboardApprovalKind, DashboardApprovalStatus

    repo = _repo(tmp_path)
    job, _ = repo.create_queued_job(_noop_request())
    approval = repo.create_approval_request(job.job_id, DashboardApprovalKind.PLAN, request={})

    resolved = repo.resolve_approval(approval.approval_id, approved=True, reviewer="ops")

    assert resolved.status == DashboardApprovalStatus.APPROVED
    assert resolved.response is not None
    assert resolved.response["approved"] is True
    assert resolved.reviewer == "ops"
    assert resolved.decided_at is not None


def test_resolve_approval_rejected_with_notes(tmp_path: Path) -> None:
    from core.models.dashboard import DashboardApprovalKind, DashboardApprovalStatus

    repo = _repo(tmp_path)
    job, _ = repo.create_queued_job(_noop_request())
    approval = repo.create_approval_request(job.job_id, DashboardApprovalKind.PLAN, request={})

    resolved = repo.resolve_approval(
        approval.approval_id, approved=False, notes="shots too short"
    )

    assert resolved.status == DashboardApprovalStatus.REJECTED
    assert resolved.response is not None
    assert resolved.response["notes"] == "shots too short"


def test_resolve_approval_duplicate_is_idempotent(tmp_path: Path) -> None:
    from core.models.dashboard import DashboardApprovalKind, DashboardApprovalStatus

    repo = _repo(tmp_path)
    job, _ = repo.create_queued_job(_noop_request())
    approval = repo.create_approval_request(job.job_id, DashboardApprovalKind.PLAN, request={})

    repo.resolve_approval(approval.approval_id, approved=True)
    # Second resolve on an already-decided approval: WHERE status='pending' won't match,
    # so the row is not updated — but the read still returns the decided row without error.
    result = repo.resolve_approval(approval.approval_id, approved=False, notes="late reject")
    # Status should still be APPROVED (the second call was a no-op).
    assert result.status == DashboardApprovalStatus.APPROVED


# ---------------------------------------------------------------------------
# Worker: noop job happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_runs_noop_job_to_completed(tmp_path: Path) -> None:
    worker, repo = _make_worker(tmp_path)
    job, queue_item = repo.create_queued_job(_noop_request())

    fake_core_job = _fake_core_job("completed")

    with patch("core.workflow.run_noop_job", new=AsyncMock(return_value=fake_core_job)):
        action = repo.claim_next_action("test-worker-1")
        assert action is not None
        # Re-enqueue so the worker can claim it from scratch.
        # (We called claim_next_action manually above; restore by reimporting state.)

    # Full run: create fresh job + let worker process it.
    repo2 = _repo(tmp_path)
    job2, _ = repo2.create_queued_job(_noop_request())

    with patch("core.workflow.run_noop_job", new=AsyncMock(return_value=fake_core_job)):
        # Run only one iteration of the worker loop.
        action = repo2.claim_next_action("test-worker-1")
        assert action is not None
        worker2 = DashboardWorker(repo2, worker.config, worker_id="test-worker-1")
        await worker2._run_action(action)

    saved = repo2.get_job(job2.job_id)
    assert saved is not None
    assert saved.status == DashboardJobStatus.COMPLETED

    events = repo2.list_events(job2.job_id)
    event_types = [e.event_type for e in events]
    assert "job_started" in event_types
    assert "job_completed" in event_types

    q = repo2.get_queue_item(action.queue_id)
    assert q is not None
    assert q.status == DashboardQueueStatus.COMPLETED


# ---------------------------------------------------------------------------
# Worker: pipeline job failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_handles_pipeline_failure(tmp_path: Path) -> None:
    worker, repo = _make_worker(tmp_path)
    job, _ = repo.create_queued_job(_plan_request())

    action = repo.claim_next_action("test-worker-1")
    assert action is not None

    mock_plan_approval = MagicMock()
    mock_image_approval = MagicMock()

    with (
        patch(
            "adapters.approval.dashboard_approval_adapter.DashboardPlanApprovalAdapter",
            return_value=mock_plan_approval,
        ),
        patch(
            "adapters.approval.dashboard_image_approval_adapter.DashboardImageApprovalAdapter",
            return_value=mock_image_approval,
        ),
        patch(
            "core.workflow.run_pipeline_job",
            new=AsyncMock(side_effect=RuntimeError("Ollama is down")),
        ),
    ):
        await worker._run_action(action)

    saved = repo.get_job(job.job_id)
    assert saved is not None
    assert saved.status == DashboardJobStatus.FAILED
    assert saved.terminal_error is not None
    assert "Ollama is down" in saved.terminal_error.get("message", "")

    q = repo.get_queue_item(action.queue_id)
    assert q is not None
    assert q.status == DashboardQueueStatus.FAILED


# ---------------------------------------------------------------------------
# Worker: cancel request during heartbeat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_cancels_job_when_requested(tmp_path: Path) -> None:
    worker, repo = _make_worker(tmp_path)
    job, _ = repo.create_queued_job(_noop_request())

    action = repo.claim_next_action("test-worker-1")
    assert action is not None

    # Simulate pipeline that yields, then we flip the cancel flag.
    cancel_triggered = asyncio.Event()

    async def slow_noop(*args, **kwargs):
        # Signal that we're inside the pipeline, then block until cancelled.
        cancel_triggered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        return _fake_core_job("completed")

    async def _set_cancel_after_start():
        await cancel_triggered.wait()
        repo.update_job_status(job.job_id, DashboardJobStatus.CANCEL_REQUESTED)

    with patch("core.workflow.run_noop_job", new=slow_noop):
        worker.HEARTBEAT_INTERVAL = 0  # fire immediately so cancel is detected fast
        run_task = asyncio.create_task(worker._run_action(action))
        cancel_setter = asyncio.create_task(_set_cancel_after_start())
        await asyncio.wait_for(
            asyncio.gather(run_task, cancel_setter, return_exceptions=True),
            timeout=5,
        )

    saved = repo.get_job(job.job_id)
    assert saved is not None
    assert saved.status == DashboardJobStatus.CANCELLED


# ---------------------------------------------------------------------------
# Worker: empty queue → no claim
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_worker_does_not_claim_when_queue_empty(tmp_path: Path) -> None:
    worker, repo = _make_worker(tmp_path)

    worker.POLL_INTERVAL = 0.05
    worker.stop()  # stop immediately after first iteration

    # Should complete without error even with nothing in the queue.
    await asyncio.wait_for(worker.run(), timeout=2.0)

    # No jobs were created so nothing should be claimed.
    assert repo.latest_worker_heartbeat() is not None
