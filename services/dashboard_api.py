from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.config import AppConfig, load_app_config
from core.models.dashboard import (
    CreateDashboardJobRequest,
    DashboardJobStatus,
)
from scripts.check_runtime_readiness import (
    CheckResult,
    check_service_health,
    collect_readiness_results,
)
from services.dashboard_repository import DashboardRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _request_id() -> str:
    return f"req_{secrets.token_hex(8)}"


def _base_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": _request_id(),
        "server_time": _utc_now().isoformat(),
        **payload,
    }


def _result_to_dict(result: CheckResult) -> dict[str, str]:
    return {
        "name": result.name,
        "status": result.status.lower(),
        "detail": result.detail,
    }


def _make_repository(config: AppConfig) -> DashboardRepository:
    return DashboardRepository(Path(config.settings.sqlite_path))


def create_app(
    *,
    repository: DashboardRepository | None = None,
    config_loader: Callable[[], AppConfig] = load_app_config,
):
    """Create the dashboard FastAPI app.

    FastAPI is imported lazily so repository/unit tests can run without the
    optional dashboard dependencies installed. Install with:
    `pip install -e ".[dashboard]"`.
    """

    try:
        from fastapi import Depends, FastAPI, HTTPException, Request, status
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "Dashboard API requires FastAPI. Install with `pip install -e \".[dashboard]\"`."
        ) from exc

    config = config_loader()
    repo = repository or _make_repository(config)

    app = FastAPI(title="video_me Dashboard API", version="0.1.0")

    def require_write_auth(request: Request) -> None:
        token = os.getenv("VIDEO_ME_DASHBOARD_TOKEN")
        if not token:
            return
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {token}"
        if not secrets.compare_digest(auth, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "UNAUTHORIZED",
                    "message": "Missing or invalid dashboard bearer token.",
                    "retryable": False,
                },
            )

    @app.get("/api/health/live")
    def live() -> dict[str, Any]:
        return _base_response({"status": "ok"})

    @app.get("/api/health/ready")
    def ready() -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        status_value = "ok"
        try:
            repo.ping()
            checks.append({"name": "database", "status": "ok"})
        except Exception as exc:
            status_value = "fail"
            checks.append({"name": "database", "status": "fail", "message": str(exc)})

        heartbeat = repo.latest_worker_heartbeat()
        if heartbeat is None:
            if status_value != "fail":
                status_value = "degraded"
            checks.append({"name": "worker", "status": "warn", "message": "No worker heartbeat"})
        else:
            age_sec = (_utc_now() - heartbeat.last_heartbeat_at).total_seconds()
            worker_status = "ok" if age_sec <= 120 else "warn"
            if worker_status == "warn" and status_value != "fail":
                status_value = "degraded"
            checks.append(
                {
                    "name": "worker",
                    "status": worker_status,
                    "worker_id": heartbeat.worker_id,
                    "age_sec": age_sec,
                    "current_job_id": heartbeat.current_job_id,
                }
            )

        return _base_response({"status": status_value, "checks": checks})

    @app.get("/api/runtime/readiness")
    def runtime_readiness(
        strict: bool = True,
        skip_services: bool = False,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        results = collect_readiness_results(
            config,
            code_test=not strict,
            skip_services=skip_services,
            allow_missing_services=not strict,
            timeout=timeout,
        )
        has_fail = any(result.status == "FAIL" for result in results)
        has_warn = any(result.status == "WARN" for result in results)
        readiness_status = "fail" if has_fail else "warn" if has_warn else "ok"
        return _base_response(
            {
                "mode": "strict" if strict else "code_test",
                "status": readiness_status,
                "checks": [_result_to_dict(result) for result in results],
            }
        )

    @app.get("/api/runtime/services")
    def runtime_services(timeout: float = 3.0) -> dict[str, Any]:
        results = check_service_health(
            config.settings,
            timeout=timeout,
            allow_missing_services=True,
        )
        return _base_response({"services": [_result_to_dict(result) for result in results]})

    @app.get("/api/config/defaults")
    def config_defaults() -> dict[str, Any]:
        settings = config.settings
        return _base_response(
            {
                "render_adapter": settings.render_adapter,
                "video_adapter": settings.video_adapter,
                "tts_adapter": settings.tts_adapter,
                "target_language": settings.target_language,
                "image_candidates": settings.image_candidates,
                "approval_required": not (
                    settings.auto_approve_plan and settings.auto_approve_images
                ),
                "max_gpu_jobs": 1,
            }
        )

    @app.post("/api/jobs")
    def create_job(
        request: CreateDashboardJobRequest,
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        if not request.rights_cleared and request.phase != "noop":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "RIGHTS_NOT_CLEARED",
                    "message": "Confirm rights clearance before queueing a real pipeline job.",
                    "retryable": False,
                },
            )

        job, queue_item = repo.create_queued_job(request)
        return _base_response(
            {
                "job_id": job.job_id,
                "status": job.status.value,
                "queue_id": queue_item.queue_id,
                "links": {
                    "detail": f"/api/jobs/{job.job_id}",
                    "events": f"/api/jobs/{job.job_id}/events",
                    "stream": f"/api/jobs/{job.job_id}/stream",
                },
            }
        )

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50) -> dict[str, Any]:
        jobs = repo.list_jobs(limit=limit)
        return _base_response(
            {"items": [job.model_dump(mode="json") for job in jobs], "next_cursor": None}
        )

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        detail = repo.get_job_detail(job_id)
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "JOB_NOT_FOUND",
                    "message": f"Dashboard job not found: {job_id}",
                    "retryable": False,
                },
            )
        return _base_response(detail.model_dump(mode="json"))

    @app.get("/api/jobs/{job_id}/events")
    def get_job_events(
        job_id: str,
        after_event_id: int = 0,
        limit: int = 200,
    ) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "JOB_NOT_FOUND",
                    "message": f"Dashboard job not found: {job_id}",
                    "retryable": False,
                },
            )
        events = repo.list_events(job_id, after_event_id=after_event_id, limit=limit)
        latest = events[-1].event_id if events else after_event_id
        return _base_response(
            {
                "items": [event.model_dump(mode="json") for event in events],
                "latest_event_id": latest,
            }
        )

    @app.get("/api/jobs/{job_id}/artifacts")
    def get_job_artifacts(job_id: str) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "JOB_NOT_FOUND",
                    "message": f"Dashboard job not found: {job_id}",
                    "retryable": False,
                },
            )
        artifacts = repo.list_artifacts(job_id)
        return _base_response(
            {"items": [artifact.model_dump(mode="json") for artifact in artifacts]}
        )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(
        job_id: str,
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "JOB_NOT_FOUND",
                    "message": f"Dashboard job not found: {job_id}",
                    "retryable": False,
                },
            )
        job = repo.update_job_status(job_id, DashboardJobStatus.CANCEL_REQUESTED)
        repo.record_event(job_id, "cancel_requested", "Cancellation requested from dashboard API.")
        return _base_response({"job_id": job.job_id, "status": job.status.value})

    return app
