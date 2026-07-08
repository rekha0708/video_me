from __future__ import annotations

import json
import os
import secrets
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from fastapi import File, Form, UploadFile
except ModuleNotFoundError:  # allow import for helper tests without fastapi
    File = Form = UploadFile = None  # type: ignore[assignment,misc]

from core.config import AppConfig, load_app_config, load_yaml_model
from core.storage import create_artifact_store
from core.models.dashboard import (
    ChatRequest,
    CreateDashboardJobRequest,
    DashboardApprovalStatus,
    DashboardJobStatus,
    DashboardQueueAction,
)
from scripts.check_runtime_readiness import (
    CheckResult,
    check_service_health,
    collect_readiness_results,
)
from services.dashboard_repository import DashboardRepository

_TERMINAL_STATUSES = {
    DashboardJobStatus.COMPLETED,
    DashboardJobStatus.FAILED,
    DashboardJobStatus.BLOCKED,
    DashboardJobStatus.CANCELLED,
}


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


# Pipeline stage → macro phase, for deriving stepper state on phase="all" jobs.
# "video_model_load" is the synthetic stage emitted by core/gpu_sequencer.py.
_STAGE_TO_MACRO = {
    "fetch_media": "transcribe",
    "transcribe": "transcribe",
    "analyze_content": "transcribe",
    "adapt_script": "script_plan",
    "plan_shots": "script_plan",
    "render_overlays": "script_plan",
    "render_character": "render",
    "synthesize_voice": "render",
    "generate_video": "render",
    "lip_sync": "render",
    "video_model_load": "render",
    "assemble_video": "assemble",
    "publish": "assemble",
}

_MACRO_ORDER = ["transcribe", "script_plan", "render", "assemble"]


def _workspace_path(path_value: str, *, cwd: Path | None = None) -> Path:
    """Map training TOML paths from /workspace/video_me to this checkout."""
    cwd = (cwd or Path.cwd()).resolve()
    if path_value.startswith("/workspace/video_me/"):
        return cwd / path_value.removeprefix("/workspace/video_me/")
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else cwd / path


def _find_lora_config_path(member_id: str, *, cwd: Path | None = None) -> Path:
    cwd = (cwd or Path.cwd()).resolve()
    normalized = member_id.strip().lower()
    matches = sorted(cwd.glob(f"assets/**/training/musubi_dataset_{normalized}.toml"))
    if not matches:
        matches = sorted(cwd.glob(f"assets/**/training/kohya_config_{normalized}.toml"))
    if not matches:
        raise FileNotFoundError(
            f"No LoRA training config found for member '{member_id}' "
            f"(expected assets/**/training/musubi_dataset_{normalized}.toml "
            f"or kohya_config_{normalized}.toml)."
        )
    return matches[0]


def _lora_dataset_image_dir(config_path: Path, *, cwd: Path | None = None) -> Path:
    data = tomllib.loads(config_path.read_text())
    for dataset in data.get("datasets", []):
        image_dir = dataset.get("image_directory")
        if image_dir:
            return _workspace_path(str(image_dir), cwd=cwd)
        for subset in dataset.get("subsets", []):
            image_dir = subset.get("image_dir")
            if image_dir:
                return _workspace_path(str(image_dir), cwd=cwd)
    for group in data.get("dataset", {}).get("general", []):
        for subset in group.get("subsets", []):
            image_dir = subset.get("image_dir")
            if image_dir:
                return _workspace_path(str(image_dir), cwd=cwd)
    raise ValueError(f"No dataset image directory found in {config_path}")


def _next_training_image_path(image_dir: Path, member_id: str, suffix: str) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    prefix = member_id.strip().lower()
    max_seen = 0
    for image in image_dir.iterdir():
        if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        stem = image.stem.lower()
        if not stem.startswith(prefix):
            continue
        tail = stem.removeprefix(prefix).lstrip("_-")
        if tail.isdigit():
            max_seen = max(max_seen, int(tail))
    return image_dir / f"{prefix}_{max_seen + 1:03d}{suffix.lower()}"


def _artifact_flags(artifact_store: Any, work_dir: Path, job_id: str) -> dict[str, bool]:
    """Which artifact cards the job page should show, based on what actually exists.

    Robust to how the job was run (phased vs phase="all" vs story-seeded) because it
    looks at artifacts, not phase bookkeeping.
    """
    has_plan = artifact_store.has(job_id, "plan_shots")
    visuals_data = artifact_store.get_json(job_id, "analyze_visuals")
    return {
        "transcript": artifact_store.has(job_id, "transcribe"),
        # Only show when the VLM actually found settings (empty for story jobs).
        "visuals": bool(visuals_data and visuals_data.get("segments")),
        "script": artifact_store.has(job_id, "adapt_script") or has_plan,
        "renders": has_plan
        and ((work_dir / "renders").exists() or (work_dir / "user_images").exists()),
        "video": (work_dir / "assembled" / "final.mp4").exists(),
    }


def _stepper_state(job: Any, flags: dict[str, bool]) -> dict[str, Any]:
    """Derive the phase stepper's active phase + completed phases.

    Phased jobs pass through unchanged. For phase="all" jobs — whose
    completed_phases contains only "all" and whose phase never matches a
    stepper node — the active macro phase comes from current_stage and the
    completed set from artifact existence.
    """
    status = getattr(job.status, "value", str(job.status))
    if job.phase != "all":
        return {"phase": job.phase, "completed": list(job.completed_phases or [])}

    if status == "completed":
        return {"phase": "assemble", "completed": list(_MACRO_ORDER)}

    active = _STAGE_TO_MACRO.get(job.current_stage or "")
    if active is None:
        # No stage recorded yet (e.g. queued) — infer from artifacts.
        if flags["video"]:
            active = "assemble"
        elif flags["renders"]:
            active = "render"
        elif flags["script"] or flags["transcript"]:
            active = "script_plan"
        else:
            active = "transcribe"
    return {"phase": active, "completed": _MACRO_ORDER[: _MACRO_ORDER.index(active)]}


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
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, status
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "Dashboard API requires FastAPI. Install with `pip install -e \".[dashboard]\"`."
        ) from exc

    config = config_loader()
    repo = repository or _make_repository(config)
    artifact_store = create_artifact_store(config.settings)

    app = FastAPI(title="video_me Dashboard API", version="0.1.0")

    def require_write_auth(
        authorization: str | None = Header(default=None),
    ) -> None:
        token = os.getenv("VIDEO_ME_DASHBOARD_TOKEN")
        if not token:
            return
        if authorization is None or not secrets.compare_digest(
            authorization, f"Bearer {token}"
        ):
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

    _VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    @app.get("/api/local-videos")
    def list_local_videos(dir: str | None = None) -> dict[str, Any]:
        if dir:
            video_dir = Path(dir).expanduser().resolve()
        else:
            video_dir = config.settings.local_video_dir.expanduser().resolve()
        if not video_dir.exists():
            return _base_response({"videos": [], "dir": str(video_dir), "error": "Directory not found"})
        if not video_dir.is_dir():
            return _base_response({"videos": [], "dir": str(video_dir), "error": "Path is not a directory"})
        videos = []
        for f in sorted(video_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _VIDEO_EXTENSIONS:
                stat = f.stat()
                videos.append({
                    "name": f.name,
                    "uri": f"file://{f}",
                    "size_mb": round(stat.st_size / 1_048_576, 1),
                })
        return _base_response({"videos": videos, "dir": str(video_dir)})

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    @app.get("/api/casts")
    def list_casts() -> dict[str, Any]:
        from core.models.profile import Cast

        casts_dir = Path("config/casts")
        result = []
        if casts_dir.is_dir():
            for yml in sorted(casts_dir.glob("*.yaml")):
                try:
                    cast = load_yaml_model(yml, Cast)
                    result.append({
                        "id": cast.id,
                        "species": cast.species,
                        "member_count": len(cast.members),
                        "members": [{"id": m.id, "name": m.name} for m in cast.members],
                    })
                except Exception:
                    pass
        return _base_response({"casts": result, "default": config.cast.id})

    @app.get("/api/local-images")
    def list_local_images(dir: str | None = None) -> dict[str, Any]:
        import base64 as b64

        if not dir:
            return _base_response({"images": [], "dir": "", "error": "Provide a dir parameter"})
        image_dir = Path(dir).expanduser().resolve()
        if not image_dir.exists():
            return _base_response({"images": [], "dir": str(image_dir), "error": "Directory not found"})
        if not image_dir.is_dir():
            return _base_response({"images": [], "dir": str(image_dir), "error": "Path is not a directory"})
        data_root = Path(config.settings.data_dir).resolve()
        images = []
        for f in sorted(image_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS:
                entry: dict[str, Any] = {
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(f.stat().st_size / 1_048_576, 1),
                }
                try:
                    f.resolve().relative_to(data_root)
                    entry["path_b64"] = b64.urlsafe_b64encode(str(f).encode()).decode()
                except ValueError:
                    pass
                images.append(entry)
        return _base_response({"images": images, "dir": str(image_dir)})

    @app.post("/api/uploads/character-image")
    async def upload_character_image(
        member_id: str = Form(...),
        file: UploadFile = File(...),
        cast_ref: str = Form(default=""),
    ) -> dict[str, Any]:
        if cast_ref:
            from core.models.profile import Cast
            cast_path = Path(f"config/casts/{cast_ref}.yaml")
            if not cast_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "UNKNOWN_CAST", "message": f"No cast config for '{cast_ref}'", "retryable": False},
                )
            target_cast = load_yaml_model(cast_path, Cast)
            cast_ids = {m.id for m in target_cast.members}
        else:
            cast_ids = {m.id for m in config.cast.members}
        if member_id not in cast_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_MEMBER", "message": f"Unknown cast member: {member_id}. Expected: {sorted(cast_ids)}", "retryable": False},
            )
        ext = Path(file.filename or "img.png").suffix.lower()
        if ext not in _IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_FORMAT", "message": f"Image must be PNG, JPG, or WebP (got {ext})", "retryable": False},
            )
        content = await file.read()
        if len(content) > 10 * 1_048_576:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "FILE_TOO_LARGE", "message": "Image must be under 10 MB", "retryable": False},
            )
        import uuid
        token = uuid.uuid4().hex[:12]
        dest_dir = Path(config.settings.data_dir) / "uploads" / token
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{member_id}{ext}"
        dest.write_bytes(content)
        return _base_response({"path": str(dest), "member_id": member_id, "filename": dest.name})

    @app.post("/api/uploads/lora-training-image")
    async def upload_lora_training_image(
        member_id: str = Form(...),
        file: UploadFile = File(...),
        cast_ref: str = Form(default=""),
        caption: str = Form(default=""),
    ) -> dict[str, Any]:
        from core.models.profile import Cast

        target_cast = config.cast
        if cast_ref and cast_ref != config.cast.id:
            cast_path = Path(f"config/casts/{cast_ref}.yaml")
            if not cast_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "UNKNOWN_CAST", "message": f"No cast config for '{cast_ref}'", "retryable": False},
                )
            target_cast = load_yaml_model(cast_path, Cast)

        member = next((m for m in target_cast.members if m.id == member_id), None)
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_MEMBER", "message": f"Unknown cast member: {member_id}", "retryable": False},
            )

        ext = Path(file.filename or "img.png").suffix.lower()
        if ext not in _IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "INVALID_FORMAT", "message": f"Image must be PNG, JPG, or WebP (got {ext})", "retryable": False},
            )
        content = await file.read()
        if len(content) > 25 * 1_048_576:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "FILE_TOO_LARGE", "message": "Image must be under 25 MB", "retryable": False},
            )

        try:
            config_path = _find_lora_config_path(member_id)
            image_dir = _lora_dataset_image_dir(config_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "LORA_CONFIG_MISSING", "message": str(exc), "retryable": False},
            ) from exc

        dest = _next_training_image_path(image_dir, member_id, ext)
        dest.write_bytes(content)

        trigger = str(member.lora_ref).replace("\\", "/").rstrip("/").split("/")[-1]
        final_caption = caption.strip() or f"{target_cast.id}_{trigger}, {member.visual_descriptor.strip()}"
        dest.with_suffix(".txt").write_text(final_caption + "\n")

        return _base_response({
            "path": str(dest),
            "caption_path": str(dest.with_suffix(".txt")),
            "member_id": member_id,
            "config_path": str(config_path),
        })

    @app.post("/api/jobs")
    def create_job(
        body: CreateDashboardJobRequest,
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        if not body.rights_cleared and body.phase != "noop":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "RIGHTS_NOT_CLEARED",
                    "message": "Confirm rights clearance before queueing a real pipeline job.",
                    "retryable": False,
                },
            )

        if body.source.kind in ("story", "story_images") and body.phase in (
            "script_plan", "render", "assemble",
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_PHASE_FOR_STORY",
                    "message": (
                        "Story jobs must start at 'transcribe' (analyze story) or 'all'. "
                        "Use the Advance button to continue from a later phase."
                    ),
                    "retryable": False,
                },
            )

        if body.source.kind == "story_images":
            # Fail fast with a clean 400 instead of a deep worker crash: every
            # image key must be a member of the selected cast and its file must
            # exist on disk (uploaded via /api/uploads/character-image).
            from core.models.profile import Cast

            if body.cast_ref and body.cast_ref != config.cast.id:
                cast_path = Path(f"config/casts/{body.cast_ref}.yaml")
                if not cast_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"code": "UNKNOWN_CAST",
                                "message": f"No cast config for '{body.cast_ref}'",
                                "retryable": False},
                    )
                cast_ids = {m.id for m in load_yaml_model(cast_path, Cast).members}
            else:
                cast_ids = {m.id for m in config.cast.members}

            for member_id, image_path in body.character_images.items():
                if member_id not in cast_ids:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"code": "INVALID_MEMBER",
                                "message": f"Unknown cast member: {member_id}. "
                                           f"Expected: {sorted(cast_ids)}",
                                "retryable": False},
                    )
                if not Path(image_path).is_file():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"code": "BAD_CHARACTER_IMAGE",
                                "message": f"Image for '{member_id}' not found: {image_path}",
                                "retryable": False},
                    )

        if body.phase == "lora_train":
            from core.models.profile import Cast

            if body.cast_ref and body.cast_ref != config.cast.id:
                cast_path = Path(f"config/casts/{body.cast_ref}.yaml")
                if not cast_path.exists():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"code": "UNKNOWN_CAST",
                                "message": f"No cast config for '{body.cast_ref}'",
                                "retryable": False},
                    )
                cast_ids = {m.id for m in load_yaml_model(cast_path, Cast).members}
            else:
                cast_ids = {m.id for m in config.cast.members}

            training = body.lora_training
            assert training is not None
            member_id = training.cast_member_id
            if member_id not in cast_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "INVALID_MEMBER",
                            "message": f"Unknown cast member: {member_id}. Expected: {sorted(cast_ids)}",
                            "retryable": False},
                )
            try:
                _find_lora_config_path(member_id)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "LORA_CONFIG_MISSING", "message": str(exc), "retryable": False},
                ) from exc
            for image_path in training.image_paths:
                if not Path(image_path).is_file():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"code": "BAD_LORA_IMAGE",
                                "message": f"Training image not found: {image_path}",
                                "retryable": False},
                    )

        job, queue_item = repo.create_queued_job(body)
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

    @app.get("/api/jobs/{job_id}/transcript")
    def get_transcript(job_id: str) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        data = artifact_store.get_json(job_id, "transcribe")
        if data is None:
            return _base_response({"transcript": None, "message": "Transcript artifact not yet available."})
        segments = data.get("segments", [])
        return _base_response({
            "transcript": {
                "language": data.get("language", ""),
                "full_text": data.get("full_text", ""),
                "duration": round(segments[-1]["end"], 1) if segments else 0,
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in segments
                ],
            }
        })

    @app.get("/api/jobs/{job_id}/script")
    def get_script(job_id: str) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        data = artifact_store.get_json(job_id, "adapt_script")
        if data is None:
            return _base_response({"script": None, "message": "Script not yet available."})
        return _base_response({
            "script": {
                "mode": data.get("mode", ""),
                "learning_objective": data.get("learning_objective", {}),
                "caption_text": data.get("caption_text", ""),
                "scenes": data.get("scenes", []),
            }
        })

    @app.get("/api/jobs/{job_id}/plan")
    def get_plan(job_id: str) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        data = artifact_store.get_json(job_id, "plan_shots")
        if data is None:
            return _base_response({"plan": None, "message": "Storyboard not yet available."})
        import base64
        shots = data.get("shots", [])
        for shot in shots:
            png_uri = (shot.get("overlay") or {}).get("png_uri")
            if png_uri:
                shot["overlay_png_b64"] = base64.urlsafe_b64encode(str(png_uri).encode()).decode()
        return _base_response({"plan": {"shots": shots}})

    @app.get("/api/jobs/{job_id}/visuals")
    def get_visuals(job_id: str) -> dict[str, Any]:
        """Per-segment settings the VLM extracted from the source video.

        Lets the operator verify the observed backgrounds before the expensive
        Flux render stage. Empty for story jobs / when extraction found nothing.
        """
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        data = artifact_store.get_json(job_id, "analyze_visuals")
        if data is None:
            return _base_response({"visuals": None, "message": "Visual analysis not run for this job."})
        return _base_response({
            "visuals": {
                "summary": data.get("summary", ""),
                "segments": data.get("segments", []),
            }
        })

    @app.get("/api/jobs/{job_id}/renders")
    def get_renders(job_id: str) -> dict[str, Any]:
        import base64

        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        storyboard_data = artifact_store.get_json(job_id, "plan_shots")
        if storyboard_data is None:
            return _base_response({"shots": [], "message": "Storyboard not yet available."})

        work_dir = Path(config.settings.data_dir) / "jobs" / job_id
        shots_out: list[dict[str, Any]] = []
        for shot in storyboard_data.get("shots", []):
            shot_id = shot.get("shot_id", "")
            speakers = shot.get("characters_on_screen") or []
            speaker_id = speakers[0] if speakers else None

            candidate_uris: list[str] = []
            if speaker_id:
                # Renders are shot-scoped (renders/{shot_id}/{speaker_id}/); fall
                # back to the legacy per-speaker path for pre-existing jobs.
                render_dir = work_dir / "renders" / shot_id / speaker_id
                if not render_dir.is_dir():
                    render_dir = work_dir / "renders" / speaker_id
                candidate_uris = [
                    base64.urlsafe_b64encode(str(p).encode()).decode()
                    for p in sorted(render_dir.glob("render_??.png"))
                ]

            winner_index = None
            reasoning = ""
            critique_path = work_dir / "critique" / f"{shot_id}.json"
            if critique_path.exists():
                critique_data = json.loads(critique_path.read_text())
                winner_index = critique_data.get("winner_index")
                reasoning = critique_data.get("overall_reasoning", "")

            if not candidate_uris and speaker_id:
                # Story+images job: no Flux renders — show the user-provided
                # reference image for this shot's speaker instead.
                user_images = sorted((work_dir / "user_images").glob(f"{speaker_id}.*"))
                if user_images:
                    candidate_uris = [
                        base64.urlsafe_b64encode(str(user_images[0]).encode()).decode()
                    ]
                    winner_index = 0
                    reasoning = reasoning or "User-provided reference image."

            shots_out.append({
                "shot_id": shot_id,
                "speaker": speaker_id,
                "setting": shot.get("setting", ""),
                "action": shot.get("action", ""),
                "candidate_paths_b64": candidate_uris,
                "winner_index": winner_index,
                "reasoning": reasoning,
            })
        return _base_response({"shots": shots_out})

    @app.get("/api/jobs/{job_id}/video")
    def get_video(job_id: str) -> dict[str, Any]:
        import base64

        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        work_dir = Path(config.settings.data_dir) / "jobs" / job_id
        video_path = work_dir / "assembled" / "final.mp4"
        if not video_path.exists():
            return _base_response({"available": False})
        encoded = base64.urlsafe_b64encode(str(video_path).encode()).decode()
        return _base_response({"available": True, "path_b64": encoded})

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

    # ------------------------------------------------------------------
    # D5 — Approval endpoints
    # ------------------------------------------------------------------

    @app.get("/api/jobs/{job_id}/approval")
    def get_job_approval(job_id: str) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        approval = repo.get_pending_approval(job_id)
        if approval is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NO_PENDING_APPROVAL",
                        "message": "No pending approval for this job.",
                        "retryable": False},
            )
        return _base_response(approval.model_dump(mode="json"))

    @app.post("/api/jobs/{job_id}/approve")
    def approve_job(
        job_id: str,
        body: dict[str, Any],
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        approval = repo.get_pending_approval(job_id)
        if approval is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "NO_PENDING_APPROVAL",
                        "message": "No pending approval to approve.",
                        "retryable": False},
            )
        if approval.status != DashboardApprovalStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "ALREADY_DECIDED",
                        "message": f"Approval already {approval.status.value}.",
                        "retryable": False},
            )
        reviewer: str | None = body.get("reviewer")
        picks: dict[str, Any] | None = body.get("picks")
        resolved = repo.resolve_approval(
            approval.approval_id, approved=True, picks=picks, reviewer=reviewer
        )
        repo.record_event(job_id, "approval_granted", "Operator approved from dashboard.",
                          payload={"approval_id": resolved.approval_id})
        return _base_response({"approval_id": resolved.approval_id, "status": resolved.status.value})

    @app.post("/api/jobs/{job_id}/reject")
    def reject_job(
        job_id: str,
        body: dict[str, Any],
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        approval = repo.get_pending_approval(job_id)
        if approval is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "NO_PENDING_APPROVAL",
                        "message": "No pending approval to reject.",
                        "retryable": False},
            )
        if approval.status != DashboardApprovalStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "ALREADY_DECIDED",
                        "message": f"Approval already {approval.status.value}.",
                        "retryable": False},
            )
        notes: str = body.get("notes", "")
        reviewer: str | None = body.get("reviewer")
        resolved = repo.resolve_approval(
            approval.approval_id, approved=False, notes=notes, reviewer=reviewer
        )
        repo.record_event(job_id, "approval_rejected", f"Operator rejected: {notes}",
                          payload={"approval_id": resolved.approval_id, "notes": notes})
        return _base_response({"approval_id": resolved.approval_id, "status": resolved.status.value})

    # ------------------------------------------------------------------
    # Phase advancement
    # ------------------------------------------------------------------

    _PHASE_SEQUENCE = ["transcribe", "script_plan", "render", "assemble"]

    @app.post("/api/jobs/{job_id}/advance")
    def advance_job_phase(
        job_id: str,
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        """Re-queue the same job for the next phase (e.g. transcribe → script_plan)."""
        job = repo.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        if job.status != DashboardJobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "JOB_NOT_COMPLETE",
                    "message": f"Job must be in 'completed' status to advance (current: {job.status.value}).",
                    "retryable": False,
                },
            )
        try:
            idx = _PHASE_SEQUENCE.index(job.phase)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "NO_NEXT_PHASE",
                    "message": f"Phase '{job.phase}' is not part of the phase sequence or has no next phase.",
                    "retryable": False,
                },
            )
        if idx >= len(_PHASE_SEQUENCE) - 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "NO_NEXT_PHASE",
                    "message": f"Phase '{job.phase}' is the last phase — nothing to advance to.",
                    "retryable": False,
                },
            )
        next_phase = _PHASE_SEQUENCE[idx + 1]

        # Build updated request payload with the new phase.
        new_request = {**job.request, "phase": next_phase}

        # Update the job record's phase + re-queue.
        repo.update_job_phase(job_id, next_phase)
        repo.update_job_status(job_id, DashboardJobStatus.QUEUED)
        queue_item = repo.enqueue_action(
            job_id,
            DashboardQueueAction.RESUME,
            payload=new_request,
        )
        repo.record_event(
            job_id,
            "phase_advanced",
            f"Job advanced from '{job.phase}' to '{next_phase}'.",
            payload={"from_phase": job.phase, "to_phase": next_phase, "queue_id": queue_item.queue_id},
        )
        return _base_response({
            "job_id": job_id,
            "from_phase": job.phase,
            "to_phase": next_phase,
            "queue_id": queue_item.queue_id,
            "status": DashboardJobStatus.QUEUED.value,
        })

    _RETRYABLE_OVERRIDE_KEYS = (
        "video_adapter", "render_adapter", "tts_adapter", "llm_model",
    )

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(
        job_id: str,
        body: dict[str, Any] | None = Body(default=None),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        """Re-queue a failed OR completed job for the SAME phase it was on.

        script_plan/render/assemble already resume from cached per-stage
        artifacts (see core/workflow.py's `_stage()` helper), so re-queuing
        skips whatever already succeeded — a failed job picks up again at
        the stage that failed; a completed phase="all" job skips straight
        past the (already-rendered) images and redoes voice/video/assemble
        fresh. That makes this the fast way to re-test generate_video or
        assemble_video changes without paying for a fresh render.

        Optional JSON body can override video_adapter/render_adapter/
        tts_adapter/llm_model for the re-run (e.g. compare ltx vs wan
        against the same cached character renders) without touching the
        job's original overrides.
        """
        job = repo.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        if job.status not in (DashboardJobStatus.FAILED, DashboardJobStatus.COMPLETED):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "JOB_NOT_RETRYABLE",
                    "message": (
                        "Job must be in 'failed' or 'completed' status to retry "
                        f"(current: {job.status.value})."
                    ),
                    "retryable": False,
                },
            )

        retry_request = {**job.request, "phase": job.phase}
        if body:
            overrides = dict(retry_request.get("overrides") or {})
            for key in _RETRYABLE_OVERRIDE_KEYS:
                value = body.get(key)
                if value:
                    overrides[key] = value
            retry_request["overrides"] = overrides

        repo.update_job_status(job_id, DashboardJobStatus.QUEUED)
        queue_item = repo.enqueue_action(
            job_id,
            DashboardQueueAction.RESUME,
            payload=retry_request,
        )
        repo.record_event(
            job_id,
            "job_retried",
            f"Job re-queued for phase '{job.phase}' (was {job.status.value}).",
            payload={"phase": job.phase, "queue_id": queue_item.queue_id},
        )
        return _base_response({
            "job_id": job_id,
            "phase": job.phase,
            "queue_id": queue_item.queue_id,
            "status": DashboardJobStatus.QUEUED.value,
        })

    # ------------------------------------------------------------------
    # Chat — Pipeline Assistant (per-job, persisted in SQLite)
    # ------------------------------------------------------------------

    @app.post("/api/jobs/{job_id}/chat")
    async def chat_with_job(job_id: str, body: ChatRequest):
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        from fastapi.responses import StreamingResponse
        from services.chat_service import chat_stream

        return StreamingResponse(
            chat_stream(job_id, body.message, repo, config.settings),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/jobs/{job_id}/chat/history")
    def get_chat_history(job_id: str):
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        messages = repo.get_chat_history(job_id, limit=50)
        return _base_response({"messages": [m.model_dump(mode="json") for m in messages]})

    @app.delete("/api/jobs/{job_id}/chat/history")
    def clear_chat_history(job_id: str, _: None = Depends(require_write_auth)):
        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )
        repo.clear_chat_history(job_id)
        return _base_response({"cleared": True})

    # ------------------------------------------------------------------
    # D4 + D7 — SSE live event stream
    # ------------------------------------------------------------------

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job_events(
        job_id: str,
        after_event_id: int = 0,
    ):
        import asyncio as _asyncio

        from fastapi.responses import StreamingResponse

        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}",
                        "retryable": False},
            )

        async def _generate():
            last_id = after_event_id
            idle_ticks = 0
            while True:
                events = repo.list_events(job_id, after_event_id=last_id, limit=50)
                for ev in events:
                    data = json.dumps(ev.model_dump(mode="json"), default=str)
                    yield f"data: {data}\n\n"
                    last_id = ev.event_id
                    idle_ticks = 0
                job_rec = repo.get_job(job_id)
                if job_rec and job_rec.status in _TERMINAL_STATUSES:
                    # Send final status event then close the stream.
                    final = json.dumps({"status": job_rec.status.value}, default=str)
                    yield f"event: done\ndata: {final}\n\n"
                    return
                idle_ticks += 1
                if idle_ticks % 15 == 0:  # every ~30 s (2 s × 15) keep connection alive
                    yield ": keepalive\n\n"
                await _asyncio.sleep(2)

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # D4 — Browser UI routes (Jinja2 templates)
    # ------------------------------------------------------------------

    _templates_dir = Path(__file__).parent / "templates"
    _static_dir = Path(__file__).parent / "static"

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        from starlette.staticfiles import StaticFiles

        _static_dir.mkdir(parents=True, exist_ok=True)
        _templates_dir.mkdir(parents=True, exist_ok=True)

        if _static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        _jinja_env = Environment(
            loader=FileSystemLoader(str(_templates_dir)),
            autoescape=select_autoescape(["html"]),
        )

        def _b64path(path: str) -> str:
            """Encode a local file path for the /img/<b64> route (base64url, no padding issues)."""
            import base64
            return base64.urlsafe_b64encode(str(path).encode()).decode()

        _jinja_env.filters["b64path"] = _b64path

        @app.get("/img/{path_b64}", include_in_schema=False)
        def serve_render_image(path_b64: str):
            import base64
            from fastapi.responses import FileResponse

            try:
                decoded = base64.urlsafe_b64decode(path_b64.encode()).decode()
            except Exception:
                raise HTTPException(status_code=400, detail="invalid path encoding")

            path = Path(decoded).resolve()
            data_dir = Path(config.settings.data_dir).resolve()
            if not path.is_relative_to(data_dir):
                raise HTTPException(status_code=403, detail="path outside data dir")
            if not path.is_file():
                raise HTTPException(status_code=404, detail="image not found")
            return FileResponse(str(path))

        def _render(template_name: str, **ctx_vars: Any):
            from fastapi.responses import HTMLResponse

            tmpl = _jinja_env.get_template(template_name)
            html = tmpl.render(**ctx_vars)
            return HTMLResponse(content=html)

        @app.get("/", include_in_schema=False)
        def ui_jobs_list():
            jobs = repo.list_jobs(limit=100)
            worker_hb = repo.latest_worker_heartbeat()
            return _render("jobs_list.html", jobs=jobs, worker=worker_hb)

        @app.get("/jobs/new", include_in_schema=False)
        def ui_new_job():
            return _render("job_new.html",
                          cast_members=config.cast.members,
                          default_cast_id=config.cast.id)

        @app.get("/jobs/{job_id}", include_in_schema=False)
        def ui_job_detail(job_id: str):
            detail = repo.get_job_detail(job_id, event_limit=200)
            if detail is None:
                from fastapi.responses import HTMLResponse
                return HTMLResponse("<h1>Job not found</h1>", status_code=404)
            work_dir = Path(config.settings.data_dir) / "jobs" / job_id
            flags = _artifact_flags(artifact_store, work_dir, job_id)
            return _render(
                "job_detail.html",
                detail=detail,
                artifact_flags=flags,
                stepper=_stepper_state(detail.job, flags),
            )

        @app.get("/health", include_in_schema=False)
        def ui_health():
            results = collect_readiness_results(
                config,
                code_test=True,
                skip_services=False,
                allow_missing_services=True,
                timeout=3.0,
            )
            worker_hb = repo.latest_worker_heartbeat()
            return _render("health.html", results=results, worker=worker_hb, now=_utc_now())

    except ImportError:
        # jinja2 / starlette not installed — UI routes silently unavailable.
        pass

    return app
