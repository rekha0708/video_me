from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
import hashlib
import hmac
import ipaddress
import json
import mimetypes
import os
import secrets
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

try:
    from fastapi import File, Form, Request, UploadFile
except ModuleNotFoundError:  # allow import for helper tests without fastapi
    File = Form = Request = UploadFile = None  # type: ignore[assignment,misc]

from core.config import AppConfig, load_app_config, load_yaml_model
from core.storage import create_artifact_store
from core.models.dashboard import (
    ChatRequest,
    CreateDashboardJobRequest,
    DashboardEvent,
    DashboardApprovalStatus,
    DashboardAssetKind,
    DashboardJobStatus,
    DashboardQueueAction,
    WAN_ANIMATE_MAX_DRIVER_RANGE_SEC,
)
from core.wan_animate_readiness import (
    wan_animate_model_readiness,
    wan_flux_retarget_readiness,
)
from scripts.check_runtime_readiness import (
    CheckResult,
    check_service_health,
    collect_readiness_results,
)
from services.dashboard_assets import (
    DashboardAssetAccessError,
    DashboardAssetError,
    DashboardAssetKindError,
    DashboardAssetMetadataError,
    DashboardAssetNotFoundError,
    DashboardAssetPathError,
    DashboardAssetQuotaError,
    DashboardAssetStateError,
    DashboardAssetStore,
    collect_animate_asset_requirements,
)
from services.dashboard_media import (
    MediaIngestError,
    copy_local_file_limited,
    download_public_video_url,
    normalize_image,
    probe_video,
    stream_upload,
)
from services.dashboard_repository import DashboardRepository, make_dashboard_job_id
from services.gpu_status import collect_gpu_status
from services.gpu_watermarks import reset_watermarks, update_watermarks

# Min/max GPU/VRAM stats seen since this API process started (or since the
# last POST /api/runtime/gpu-watermarks/reset) — see services/gpu_watermarks.py.
# Deliberately process-lifetime, not persisted: same scope as WorkerHeartbeat
# and other in-memory dashboard state, resets on `restart_dashboard.sh`.
_gpu_watermarks: dict[str, Any] = {}

_TERMINAL_STATUSES = {
    DashboardJobStatus.COMPLETED,
    DashboardJobStatus.FAILED,
    DashboardJobStatus.BLOCKED,
    DashboardJobStatus.CANCELLED,
}

_DASHBOARD_TIME_ZONE = "America/Los_Angeles"
_DASHBOARD_TZ = ZoneInfo(_DASHBOARD_TIME_ZONE)


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


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_dashboard_time(value: Any, fmt: str = "%H:%M:%S") -> str:
    dt = _coerce_datetime(value)
    if dt is None:
        return ""
    return dt.astimezone(_DASHBOARD_TZ).strftime(fmt)


def _event_value(event: DashboardEvent | dict[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


_COST_GROUP_ORDER = [
    "prep",
    "script",
    "render",
    "voice",
    "video",
    "enhance",
    "assemble",
    "training",
    "other",
]

_COST_GROUP_LABELS = {
    "prep": "Prep / transcript",
    "script": "Script / plan",
    "render": "Image render",
    "voice": "Voice / source audio",
    "video": "Video + lip-sync",
    "enhance": "Video enhancement",
    "assemble": "Assembly",
    "training": "Training",
    "other": "Other",
}

_COST_STAGE_GROUPS = {
    "fetch_media": "prep",
    "transcribe": "prep",
    "whisper_model_unload": "prep",
    "analyze_content": "prep",
    "analyze_visuals": "prep",
    "adapt_script": "script",
    "plan_shots": "script",
    "critique_plan": "script",
    "render_overlays": "script",
    "render_loop": "render",
    "render_character": "render",
    "slice_source_audio": "voice",
    "synthesize_voice": "voice",
    "fit_voice_audio": "voice",
    "voice_model_unload": "voice",
    "voice_model_load": "voice",
    "fish_s2_process_start": "voice",
    "fish_s2_process_stop": "voice",
    "video_model_load": "video",
    "video_model_unload": "video",
    "wan_model_unload": "video",
    "video_loop": "video",
    "generate_video": "video",
    "lip_sync": "video",
    "lip_sync_qa": "video",
    "shot_complete": "video",
    "video_enhance": "enhance",
    "assemble_video": "assemble",
    "publish": "assemble",
    "lora_train": "training",
    "animate_gpu_reset": "prep",
    "animate_validate": "prep",
    "wan_animate_service_start": "prep",
    "canonical_look": "render",
    "animate_preprocess": "video",
    "animate_audio": "voice",
    "animate_generate": "video",
    "animate_lipsync": "video",
    "animate_mux": "assemble",
    "animate_export": "enhance",
}


def _request_gpu_price_per_hour(request: Any) -> float:
    if isinstance(request, dict):
        raw = request.get("gpu_price_per_hour", 0.0)
    else:
        raw = getattr(request, "gpu_price_per_hour", 0.0)
    try:
        return max(0.0, float(raw or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def _format_cost(cost: float) -> str:
    return f"${cost:.2f}"


def _pop_stage_start(
    starts: dict[tuple[str, str], list[datetime]],
    stage_name: str,
    shot_id: str,
) -> datetime | None:
    keys = [(stage_name, shot_id), (stage_name, "")]
    for key in keys:
        values = starts.get(key)
        if values:
            return values.pop(0)

    candidates = [
        (key, values[0])
        for key, values in starts.items()
        if key[0] == stage_name and values
    ]
    if not candidates:
        return None
    key, _ = min(candidates, key=lambda item: item[1])
    return starts[key].pop(0)


def _build_cost_summary(
    events: list[DashboardEvent] | list[dict[str, Any]],
    request: Any,
) -> dict[str, Any]:
    price_per_hour = _request_gpu_price_per_hour(request)
    summary: dict[str, Any] = {
        "enabled": price_per_hour > 0,
        "price_per_hour": price_per_hour,
        "price_per_hour_display": _format_cost(price_per_hour),
        "timezone": _DASHBOARD_TIME_ZONE,
        "groups": [],
        "messages": [],
        "total_seconds": 0.0,
        "total_duration": _format_duration(0.0),
        "total_cost": 0.0,
        "total_cost_display": _format_cost(0.0),
        "shot_count": 0,
    }
    if price_per_hour <= 0:
        return summary

    starts: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    groups: dict[str, dict[str, Any]] = {
        key: {
            "key": key,
            "label": _COST_GROUP_LABELS[key],
            "seconds": 0.0,
            "cost": 0.0,
            "shots": set(),
            "stage_count": 0,
        }
        for key in _COST_GROUP_ORDER
    }

    for event in sorted(events, key=lambda ev: int(_event_value(ev, "event_id", 0) or 0)):
        stage_name = _event_value(event, "stage_name")
        event_type = _event_value(event, "event_type")
        if not stage_name or event_type not in {"stage_started", "stage_completed", "stage_failed"}:
            continue
        event_dt = _coerce_datetime(_event_value(event, "created_at"))
        if event_dt is None:
            continue
        shot_id = str(_event_value(event, "shot_id") or "")
        key = (str(stage_name), shot_id)
        if event_type == "stage_started":
            starts[key].append(event_dt)
            continue

        start_dt = _pop_stage_start(starts, str(stage_name), shot_id)
        if start_dt is None:
            continue
        seconds = max((event_dt - start_dt).total_seconds(), 0.0)
        group_key = _COST_STAGE_GROUPS.get(str(stage_name), "other")
        group = groups[group_key]
        group["seconds"] += seconds
        group["stage_count"] += 1
        if shot_id:
            group["shots"].add(shot_id)

    group_list: list[dict[str, Any]] = []
    all_shots: set[str] = set()
    total_seconds = 0.0
    for key in _COST_GROUP_ORDER:
        group = groups[key]
        seconds = float(group["seconds"])
        if seconds <= 0:
            continue
        cost = seconds / 3600.0 * price_per_hour
        shots = set(group["shots"])
        all_shots.update(shots)
        shot_count = len(shots)
        row = {
            "key": key,
            "label": group["label"],
            "seconds": round(seconds, 3),
            "duration": _format_duration(seconds),
            "cost": round(cost, 4),
            "cost_display": _format_cost(cost),
            "shot_count": shot_count,
            "stage_count": int(group["stage_count"]),
            "avg_seconds_per_shot": round(seconds / shot_count, 3) if shot_count else None,
            "avg_duration_per_shot": _format_duration(seconds / shot_count) if shot_count else None,
            "avg_cost_per_shot": round(cost / shot_count, 4) if shot_count else None,
            "avg_cost_per_shot_display": _format_cost(cost / shot_count) if shot_count else None,
        }
        group_list.append(row)
        total_seconds += seconds

    total_cost = total_seconds / 3600.0 * price_per_hour
    summary.update(
        {
            "groups": group_list,
            "total_seconds": round(total_seconds, 3),
            "total_duration": _format_duration(total_seconds),
            "total_cost": round(total_cost, 4),
            "total_cost_display": _format_cost(total_cost),
            "shot_count": len(all_shots),
        }
    )

    messages: list[str] = []
    for group in group_list:
        shot_count = int(group["shot_count"])
        if shot_count:
            noun = "shot" if shot_count == 1 else "shots"
            messages.append(
                f"{group['label']} took {group['duration']} for {shot_count} {noun}, "
                f"costing {group['cost_display']} "
                f"(avg {group['avg_duration_per_shot']} / {group['avg_cost_per_shot_display']} per shot)."
            )
        else:
            messages.append(
                f"{group['label']} took {group['duration']}, costing {group['cost_display']}."
            )
    if total_seconds > 0:
        messages.append(
            f"Total measured stage runtime: {summary['total_duration']}, "
            f"estimated GPU spend {summary['total_cost_display']} at "
            f"{summary['price_per_hour_display']}/hr."
        )
    summary["messages"] = messages
    return summary


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
    "voice_model_unload": "render",
    "generate_video": "render",
    "lip_sync": "render",
    "video_model_load": "render",
    "assemble_video": "assemble",
    "publish": "assemble",
}

_MACRO_ORDER = ["transcribe", "script_plan", "render", "assemble"]

_ANIMATE_MACRO_ORDER = [
    "animate_validate",
    "canonical_look",
    "animate_preprocess",
    "animate_audio",
    "animate_generate",
    "animate_finish",
]

_ANIMATE_STAGE_TO_MACRO = {
    "wan_animate_direct": "animate_validate",
    "animate_gpu_reset": "animate_validate",
    "animate_validate": "animate_validate",
    "canonical_look": "canonical_look",
    "animate_preprocess": "animate_preprocess",
    "animate_audio": "animate_audio",
    "voice_model_load": "animate_audio",
    "voice_model_unload": "animate_audio",
    "video_model_load": "animate_generate",
    "animate_generate": "animate_generate",
    "video_model_unload": "animate_generate",
    "animate_lipsync": "animate_finish",
    "animate_mux": "animate_finish",
    "animate_export": "animate_finish",
}


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
        "shot_attempts": (work_dir / "shot_attempts").exists(),
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
    request = getattr(job, "request", {}) or {}
    if request.get("workflow_kind") == "wan_animate_direct":
        if status == "completed":
            return {
                "phase": "animate_finish",
                "completed": list(_ANIMATE_MACRO_ORDER),
            }
        active = _ANIMATE_STAGE_TO_MACRO.get(
            getattr(job, "current_stage", None) or "",
            "animate_validate",
        )
        return {
            "phase": active,
            "completed": _ANIMATE_MACRO_ORDER[: _ANIMATE_MACRO_ORDER.index(active)],
        }
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
    return {
        "phase": active,
        "completed": _MACRO_ORDER[: _MACRO_ORDER.index(active)],
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
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, status
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "Dashboard API requires FastAPI. Install with `pip install -e \".[dashboard]\"`."
        ) from exc

    config = config_loader()
    repo = repository or _make_repository(config)
    artifact_store = create_artifact_store(config.settings)
    asset_owner_id = os.getenv("VIDEO_ME_DASHBOARD_ASSET_OWNER", "dashboard-local")
    asset_store = DashboardAssetStore(
        repo.db_path,
        Path(config.settings.data_dir) / "dashboard_assets",
        allowed_server_roots=(config.settings.local_video_dir,),
        max_total_bytes=config.settings.dashboard_asset_quota_bytes,
    )
    # Startup cleanup is intentionally limited to unclaimed assets. Claimed
    # job inputs remain immutable and available to a queued/running worker.
    asset_store.expire_staged(delete_files=True)
    asset_store.delete_orphaned_claims()

    signing_seed = os.getenv("VIDEO_ME_DASHBOARD_TOKEN", "").encode("utf-8")
    asset_signing_key = (
        hashlib.sha256(b"video-me-dashboard-assets\0" + signing_seed).digest()
        if signing_seed
        else secrets.token_bytes(32)
    )

    app = FastAPI(title="video_me Dashboard API", version="0.1.0")

    def _signed_value(purpose: str, value: str) -> str:
        digest = hmac.new(
            asset_signing_key,
            f"{purpose}\0{value}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

    def _asset_media_url(asset_id: str) -> str:
        signature = _signed_value("asset-media", asset_id)
        return f"/api/assets/{asset_id}/media?token={signature}"

    def _asset_payload(record: Any) -> dict[str, Any]:
        payload = record.model_dump(mode="json", exclude={"owner_id", "storage_path"})
        payload["media_url"] = _asset_media_url(record.asset_id)
        return payload

    def _asset_http_error(exc: Exception) -> HTTPException:
        if isinstance(exc, DashboardAssetNotFoundError):
            code, http_status = "ASSET_NOT_FOUND", status.HTTP_404_NOT_FOUND
        elif isinstance(exc, DashboardAssetAccessError):
            code, http_status = "ASSET_FORBIDDEN", status.HTTP_403_FORBIDDEN
        elif isinstance(exc, DashboardAssetKindError):
            code, http_status = "ASSET_KIND_MISMATCH", status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, DashboardAssetMetadataError):
            code, http_status = "INVALID_ASSET_METADATA", status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, DashboardAssetQuotaError):
            code, http_status = "ASSET_QUOTA_EXCEEDED", 507
        elif isinstance(exc, DashboardAssetStateError):
            code, http_status = "INVALID_ASSET_STATE", status.HTTP_409_CONFLICT
        elif isinstance(exc, DashboardAssetPathError):
            code, http_status = "INVALID_ASSET_PATH", status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, MediaIngestError):
            code = exc.code
            http_status = (
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                if exc.code == "FILE_TOO_LARGE"
                else status.HTTP_400_BAD_REQUEST
            )
        else:
            code, http_status = "INVALID_ASSET", status.HTTP_400_BAD_REQUEST
        return HTTPException(
            status_code=http_status,
            detail={"code": code, "message": str(exc), "retryable": False},
        )

    def _server_file_id(root_index: int, relative_path: str) -> str:
        raw = json.dumps(
            {"root": root_index, "path": relative_path},
            separators=(",", ":"),
        ).encode("utf-8")
        payload = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return f"{payload}.{_signed_value('server-file', payload)}"

    def _decode_server_file_id(file_id: str) -> Path:
        try:
            payload, signature = file_id.split(".", 1)
            if not secrets.compare_digest(signature, _signed_value("server-file", payload)):
                raise ValueError("signature mismatch")
            padded = payload + "=" * (-len(payload) % 4)
            decoded = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
            root_index = int(decoded["root"])
            relative = str(decoded["path"])
            root = asset_store.allowed_server_roots[root_index]
        except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DashboardAssetPathError("invalid or expired server-file selection") from exc
        return asset_store.validate_server_path(
            root / relative,
            expected_kind=DashboardAssetKind.VIDEO,
        )

    def require_write_auth(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> None:
        token = os.getenv("VIDEO_ME_DASHBOARD_TOKEN")
        if not token:
            client_host = request.client.host if request.client is not None else ""
            try:
                is_loopback = ipaddress.ip_address(
                    client_host.split("%", 1)[0]
                ).is_loopback
            except ValueError:
                # Starlette's in-process TestClient uses this sentinel host.
                is_loopback = client_host == "testclient"
            if is_loopback:
                return
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "LOCAL_DASHBOARD_ONLY",
                    "message": (
                        "Dashboard writes are restricted to loopback while "
                        "VIDEO_ME_DASHBOARD_TOKEN is unset. Use an SSH tunnel or "
                        "configure bearer-token injection at the trusted proxy."
                    ),
                    "retryable": False,
                },
            )
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

    @app.get("/api/runtime/gpu-status")
    def runtime_gpu_status(log_lines: int = 120) -> dict[str, Any]:
        log_lines = max(10, min(int(log_lines), 500))
        status = collect_gpu_status(workspace=Path.cwd(), log_lines=log_lines)
        watermarks = update_watermarks(_gpu_watermarks, status)
        return _base_response({**status, "watermarks": watermarks})

    @app.post("/api/runtime/gpu-watermarks/reset")
    def runtime_gpu_watermarks_reset(
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        reset_watermarks(_gpu_watermarks)
        return _base_response({"watermarks": _gpu_watermarks})

    @app.get("/api/config/defaults")
    def config_defaults() -> dict[str, Any]:
        settings = config.settings
        return _base_response(
            {
                "render_adapter": settings.render_adapter,
                "video_adapter": settings.video_adapter,
                "lipsync_adapter": settings.lipsync_adapter,
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
    _VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}

    def _existing_lora_path(member: Any, params: Any | None) -> Path | None:
        lora_dir = Path(config.settings.lora_dir).expanduser()
        explicit = str(getattr(params, "lora_file", "") or "").strip()
        if explicit:
            candidate = lora_dir / explicit
            return candidate if candidate.is_file() else None
        parts = Path(str(member.lora_ref)).parts
        if parts and parts[0] == "loras":
            parts = parts[1:]
        stem = "_".join(parts)
        for extension in (".safetensors", ".pt", ".ckpt"):
            candidate = lora_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate
        return None

    def _existing_voice_path(member: Any, params: Any | None) -> Path | None:
        voice_ref = str(
            getattr(params, "voice_file", "") or member.voice_profile_ref or ""
        ).strip()
        parts = Path(voice_ref).parts
        if parts and parts[0] == "voices":
            parts = parts[1:]
        base = Path(config.settings.voice_dir).expanduser().joinpath(*parts)
        candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in (".wav", ".mp3", ".flac", ".ogg")]
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    def _animate_cast_catalog() -> list[dict[str, Any]]:
        from core.cast_params import load_cast_params
        from core.models.profile import Cast

        casts: list[Any] = []
        casts_dir = Path("config/casts")
        if casts_dir.is_dir():
            for yaml_path in sorted(casts_dir.glob("*.yaml")):
                try:
                    casts.append(load_yaml_model(yaml_path, Cast))
                except Exception:
                    continue
        if not any(cast.id == config.cast.id for cast in casts):
            casts.append(config.cast)

        result: list[dict[str, Any]] = []
        for cast in casts:
            try:
                params_by_member = load_cast_params(cast.id, casts_dir)
            except Exception:
                params_by_member = {}
            members = []
            for member in cast.members:
                params = params_by_member.get(member.id)
                lora_path = _existing_lora_path(member, params)
                voice_path = _existing_voice_path(member, params)
                members.append(
                    {
                        "id": member.id,
                        "name": member.name,
                        "visual_descriptor": member.visual_descriptor,
                        "has_lora": lora_path is not None,
                        "has_voice": voice_path is not None,
                    }
                )
            result.append(
                {
                    "id": cast.id,
                    "name": cast.id.replace("_", " ").title(),
                    "species": cast.species,
                    "member_count": len(members),
                    "members": members,
                }
            )
        return result

    def _wan_animate_install_readiness() -> tuple[bool, str]:
        settings = config.settings
        python_value = str(settings.wan_animate_python)
        python_ready = Path(python_value).expanduser().is_file() or shutil.which(python_value) is not None
        model_dir = Path(settings.wan_animate_model_dir).expanduser()
        required_paths = [
            ("Wan Animate Python", python_ready),
            ("Wan 2.2 repository", Path(settings.wan_animate_repo_dir).expanduser().is_dir()),
        ]
        missing = [name for name, ready in required_paths if not ready]
        if missing:
            return False, "Missing " + ", ".join(missing) + ". Run setup_gpu.sh --with-wan-animate."
        model_ready, model_reason = wan_animate_model_readiness(model_dir)
        if not model_ready:
            return False, model_reason
        return True, "Wan Animate is installed; its deferred-loading service starts on demand."

    def _http_service_readiness(base_url: str, label: str) -> dict[str, Any]:
        url = f"{base_url.rstrip('/')}/health"
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=0.75) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ready = payload.get("status") == "ok"
            reason = "Service is ready." if ready else str(payload.get("reason") or payload)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            ready = False
            reason = f"{label} is unreachable at {base_url}: {exc}"
        return {"ready": ready, "reason": reason, "base_url": base_url}

    @app.get("/api/casts")
    def list_casts() -> dict[str, Any]:
        return _base_response({"casts": _animate_cast_catalog(), "default": config.cast.id})

    @app.get("/api/animate/options")
    def animate_options() -> dict[str, Any]:
        backend_ready, backend_reason = _wan_animate_install_readiness()
        flux_retarget_ready, flux_retarget_reason = wan_flux_retarget_readiness(
            config.settings.wan_animate_model_dir
        )
        lipsync_readiness = {
            "latentsync": _http_service_readiness(
                config.settings.latentsync_base_url, "LatentSync"
            ),
            "musetalk": _http_service_readiness(
                config.settings.musetalk_base_url, "MuseTalk"
            ),
        }
        return _base_response(
            {
                "casts": _animate_cast_catalog(),
                "default": config.cast.id,
                "defaults": {"cast_ref": config.cast.id},
                "readiness": {
                    "ready": backend_ready,
                    "reason": backend_reason,
                    "wan_animate": {"ready": backend_ready, "reason": backend_reason},
                    "lipsync": lipsync_readiness,
                },
                "features": {
                    "flux2_edit_enabled": config.settings.flux2_edit_enabled,
                    # The canonical-look edit prepends one identity image to
                    # the user-provided complete-look references.
                    "flux2_edit_max_user_references": max(
                        0, config.settings.flux2_edit_max_references - 1
                    ),
                    "flux2_edit_reason": (
                        "FLUX.2 reference-image editing is disabled. Set "
                        "VIDEO_ME_FLUX2_EDIT_ENABLED=true after validating the installed "
                        "Musubi revision; text-directed complete-look styling remains available."
                    ),
                    "wan_flux_retarget_enabled": flux_retarget_ready,
                    "wan_flux_retarget_reason": flux_retarget_reason,
                },
                "limits": {
                    "max_driver_range_sec": WAN_ANIMATE_MAX_DRIVER_RANGE_SEC,
                },
            }
        )

    @app.get("/api/assets/video/server-files")
    def list_server_video_assets(
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for root_index, root in enumerate(asset_store.allowed_server_roots):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue
                try:
                    relative = str(path.resolve().relative_to(root))
                except (OSError, ValueError):
                    continue
                files.append(
                    {
                        "file_id": _server_file_id(root_index, relative),
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                    }
                )
                if len(files) >= 500:
                    break
            if len(files) >= 500:
                break
        return _base_response({"files": files})

    async def _register_video_asset(
        *,
        destination: Path,
        asset_id: str,
        original_name: str,
    ) -> Any:
        try:
            metadata = await probe_video(
                destination,
                ffprobe_bin=config.settings.ffprobe_bin,
            )
            container = str(metadata.get("container") or "").lower()
            suffix = destination.suffix.lower()
            if "webm" in container or suffix == ".webm":
                safe_mime = "video/webm"
            elif "matroska" in container or suffix == ".mkv":
                safe_mime = "video/x-matroska"
            elif "avi" in container or suffix == ".avi":
                safe_mime = "video/x-msvideo"
            elif suffix == ".mov":
                safe_mime = "video/quicktime"
            else:
                safe_mime = "video/mp4"
            return await asyncio.to_thread(
                asset_store.create_staged,
                owner_id=asset_owner_id,
                kind=DashboardAssetKind.VIDEO,
                original_name=original_name,
                mime_type=safe_mime,
                storage_path=destination,
                asset_id=asset_id,
                metadata=metadata,
            )
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    @app.post("/api/assets/video/upload")
    async def upload_video_asset(
        file: UploadFile = File(...),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        extension = Path(file.filename or "driver.mp4").suffix.lower()
        if extension not in _VIDEO_EXTENSIONS:
            raise _asset_http_error(MediaIngestError(
                "INVALID_FORMAT", "Video must be MP4, MOV, WebM, or MKV"
            ))
        asset_id, destination = asset_store.allocate_path(
            DashboardAssetKind.VIDEO,
            suffix=extension,
        )
        try:
            await stream_upload(file, destination, max_bytes=2 * 1024 * 1024 * 1024)
            record = await _register_video_asset(
                destination=destination,
                asset_id=asset_id,
                original_name=file.filename or destination.name,
            )
        except (DashboardAssetError, MediaIngestError) as exc:
            raise _asset_http_error(exc) from exc
        return _base_response({"asset": _asset_payload(record)})

    @app.post("/api/assets/video/from-url")
    async def import_video_asset_from_url(
        body: dict[str, Any] = Body(...),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        url = str(body.get("url") or "").strip()
        extension = Path(urlparse(url).path).suffix.lower()
        if extension not in _VIDEO_EXTENSIONS:
            extension = ".mp4"
        asset_id, destination = asset_store.allocate_path(
            DashboardAssetKind.VIDEO,
            suffix=extension,
        )
        try:
            _, _, final_url = await download_public_video_url(url, destination)
            record = await _register_video_asset(
                destination=destination,
                asset_id=asset_id,
                original_name=Path(urlparse(final_url).path).name or "remote-video",
            )
        except (DashboardAssetError, MediaIngestError) as exc:
            raise _asset_http_error(exc) from exc
        return _base_response({"asset": _asset_payload(record)})

    @app.post("/api/assets/video/from-server-file")
    async def import_video_asset_from_server(
        body: dict[str, Any] = Body(...),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        destination: Path | None = None
        try:
            source = _decode_server_file_id(str(body.get("file_id") or ""))
            asset_id, destination = asset_store.allocate_path(
                DashboardAssetKind.VIDEO,
                suffix=source.suffix.lower(),
            )
            await asyncio.to_thread(
                copy_local_file_limited,
                source,
                destination,
                max_bytes=2 * 1024 * 1024 * 1024,
            )
            record = await _register_video_asset(
                destination=destination,
                asset_id=asset_id,
                original_name=source.name,
            )
        except (DashboardAssetError, MediaIngestError) as exc:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise _asset_http_error(exc) from exc
        return _base_response({"asset": _asset_payload(record)})

    @app.post("/api/assets/image/upload")
    async def upload_image_asset(
        file: UploadFile = File(...),
        purpose: str = Form(default="character"),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        extension = Path(file.filename or "image.png").suffix.lower()
        if extension not in _IMAGE_EXTENSIONS:
            raise _asset_http_error(MediaIngestError(
                "INVALID_FORMAT", "Image must be PNG, JPG, JPEG, or WebP"
            ))
        if purpose not in {"character", "garment", "accessory"}:
            raise _asset_http_error(MediaIngestError("INVALID_PURPOSE", "Invalid image purpose"))
        asset_id, destination = asset_store.allocate_path(
            DashboardAssetKind.IMAGE,
            suffix=".png",
        )
        incoming = destination.with_suffix(f"{extension}.incoming")
        try:
            await stream_upload(file, incoming, max_bytes=25 * 1024 * 1024)
            metadata = await asyncio.to_thread(normalize_image, incoming, destination)
            metadata["purpose"] = purpose
            record = asset_store.create_staged(
                owner_id=asset_owner_id,
                kind=DashboardAssetKind.IMAGE,
                original_name=file.filename or "image.png",
                mime_type="image/png",
                storage_path=destination,
                asset_id=asset_id,
                metadata=metadata,
            )
        except (DashboardAssetError, MediaIngestError) as exc:
            destination.unlink(missing_ok=True)
            raise _asset_http_error(exc) from exc
        finally:
            incoming.unlink(missing_ok=True)
        return _base_response({"asset": _asset_payload(record)})

    @app.get("/api/assets/{asset_id}")
    def get_dashboard_asset(
        asset_id: str,
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        try:
            record, _ = asset_store.resolve(asset_id, owner_id=asset_owner_id)
        except DashboardAssetError as exc:
            raise _asset_http_error(exc) from exc
        return _base_response({"asset": _asset_payload(record)})

    @app.get("/api/assets/{asset_id}/media", include_in_schema=False)
    def serve_dashboard_asset(asset_id: str, token: str = ""):
        from fastapi.responses import FileResponse

        expected = _signed_value("asset-media", asset_id)
        if not token or not secrets.compare_digest(token, expected):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid asset token")
        try:
            record, path = asset_store.resolve(asset_id, owner_id=asset_owner_id)
        except DashboardAssetError as exc:
            raise _asset_http_error(exc) from exc
        return FileResponse(
            str(path),
            media_type=record.mime_type,
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    @app.delete("/api/assets/{asset_id}")
    def delete_dashboard_asset(
        asset_id: str,
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        try:
            deleted = asset_store.delete_staged(asset_id, owner_id=asset_owner_id)
        except DashboardAssetError as exc:
            raise _asset_http_error(exc) from exc
        if not deleted:
            raise _asset_http_error(DashboardAssetNotFoundError(f"dashboard asset not found: {asset_id}"))
        return _base_response({"deleted": True, "asset_id": asset_id})

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
        _: None = Depends(require_write_auth),
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

    @app.post("/api/uploads/wan-animate-driver")
    async def upload_wan_animate_driver(
        file: UploadFile = File(...),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        """Stream a driver video to a server-local, job-safe staging directory."""
        ext = Path(file.filename or "driver.mp4").suffix.lower()
        if ext not in _VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_FORMAT",
                    "message": "Driver video must be MP4, MOV, WebM, or MKV",
                    "retryable": False,
                },
            )
        import uuid
        token = uuid.uuid4().hex
        dest_dir = Path(config.settings.data_dir) / "uploads" / "wan_animate" / token
        dest_dir.mkdir(parents=True, exist_ok=False)
        dest = dest_dir / f"driver{ext}"
        size = 0
        max_size = 2 * 1024 * 1024 * 1024
        try:
            with dest.open("wb") as handle:
                while chunk := await file.read(8 * 1024 * 1024):
                    size += len(chunk)
                    if size > max_size:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail={
                                "code": "FILE_TOO_LARGE",
                                "message": "Driver video must be at most 2 GiB",
                                "retryable": False,
                            },
                        )
                    handle.write(chunk)
            probe = await asyncio.create_subprocess_exec(
                config.settings.ffprobe_bin,
                "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,codec_name:format=duration",
                "-of", "json", str(dest),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await probe.communicate()
            if probe.returncode != 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "UNREADABLE_VIDEO",
                        "message": "Driver video could not be decoded: "
                        + stderr.decode(errors="replace")[-500:],
                        "retryable": False,
                    },
                )
            metadata = json.loads(stdout)
            streams = metadata.get("streams") or []
            if len(streams) != 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "NO_VIDEO", "message": "Driver must contain one readable video stream", "retryable": False},
                )
            duration = float((metadata.get("format") or {}).get("duration") or 0)
            width = int(streams[0].get("width") or 0)
            height = int(streams[0].get("height") or 0)
            if duration <= 0 or duration > 600:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "INVALID_DURATION", "message": "Driver duration must be greater than 0 and at most 10 minutes", "retryable": False},
                )
            if min(width, height) < 256:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "VIDEO_TOO_SMALL", "message": "Driver's shorter side must be at least 256 pixels", "retryable": False},
                )
        except Exception:
            dest.unlink(missing_ok=True)
            try:
                dest_dir.rmdir()
            except OSError:
                pass
            raise
        return _base_response({
            "path": str(dest.resolve()),
            "filename": file.filename or dest.name,
            "size_bytes": size,
            "duration_sec": duration,
            "width": width,
            "height": height,
            "codec": streams[0].get("codec_name"),
        })

    @app.post("/api/uploads/lora-training-image")
    async def upload_lora_training_image(
        member_id: str = Form(...),
        file: UploadFile = File(...),
        cast_ref: str = Form(default=""),
        caption: str = Form(default=""),
        _: None = Depends(require_write_auth),
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
                    "message": "Confirm rights clearance before queueing a real job.",
                    "retryable": False,
                },
            )

        if body.workflow_kind == "wan_animate_direct":
            from core.cast_params import load_cast_params
            from core.models.profile import Cast

            assert body.animate is not None
            character = body.animate.character
            cast_ref = character.cast_ref
            member_id = character.member_id
            requires_member = (
                character.look_source in {"auto_lora", "styled_lora"}
                or body.animate.audio.mode == "cast_voice"
            )
            if bool(cast_ref) != bool(member_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "INCOMPLETE_CAST_TARGET",
                        "message": "Provide both cast and member, or neither for an exact uploaded image.",
                        "retryable": False,
                    },
                )
            target_cast = config.cast if cast_ref == config.cast.id else None
            if target_cast is None and cast_ref:
                for cast_path in sorted(Path("config/casts").glob("*.yaml")):
                    try:
                        candidate = load_yaml_model(cast_path, Cast)
                    except Exception:
                        continue
                    if candidate.id == cast_ref:
                        target_cast = candidate
                        break
            if target_cast is None and (requires_member or cast_ref):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "UNKNOWN_CAST",
                        "message": "Choose a configured cast for the Animate target.",
                        "retryable": False,
                    },
                )
            member = (
                next((item for item in target_cast.members if item.id == member_id), None)
                if target_cast is not None and member_id
                else None
            )
            if member_id and member is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "INVALID_MEMBER",
                        "message": (
                            f"Unknown target member {member_id!r} for cast {cast_ref!r}."
                        ),
                        "retryable": False,
                    },
                )

            params = (
                load_cast_params(target_cast.id).get(member.id)
                if target_cast is not None and member is not None
                else None
            )
            if character.look_source in {"auto_lora", "styled_lora"}:
                assert member is not None
                effective_render_adapter = (
                    body.overrides.render_adapter or config.settings.render_adapter
                )
                if effective_render_adapter != "musubi_flux":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "FLUX2_RENDERER_REQUIRED",
                            "message": (
                                "Generated Animate looks require render_adapter=musubi_flux; "
                                f"{effective_render_adapter!r} cannot apply the trained FLUX.2 "
                                "LoRA and complete-look controls reliably."
                            ),
                            "retryable": False,
                        },
                    )
                if _existing_lora_path(member, params) is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "LORA_NOT_READY",
                            "message": (
                                f"The FLUX.2 LoRA for {member.name} is not available under "
                                f"{config.settings.lora_dir}."
                            ),
                            "retryable": False,
                        },
                    )

            if body.animate.advanced.use_flux_retarget:
                flux_ready, flux_reason = wan_flux_retarget_readiness(
                    config.settings.wan_animate_model_dir
                )
                if not flux_ready:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "WAN_FLUX_RETARGET_NOT_READY",
                            "message": flux_reason,
                            "retryable": False,
                        },
                    )
            if body.animate.audio.mode == "cast_voice":
                assert member is not None
                if _existing_voice_path(member, params) is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "VOICE_NOT_READY",
                            "message": f"The configured cast voice for {member.name} is unavailable.",
                            "retryable": False,
                        },
                    )

            wardrobe = character.wardrobe
            wardrobe_refs = (
                [*wardrobe.garment_asset_ids, *wardrobe.accessory_asset_ids]
                if wardrobe is not None
                else []
            )
            if wardrobe_refs and not config.settings.flux2_edit_enabled:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "FLUX2_EDIT_NOT_READY",
                        "message": (
                            "Reference-image complete-look styling is disabled. Use text-only "
                            "styling direction or enable VIDEO_ME_FLUX2_EDIT_ENABLED after a "
                            "Hopper smoke test."
                        ),
                        "retryable": False,
                    },
                )
            max_user_references = max(0, config.settings.flux2_edit_max_references - 1)
            if len(wardrobe_refs) > max_user_references:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "TOO_MANY_FLUX2_REFERENCES",
                        "message": (
                            f"Use at most {max_user_references} complete-look reference images; "
                            "one FLUX.2 control slot is reserved for cast identity."
                        ),
                        "retryable": False,
                    },
                )

            try:
                asset_store.validate_animate_assets(
                    body.animate,
                    owner_id=asset_owner_id,
                )
            except DashboardAssetError as exc:
                raise _asset_http_error(exc) from exc

            job_id = make_dashboard_job_id()
            asset_ids = list(collect_animate_asset_requirements(body.animate))
            try:
                asset_store.claim_assets(
                    asset_ids,
                    owner_id=asset_owner_id,
                    job_id=job_id,
                )
                job, queue_item = repo.create_queued_job(body, job_id=job_id)
            except Exception:
                asset_store.release_claims(
                    job_id=job_id,
                    owner_id=asset_owner_id,
                    asset_ids=asset_ids,
                )
                raise
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
        cost_events = repo.list_events(job_id, limit=5000)
        payload = detail.model_dump(mode="json")
        payload["cost_summary"] = _build_cost_summary(
            cost_events,
            detail.job.request,
        )
        return _base_response(payload)

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

    @app.get("/api/jobs/{job_id}/shot_attempts")
    def get_shot_attempts(job_id: str) -> dict[str, Any]:
        import base64

        if repo.get_job(job_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}", "retryable": False},
            )
        work_dir = Path(config.settings.data_dir) / "jobs" / job_id
        attempt_dir = work_dir / "shot_attempts"
        if not attempt_dir.exists():
            return _base_response({"shots": []})

        def _path_b64(value: str | None) -> str | None:
            if not value:
                return None
            path = Path(value)
            if not path.exists():
                return None
            return base64.urlsafe_b64encode(str(path).encode()).decode()

        shots: list[dict[str, Any]] = []
        for path in sorted(attempt_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for run in data.get("runs") or []:
                run["raw_clip_b64"] = _path_b64(run.get("raw_clip_snapshot_uri") or run.get("raw_clip_uri"))
                run["selected_clip_b64"] = _path_b64(run.get("selected_clip_snapshot_uri") or run.get("selected_clip_uri"))
                run["audio_b64"] = _path_b64(run.get("audio_snapshot_uri") or run.get("audio_uri"))
            latest = data.get("latest") or {}
            latest["raw_clip_b64"] = _path_b64(latest.get("raw_clip_snapshot_uri") or latest.get("raw_clip_uri"))
            latest["selected_clip_b64"] = _path_b64(latest.get("selected_clip_snapshot_uri") or latest.get("selected_clip_uri"))
            latest["audio_b64"] = _path_b64(latest.get("audio_snapshot_uri") or latest.get("audio_uri"))
            shots.append(data)
        return _base_response({"shots": shots})

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

    @app.post("/api/jobs/bulk-cancel")
    def bulk_cancel_jobs(
        body: dict[str, Any],
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        job_ids = body.get("job_ids") or []
        results: dict[str, str] = {}
        for job_id in job_ids:
            job = repo.get_job(job_id)
            if job is None:
                results[job_id] = "not_found"
            elif job.status in _TERMINAL_STATUSES or job.status == DashboardJobStatus.CANCEL_REQUESTED:
                results[job_id] = "already_terminal"
            else:
                repo.update_job_status(job_id, DashboardJobStatus.CANCEL_REQUESTED)
                repo.record_event(
                    job_id, "cancel_requested",
                    "Cancellation requested from dashboard API (bulk action).",
                )
                results[job_id] = "cancel_requested"
        return _base_response({"results": results})

    @app.post("/api/jobs/bulk-delete")
    def bulk_delete_jobs(
        body: dict[str, Any],
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        job_ids = body.get("job_ids") or []
        results: dict[str, str] = {}
        for job_id in job_ids:
            job = repo.get_job(job_id)
            if job is None:
                results[job_id] = "not_found"
            elif job.status not in _TERMINAL_STATUSES:
                results[job_id] = "skipped_active"
            else:
                repo.delete_job(job_id)
                asset_store.delete_claimed_for_job(job_id)
                results[job_id] = "deleted"
        return _base_response({"results": results})

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
        "video_adapter", "lipsync_adapter", "render_adapter", "tts_adapter", "llm_model",
        "whisper_language",
        "max_shot_duration_sec", "lipsync_failure_policy", "lipsync_max_retries",
        "av_sync_duration_tolerance_sec", "av_sync_failure_policy",
        "video_upscale_enabled", "video_upscale_target_fps",
        "video_enhance_enabled", "video_enhance_adapter", "video_enhance_target_fps",
    )

    _RERUN_PHASES = ("transcribe", "script_plan", "render", "assemble", "all")
    _RERUN_RENDER_MODES = ("full", "source_audio", "re_voice")
    _RERUN_AUDIO_PROFILES = ("auto", "single_speaker", "singing", "multi_speaker")

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(
        job_id: str,
        body: dict[str, Any] | None = Body(default=None),
        _: None = Depends(require_write_auth),
    ) -> dict[str, Any]:
        """Re-queue a failed OR completed job, optionally from a different phase.

        script_plan/render/assemble already resume from cached per-stage
        artifacts (see core/workflow.py's `_stage()` helper), so re-queuing
        skips whatever already succeeded — a failed job picks up again at
        the stage that failed; a completed phase="all" job skips straight
        past the (already-rendered) images and redoes voice/video/assemble
        fresh. That makes this the fast way to re-test generate_video or
        assemble_video changes without paying for a fresh render.

        Optional JSON body fields:
        - ``phase`` — run from a specific phase instead of the job's
          original phase.  E.g. ``"assemble"`` to redo only the final
          concat, or ``"render"`` to redo voice/video/assemble while
          keeping cached character images.
        - ``render_mode`` — switch between full/source_audio/re_voice for this
          retry. Timed modes require a matching timed plan artifact.
        - ``audio_profile`` — retry transcript/planning with a different
          source-audio validation profile.
        - ``video_adapter`` / ``lipsync_adapter`` / ``render_adapter`` / ``tts_adapter`` /
          ``llm_model`` — swap an adapter for this re-run without
          touching the job's original overrides.
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

        if job.request.get("workflow_kind") == "wan_animate_direct":
            requested_phase = (body or {}).get("phase", "all")
            unsupported = set(body or {}) - {"phase"}
            if requested_phase != "all" or unsupported:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "INVALID_ANIMATE_RETRY",
                        "message": (
                            "Direct Animate retries rerun the same versioned request with semantic "
                            "stage caches. Only phase='all' is valid; create a new Animate job to "
                            "change its controls."
                        ),
                        "retryable": False,
                    },
                )
            try:
                validated = CreateDashboardJobRequest.model_validate(
                    {**job.request, "phase": "all"}
                )
                assert validated.animate is not None
                asset_store.validate_animate_assets(
                    validated.animate,
                    owner_id=asset_owner_id,
                    job_id=job_id,
                )
            except DashboardAssetError as exc:
                raise _asset_http_error(exc) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "INVALID_STORED_ANIMATE_REQUEST",
                        "message": str(exc),
                        "retryable": False,
                    },
                ) from exc
            repo.update_job_status(job_id, DashboardJobStatus.QUEUED)
            queue_item = repo.enqueue_action(
                job_id,
                DashboardQueueAction.RESUME,
                payload=validated.model_dump(mode="json"),
            )
            repo.record_event(
                job_id,
                "job_retried",
                f"Direct Animate job re-queued with semantic stage caching (was {job.status.value}).",
                payload={"phase": "all", "queue_id": queue_item.queue_id},
            )
            return _base_response(
                {
                    "job_id": job_id,
                    "phase": "all",
                    "queue_id": queue_item.queue_id,
                    "status": DashboardJobStatus.QUEUED.value,
                }
            )

        rerun_phase = job.phase
        if body and body.get("phase"):
            requested_phase = body["phase"]
            if requested_phase not in _RERUN_PHASES:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "code": "INVALID_PHASE",
                        "message": (
                            f"Invalid phase '{requested_phase}'. "
                            f"Must be one of: {', '.join(_RERUN_PHASES)}."
                        ),
                        "retryable": False,
                    },
                )
            rerun_phase = requested_phase

        retry_request = {**job.request, "phase": rerun_phase}
        if body:
            if body.get("render_mode"):
                requested_mode = body["render_mode"]
                if requested_mode not in _RERUN_RENDER_MODES:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "code": "INVALID_RENDER_MODE",
                            "message": (
                                f"Invalid render_mode '{requested_mode}'. "
                                f"Must be one of: {', '.join(_RERUN_RENDER_MODES)}."
                            ),
                            "retryable": False,
                        },
                )
                retry_request["render_mode"] = requested_mode
            if body.get("audio_profile"):
                requested_profile = body["audio_profile"]
                if requested_profile not in _RERUN_AUDIO_PROFILES:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail={
                            "code": "INVALID_AUDIO_PROFILE",
                            "message": (
                                f"Invalid audio_profile '{requested_profile}'. "
                                f"Must be one of: {', '.join(_RERUN_AUDIO_PROFILES)}."
                            ),
                            "retryable": False,
                        },
                    )
                retry_request["audio_profile"] = requested_profile
            overrides = dict(retry_request.get("overrides") or {})
            for key in _RETRYABLE_OVERRIDE_KEYS:
                if key in body and body[key] is not None:
                    value = body[key]
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
            f"Job re-queued for phase '{rerun_phase}' (was {job.status.value}).",
            payload={"phase": rerun_phase, "queue_id": queue_item.queue_id},
        )
        return _base_response({
            "job_id": job_id,
            "phase": rerun_phase,
            "queue_id": queue_item.queue_id,
            "status": DashboardJobStatus.QUEUED.value,
        })

    # ------------------------------------------------------------------
    # Chat — Pipeline Assistant (per-job, persisted in SQLite)
    # ------------------------------------------------------------------

    @app.post("/api/jobs/{job_id}/chat")
    async def chat_with_job(
        job_id: str,
        body: ChatRequest,
        _: None = Depends(require_write_auth),
    ):
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
        _jinja_env.filters["dashboard_time"] = _format_dashboard_time

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
            jobs_dir = (data_dir / "jobs").resolve()
            # `/img` predates opaque dashboard inputs and is used only for
            # generated job previews. Never let a caller turn a leaked asset
            # ID into a predictable dashboard_assets path that bypasses the
            # signed media URL.
            if not path.is_relative_to(jobs_dir):
                raise HTTPException(status_code=403, detail="path outside job preview directory")
            if not path.is_file():
                raise HTTPException(status_code=404, detail="image not found")
            return FileResponse(
                str(path), headers={"X-Content-Type-Options": "nosniff"}
            )

        def _render(template_name: str, **ctx_vars: Any):
            from fastapi.responses import HTMLResponse

            tmpl = _jinja_env.get_template(template_name)
            html = tmpl.render(**ctx_vars)
            return HTMLResponse(content=html)

        @app.get("/", include_in_schema=False)
        def ui_jobs_list():
            jobs = repo.list_jobs(limit=100)
            worker_hb = repo.latest_worker_heartbeat()
            return _render("jobs_list.html", jobs=jobs, worker=worker_hb, active="jobs")

        @app.get("/jobs/new", include_in_schema=False)
        def ui_new_job():
            worker_hb = repo.latest_worker_heartbeat()
            return _render(
                "job_new.html",
                cast_members=config.cast.members,
                default_cast_id=config.cast.id,
                worker=worker_hb,
                active="jobs",
            )

        @app.get("/animate/new", include_in_schema=False)
        def ui_new_animate_job():
            """Dedicated creator for direct Wan 2.2 Animate jobs."""
            worker_hb = repo.latest_worker_heartbeat()
            return _render(
                "animate_new.html",
                worker=worker_hb,
                active="animate",
            )

        @app.get("/jobs/{job_id}", include_in_schema=False)
        def ui_job_detail(job_id: str):
            detail = repo.get_job_detail(job_id, event_limit=200)
            if detail is None:
                from fastapi.responses import HTMLResponse
                return HTMLResponse("<h1>Job not found</h1>", status_code=404)
            work_dir = Path(config.settings.data_dir) / "jobs" / job_id
            flags = _artifact_flags(artifact_store, work_dir, job_id)
            worker_hb = repo.latest_worker_heartbeat()
            cost_events = repo.list_events(job_id, limit=5000)
            cost_summary = _build_cost_summary(cost_events, detail.job.request)
            return _render(
                "job_detail.html",
                detail=detail,
                artifact_flags=flags,
                stepper=_stepper_state(detail.job, flags),
                cost_summary=cost_summary,
                artifacts=repo.list_artifacts(job_id),
                worker=worker_hb,
                active="jobs",
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
            return _render(
                "health.html",
                results=results,
                worker=worker_hb,
                now=_utc_now(),
                active="health",
            )

        @app.get("/gpu", include_in_schema=False)
        def ui_gpu_status():
            worker_hb = repo.latest_worker_heartbeat()
            return _render("gpu_status.html", worker=worker_hb, active="gpu")

    except ImportError:
        # jinja2 / starlette not installed — UI routes silently unavailable.
        pass

    return app
