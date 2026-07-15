"""Direct Wan 2.2 Animate orchestration for dashboard Animate Studio jobs.

This module deliberately does not route direct Animate jobs through the story
pipeline.  A direct job has one driving-video range, one canonical character
look, one target cast member, and one output.  Wan performs its own internal
77-frame chunking while the exact same approved reference image is reused for
the whole range.

The workflow accepts injected dependencies so its cache, sequencing, and media
semantics can be tested without loading GPU models.  The dashboard worker uses
``build_default_dependencies`` for the production adapters.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel

from core.cast_params import load_cast_params
from core.gpu_sequencer import (
    ensure_video_model_unloaded,
    free_comfyui,
    prepare_video_model,
    prepare_voice_model,
    stop_fish_s2_process,
)
from core.models.capabilities import (
    AudioTrack,
    ImageApprovalRequest,
    ImageCritiqueResult,
    LipSyncRequest,
    RenderCharacterRequest,
    TranscribeRequest,
    VideoClip,
    VideoDriver,
    VideoRequest,
    VoiceRequest,
)
from core.models.content import Shot
from core.models.dashboard import WAN_ANIMATE_MAX_DRIVER_RANGE_SEC
from core.wan_animate_readiness import wan_flux_retarget_readiness

logger = logging.getLogger(__name__)

DIRECT_WORKFLOW_VERSION = "3"
DIRECT_SHOT_ID = "animate_direct"


class AssetResolver(Protocol):
    def __call__(self, asset_id: str, expected_kind: str) -> Any: ...


@dataclass(frozen=True)
class ResolvedAnimateAsset:
    asset_id: str
    kind: str
    path: Path
    sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnimateWorkflowDependencies:
    """Adapters and services used by one direct Animate job.

    Only ``resolve_asset`` and ``video`` are universally required.  Generated
    look, cast voice, lip-sync, and approval dependencies are required only
    when their corresponding request options are selected.
    """

    resolve_asset: AssetResolver
    video: Any
    render: Any | None = None
    image_approval: Any | None = None
    transcriber: Any | None = None
    voice: Any | None = None
    lipsync: Any | None = None


class AnimateWorkflowResult(BaseModel):
    job_id: str
    raw_video_uri: str
    final_video_uri: str
    canonical_look_uri: str
    audio_uri: str | None = None
    duration_sec: float
    manifests_dir: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return value


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(_dump(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_revision(path: Path) -> dict[str, Any]:
    """Cheap revision token for large model trees without hashing 72 GB of weights."""

    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        return {"path": str(resolved), "missing": True}
    head = ""
    git_dir = resolved / ".git"
    try:
        head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head_text.startswith("ref: "):
            ref_path = git_dir / head_text[5:]
            head = ref_path.read_text(encoding="utf-8").strip()
        else:
            head = head_text
    except OSError:
        pass
    # A Git source checkout is identified solely by its commit. Importing Wan
    # creates __pycache__/pyc files, whose mtimes must not invalidate a retry.
    if head:
        return {"path": str(resolved), "git_head": head}

    entries: list[tuple[str, int, str | None]] = []
    try:
        for candidate in sorted(resolved.rglob("*")):
            if (
                not candidate.is_file()
                or ".git" in candidate.parts
                or "__pycache__" in candidate.parts
                or candidate.suffix == ".pyc"
            ):
                continue
            stat = candidate.stat()
            content_sha = None
            if stat.st_size <= 2 * 1024 * 1024 and candidate.suffix.lower() in {
                ".json", ".yaml", ".yml", ".toml", ".txt", ".py"
            }:
                content_sha = _sha256_file(candidate)
            entries.append(
                (str(candidate.relative_to(resolved)), stat.st_size, content_sha)
            )
    except OSError:
        pass
    return {
        "path": str(resolved),
        "git_head": None,
        "tree": _fingerprint(entries),
        "file_count": len(entries),
    }


def _file_revision(path: Path, *, hash_limit: int = 2 * 1024 * 1024) -> dict[str, Any]:
    """Return a stable identity for a file without hashing multi-GB checkpoints."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    size = resolved.stat().st_size
    revision: dict[str, Any] = {"path": str(resolved), "size": size}
    if size <= hash_limit:
        revision["sha256"] = _sha256_file(resolved)
    return revision


def _settings_snapshot(settings: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Capture only semantic settings; omit process state and absent settings."""

    return {
        name: _dump(getattr(settings, name))
        for name in names
        if hasattr(settings, name)
    }


def _adapter_snapshot(adapter: Any | None, attributes: tuple[str, ...]) -> dict[str, Any] | None:
    """Describe adapter implementation and immutable construction parameters."""

    if adapter is None:
        return None
    snapshot: dict[str, Any] = {
        "class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "version": getattr(adapter, "version", None),
    }
    parameters: dict[str, Any] = {}
    for attribute in attributes:
        if not hasattr(adapter, attribute):
            continue
        value = getattr(adapter, attribute)
        if not callable(value):
            parameters[attribute.removeprefix("_")] = _dump(value)
    snapshot["parameters"] = parameters
    source = inspect.getsourcefile(type(adapter))
    if source and Path(source).is_file():
        snapshot["implementation"] = _file_revision(Path(source))
    return snapshot


def _render_snapshot(adapter: Any | None) -> dict[str, Any] | None:
    snapshot = _adapter_snapshot(
        adapter,
        (
            "_lora_dir",
            "_lora_weight",
            "_steps",
            "_width",
            "_height",
            "_num_images",
            "_allow_placeholder_lora",
            "_guidance_scale",
            "_enable_image_edit",
            "_max_control_images",
        ),
    )
    if snapshot is None:
        return None
    module = inspect.getmodule(type(adapter))
    runtime_files: dict[str, Any] = {}
    if module is not None:
        for name in ("_MUSUBI_SCRIPT", "_DIT", "_VAE", "_TEXT_ENCODER"):
            value = getattr(module, name, None)
            if isinstance(value, (str, Path)):
                runtime_files[name.removeprefix("_").lower()] = _file_revision(Path(value))
    if runtime_files:
        snapshot["runtime_files"] = runtime_files
    return snapshot


def _video_snapshot(adapter: Any | None) -> dict[str, Any] | None:
    return _adapter_snapshot(
        adapter,
        (
            "_mode",
            "_fps",
            "_resolution_area",
            "_subject_selection",
            "_retarget_pose",
            "_use_flux_retarget",
            "_refert_num",
            "_sampling_steps",
            "_mask_iterations",
            "_mask_kernel",
            "_mask_w_len",
            "_mask_h_len",
            "_ffmpeg_bin",
            "_ffprobe_bin",
        ),
    )


def _transcriber_snapshot(adapter: Any | None) -> dict[str, Any] | None:
    return _adapter_snapshot(
        adapter,
        (
            "_model_size",
            "_device",
            "_compute_type",
            "_beam_size",
            "_vad_filter",
            "_download_root",
            "_local_files_only",
            "_revision",
            "_language",
        ),
    )


def _voice_snapshot(adapter: Any | None) -> dict[str, Any] | None:
    return _adapter_snapshot(
        adapter,
        ("_base_url", "_voice_dir", "_timeout_sec", "_sample_rate"),
    )


def _lipsync_snapshot(adapter: Any | None) -> dict[str, Any] | None:
    return _adapter_snapshot(
        adapter,
        ("_base_url", "_inference_steps", "_guidance_scale", "_timeout_sec"),
    )


def _stable_wan_health(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep service identity/capabilities while excluding live load-state fields."""

    keys = (
        "status",
        "flash_attn_3",
        "require_flash_attn_3",
        "model_dir",
        "service_version",
        "model_revision",
        "wan_revision",
    )
    return {key: _dump(payload[key]) for key in keys if key in payload}


def _wan_runtime_revisions() -> dict[str, Any]:
    """Fingerprint the local service code that executes cached Wan stages."""

    repo_root = Path(__file__).resolve().parents[1]
    return {
        "preprocessor": _file_revision(repo_root / "services" / "wan_animate_preprocess.py"),
        "server": _file_revision(repo_root / "services" / "wan_animate_server.py"),
    }


def _fallback_lora_path(member: Any, lora_dir: Path) -> Path | None:
    parts = Path(str(member.lora_ref)).parts
    if parts and parts[0] == "loras":
        parts = parts[1:]
    stem = "_".join(parts)
    for extension in (".safetensors", ".pt", ".ckpt"):
        candidate = lora_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


async def _character_provenance(options: Any, config: Any, deps: "AnimateWorkflowDependencies") -> dict[str, Any]:
    character = options.character
    provenance: dict[str, Any] = {"asset_sha256": {}}
    asset_ids: list[str] = []
    if character.exact_image_asset_id:
        asset_ids.append(character.exact_image_asset_id)
    wardrobe = character.wardrobe
    if wardrobe is not None:
        asset_ids.extend(wardrobe.garment_asset_ids)
        asset_ids.extend(wardrobe.accessory_asset_ids)
    for asset_id in asset_ids:
        asset = await _resolve_asset(deps.resolve_asset, asset_id, "image")
        provenance["asset_sha256"][asset_id] = await _asset_sha(asset)

    if character.member_id:
        member = next(
            (item for item in config.cast.members if item.id == character.member_id), None
        )
        if member is not None:
            provenance["member"] = _dump(member)
            params = load_cast_params(config.cast.id).get(member.id)
            provenance["render_params"] = _dump(params) if params else None
            lora_dir = Path(config.settings.lora_dir)
            lora = (
                lora_dir / params.lora_file
                if params and params.lora_file
                else _fallback_lora_path(member, lora_dir)
            )
            if lora is not None and lora.is_file():
                provenance["lora"] = {
                    "path": str(lora.resolve()),
                    "sha256": await asyncio.to_thread(_sha256_file, lora),
                    "source": "cast_params" if params and params.lora_file else "member_fallback",
                }
    return provenance


async def _voice_provenance(options: Any, config: Any) -> dict[str, Any] | None:
    if options.audio.mode != "cast_voice":
        return None
    member_id = options.audio.voice_member_id
    member = next((item for item in config.cast.members if item.id == member_id), None)
    if member is None:
        return {"member_id": member_id, "missing": True}
    params = load_cast_params(config.cast.id).get(member.id)
    voice_ref = params.voice_file if params and params.voice_file else member.voice_profile_ref
    parts = Path(voice_ref).parts
    if parts and parts[0] == "voices":
        parts = parts[1:]
    stem = Path(config.settings.voice_dir, *parts)
    path = next(
        (candidate for suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a")
         if (candidate := stem.with_suffix(suffix)).is_file()),
        None,
    )
    return {
        "member": _dump(member),
        "voice_ref": voice_ref,
        "voice_sha256": (
            await asyncio.to_thread(_sha256_file, path) if path is not None else None
        ),
    }


async def _asset_sha(asset: ResolvedAnimateAsset) -> str:
    if asset.sha256:
        return asset.sha256
    return await asyncio.to_thread(_sha256_file, asset.path)


def _manifest_path(work_dir: Path, stage_name: str) -> Path:
    return work_dir / "animate_manifests" / f"{stage_name}.json"


def _read_manifest(work_dir: Path, stage_name: str, fingerprint: str) -> dict[str, Any] | None:
    path = _manifest_path(work_dir, stage_name)
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        manifest.get("workflow_version") != DIRECT_WORKFLOW_VERSION
        or manifest.get("fingerprint") != fingerprint
    ):
        return None
    for output in (manifest.get("outputs") or {}).values():
        output_path = Path(str(output.get("path", "")))
        if not output_path.is_file():
            return None
        expected_sha = str(output.get("sha256") or "")
        if expected_sha and _sha256_file(output_path) != expected_sha:
            return None
    return manifest


def _write_manifest(
    work_dir: Path,
    stage_name: str,
    fingerprint: str,
    inputs: Any,
    outputs: dict[str, Path],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = _manifest_path(work_dir, stage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_payload: dict[str, Any] = {}
    for name, output in outputs.items():
        if not output.is_file():
            raise FileNotFoundError(f"{stage_name} did not produce {name}: {output}")
        output_payload[name] = {
            "path": str(output.resolve()),
            "sha256": _sha256_file(output),
            "size_bytes": output.stat().st_size,
        }
    payload = {
        "schema_version": 1,
        "workflow_version": DIRECT_WORKFLOW_VERSION,
        "stage": stage_name,
        "fingerprint": fingerprint,
        "inputs": _dump(inputs),
        "outputs": output_payload,
        "metadata": _dump(metadata or {}),
        "created_at": _utc_now(),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
    return payload


def _notify(
    hook: Callable[..., None] | None,
    stage_name: str,
    event_type: str,
    message: str,
) -> None:
    if hook:
        hook(stage_name, event_type, shot_id=DIRECT_SHOT_ID, message=message)


async def _run_visible_stage(
    stage_name: str,
    message: str,
    hook: Callable[..., None] | None,
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    _notify(hook, stage_name, "stage_started", message)
    try:
        result = await operation()
    except asyncio.CancelledError:
        _notify(hook, stage_name, "stage_failed", f"{stage_name} cancelled")
        raise
    except Exception as exc:
        _notify(hook, stage_name, "stage_failed", f"{stage_name} failed: {exc}")
        raise
    _notify(hook, stage_name, "stage_completed", f"{stage_name} completed")
    return result


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _record_value(record: Any, *names: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        for name in names:
            if name in record:
                return record[name]
        return default
    for name in names:
        if hasattr(record, name):
            return getattr(record, name)
    return default


async def _resolve_asset(
    resolver: AssetResolver,
    asset_id: str,
    expected_kind: str,
) -> ResolvedAnimateAsset:
    record = await _maybe_await(resolver(asset_id, expected_kind))
    resolved_path: Path | None = None
    if isinstance(record, tuple) and len(record) == 2:
        record, raw_resolved_path = record
        resolved_path = Path(raw_resolved_path).expanduser().resolve()
    if isinstance(record, ResolvedAnimateAsset):
        asset = record
    else:
        kind = str(_record_value(record, "kind", "asset_kind", default=""))
        raw_path = resolved_path or _record_value(
            record, "normalized_path", "storage_path", "path", "local_path", default=""
        )
        metadata = _record_value(record, "metadata", default={}) or {}
        asset = ResolvedAnimateAsset(
            asset_id=str(_record_value(record, "asset_id", default=asset_id)),
            kind=kind,
            path=Path(str(raw_path)).expanduser().resolve(),
            sha256=str(_record_value(record, "sha256", "content_sha256", default="") or ""),
            metadata=dict(metadata),
        )
    if asset.kind and asset.kind != expected_kind:
        raise ValueError(
            f"Asset {asset_id} is {asset.kind!r}; direct Animate expected {expected_kind!r}"
        )
    if not asset.path.is_file():
        raise FileNotFoundError(f"Resolved asset {asset_id} is missing: {asset.path}")
    return asset


async def _wan_health_payload(base_url: str) -> dict[str, Any] | None:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/health")
            response.raise_for_status()
        return dict(response.json())
    except Exception:
        return None


async def ensure_wan_animate_process_running(
    settings: Any,
    *,
    notify: Callable[..., None] | None = None,
    poll_sec: float = 2.0,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Ensure the lightweight deferred-loading service is reachable.

    Dashboard jobs can choose Animate even when the global video adapter is a
    different backend, so ``start_services.sh`` may not have launched port
    8033.  Local URLs are started on demand; remote URLs fail clearly rather
    than starting an unrelated local process.
    """

    base_url = str(settings.wan_animate_base_url).rstrip("/")
    payload = await _wan_health_payload(base_url)
    if payload is not None:
        _validate_wan_health(
            payload, expected_model_dir=getattr(settings, "wan_animate_model_dir", None)
        )
        return payload

    parsed = urlparse(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(f"Remote Wan Animate service is unreachable: {base_url}")

    python_bin = Path(str(settings.wan_animate_python)).expanduser()
    if not python_bin.is_file():
        raise FileNotFoundError(
            f"Wan Animate Python is missing: {python_bin}. Run setup_gpu.sh --with-wan-animate."
        )
    wan_dir = Path(settings.wan_animate_repo_dir).expanduser().resolve()
    model_dir = Path(settings.wan_animate_model_dir).expanduser().resolve()
    if not wan_dir.is_dir():
        raise FileNotFoundError(f"Wan repository is missing: {wan_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Wan Animate checkpoint is missing: {model_dir}")

    _notify(
        notify,
        "wan_animate_service_start",
        "stage_started",
        "Wan Animate service is not running; starting its deferred-loading server",
    )
    port = parsed.port or 8033
    repo_root = Path(__file__).resolve().parent.parent
    log_dir = Path(settings.data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "wan_animate.log"
    env = {
        **os.environ,
        "WAN_DIR": str(wan_dir),
        "WAN_ANIMATE_MODEL_DIR": str(model_dir),
        "WAN_ANIMATE_DATA_ROOT": str(Path(settings.wan_animate_data_root).resolve()),
        "WAN_REQUIRE_FLASH_ATTN_3": "true",
    }
    with log_path.open("ab") as log_file:
        await asyncio.create_subprocess_exec(
            str(python_bin),
            "-m",
            "uvicorn",
            "services.wan_animate_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )

    timeout = float(
        timeout_sec
        if timeout_sec is not None
        else getattr(settings, "wan_animate_service_start_timeout_sec", 60.0)
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_sec)
        payload = await _wan_health_payload(base_url)
        if payload is not None:
            _validate_wan_health(payload, expected_model_dir=model_dir)
            _notify(
                notify,
                "wan_animate_service_start",
                "stage_completed",
                "Wan Animate deferred-loading service is ready",
            )
            return payload
    raise TimeoutError(
        f"Wan Animate service did not start within {timeout:.0f}s; check {log_path}"
    )


def _validate_wan_health(
    payload: dict[str, Any], *, expected_model_dir: str | Path | None = None
) -> None:
    if payload.get("status") != "ok":
        raise RuntimeError(f"Wan Animate service is unhealthy: {payload}")
    if payload.get("require_flash_attn_3", True) and not payload.get("flash_attn_3", False):
        raise RuntimeError(
            "Wan Animate service is running without required FlashAttention-3 "
            "(flash_attn_interface / Hopper sm_90a)"
        )
    if payload.get("error"):
        raise RuntimeError(f"Wan Animate service load error: {payload['error']}")
    if expected_model_dir is not None:
        reported_model_dir = payload.get("model_dir")
        if not reported_model_dir:
            raise RuntimeError(
                "Wan Animate health response does not identify its model_dir; "
                "restart the service with the current video_me server"
            )
        expected = Path(str(expected_model_dir)).expanduser().resolve()
        reported = Path(str(reported_model_dir)).expanduser().resolve()
        if reported != expected:
            raise RuntimeError(
                "Wan Animate service checkpoint mismatch: "
                f"expected {expected}, service reports {reported}. Restart the service "
                "with WAN_ANIMATE_MODEL_DIR set to the configured checkpoint."
            )


async def cleanup_wan_animate_processes(*, kill_service: bool = False) -> None:
    """Best-effort cancellation cleanup for subprocesses that outlive awaiters."""

    patterns = ["services.wan_animate_preprocess"]
    if kill_service:
        patterns.append("services.wan_animate_server:app")
    for pattern in patterns:
        try:
            process = await asyncio.create_subprocess_exec(
                "pkill",
                "-f",
                pattern,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
        except Exception:
            logger.warning("Could not terminate %s during cancellation", pattern, exc_info=True)


def _default_asset_resolver(asset_store: Any, *, job_id: str) -> AssetResolver:
    def resolve(asset_id: str, expected_kind: str) -> Any:
        for method_name in ("resolve", "resolve_asset", "get_asset", "get"):
            method = getattr(asset_store, method_name, None)
            if method is None:
                continue
            try:
                return method(asset_id, expected_kind=expected_kind, job_id=job_id)
            except TypeError:
                try:
                    return method(asset_id, expected_kind=expected_kind)
                except TypeError:
                    try:
                        return method(asset_id, expected_kind)
                    except TypeError:
                        return method(asset_id)
        raise TypeError("Asset store does not expose resolve/get_asset/get")

    return resolve


def build_default_dependencies(
    *,
    config: Any,
    job_id: str,
    asset_store: Any,
    options: Any,
    image_approval: Any | None,
    stage_hook: Callable[..., None] | None,
) -> AnimateWorkflowDependencies:
    """Build production adapters while keeping the direct workflow injectable."""

    from adapters.generate_video.wan_animate_adapter import WanAnimateAdapter
    from adapters.transcribe.whisper_adapter import WhisperAdapter
    from core.workflow import _make_lipsync_adapter, _make_render_adapter, _make_tts_adapter

    settings = config.settings
    work_dir = Path(settings.data_dir) / "jobs" / job_id
    advanced = options.advanced

    def advanced_int(name: str, default: int) -> int:
        value = getattr(advanced, name, None)
        return default if value is None else int(value)

    video = WanAnimateAdapter(
        work_dir=work_dir / "video" / "wan_animate",
        base_url=settings.wan_animate_base_url,
        python_bin=settings.wan_animate_python,
        wan_dir=settings.wan_animate_repo_dir,
        model_dir=settings.wan_animate_model_dir,
        mode=options.mode,
        driver_source="upload",
        timeline="sequential",
        fps=settings.wan_animate_fps,
        resolution_area=options.output.generation_area,
        subject_selection=options.driver.subject_selection,
        retarget_pose=bool(getattr(advanced, "retarget_pose", False)),
        use_flux_retarget=bool(getattr(advanced, "use_flux_retarget", False)),
        refert_num=int(getattr(advanced, "refert_num", 1)),
        sampling_steps=int(getattr(advanced, "sampling_steps", 20)),
        mask_iterations=advanced_int("mask_iterations", 3),
        mask_kernel=advanced_int("mask_kernel", 7),
        mask_w_len=advanced_int("mask_w_len", 1),
        mask_h_len=advanced_int("mask_h_len", 1),
        ffmpeg_bin=settings.ffmpeg_bin,
        ffprobe_bin=settings.ffprobe_bin,
    )

    render = None
    if options.character.look_source in {"auto_lora", "styled_lora"}:
        render = _make_render_adapter(settings, work_dir)

    transcriber = voice = None
    if options.audio.mode == "cast_voice":
        transcriber = WhisperAdapter(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=settings.whisper_download_root,
            local_files_only=settings.whisper_local_files_only,
            revision=settings.whisper_model_revision,
            vad_filter=settings.whisper_vad_filter,
            language=settings.whisper_language,
            stage_hook=stage_hook,
        )
        voice = _make_tts_adapter(settings, work_dir)

    lipsync = None
    if options.lipsync.enabled:
        lip_settings = settings.model_copy(
            update={"video_adapter": "wan_animate", "lipsync_adapter": options.lipsync.backend}
        )
        lipsync = _make_lipsync_adapter(lip_settings, work_dir)

    return AnimateWorkflowDependencies(
        resolve_asset=_default_asset_resolver(asset_store, job_id=job_id),
        video=video,
        render=render,
        image_approval=image_approval,
        transcriber=transcriber,
        voice=voice,
        lipsync=lipsync,
    )


async def _probe_media(path: Path, ffprobe_bin: str) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise ValueError(
            f"Driving video is unreadable: {stderr.decode(errors='replace')[-1000:]}"
        )
    payload = json.loads(stdout)
    duration = float((payload.get("format") or {}).get("duration") or 0)
    streams = payload.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if duration <= 0 or not video_streams:
        raise ValueError("Driving asset must contain a finite, positive-duration video stream")
    return {
        "duration_sec": duration,
        "has_audio": bool(audio_streams),
        "video": video_streams[0],
        "audio": audio_streams[0] if audio_streams else None,
    }


def _selected_range(driver: Any, duration_sec: float) -> tuple[float, float]:
    if driver.timeline == "full_driver":
        return 0.0, duration_sec
    start = float(driver.start_sec or 0.0)
    end = float(driver.end_sec or 0.0)
    if start < 0 or end <= start or end > duration_sec + 0.05:
        raise ValueError(
            f"Invalid selected driver range {start:.3f}-{end:.3f}s for {duration_sec:.3f}s asset"
        )
    return start, min(end, duration_sec)


_STYLE_TARGET_LABELS = {
    "clothing": "clothing or dress",
    "jewelry": "jewelry",
    "bags": "bags",
    "footwear": "footwear",
    "makeup": "makeup or lipstick",
    "hair": "hair styling",
    "other": "other styling details",
}


def _complete_look_change_targets(wardrobe: Any | None) -> list[str]:
    """Return a stable, backwards-compatible scope for the requested edit."""

    if wardrobe is None:
        return []
    targets: list[str] = []

    def add(target: str, when: Any) -> None:
        if when and target not in targets:
            targets.append(target)

    explicit_targets = getattr(wardrobe, "change_targets", []) or []
    for target in explicit_targets:
        add(str(target), target in _STYLE_TARGET_LABELS)
    if targets:
        return targets

    # Version-1 clients historically used these free-form fields for mixed
    # jewelry, bags, makeup, and preservation instructions. Without explicit
    # scope, a restrictive "change only" clause could contradict their text.
    if (getattr(wardrobe, "accessories", []) or []) or str(
        getattr(wardrobe, "details", "") or ""
    ).strip():
        return []

    add(
        "clothing",
        any(
            (
                str(getattr(wardrobe, "clothing_type", "") or "").strip(),
                str(getattr(wardrobe, "primary_color", "") or "").strip(),
                str(getattr(wardrobe, "material_pattern", "") or "").strip(),
                getattr(wardrobe, "garment_asset_ids", []) or [],
            )
        ),
    )
    add("jewelry", getattr(wardrobe, "jewelry", []) or [])
    add("bags", getattr(wardrobe, "bags", []) or [])
    add("footwear", str(getattr(wardrobe, "footwear", "") or "").strip())
    add("makeup", str(getattr(wardrobe, "makeup", "") or "").strip())
    add("hair", str(getattr(wardrobe, "hair", "") or "").strip())
    return targets


def _wardrobe_prompt(wardrobe: Any | None) -> str:
    if wardrobe is None:
        return ""
    change_targets = _complete_look_change_targets(wardrobe)
    explicit_targets = set(getattr(wardrobe, "change_targets", []) or [])

    def scoped(target: str, value: Any) -> Any:
        if explicit_targets and target not in explicit_targets:
            return ""
        return value

    values = [
        (
            "requested styling change scope",
            ", ".join(_STYLE_TARGET_LABELS[target] for target in change_targets),
        ),
        (
            "clothing or dress",
            scoped("clothing", getattr(wardrobe, "clothing_type", "")),
        ),
        (
            "clothing color or palette",
            scoped("clothing", getattr(wardrobe, "primary_color", "")),
        ),
        (
            "clothing material or pattern",
            scoped("clothing", getattr(wardrobe, "material_pattern", "")),
        ),
        (
            "jewelry",
            scoped("jewelry", ", ".join(getattr(wardrobe, "jewelry", []) or [])),
        ),
        ("bags", scoped("bags", ", ".join(getattr(wardrobe, "bags", []) or []))),
        (
            "footwear, including sandals or boots",
            scoped("footwear", getattr(wardrobe, "footwear", "")),
        ),
        (
            "makeup or lipstick",
            scoped("makeup", getattr(wardrobe, "makeup", "")),
        ),
        ("hair styling", scoped("hair", getattr(wardrobe, "hair", ""))),
        (
            "other jewelry, bags, or accessories",
            scoped(
                "other", ", ".join(getattr(wardrobe, "accessories", []) or [])
            ),
        ),
        (
            "custom directions within the selected scope and preservation directions",
            getattr(wardrobe, "details", ""),
        ),
    ]
    return "; ".join(
        f"{label}: {str(value).strip()}"
        for label, value in values
        if str(value).strip()
    )


async def _approve_candidates(
    candidates: list[str],
    *,
    member: Any,
    cast_id: str,
    approval: Any,
    wardrobe_prompt: str,
) -> str:
    if not candidates:
        raise RuntimeError("Character renderer returned no canonical-look candidates")
    shot = Shot(
        shot_id="canonical_look",
        scene_ref="animate_direct",
        characters_on_screen=[member.id],
        setting="canonical character reference",
        camera="full body",
        action=wardrobe_prompt or "default complete cast look",
        duration_sec=1.0,
    )
    critique = ImageCritiqueResult(
        winner_index=0,
        winner_uri=candidates[0],
        candidate_uris=candidates,
        overall_reasoning=(
            "Canonical look candidates generated once for this Animate job; "
            "the approved image is reused byte-identically for the full driver."
        ),
        origin="single",
    )
    result = await approval.run(
        ImageApprovalRequest(shots=[shot], critique_results=[critique], cast_id=cast_id)
    )
    if len(result.approved_uris) != 1:
        raise RuntimeError("Look approval did not return exactly one canonical image")
    return result.approved_uris[0]


async def _render_canonical_look(
    options: Any,
    config: Any,
    deps: AnimateWorkflowDependencies,
    work_dir: Path,
) -> Path:
    character = options.character
    look_dir = work_dir / "canonical_look"
    look_dir.mkdir(parents=True, exist_ok=True)

    if character.look_source == "exact_image":
        asset = await _resolve_asset(
            deps.resolve_asset, character.exact_image_asset_id, "image"
        )
        suffix = asset.path.suffix.lower() if asset.path.suffix else ".png"
        output = look_dir / f"reference{suffix}"
        if asset.path.resolve() != output.resolve():
            shutil.copyfile(asset.path, output)
        return output

    if deps.render is None or deps.image_approval is None:
        raise RuntimeError("Generated look mode requires renderer and look approval adapters")
    member = next(
        (item for item in config.cast.members if item.id == character.member_id), None
    )
    if member is None:
        raise ValueError(
            f"Unknown target member {character.member_id!r} in cast {config.cast.id!r}"
        )
    params = load_cast_params(config.cast.id).get(member.id)
    wardrobe = character.wardrobe if character.look_source == "styled_lora" else None
    wardrobe_prompt = _wardrobe_prompt(wardrobe)
    change_targets = _complete_look_change_targets(wardrobe)
    negative_prompt = (
        str(getattr(wardrobe, "negative_constraints", "") or "") if wardrobe else ""
    )
    garment_ids = list(getattr(wardrobe, "garment_asset_ids", []) or []) if wardrobe else []
    accessory_ids = list(getattr(wardrobe, "accessory_asset_ids", []) or []) if wardrobe else []
    supplied_reference_count = len(garment_ids) + len(accessory_ids)
    max_control_images = max(
        1, int(getattr(config.settings, "flux2_edit_max_references", 4))
    )
    # The identity-base render is itself the first FLUX.2 control image. Reject
    # an oversized request before paying for that base render.
    if supplied_reference_count + 1 > max_control_images:
        raise ValueError(
            "Too many complete-look reference images: FLUX.2 accepts "
            f"{max_control_images} total controls, and the cast identity base uses one; "
            f"upload at most {max_control_images - 1} clothing/styling images."
        )
    reference_assets = [
        await _resolve_asset(deps.resolve_asset, asset_id, "image")
        for asset_id in garment_ids + accessory_ids
    ]

    common = dict(
        member=member,
        setting="a clean neutral premium studio backdrop",
        camera="full body, head-to-toe portrait",
        other_members=[],
        lora_file=params.lora_file if params else "",
        lora_weight=params.lora_weight if params else None,
        steps=params.steps if params else None,
        guidance_scale=params.guidance_scale if params else None,
        trigger=params.trigger if params else "",
        style_suffix=params.style_suffix if params else "",
    )

    if reference_assets:
        if not bool(getattr(config.settings, "flux2_edit_enabled", False)):
            raise RuntimeError(
                "Image-directed complete-look styling is unavailable until "
                "VIDEO_ME_FLUX2_EDIT_ENABLED=true and the Hopper smoke test passes"
            )
        base = await deps.render.run(
            RenderCharacterRequest(
                **common,
                shot_id="canonical_identity_base",
                action=(
                    "neutral natural standing pose, entire body visible, consistent identity, "
                    "coherent understated baseline styling, no crop"
                ),
                num_images=1,
            )
        )
        if not base.images:
            raise RuntimeError(
                "FLUX.2 did not produce the identity base for complete-look editing"
            )
        controls = [base.images[0], *[str(asset.path) for asset in reference_assets]]
        preserved_details = ["face", "facial structure", "skin tone", "body proportions"]
        if change_targets:
            if "hair" not in change_targets:
                preserved_details.append("hair")
            if "makeup" not in change_targets:
                preserved_details.append("makeup")
            scope_instruction = (
                "the requested styling change scope is authoritative; "
                "preserve every untargeted styling category from the identity base; "
                "for a selected category without detailed direction, choose a tasteful, "
                "coherent option; "
            )
        else:
            scope_instruction = (
                "apply only the styling described by the text and reference images; "
                "leave unrelated aspects of the identity-base look unchanged where possible; "
            )
        reference_roles = ["control image 1 is the cast identity and baseline look"]
        if garment_ids:
            reference_roles.append(
                f"the next {len(garment_ids)} control image(s) are clothing or dress references"
            )
        if accessory_ids:
            reference_roles.append(
                f"the final {len(accessory_ids)} control image(s) are jewelry, bag, "
                "footwear, makeup, or other styling references"
            )
        rendered = await deps.render.run(
            RenderCharacterRequest(
                **common,
                shot_id="canonical_outfit_edit",
                action=(
                    f"preserve the exact person's {', '.join(preserved_details)}; "
                    "apply this requested complete-look styling: "
                    f"{wardrobe_prompt or 'the supplied styling references'}; "
                    f"{scope_instruction}"
                    f"{'; '.join(reference_roles)}; never copy another person's identity, "
                    "body, pose, or background from a styling reference; full body visible, "
                    "natural fashion pose"
                ),
                control_image_uris=controls,
                negative_prompt=negative_prompt,
            )
        )
    else:
        rendered = await deps.render.run(
            RenderCharacterRequest(
                **common,
                shot_id="canonical_look_candidates",
                action=(
                    "neutral natural standing pose, entire body visible from head to toe, "
                    "single person, clear face and hands, no crop"
                    + (
                        f", complete-look styling specification: {wardrobe_prompt}; "
                        "keep the cast member's identity unchanged and use tasteful, coherent "
                        "defaults for unspecified styling categories"
                        if wardrobe_prompt
                        else ""
                    )
                ),
                negative_prompt=negative_prompt,
            )
        )

    approved = await _approve_candidates(
        rendered.images,
        member=member,
        cast_id=config.cast.id,
        approval=deps.image_approval,
        wardrobe_prompt=wardrobe_prompt,
    )
    source = Path(approved).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Approved canonical look is missing: {source}")
    output = look_dir / f"reference{source.suffix.lower() or '.png'}"
    if source.resolve() != output.resolve():
        shutil.copyfile(source, output)
    return output


async def _extract_audio(
    source: Path,
    output: Path,
    start_sec: float,
    end_sec: float,
    ffmpeg_bin: str,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{end_sec - start_sec:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        f"apad,atrim=0:{end_sec - start_sec:.6f},asetpts=N/SR/TB",
        "-c:a",
        "pcm_s16le",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"Driving-audio extraction failed: {stderr.decode(errors='replace')[-1500:]}"
        )
    return output


async def _make_silence(path: Path, duration: float, ffmpeg_bin: str) -> None:
    process = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        f"{max(duration, 0.001):.6f}",
        "-c:a",
        "pcm_s16le",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Silence generation failed: {stderr.decode(errors='replace')[-800:]}")


async def _concat_audio(parts: list[Path], output: Path, duration: float, ffmpeg_bin: str) -> None:
    concat_file = output.parent / "cast_voice_concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{part.resolve()}'" for part in parts), encoding="utf-8"
    )
    process = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-af",
        f"apad,atrim=0:{duration:.6f},asetpts=N/SR/TB",
        "-ar",
        "44100",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Cast-voice assembly failed: {stderr.decode(errors='replace')[-1500:]}")


async def _build_cast_voice(
    source_audio: Path,
    duration_sec: float,
    options: Any,
    config: Any,
    deps: AnimateWorkflowDependencies,
    work_dir: Path,
) -> Path:
    if deps.transcriber is None or deps.voice is None:
        raise RuntimeError("Cast voice requires transcription and voice adapters")
    transcription_error: BaseException | None = None
    try:
        transcript = await deps.transcriber.run(
            TranscribeRequest(
                audio_uri=str(source_audio),
                isolate_vocals=bool(
                    getattr(
                        getattr(config, "settings", None),
                        "whisper_isolate_vocals",
                        False,
                    )
                ),
            )
        )
    except BaseException as exc:
        transcription_error = exc
        raise
    finally:
        try:
            unload = getattr(deps.transcriber, "unload", None)
            if unload is not None:
                await asyncio.shield(_maybe_await(unload()))
        except BaseException:
            if transcription_error is None:
                raise
            logger.exception("Whisper cleanup failed after transcription failure")

    segments: list[tuple[Any, float, float]] = []
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        start = max(0.0, min(float(segment.start), duration_sec))
        end = max(start, min(float(segment.end), duration_sec))
        if end - start > 0.02:
            segments.append((segment, start, end))
    segments.sort(key=lambda item: (item[1], item[2]))
    if not segments:
        raise RuntimeError("No speech was detected in the selected driving-video range")
    previous_end = 0.0
    for _segment, start, end in segments:
        if start < previous_end - 0.05:
            raise RuntimeError(
                "Whisper returned overlapping speech segments that cannot be safely "
                "re-voiced without mixing voices; adjust the selected range or use "
                "source audio."
            )
        previous_end = max(previous_end, end)

    member_id = options.audio.voice_member_id or options.character.member_id
    member = next((item for item in config.cast.members if item.id == member_id), None)
    if member is None:
        raise ValueError(f"Cast voice member {member_id!r} is not in cast {config.cast.id!r}")
    params = load_cast_params(config.cast.id).get(member.id)
    voice_ref = params.voice_file if params and params.voice_file else member.voice_profile_ref
    voice_error: BaseException | None = None
    try:
        await prepare_voice_model(deps.voice, config.settings)

        from core.workflow import _fit_audio_to_duration

        pieces_dir = work_dir / "audio" / "cast_voice_parts"
        pieces_dir.mkdir(parents=True, exist_ok=True)
        parts: list[Path] = []
        cursor = 0.0
        for index, (segment, start, end) in enumerate(segments):
            # Tolerate only sub-50ms timestamp jitter by trimming it. Larger
            # overlap was rejected above instead of silently moving dialogue.
            slot_start = max(start, cursor)
            if end - slot_start <= 0.02:
                continue
            if slot_start > cursor + 0.01:
                silence = pieces_dir / f"{index:04d}_silence.wav"
                await _make_silence(
                    silence, slot_start - cursor, config.settings.ffmpeg_bin
                )
                parts.append(silence)
            track = await deps.voice.run(
                VoiceRequest(
                    text=segment.text.strip(),
                    voice_profile_ref=voice_ref,
                    speaker_id=member.id,
                    language=transcript.language or "en",
                )
            )
            fitted = pieces_dir / f"{index:04d}_voice.wav"
            await _fit_audio_to_duration(
                track,
                end - slot_start,
                fitted,
                config.settings.ffmpeg_bin,
                config.settings.ffprobe_bin,
            )
            parts.append(fitted)
            cursor = end
        if cursor < duration_sec - 0.01:
            silence = pieces_dir / "9999_silence.wav"
            await _make_silence(silence, duration_sec - cursor, config.settings.ffmpeg_bin)
            parts.append(silence)
        if not parts:
            raise RuntimeError("Cast voice produced no timed audio pieces")

        output = work_dir / "audio" / "cast_voice.wav"
        await _concat_audio(parts, output, duration_sec, config.settings.ffmpeg_bin)
        return output
    except BaseException as exc:
        voice_error = exc
        raise
    finally:
        async def release_voice_gpu() -> None:
            errors: list[BaseException] = []
            try:
                await ensure_video_model_unloaded(deps.voice)
            except BaseException as exc:
                errors.append(exc)
            # Fish S2 retains CUDA allocations after /unload. Always stop the
            # process, including when preparation or synthesis failed.
            try:
                await stop_fish_s2_process()
            except BaseException as exc:
                errors.append(exc)
            if errors:
                raise RuntimeError("Cast-voice GPU cleanup failed") from errors[0]

        try:
            await asyncio.shield(release_voice_gpu())
        except BaseException:
            if voice_error is None:
                raise
            logger.exception("Fish/voice cleanup failed after cast-voice failure")


async def _mux_audio(video: Path, audio: Path, output: Path, ffmpeg_bin: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Audio mux failed: {stderr.decode(errors='replace')[-1500:]}")
    return output


async def _export_video(
    source: Path,
    output: Path,
    output_options: Any,
    ffmpeg_bin: str,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_options.export == "generated" and output_options.target_fps == "generated":
        if source.resolve() != output.resolve():
            shutil.copyfile(source, output)
        return output

    filters: list[str] = []
    if output_options.export == "scale_1080p":
        # Contract says 1080-pixel long edge and preserve aspect ratio.  -2 asks
        # ffmpeg to keep dimensions even for H.264 without padding/stretching.
        filters.append("scale='if(gte(iw,ih),1080,-2)':'if(gte(iw,ih),-2,1080)':flags=lanczos")
    elif output_options.export == "vertical_1080x1920":
        # Optional vertical canvas: fit the complete source inside 9:16, then
        # letterbox it. This never crops or stretches the generated frames.
        filters.append(
            "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    if output_options.target_fps == 48:
        filters.append("minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir")
    process = await asyncio.create_subprocess_exec(
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Animate export failed: {stderr.decode(errors='replace')[-1500:]}")
    return output


async def _probe_duration(path: Path, ffprobe_bin: str) -> float:
    metadata = await _probe_media(path, ffprobe_bin)
    return float(metadata["duration_sec"])


def _assert_duration_close(
    stage: str,
    actual_sec: float,
    expected_sec: float,
    tolerance_sec: float,
) -> None:
    if not math.isfinite(actual_sec) or actual_sec <= 0:
        raise RuntimeError(f"{stage} produced an invalid duration: {actual_sec!r}")
    difference = abs(actual_sec - expected_sec)
    if difference > tolerance_sec:
        raise RuntimeError(
            f"{stage} duration {actual_sec:.3f}s differs from the selected driver "
            f"range {expected_sec:.3f}s by {difference:.3f}s (allowed "
            f"{tolerance_sec:.3f}s); refusing a silently truncated output"
        )


async def run_wan_animate_direct_job(
    request: Any,
    app_config: Any,
    job_id: str,
    *,
    asset_store: Any | None = None,
    image_approval: Any | None = None,
    stage_hook: Callable[..., None] | None = None,
    dependencies: AnimateWorkflowDependencies | None = None,
) -> AnimateWorkflowResult:
    """Run one versioned direct Animate job with semantic stage caching."""

    options = request.animate
    if options is None:
        raise ValueError("workflow_kind='wan_animate_direct' requires animate options")
    if options.schema_version != 1:
        raise ValueError(f"Unsupported direct Animate schema version: {options.schema_version}")

    config = app_config
    if options.character.look_source in {"auto_lora", "styled_lora"} and getattr(
        config.settings, "render_adapter", "musubi_flux"
    ) != "musubi_flux":
        raise RuntimeError(
            "Generated direct Animate looks require render_adapter=musubi_flux so the "
            "cast's trained FLUX.2 LoRA and complete-look controls cannot be silently ignored"
        )
    if options.advanced.use_flux_retarget:
        flux_ready, flux_reason = wan_flux_retarget_readiness(
            config.settings.wan_animate_model_dir
        )
        if not flux_ready:
            raise RuntimeError(flux_reason)
    work_dir = Path(config.settings.data_dir) / "jobs" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(config.settings.wan_animate_data_root).expanduser().resolve()
    try:
        work_dir.resolve().relative_to(prepared_root)
    except ValueError as exc:
        raise ValueError(
            f"Direct Animate job data directory {work_dir.resolve()} is outside "
            f"WAN_ANIMATE_DATA_ROOT={prepared_root}; align the two paths before queueing"
        ) from exc

    production_dependencies = dependencies is None
    if production_dependencies:
        if asset_store is None:
            raise TypeError("asset_store is required when dependencies are not injected")
        dependencies = build_default_dependencies(
            config=config,
            job_id=job_id,
            asset_store=asset_store,
            options=options,
            image_approval=image_approval,
            stage_hook=stage_hook,
        )
    deps = dependencies

    try:
        driver_asset = await _resolve_asset(
            deps.resolve_asset, options.driver.asset_id, "video"
        )
        driver_sha = await _asset_sha(driver_asset)

        async def readiness() -> dict[str, Any]:
            health = await ensure_wan_animate_process_running(
                config.settings, notify=stage_hook
            )
            if options.character.look_source in {"auto_lora", "styled_lora"}:
                if deps.render is None:
                    raise RuntimeError("Generated look mode has no configured renderer")
                render_health = await deps.render.health()
                if render_health.status == "down":
                    raise RuntimeError(f"Character renderer unavailable: {render_health.reason}")
            if options.lipsync.enabled:
                if deps.lipsync is None:
                    raise RuntimeError("Lip-sync is enabled but no adapter is configured")
                lip_health = await deps.lipsync.health()
                if lip_health.status == "down":
                    raise RuntimeError(f"Lip-sync service unavailable: {lip_health.reason}")
            return health

        health_payload = await _run_visible_stage(
            "animate_validate",
            "Validating direct Animate assets, models, and services",
            stage_hook,
            readiness,
        )
        if production_dependencies:
            async def reset_gpu_before_pipeline() -> None:
                # Default adapters may point at already-running services left by
                # a prior dashboard job. Start every production direct job from
                # a known VRAM state before FLUX/Whisper/Fish preprocessing.
                await ensure_video_model_unloaded(deps.video, notify=stage_hook)
                await free_comfyui(config.settings.comfyui_base_url)
                await stop_fish_s2_process()

            await _run_visible_stage(
                "animate_gpu_reset",
                "Clearing prior video, ComfyUI, and voice models from GPU memory",
                stage_hook,
                reset_gpu_before_pipeline,
            )

        wan_repository_revision = (
            _directory_revision(Path(config.settings.wan_animate_repo_dir))
            if getattr(config.settings, "wan_animate_repo_dir", None)
            else None
        )
        wan_model_revision = _directory_revision(
            Path(config.settings.wan_animate_model_dir)
        )
        wan_runtime_revisions = _wan_runtime_revisions()
        video_semantics = _video_snapshot(deps.video)
        driver_metadata = await _probe_media(driver_asset.path, config.settings.ffprobe_bin)
        start_sec, end_sec = _selected_range(options.driver, driver_metadata["duration_sec"])
        duration_sec = end_sec - start_sec
        if duration_sec > WAN_ANIMATE_MAX_DRIVER_RANGE_SEC:
            raise ValueError(
                "Direct Animate range exceeds the safe "
                f"{WAN_ANIMATE_MAX_DRIVER_RANGE_SEC:.0f}-second limit; select a shorter range"
            )
        if options.audio.mode in {"driver", "cast_voice"} and not driver_metadata["has_audio"]:
            raise ValueError(
                f"Audio mode {options.audio.mode!r} requires an audio stream in the driving video"
            )

        uses_character_renderer = options.character.look_source in {
            "auto_lora",
            "styled_lora",
        }
        character_inputs = {
            "character": options.character,
            "provenance": await _character_provenance(options, config, deps),
            "renderer": _render_snapshot(deps.render) if uses_character_renderer else None,
            "render_settings": (
                _settings_snapshot(
                    config.settings,
                    (
                        "render_adapter",
                        "image_candidates",
                        "render_allow_placeholder_lora",
                        "lora_dir",
                        "flux2_edit_enabled",
                        "flux2_edit_max_references",
                    ),
                )
                if uses_character_renderer
                else None
            ),
        }
        look_fp = _fingerprint(character_inputs)
        look_manifest = _read_manifest(work_dir, "canonical_look", look_fp)
        if look_manifest:
            canonical_look = Path(look_manifest["outputs"]["reference"]["path"])
            _notify(
                stage_hook,
                "canonical_look",
                "stage_completed",
                "Canonical look reused from matching manifest",
            )
        else:
            canonical_look = await _run_visible_stage(
                "canonical_look",
                "Resolving one canonical character look for the full driving video",
                stage_hook,
                lambda: _render_canonical_look(options, config, deps, work_dir),
            )
            look_manifest = _write_manifest(
                work_dir,
                "canonical_look",
                look_fp,
                character_inputs,
                {"reference": canonical_look},
                metadata={"look_source": options.character.look_source},
            )
        look_sha = str(look_manifest["outputs"]["reference"]["sha256"])

        driver = VideoDriver(
            uri=str(driver_asset.path),
            start_sec=start_sec,
            end_sec=end_sec,
            mode=options.mode,
        )
        video_request = VideoRequest(
            image_uri=str(canonical_look),
            action="transfer the selected person's motion to the canonical character",
            duration_sec=duration_sec,
            shot_id=DIRECT_SHOT_ID,
            driver=driver,
        )
        prep_inputs = {
            "driver_sha256": driver_sha,
            "range": [start_sec, end_sec],
            "look_sha256": look_sha,
            "mode": options.mode,
            "generation_area": options.output.generation_area,
            "subject_selection": options.driver.subject_selection,
            "advanced": options.advanced,
            "preprocessor": video_semantics,
            "preprocessor_runtime": wan_runtime_revisions["preprocessor"],
            "wan_repository": wan_repository_revision,
            "model_revision": wan_model_revision,
            "preprocess_settings": _settings_snapshot(
                config.settings,
                (
                    "wan_animate_fps",
                    "ffmpeg_bin",
                    "ffprobe_bin",
                ),
            ),
        }
        prep_fp = _fingerprint(prep_inputs)
        prep_manifest = _read_manifest(work_dir, "animate_preprocess", prep_fp)
        if prep_manifest:
            driver.prepared_dir = str(prep_manifest["metadata"]["prepared_dir"])
            assert video_request.driver is not None
            video_request.driver.prepared_dir = driver.prepared_dir
            _notify(
                stage_hook,
                "animate_preprocess",
                "stage_completed",
                "Wan Animate preprocessing reused from matching manifest",
            )
        else:
            async def preprocess() -> Any:
                prepared = await deps.video.prepare_inputs([video_request])
                result = prepared.get(DIRECT_SHOT_ID)
                if result is None:
                    raise RuntimeError("Wan Animate preprocessor returned no direct-job input")
                driver.prepared_dir = result.prepared_dir
                assert video_request.driver is not None
                video_request.driver.prepared_dir = result.prepared_dir
                return result

            prepared = await _run_visible_stage(
                "animate_preprocess",
                "Preparing pose, face, and replacement-mask inputs on CUDA",
                stage_hook,
                preprocess,
            )
            prepared_dir = Path(prepared.prepared_dir)
            required = ["src_ref.png", "src_pose.mp4", "src_face.mp4"]
            if options.mode == "replace":
                required += ["src_bg.mp4", "src_mask.mp4"]
            prep_manifest = _write_manifest(
                work_dir,
                "animate_preprocess",
                prep_fp,
                prep_inputs,
                {name: prepared_dir / name for name in required},
                metadata={
                    "prepared_dir": str(prepared_dir.resolve()),
                    "cache_hit": bool(getattr(prepared, "cache_hit", False)),
                },
            )

        audio_path: Path | None = None
        if options.audio.mode != "none":
            cast_voice_inputs = (
                {
                    "voice_provenance": await _voice_provenance(options, config),
                    "transcriber": _transcriber_snapshot(deps.transcriber),
                    "voice_adapter": _voice_snapshot(deps.voice),
                    "whisper_settings": _settings_snapshot(
                        config.settings,
                        (
                            "whisper_model_size",
                            "whisper_device",
                            "whisper_compute_type",
                            "whisper_download_root",
                            "whisper_local_files_only",
                            "whisper_model_revision",
                            "whisper_language",
                            "whisper_vad_filter",
                            "whisper_isolate_vocals",
                        ),
                    ),
                    "tts_settings": _settings_snapshot(
                        config.settings,
                        (
                            "tts_adapter",
                            "tts_base_url",
                            "fish_s2_base_url",
                            "fish_s2_speech_dir",
                            "fish_s2_venv_python",
                            "fish_s2_load_timeout_sec",
                        ),
                    ),
                }
                if options.audio.mode == "cast_voice"
                else None
            )
            audio_inputs = {
                "driver_sha256": driver_sha,
                "range": [start_sec, end_sec],
                "audio": options.audio,
                "cast_voice": cast_voice_inputs,
            }
            audio_fp = _fingerprint(audio_inputs)
            audio_manifest = _read_manifest(work_dir, "animate_audio", audio_fp)
            if audio_manifest:
                audio_path = Path(audio_manifest["outputs"]["audio"]["path"])
                _notify(
                    stage_hook,
                    "animate_audio",
                    "stage_completed",
                    "Direct Animate audio reused from matching manifest",
                )
            else:
                async def make_audio() -> Path:
                    extracted = work_dir / "audio" / "driver_audio.wav"
                    await _extract_audio(
                        driver_asset.path,
                        extracted,
                        start_sec,
                        end_sec,
                        config.settings.ffmpeg_bin,
                    )
                    if options.audio.mode == "driver":
                        return extracted
                    return await _build_cast_voice(
                        extracted, duration_sec, options, config, deps, work_dir
                    )

                audio_path = await _run_visible_stage(
                    "animate_audio",
                    (
                        "Extracting driving-video audio"
                        if options.audio.mode == "driver"
                        else "Transcribing and re-voicing driving speech with the selected cast voice"
                    ),
                    stage_hook,
                    make_audio,
                )
                _write_manifest(
                    work_dir,
                    "animate_audio",
                    audio_fp,
                    audio_inputs,
                    {"audio": audio_path},
                    metadata={"mode": options.audio.mode},
                )

        generation_inputs = {
            "preprocess_fingerprint": prep_fp,
            "look_sha256": look_sha,
            "mode": options.mode,
            "generation_area": options.output.generation_area,
            "advanced": options.advanced,
            "video_adapter": video_semantics,
            "service": _stable_wan_health(health_payload),
            "server_runtime": wan_runtime_revisions["server"],
            "wan_repository": wan_repository_revision,
            "model_revision": wan_model_revision,
        }
        generation_fp = _fingerprint(generation_inputs)
        generation_manifest = _read_manifest(work_dir, "animate_generate", generation_fp)
        if generation_manifest:
            raw_path = Path(generation_manifest["outputs"]["raw_video"]["path"])
            raw_duration = float(
                generation_manifest.get("metadata", {}).get("duration_sec")
                or await _probe_duration(raw_path, config.settings.ffprobe_bin)
            )
            raw_clip = VideoClip(
                uri=str(raw_path), duration_sec=raw_duration, shot_id=DIRECT_SHOT_ID
            )
            _notify(
                stage_hook,
                "animate_generate",
                "stage_completed",
                "Raw Wan Animate video reused from matching manifest",
            )
        else:
            await free_comfyui(config.settings.comfyui_base_url)
            await prepare_video_model(deps.video, config.settings, notify=stage_hook)
            raw_clip = await _run_visible_stage(
                "animate_generate",
                "Generating the full direct Animate range with one Wan model load",
                stage_hook,
                lambda: deps.video.run(video_request),
            )
            raw_path = Path(raw_clip.uri)
            raw_duration = await _probe_duration(raw_path, config.settings.ffprobe_bin)
            raw_clip = raw_clip.model_copy(update={"duration_sec": raw_duration})
            generation_manifest = _write_manifest(
                work_dir,
                "animate_generate",
                generation_fp,
                generation_inputs,
                {"raw_video": raw_path},
                metadata={"duration_sec": raw_duration},
            )
        duration_tolerance = max(
            0.05,
            float(getattr(config.settings, "av_sync_duration_tolerance_sec", 0.35)),
        )
        _assert_duration_close(
            "Wan Animate generation",
            raw_duration,
            duration_sec,
            duration_tolerance,
        )
        # Mandatory phase boundary: do not co-reside Wan and lip-sync/enhance.
        await ensure_video_model_unloaded(deps.video, notify=stage_hook)

        selected_path = Path(raw_clip.uri)
        if options.lipsync.enabled:
            if audio_path is None:
                raise ValueError("Lip-sync cannot run when Animate audio mode is 'none'")
            lipsync_inputs = {
                "raw_sha256": generation_manifest["outputs"]["raw_video"]["sha256"],
                "audio_sha256": _sha256_file(audio_path),
                "lipsync": options.lipsync,
                "adapter": _lipsync_snapshot(deps.lipsync),
                "settings": _settings_snapshot(
                    config.settings,
                    (
                        "lipsync_adapter",
                        "lipsync_base_url",
                        "musetalk_base_url",
                        "latentsync_base_url",
                        "latentsync_inference_steps",
                        "latentsync_guidance_scale",
                        "lipsync_failure_policy",
                        "lipsync_max_retries",
                    ),
                ),
            }
            lipsync_fp = _fingerprint(lipsync_inputs)
            lipsync_manifest = _read_manifest(work_dir, "animate_lipsync", lipsync_fp)
            if lipsync_manifest:
                selected_path = Path(lipsync_manifest["outputs"]["video"]["path"])
                _notify(
                    stage_hook,
                    "animate_lipsync",
                    "stage_completed",
                    "Lip-synced video reused from matching manifest",
                )
            else:
                synced = await _run_visible_stage(
                    "animate_lipsync",
                    f"Repairing mouth timing with {options.lipsync.backend}",
                    stage_hook,
                    lambda: deps.lipsync.run(
                        LipSyncRequest(
                            video_uri=str(selected_path),
                            audio_uri=str(audio_path),
                            shot_id=DIRECT_SHOT_ID,
                        )
                    ),
                )
                if getattr(deps.lipsync, "last_application_status", None) == "passthrough":
                    raise RuntimeError(
                        "Lip-sync was requested, but MuseTalk could not detect a usable "
                        "face and returned the video unchanged. Review the character/driver "
                        "framing or use LatentSync."
                    )
                selected_path = Path(synced.uri)
                synced_duration = await _probe_duration(
                    selected_path, config.settings.ffprobe_bin
                )
                _assert_duration_close(
                    "Lip-sync",
                    synced_duration,
                    duration_sec,
                    duration_tolerance,
                )
                _write_manifest(
                    work_dir,
                    "animate_lipsync",
                    lipsync_fp,
                    lipsync_inputs,
                    {"video": selected_path},
                    metadata={"backend": options.lipsync.backend},
                )

        if audio_path is not None:
            mux_inputs = {
                "video_sha256": _sha256_file(selected_path),
                "audio_sha256": _sha256_file(audio_path),
                "ffmpeg_bin": config.settings.ffmpeg_bin,
            }
            mux_fp = _fingerprint(mux_inputs)
            mux_manifest = _read_manifest(work_dir, "animate_mux", mux_fp)
            if mux_manifest:
                selected_path = Path(mux_manifest["outputs"]["video"]["path"])
                _notify(
                    stage_hook,
                    "animate_mux",
                    "stage_completed",
                    "Final audio mux reused from matching manifest",
                )
            else:
                muxed_path = work_dir / "assembled" / "with_audio.mp4"
                selected_path = await _run_visible_stage(
                    "animate_mux",
                    "Muxing the selected audio onto the generated video",
                    stage_hook,
                    lambda: _mux_audio(
                        selected_path, audio_path, muxed_path, config.settings.ffmpeg_bin
                    ),
                )
                _write_manifest(
                    work_dir,
                    "animate_mux",
                    mux_fp,
                    mux_inputs,
                    {"video": selected_path},
                )

        final_inputs = {
            "selected_sha256": _sha256_file(selected_path),
            "output": options.output,
            "audio_mode": options.audio.mode,
        }
        final_fp = _fingerprint(final_inputs)
        final_manifest = _read_manifest(work_dir, "animate_export", final_fp)
        if final_manifest:
            final_path = Path(final_manifest["outputs"]["final_video"]["path"])
            _notify(
                stage_hook,
                "animate_export",
                "stage_completed",
                "Final Animate export reused from matching manifest",
            )
        else:
            final_path = work_dir / "assembled" / "final.mp4"
            final_path = await _run_visible_stage(
                "animate_export",
                "Exporting final video with aspect-preserving output settings",
                stage_hook,
                lambda: _export_video(
                    selected_path,
                    final_path,
                    options.output,
                    config.settings.ffmpeg_bin,
                ),
            )
            final_manifest = _write_manifest(
                work_dir,
                "animate_export",
                final_fp,
                final_inputs,
                {"final_video": final_path},
            )

        final_duration = await _probe_duration(final_path, config.settings.ffprobe_bin)
        _assert_duration_close(
            "Final Animate export",
            final_duration,
            duration_sec,
            duration_tolerance,
        )
        result = AnimateWorkflowResult(
            job_id=job_id,
            raw_video_uri=str(Path(raw_clip.uri).resolve()),
            final_video_uri=str(final_path.resolve()),
            canonical_look_uri=str(canonical_look.resolve()),
            audio_uri=str(audio_path.resolve()) if audio_path else None,
            duration_sec=final_duration,
            manifests_dir=str((work_dir / "animate_manifests").resolve()),
        )
        result_path = work_dir / "animate_direct_result.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result
    except asyncio.CancelledError:
        # Wan inference runs in the service's executor and preprocessing runs in
        # a child process; cancelling the Python await alone does not release
        # either GPU allocation.  The worker is single-job, so killing these
        # job-scoped services on cancellation is the safest deterministic reset.
        await asyncio.shield(cleanup_wan_animate_processes(kill_service=True))
        raise
