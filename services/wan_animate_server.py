"""Deferred-loading HTTP service for official Wan2.2-Animate-14B."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from core.wan_animate_readiness import (
    WAN_ANIMATE_REQUIRED_MODEL_FILES,
    wan_animate_component_ready,
)

logger = logging.getLogger(__name__)

WAN_DIR = Path(os.getenv("WAN_DIR", "/workspace/Wan2.2"))
WAN_ANIMATE_MODEL_DIR = Path(
    os.getenv("WAN_ANIMATE_MODEL_DIR", "/workspace/Wan2.2-Animate-14B")
)
WAN_ANIMATE_DATA_ROOT = Path(
    os.getenv("WAN_ANIMATE_DATA_ROOT", "/workspace/video_me/.local")
).resolve()
WAN_ANIMATE_OFFLOAD_MODEL = os.getenv("WAN_ANIMATE_OFFLOAD_MODEL", "true").lower() in {
    "1", "true", "yes"
}
WAN_REQUIRE_FLASH_ATTN_3 = os.getenv("WAN_REQUIRE_FLASH_ATTN_3", "true").lower() in {
    "1", "true", "yes"
}

_MODEL_REQUIRED_FILES = WAN_ANIMATE_REQUIRED_MODEL_FILES

# This flag deliberately means "an FA3 kernel executed successfully", not just
# "flash_attn_interface imported".  Wan itself falls back silently, so an
# import-only probe would make /health claim FA3 on an incompatible wheel/GPU.
FLASH_ATTN_3_AVAILABLE = False
_fa3_status: dict[str, object] | None = None
_fa3_lock = threading.Lock()

_pipeline = None
_pipeline_mode: str | None = None
_pipeline_error: str | None = None
_loading = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _probe_flash_attn_3() -> dict[str, object]:
    status: dict[str, object] = {
        "imported": False,
        "kernel_ready": False,
        "device_capability": None,
        "error": None,
    }
    try:
        import torch
        import flash_attn_interface as fa3

        status["imported"] = True
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        capability = tuple(int(part) for part in torch.cuda.get_device_capability())
        status["device_capability"] = list(capability)
        if capability != (9, 0):
            raise RuntimeError(
                f"Hopper compute capability 9.0 is required, got {capability}"
            )
        query = torch.randn(
            (1, 128, 8, 64), device="cuda", dtype=torch.bfloat16
        )
        fa3.flash_attn_func(query, query, query)
        torch.cuda.synchronize()
        status["kernel_ready"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _flash_attn_3_readiness(*, refresh: bool = False) -> dict[str, object]:
    global FLASH_ATTN_3_AVAILABLE, _fa3_status
    with _fa3_lock:
        if _fa3_status is None or refresh:
            _fa3_status = _probe_flash_attn_3()
            FLASH_ATTN_3_AVAILABLE = bool(_fa3_status["kernel_ready"])
        return dict(_fa3_status)


def _model_readiness_error() -> str | None:
    missing: list[str] = []
    for relative in _MODEL_REQUIRED_FILES:
        path = WAN_ANIMATE_MODEL_DIR / relative
        if not wan_animate_component_ready(path):
            missing.append(relative)
    if missing:
        shown = ", ".join(missing[:4])
        suffix = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        return f"Wan Animate model is incomplete: {shown}{suffix}"
    try:
        json.loads((WAN_ANIMATE_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (
                WAN_ANIMATE_MODEL_DIR
                / "diffusion_pytorch_model.safetensors.index.json"
            ).read_text(encoding="utf-8")
        )
        indexed_shards = set(index["weight_map"].values())
    except (AttributeError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return f"Wan Animate model metadata is invalid: {exc}"
    missing_indexed: list[str] = []
    for shard in sorted(str(value) for value in indexed_shards):
        path = WAN_ANIMATE_MODEL_DIR / shard
        try:
            valid = path.is_file() and path.stat().st_size > 0
        except OSError:
            valid = False
        if not valid:
            missing_indexed.append(shard)
    if missing_indexed:
        return f"Wan Animate model index references missing shard: {missing_indexed[0]}"
    return None


def _normalize_seed(seed: int) -> int:
    return seed if seed >= 0 else secrets.randbelow(2**31)


def _validate_encoded_video(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"Wan Animate output is missing: {path}") from exc
    if size <= 0:
        raise RuntimeError("Wan Animate encoder produced an empty MP4")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to validate Wan Animate output")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,duration,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-500:] or "unknown ffprobe error"
        raise RuntimeError(f"Wan Animate output failed ffprobe validation: {detail}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        codec = str(stream["codec_name"]).strip()
        duration = 0.0
        for raw_duration in (
            payload.get("format", {}).get("duration"),
            stream.get("duration"),
        ):
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                continue
            if duration > 0:
                break
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Wan Animate output has no valid encoded video stream") from exc
    if codec.lower() in {"", "n/a", "none"} or width <= 0 or height <= 0 or duration <= 0:
        raise RuntimeError("Wan Animate output has an invalid encoded video stream")


def _load_pipeline(mode: str) -> None:
    global _pipeline, _pipeline_mode, _pipeline_error, _loading
    if mode not in {"animate", "replace"}:
        _pipeline_error = f"invalid mode: {mode}"
        return
    if _pipeline is not None and _pipeline_mode != mode:
        _unload_pipeline()
    with _load_lock:
        if _pipeline is not None and _pipeline_mode == mode:
            return
        if _loading:
            return
        _loading = True
        _pipeline_error = None
    try:
        model_error = _model_readiness_error()
        if model_error:
            raise RuntimeError(model_error)
        # Re-run the tiny kernel at load time so a transient startup/driver
        # failure cached by an early health probe cannot permanently block a
        # healthy service process.
        fa3_status = _flash_attn_3_readiness(refresh=True)
        if WAN_REQUIRE_FLASH_ATTN_3 and not fa3_status["kernel_ready"]:
            raise RuntimeError(
                "FlashAttention-3 is required but its Hopper kernel is not ready: "
                f"{fa3_status['error']}"
            )
        if str(WAN_DIR) not in sys.path:
            sys.path.insert(0, str(WAN_DIR))
        import wan
        from wan.configs import WAN_CONFIGS

        config = WAN_CONFIGS["animate-14B"]
        _pipeline = wan.WanAnimate(
            config=config,
            checkpoint_dir=str(WAN_ANIMATE_MODEL_DIR),
            device_id=0,
            t5_cpu=True,
            init_on_cpu=True,
            use_relighting_lora=mode == "replace",
        )
        _pipeline_mode = mode
        logger.info("Wan Animate ready in %s mode", mode)
    except Exception as exc:
        _pipeline = None
        _pipeline_mode = None
        _pipeline_error = str(exc)
        logger.exception("Wan Animate load failed")
    finally:
        with _load_lock:
            _loading = False


def _unload_pipeline() -> None:
    global _pipeline, _pipeline_mode, _pipeline_error
    with _infer_lock:
        _pipeline = None
        _pipeline_mode = None
        _pipeline_error = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _safe_prepared_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(WAN_ANIMATE_DATA_ROOT)
    except ValueError as exc:
        raise HTTPException(400, detail="prepared_dir is outside WAN_ANIMATE_DATA_ROOT") from exc
    required = ["src_ref.png", "src_pose.mp4", "src_face.mp4"]
    if not path.is_dir() or not all((path / name).is_file() for name in required):
        raise HTTPException(400, detail="prepared_dir is missing Animate inputs")
    return path


def _inference(
    prepared_dir: Path,
    mode: str,
    fps: int,
    refert_num: int,
    sampling_steps: int,
    seed: int,
) -> Path:
    from wan.utils.utils import save_video

    with _infer_lock:
        tensor = _pipeline.generate(
            src_root_path=str(prepared_dir),
            replace_flag=mode == "replace",
            clip_len=77,
            refert_num=refert_num,
            sampling_steps=sampling_steps,
            seed=seed,
            offload_model=WAN_ANIMATE_OFFLOAD_MODEL,
        )
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        output = Path(handle.name)
    try:
        save_video(tensor[None], str(output), fps=fps, nrow=1)
        _validate_encoded_video(output)
        return output
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _delete_response_file(path: Path) -> None:
    path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Wan Animate service started with deferred model loading")
    yield
    _unload_pipeline()


app = FastAPI(title="Wan2.2 Animate", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    fa3_status = _flash_attn_3_readiness()
    model_error = _model_readiness_error()
    fa3_error = (
        str(fa3_status["error"])
        if WAN_REQUIRE_FLASH_ATTN_3 and not fa3_status["kernel_ready"]
        else None
    )
    readiness_error = _pipeline_error or model_error or fa3_error
    return JSONResponse({
        "status": "ok" if readiness_error is None else "down",
        "model_loaded": _pipeline is not None,
        "loading": _loading,
        "mode": _pipeline_mode,
        "error": readiness_error,
        "flash_attn_3": bool(fa3_status["kernel_ready"]),
        "flash_attn_3_imported": bool(fa3_status["imported"]),
        "flash_attn_3_kernel_ready": bool(fa3_status["kernel_ready"]),
        "flash_attn_3_device_capability": fa3_status["device_capability"],
        "flash_attn_3_error": fa3_status["error"],
        "require_flash_attn_3": WAN_REQUIRE_FLASH_ATTN_3,
        "offload_model": WAN_ANIMATE_OFFLOAD_MODEL,
        "model_dir": str(WAN_ANIMATE_MODEL_DIR),
    })


@app.post("/load")
async def load(mode: str = Form("animate")) -> JSONResponse:
    if _pipeline is not None and _pipeline_mode == mode:
        return JSONResponse({"status": "ok", "model_loaded": True, "mode": mode})
    if _loading:
        return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)
    if _pipeline is not None and _pipeline_mode != mode:
        await asyncio.get_running_loop().run_in_executor(None, _unload_pipeline)
    asyncio.get_running_loop().run_in_executor(None, _load_pipeline, mode)
    return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)


@app.post("/unload")
async def unload() -> JSONResponse:
    if _loading:
        raise HTTPException(409, detail="model load in progress")
    await asyncio.get_running_loop().run_in_executor(None, _unload_pipeline)
    return JSONResponse({"status": "ok", "model_loaded": False})


@app.post("/generate")
async def generate(
    prepared_dir: str = Form(...),
    mode: str = Form("animate"),
    fps: int = Form(30),
    refert_num: int = Form(1),
    sampling_steps: int = Form(20),
    seed: int = Form(-1),
) -> FileResponse:
    if mode not in {"animate", "replace"}:
        raise HTTPException(400, detail="mode must be animate or replace")
    if refert_num not in {1, 5}:
        raise HTTPException(400, detail="refert_num must be 1 or 5")
    if fps != 30:
        raise HTTPException(400, detail="fps must be 30")
    if not 10 <= sampling_steps <= 40:
        raise HTTPException(400, detail="sampling_steps must be between 10 and 40")
    path = _safe_prepared_dir(prepared_dir)
    if mode == "replace" and not all(
        (path / name).is_file() for name in ("src_bg.mp4", "src_mask.mp4")
    ):
        raise HTTPException(400, detail="replacement inputs are incomplete")

    if _pipeline is None or _pipeline_mode != mode:
        await asyncio.get_running_loop().run_in_executor(None, _load_pipeline, mode)
        while _loading:
            await asyncio.sleep(2)
    if _pipeline is None or _pipeline_mode != mode:
        raise HTTPException(503, detail=_pipeline_error or "Wan Animate is not ready")

    normalized_seed = _normalize_seed(seed)
    video = await asyncio.get_running_loop().run_in_executor(
        None, _inference, path, mode, fps, refert_num, sampling_steps, normalized_seed
    )
    return FileResponse(
        str(video),
        media_type="video/mp4",
        filename="wan_animate.mp4",
        headers={"X-Wan-Seed": str(normalized_seed)},
        background=BackgroundTask(_delete_response_file, video),
    )
