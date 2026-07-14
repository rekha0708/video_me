"""Deferred-loading HTTP service for official Wan2.2-Animate-14B."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

WAN_DIR = Path(os.getenv("WAN_DIR", "/workspace/Wan2.2"))
WAN_ANIMATE_MODEL_DIR = Path(
    os.getenv("WAN_ANIMATE_MODEL_DIR", "/workspace/Wan2.2-Animate-14B")
)
WAN_ANIMATE_DATA_ROOT = Path(
    os.getenv("WAN_ANIMATE_DATA_ROOT", "/workspace/video_me/.local")
).resolve()
WAN_ANIMATE_OFFLOAD_MODEL = os.getenv("WAN_ANIMATE_OFFLOAD_MODEL", "false").lower() in {
    "1", "true", "yes"
}
WAN_REQUIRE_FLASH_ATTN_3 = os.getenv("WAN_REQUIRE_FLASH_ATTN_3", "true").lower() in {
    "1", "true", "yes"
}

try:
    import flash_attn_interface  # noqa: F401
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

_pipeline = None
_pipeline_mode: str | None = None
_pipeline_error: str | None = None
_loading = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


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
        if WAN_REQUIRE_FLASH_ATTN_3 and not FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError("FlashAttention-3 is required but flash_attn_interface is unavailable")
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
) -> bytes:
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
        return output.read_bytes()
    finally:
        output.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Wan Animate service started with deferred model loading")
    yield
    _unload_pipeline()


app = FastAPI(title="Wan2.2 Animate", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "loading": _loading,
        "mode": _pipeline_mode,
        "error": _pipeline_error,
        "flash_attn_3": FLASH_ATTN_3_AVAILABLE,
        "require_flash_attn_3": WAN_REQUIRE_FLASH_ATTN_3,
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
) -> Response:
    if mode not in {"animate", "replace"}:
        raise HTTPException(400, detail="mode must be animate or replace")
    if refert_num not in {1, 5}:
        raise HTTPException(400, detail="refert_num must be 1 or 5")
    path = _safe_prepared_dir(prepared_dir)
    if mode == "replace" and not all((path / name).is_file() for name in ("src_bg.mp4", "src_mask.mp4")):
        raise HTTPException(400, detail="replacement inputs are incomplete")

    if _pipeline is None or _pipeline_mode != mode:
        await asyncio.get_running_loop().run_in_executor(None, _load_pipeline, mode)
        while _loading:
            await asyncio.sleep(2)
    if _pipeline is None or _pipeline_mode != mode:
        raise HTTPException(503, detail=_pipeline_error or "Wan Animate is not ready")

    video = await asyncio.get_running_loop().run_in_executor(
        None, _inference, path, mode, fps, refert_num, sampling_steps, seed
    )
    return Response(video, media_type="video/mp4")
