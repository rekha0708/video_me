"""
Wan2.2 image-to-video HTTP service — deferred-loading edition.

The model is NOT loaded at startup. The pipeline orchestrator loads it only when
the video phase begins (after render_character finishes), so Wan never competes
with Flux/Ollama for VRAM during the image-rendering phase.

API contract:
  GET  /health    → 200 {"status": "ok", "model_loaded": bool, "loading": bool,
                         "error": str|null}   — 200 whenever the process is up
  POST /load      → start loading the model in a background thread.
                    200 if already loaded, 202 if load started or in progress.
  POST /unload    → drop the pipeline, gc + empty CUDA cache. Idempotent 200.
                    Blocks until any in-flight inference finishes.
                    409 if a load is in progress (retry after it completes).
  POST /generate  → multipart/form-data:
                      image        (file, PNG/JPG)
                      prompt       (str)
                      duration_sec (float)
                      fps          (int)
                    → raw MP4 bytes
                    Safety net: if the model is unloaded, /generate blocks on a
                    load first — standalone use keeps working, paying the 4–5 min
                    load on the first request instead of at server boot.

Model notes (unchanged from the resident edition):
  - t5_cpu=True → T5 text encoder stays on CPU (runs once per inference, saves ~11 GB VRAM)
  - offload_model=True in generate: both DiTs (108 GB) can never be in VRAM together

Environment variables:
  WAN_DIR        Path to the cloned Wan2.2 repo (default: /workspace/Wan2.2)
  WAN_MODEL_DIR  Path to the downloaded I2V model (default: /workspace/Wan2.2-I2V-A14B)

Run (from /workspace/video_me):
  uvicorn services.wan_server:app --host 0.0.0.0 --port 8030
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

WAN_DIR = Path(os.getenv("WAN_DIR", "/workspace/Wan2.2"))
WAN_MODEL_DIR = Path(os.getenv("WAN_MODEL_DIR", "/workspace/Wan2.2-I2V-A14B"))

# 480p landscape — matches old _DEFAULT_SIZE = "832*480"
_MAX_AREA = 832 * 480
# Wan2.2 recommends shift=3.0 for 480p (5.0 is for 720p)
_SHIFT = 3.0

_pipeline = None             # WanI2V instance, set by /load (or /generate's safety net)
_pipeline_error: str | None = None
_infer_lock = threading.Lock()   # only one GPU inference at a time
_load_lock = threading.Lock()    # serializes load attempts / guards _loading
_loading = False


def _load_pipeline() -> None:
    """Load WanI2V into memory.  Runs in a thread executor, never on the event loop."""
    global _pipeline, _pipeline_error, _loading
    with _load_lock:
        if _pipeline is not None:
            return
        if _loading:
            return
        _loading = True
        _pipeline_error = None
    try:
        if str(WAN_DIR) not in sys.path:
            sys.path.insert(0, str(WAN_DIR))

        import wan  # noqa: PLC0415
        from wan.configs import WAN_CONFIGS  # noqa: PLC0415

        cfg = WAN_CONFIGS["i2v-A14B"]
        logger.info("Loading WanI2V from %s — takes 4–5 min on first start ...", WAN_MODEL_DIR)
        _pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=str(WAN_MODEL_DIR),
            device_id=0,
            t5_cpu=True,   # T5 on CPU: saves ~11 GB VRAM; runs once per inference
            # init_on_cpu=True (default): both 54 GB DiT models start in CPU RAM.
            # offload_model=True (in generate): swaps one DiT to GPU per denoising step.
            # Both DiTs together (108 GB) + other services (7 GB) > 80 GB, so they
            # can never both be in VRAM simultaneously — offloading is unavoidable.
            # The benefit of this resident approach vs subprocess: no 4-5 min disk
            # reload per shot; model stays in CPU RAM between calls.
        )
        logger.info("WanI2V ready")
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.error("WanI2V failed to load: %s", exc, exc_info=True)
    finally:
        with _load_lock:
            _loading = False


def _unload_pipeline() -> None:
    """Drop the pipeline and release CUDA memory. Idempotent."""
    global _pipeline, _pipeline_error
    with _infer_lock:  # wait out any in-flight inference
        _pipeline = None
        _pipeline_error = None
    import gc  # noqa: PLC0415
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("WanI2V unloaded — VRAM released")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deferred loading: the model is loaded via POST /load (or lazily by
    # /generate), never at startup — the render phase needs the VRAM first.
    if not WAN_DIR.exists():
        logger.error("WAN_DIR not found: %s", WAN_DIR)
    elif not WAN_MODEL_DIR.exists():
        logger.error("WAN_MODEL_DIR not found: %s", WAN_MODEL_DIR)
    else:
        logger.info("wan_server up — model deferred; POST /load to bring it into memory")
    yield
    global _pipeline
    _pipeline = None
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:
        pass


app = FastAPI(title="Wan2.2 image-to-video (deferred loading)", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "loading": _loading,
        "error": _pipeline_error,
    })


@app.post("/load")
async def load() -> JSONResponse:
    if _pipeline is not None:
        return JSONResponse({"status": "ok", "model_loaded": True})
    if _loading:
        return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _load_pipeline)  # fire and forget; poll /health
    return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)


@app.post("/unload")
async def unload() -> JSONResponse:
    if _loading:
        # Unloading now would race the loader thread, which could set _pipeline
        # after we return — leaving the model resident during the render phase.
        raise HTTPException(409, detail="model load in progress; retry after it completes")
    if _pipeline is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _unload_pipeline)
    return JSONResponse({"status": "ok", "model_loaded": False})


def _inference(pil_image, prompt: str, num_frames: int, fps: int) -> bytes:
    """Blocking inference — called in a thread executor, never on the event loop."""
    from wan.utils.utils import save_video  # noqa: PLC0415

    with _infer_lock:
        video_tensor = _pipeline.generate(
            prompt,
            pil_image,
            max_area=_MAX_AREA,
            frame_num=num_frames,
            shift=_SHIFT,
            sample_solver="unipc",
            sampling_steps=40,
            guide_scale=5.0,
            seed=-1,
            offload_model=True,   # required: both DiTs (108 GB) > 80 GB VRAM; one at a time
        )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = Path(f.name)

    # generate() returns (C, N, H, W) — save_video()'s unbind(2) expects an
    # extra leading batch axis so dim=2 lands on frames, not height (matches
    # Wan2.2's own generate.py: save_video(tensor=video[None], ...)).
    save_video(video_tensor[None], str(out_path), fps=fps, nrow=1)
    data = out_path.read_bytes()
    out_path.unlink(missing_ok=True)
    return data


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    duration_sec: float = Form(4.0),
    fps: int = Form(16),
) -> Response:
    if _pipeline is None:
        # Safety net for standalone use: block on a load instead of failing.
        # The orchestrated path always POSTs /load and polls /health first.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_pipeline)
        while _loading:  # another caller may hold the load; wait it out
            await asyncio.sleep(2)
        if _pipeline is None:
            detail = "WanI2V not ready" + (f": {_pipeline_error}" if _pipeline_error else "")
            raise HTTPException(503, detail=detail)

    # Wan requires frame_num = 4n+1  (81 for 5 s @ 16 fps)
    n = max(1, round(duration_sec * fps / 4))
    num_frames = 4 * n + 1

    img_bytes = await image.read()
    from PIL import Image  # noqa: PLC0415
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    loop = asyncio.get_running_loop()
    video_bytes = await loop.run_in_executor(None, _inference, pil_image, prompt, num_frames, fps)

    return Response(content=video_bytes, media_type="video/mp4")
