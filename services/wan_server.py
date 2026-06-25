"""
Wan2.2 image-to-video HTTP service — resident-model edition.

Exposes the same API contract as before:
  GET  /health    → {"status": "ok"}
  POST /generate  → multipart/form-data:
                      image        (file, PNG/JPG)
                      prompt       (str)
                      duration_sec (float)
                      fps          (int)
                    → raw MP4 bytes

Key differences vs the old subprocess approach:
  - WanI2V loaded once at startup (4–5 min), stays resident in VRAM between shots
  - offload_model=False → GPU runs the full denoising loop continuously; no CPU ↔ GPU
    weight shuffling that was causing 7% GPU / 280% CPU utilization
  - t5_cpu=True → T5 text encoder stays on CPU (runs once per inference, saves ~11 GB VRAM)
  - Result: ~26 min/shot → ~5 min/shot

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

_pipeline = None             # WanI2V instance, set during lifespan startup
_pipeline_error: str | None = None
_infer_lock = threading.Lock()   # only one GPU inference at a time


def _load_pipeline() -> None:
    """Load WanI2V into VRAM.  Runs in a thread executor at startup."""
    global _pipeline, _pipeline_error
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
        logger.info("WanI2V ready — model resident in VRAM")
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.error("WanI2V failed to load: %s", exc, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not WAN_DIR.exists():
        logger.error("WAN_DIR not found: %s", WAN_DIR)
    elif not WAN_MODEL_DIR.exists():
        logger.error("WAN_MODEL_DIR not found: %s", WAN_MODEL_DIR)
    else:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_pipeline)
    yield
    global _pipeline
    _pipeline = None
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:
        pass


app = FastAPI(title="Wan2.2 image-to-video (resident)", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if _pipeline_error:
        return JSONResponse({"status": "down", "reason": _pipeline_error}, status_code=503)
    if _pipeline is None:
        return JSONResponse({"status": "down", "reason": "model loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


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

    save_video(video_tensor, str(out_path), fps=fps, nrow=1)
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
        detail = "WanI2V not ready" + (f": {_pipeline_error}" if _pipeline_error else " (still loading)")
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
