"""
LightX2V Wan2.2 I2V HTTP service.

This service is intentionally separate from services/wan_server.py. The official
Wan server keeps the reference implementation available, while this one targets
the experimental high-throughput path:

  Wan2.2-I2V-A14B base + LightX2V Wan2.2 Distill LoRAs + 4-step inference.

API contract matches adapters/generate_video/wan_lightx2v_adapter.py:
  GET  /health
  POST /load
  POST /unload
  POST /generate

Environment variables:
  LIGHTX2V_DIR                 LightX2V checkout (default: /workspace/LightX2V)
  LIGHTX2V_I2V_MODEL_DIR       Wan2.2-I2V-A14B base model dir
                                (default: /workspace/Wan2.2-I2V-A14B)
  LIGHTX2V_I2V_LORA_DIR        Distill LoRA dir
                                (default: /workspace/Wan2.2-Distill-Loras)
  LIGHTX2V_I2V_HIGH_LORA       Optional explicit high-noise LoRA path
  LIGHTX2V_I2V_LOW_LORA        Optional explicit low-noise LoRA path
  LIGHTX2V_I2V_WIDTH           Output width (default: 400)
  LIGHTX2V_I2V_HEIGHT          Output height (default: 704)
  LIGHTX2V_I2V_INFER_FRAMES    Load-time default frame count (default: 81)
  LIGHTX2V_I2V_STEPS           Distilled inference steps (default: 4)
  LIGHTX2V_I2V_FPS             Output FPS metadata (default: 16)
  LIGHTX2V_I2V_ATTN_MODE       Attention mode written into config (default: sage_attn2)
  LIGHTX2V_I2V_OFFLOAD_MODEL   CPU/block offload for lower VRAM, slower (default: false)
  LIGHTX2V_I2V_TIMEOUT_SEC     Per-shot inference timeout (default: 1800)
  LIGHTX2V_I2V_SEED            Fixed seed, or -1 for random per shot (default: -1)
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import random
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

HIGH_LORA_FILE = "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
LOW_LORA_FILE = "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
NEGATIVE_PROMPT = (
    "overexposed, static, blurry details, subtitles, watermark, low quality, "
    "jpeg artifacts, malformed hands, malformed face, distorted limbs, crowded background"
)

LIGHTX2V_DIR = Path(os.getenv("LIGHTX2V_DIR", "/workspace/LightX2V"))
LIGHTX2V_I2V_MODEL_DIR = Path(os.getenv("LIGHTX2V_I2V_MODEL_DIR", "/workspace/Wan2.2-I2V-A14B"))
LIGHTX2V_I2V_LORA_DIR = Path(os.getenv("LIGHTX2V_I2V_LORA_DIR", "/workspace/Wan2.2-Distill-Loras"))
LIGHTX2V_I2V_HIGH_LORA = os.getenv("LIGHTX2V_I2V_HIGH_LORA", "").strip()
LIGHTX2V_I2V_LOW_LORA = os.getenv("LIGHTX2V_I2V_LOW_LORA", "").strip()
LIGHTX2V_I2V_WIDTH = int(os.getenv("LIGHTX2V_I2V_WIDTH", "400"))
LIGHTX2V_I2V_HEIGHT = int(os.getenv("LIGHTX2V_I2V_HEIGHT", "704"))
LIGHTX2V_I2V_INFER_FRAMES = int(os.getenv("LIGHTX2V_I2V_INFER_FRAMES", "81"))
LIGHTX2V_I2V_STEPS = int(os.getenv("LIGHTX2V_I2V_STEPS", "4"))
LIGHTX2V_I2V_FPS = int(os.getenv("LIGHTX2V_I2V_FPS", "16"))
LIGHTX2V_I2V_ATTN_MODE = os.getenv("LIGHTX2V_I2V_ATTN_MODE", "sage_attn2").strip()
LIGHTX2V_I2V_OFFLOAD_MODEL = os.getenv("LIGHTX2V_I2V_OFFLOAD_MODEL", "false").strip().lower() in (
    "1", "true", "yes",
)
LIGHTX2V_I2V_TIMEOUT_SEC = int(os.getenv("LIGHTX2V_I2V_TIMEOUT_SEC", "1800"))
LIGHTX2V_I2V_SEED = int(os.getenv("LIGHTX2V_I2V_SEED", "-1"))

_pipeline = None
_pipeline_error: str | None = None
_loading = False
_loaded_frames = 0
_loaded_fps = 0
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _lora_paths() -> tuple[Path, Path]:
    high = Path(LIGHTX2V_I2V_HIGH_LORA) if LIGHTX2V_I2V_HIGH_LORA else LIGHTX2V_I2V_LORA_DIR / HIGH_LORA_FILE
    low = Path(LIGHTX2V_I2V_LOW_LORA) if LIGHTX2V_I2V_LOW_LORA else LIGHTX2V_I2V_LORA_DIR / LOW_LORA_FILE
    return high, low


def _check_setup() -> None:
    if not LIGHTX2V_DIR.exists():
        raise FileNotFoundError(f"LIGHTX2V_DIR missing: {LIGHTX2V_DIR}")
    if not LIGHTX2V_I2V_MODEL_DIR.exists():
        raise FileNotFoundError(f"LIGHTX2V_I2V_MODEL_DIR missing: {LIGHTX2V_I2V_MODEL_DIR}")
    high_lora, low_lora = _lora_paths()
    if not high_lora.exists():
        raise FileNotFoundError(f"LightX2V high-noise LoRA missing: {high_lora}")
    if not low_lora.exists():
        raise FileNotFoundError(f"LightX2V low-noise LoRA missing: {low_lora}")


def _config_payload(num_frames: int, fps: int) -> dict:
    high_lora, low_lora = _lora_paths()
    return {
        "infer_steps": LIGHTX2V_I2V_STEPS,
        "target_video_length": num_frames,
        "text_len": 512,
        "target_height": LIGHTX2V_I2V_HEIGHT,
        "target_width": LIGHTX2V_I2V_WIDTH,
        "self_attn_1_type": LIGHTX2V_I2V_ATTN_MODE,
        "cross_attn_1_type": LIGHTX2V_I2V_ATTN_MODE,
        "cross_attn_2_type": LIGHTX2V_I2V_ATTN_MODE,
        "sample_guide_scale": [3.5, 3.5],
        "sample_shift": 5.0,
        "enable_cfg": False,
        "cpu_offload": LIGHTX2V_I2V_OFFLOAD_MODEL,
        "offload_granularity": "block",
        "t5_cpu_offload": LIGHTX2V_I2V_OFFLOAD_MODEL,
        "vae_cpu_offload": LIGHTX2V_I2V_OFFLOAD_MODEL,
        "use_image_encoder": False,
        "fps": fps,
        "boundary_step_index": 2,
        "denoising_step_list": [1000, 750, 500, 250],
        "lora_configs": [
            {"name": "high_noise_model", "path": str(high_lora), "strength": 1.0},
            {"name": "low_noise_model", "path": str(low_lora), "strength": 1.0},
        ],
    }


def _write_config(num_frames: int, fps: int) -> Path:
    config_path = Path(tempfile.gettempdir()) / "video_me_lightx2v_wan_i2v_config.json"
    config_path.write_text(json.dumps(_config_payload(num_frames, fps), indent=2), encoding="utf-8")
    return config_path


def _load_pipeline() -> None:
    global _pipeline, _pipeline_error, _loading, _loaded_frames, _loaded_fps
    with _load_lock:
        if _pipeline is not None:
            return
        if _loading:
            return
        _loading = True
        _pipeline_error = None

    try:
        _check_setup()
        if str(LIGHTX2V_DIR) not in sys.path:
            sys.path.insert(0, str(LIGHTX2V_DIR))

        from lightx2v import LightX2VPipeline  # noqa: PLC0415

        config_path = _write_config(LIGHTX2V_I2V_INFER_FRAMES, LIGHTX2V_I2V_FPS)
        logger.info(
            "Loading LightX2V Wan I2V (model=%s, config=%s, offload=%s)",
            LIGHTX2V_I2V_MODEL_DIR,
            config_path,
            LIGHTX2V_I2V_OFFLOAD_MODEL,
        )
        pipe = LightX2VPipeline(
            model_path=str(LIGHTX2V_I2V_MODEL_DIR),
            model_cls="wan2.2_moe_distill",
            task="i2v",
        )
        pipe.create_generator(config_json=str(config_path))
        _pipeline = pipe
        _loaded_frames = LIGHTX2V_I2V_INFER_FRAMES
        _loaded_fps = LIGHTX2V_I2V_FPS
        logger.info("LightX2V Wan I2V ready")
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.error("LightX2V Wan I2V failed to load: %s", exc, exc_info=True)
    finally:
        with _load_lock:
            _loading = False


def _unload_pipeline() -> None:
    global _pipeline, _pipeline_error, _loaded_frames, _loaded_fps
    with _infer_lock:
        _pipeline = None
        _pipeline_error = None
        _loaded_frames = 0
        _loaded_fps = 0
    gc.collect()
    try:
        import torch  # noqa: PLC0415

        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("LightX2V Wan I2V unloaded")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _check_setup()
    except Exception as exc:
        logger.error("LightX2V Wan I2V setup incomplete: %s", exc)
    else:
        logger.info("LightX2V Wan I2V service ready; POST /load to warm model")
    yield
    _unload_pipeline()


app = FastAPI(title="LightX2V Wan2.2 I2V", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    setup_error = None
    try:
        _check_setup()
    except Exception as exc:
        setup_error = str(exc)
    return JSONResponse(
        {
            "status": "ok" if setup_error is None else "down",
            "model_loaded": _pipeline is not None,
            "loading": _loading,
            "error": _pipeline_error or setup_error,
            "offload_model": LIGHTX2V_I2V_OFFLOAD_MODEL,
            "width": LIGHTX2V_I2V_WIDTH,
            "height": LIGHTX2V_I2V_HEIGHT,
            "steps": LIGHTX2V_I2V_STEPS,
        },
        status_code=200 if setup_error is None else 503,
    )


@app.post("/load")
async def load() -> JSONResponse:
    if _pipeline is not None:
        return JSONResponse({"status": "ok", "model_loaded": True})
    if _loading:
        return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _load_pipeline)
    return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)


@app.post("/unload")
async def unload() -> JSONResponse:
    if _loading:
        raise HTTPException(409, detail="model load in progress; retry after it completes")
    if _pipeline is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _unload_pipeline)
    return JSONResponse({"status": "ok", "model_loaded": False})


def _generate(image_path: Path, prompt: str, output_path: Path, infer_frames: int, fps: int) -> None:
    global _loaded_frames, _loaded_fps
    if _pipeline is None:
        raise RuntimeError("LightX2V pipeline is not loaded")
    with _infer_lock:
        if infer_frames != _loaded_frames or fps != _loaded_fps:
            _pipeline.modify_config({"target_video_length": infer_frames, "fps": fps})
            _loaded_frames = infer_frames
            _loaded_fps = fps
        seed = LIGHTX2V_I2V_SEED
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        _pipeline.generate(
            seed=seed,
            image_path=str(image_path),
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            save_result_path=str(output_path),
        )


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    duration_sec: float = Form(4.0),
    fps: int = Form(16),
    infer_frames: int = Form(0),
    shot_id: str = Form("shot"),
) -> Response:
    if _pipeline is None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_pipeline)
        while _loading:
            await asyncio.sleep(2)
        if _pipeline is None:
            detail = "LightX2V Wan I2V not ready" + (f": {_pipeline_error}" if _pipeline_error else "")
            raise HTTPException(503, detail=detail)

    effective_fps = fps if fps > 0 else LIGHTX2V_I2V_FPS
    effective_frames = infer_frames if infer_frames > 0 else 4 * max(1, round(duration_sec * effective_fps / 4)) + 1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        image_path = tmpdir_path / f"input{Path(image.filename or '.png').suffix or '.png'}"
        output_path = tmpdir_path / "output.mp4"
        image_path.write_bytes(await image.read())

        logger.info(
            "Running LightX2V Wan I2V for shot %s (frames=%s, fps=%s, size=%sx%s)",
            shot_id,
            effective_frames,
            effective_fps,
            LIGHTX2V_I2V_WIDTH,
            LIGHTX2V_I2V_HEIGHT,
        )
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    _generate,
                    image_path,
                    prompt,
                    output_path,
                    effective_frames,
                    effective_fps,
                ),
                timeout=LIGHTX2V_I2V_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, detail="LightX2V Wan I2V timed out") from exc

        if not output_path.exists():
            raise HTTPException(500, detail="LightX2V Wan I2V produced no output video")
        return Response(content=output_path.read_bytes(), media_type="video/mp4")
