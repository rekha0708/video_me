"""
Wan2.2 Speech-to-Video HTTP service wrapper (resident model).

Exposes the API contract expected by adapters/generate_video/wan_s2v_adapter.py:
  GET  /health   -> {"status": "ok"|"down", "model_loaded": bool, "loading": bool, "error": str|None}
  POST /load     -> start loading WanS2V in a background thread; idempotent when already loaded
  POST /unload   -> free the resident model (see note on the respawn strategy below)
  POST /generate -> multipart/form-data:
                      image        (file, PNG/JPG)
                      audio        (file, WAV/MP3/etc.)
                      prompt       (str)
                      duration_sec (float, advisory)
                      shot_id      (str)
                    -> raw MP4 bytes

2026-07-13: rewritten from a subprocess-per-shot wrapper (spawning generate.py
fresh for every clip) to a resident in-process model. The subprocess path paid
the full disk->VRAM checkpoint load (~4+ min) on *every shot*, not just once
per job — for a multi-shot job that dwarfed the actual generation time.
WanS2V (wan/speech2video.py) supports staying resident across .generate()
calls the same way wan.WanI2V already does for services/wan_server.py, so
this now follows that same load-once-per-job pattern.

Environment variables:
  WAN_DIR             Path to the Wan2.2 repo containing wan/ (default: /workspace/Wan2.2)
  WAN_S2V_MODEL_DIR   Path to Wan2.2-S2V-14B weights
                      (default: /workspace/Wan2.2-S2V-14B)
  WAN_S2V_SIZE        Area preset passed as max_area (default: 480*832)
  WAN_S2V_INFER_FRAMES Frames per generated S2V clip chunk (default: 80)
  WAN_S2V_FPS         FPS used to derive infer_frames when request omits it
                      (default: 16)
  WAN_S2V_TIMEOUT_SEC Per-shot generate() timeout, seconds (default: 3600)
  WAN_S2V_OFFLOAD_MODEL
                      Pass-through for WanS2V.generate()'s offload_model —
                      shuttles the model to CPU during denoising to cut GPU
                      memory use, at a real speed cost (default: false; this
                      wrapper is sized for a single high-VRAM GPU where the
                      14B model comfortably stays resident — set true on a
                      smaller/shared GPU that needs the memory back)
  WAN_S2V_SAMPLE_STEPS
                      Pass-through for generate()'s sampling_steps. Unset by
                      default, which falls through to Wan's own s2v_14B config
                      default (40) rather than us guessing a number. This is
                      the fallback speed lever if disabling offload_model
                      alone isn't enough: no Lightning/distillation LoRA
                      exists for S2V-14B as of 2026-07 (only the MoE
                      T2V-A14B/I2V-A14B variants have one — S2V is a dense,
                      differently-shaped model), so cutting steps directly is
                      the only remaining knob, at a real quality/motion-
                      coherence cost Wan's own docs acknowledge. Revisit this
                      note if lightx2v or Wan-AI ever ship a native S2V
                      distillation LoRA.
  WAN_S2V_SHIFT       Pass-through for generate()'s shift (noise schedule).
                      Default 3.0 — Wan's own docstring recommends 3.0 for
                      480p generation (vs the 5.0 default tuned for 720p+).

FlashAttention-3: Wan2.2's wan/modules/attention.py auto-detects FA3 via
`import flash_attn_interface` (falls back to flash-attn 2, then SDPA) with no
config on our side — this module just re-checks the same import so /health
and the load log can report whether it's actually active in this process's
venv. Install lives at /workspace/flash-attention/hopper (built for both
sm80 and sm90; only sm90/Hopper is needed on this pod's H200) and is
installed into .venv_wan via `python setup.py install` there.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

try:
    import flash_attn_interface  # noqa: F401  (Wan2.2's own attention.py detects this same module)
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

WAN_DIR = Path(os.getenv("WAN_DIR", "/workspace/Wan2.2"))
WAN_S2V_MODEL_DIR = Path(os.getenv("WAN_S2V_MODEL_DIR", "/workspace/Wan2.2-S2V-14B"))
WAN_S2V_SIZE = os.getenv("WAN_S2V_SIZE", "480*832")  # gram needs 9:16 aspect ratio. 480 × 832 is the 480p 9:16 preset s2v-14B supports — dropped from 720 × 1280 (2026-07-13) to cut per-shot generation time; raise back to 720*1280 (or 704*1280) once per-shot cost with FA3 + resident-model loading (both true as of 2026-07-13) is measured and back under control.
WAN_S2V_INFER_FRAMES = int(os.getenv("WAN_S2V_INFER_FRAMES", "80"))
WAN_S2V_FPS = int(os.getenv("WAN_S2V_FPS", "16"))
WAN_S2V_TIMEOUT_SEC = int(os.getenv("WAN_S2V_TIMEOUT_SEC", "3600"))
WAN_S2V_OFFLOAD_MODEL = os.getenv("WAN_S2V_OFFLOAD_MODEL", "false").strip().lower() in (
    "1", "true", "yes",
)
WAN_S2V_SAMPLE_STEPS = os.getenv("WAN_S2V_SAMPLE_STEPS", "").strip()
WAN_S2V_SHIFT = float(os.getenv("WAN_S2V_SHIFT", "3.0"))

_pipeline = None
_pipeline_error: str | None = None
_loading = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _check_setup() -> None:
    if not WAN_DIR.exists():
        raise FileNotFoundError(f"WAN_DIR missing: {WAN_DIR}")
    if not WAN_S2V_MODEL_DIR.exists():
        raise FileNotFoundError(f"WAN_S2V_MODEL_DIR missing: {WAN_S2V_MODEL_DIR}")


def _infer_frames_for_duration(duration_sec: float, fps: int) -> int:
    if duration_sec <= 0 or fps <= 0:
        return WAN_S2V_INFER_FRAMES
    return 4 * max(1, round(duration_sec * fps / 4)) + 1


def _load_pipeline() -> None:
    global _pipeline, _pipeline_error, _loading
    with _load_lock:
        if _pipeline is not None:
            return
        if _loading:
            return
        _loading = True
        _pipeline_error = None

    try:
        _check_setup()
        if str(WAN_DIR) not in sys.path:
            sys.path.insert(0, str(WAN_DIR))

        import wan  # noqa: PLC0415
        from wan.configs import WAN_CONFIGS  # noqa: PLC0415

        cfg = WAN_CONFIGS["s2v-14B"]
        logger.info(
            "Loading WanS2V (model=%s, size=%s, offload=%s, flash_attn_3=%s)",
            WAN_S2V_MODEL_DIR, WAN_S2V_SIZE, WAN_S2V_OFFLOAD_MODEL, FLASH_ATTN_3_AVAILABLE,
        )
        pipe = wan.WanS2V(
            config=cfg,
            checkpoint_dir=str(WAN_S2V_MODEL_DIR),
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=False,  # keep resident on GPU across shots in the job
            convert_model_dtype=True,
        )
        _pipeline = pipe
        logger.info("WanS2V ready")
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.error("WanS2V failed to load: %s", exc, exc_info=True)
    finally:
        with _load_lock:
            _loading = False


def _unload_pipeline() -> None:
    # A plain `_pipeline = None` + gc.collect() + torch.cuda.empty_cache() is
    # not reliable here: measured on the sibling LightX2V service, a single
    # load->unload->reload cycle in the SAME process went from a clean 67GB
    # back up to 130GB+ even with matching dtype/allocator tuning, because
    # something inside the model stack keeps CUDA-side references alive past
    # dropping our own reference. Re-exec this process fresh instead — the
    # next /load always starts from a clean CUDA context.
    def _respawn() -> None:
        time.sleep(0.5)  # let the /unload response flush before we replace ourselves
        os.execv(sys.argv[0], sys.argv)

    threading.Thread(target=_respawn, daemon=False).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _check_setup()
    except Exception as exc:
        logger.error("Wan S2V setup incomplete: %s", exc)
    else:
        logger.info("Wan S2V service ready (repo=%s, model=%s); POST /load to warm model", WAN_DIR, WAN_S2V_MODEL_DIR)
    yield


app = FastAPI(title="Wan2.2 Speech-to-Video (resident)", lifespan=lifespan)


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
            "offload_model": WAN_S2V_OFFLOAD_MODEL,
            "size": WAN_S2V_SIZE,
            "flash_attn_3": FLASH_ATTN_3_AVAILABLE,
        },
        status_code=200 if setup_error is None else 503,
    )


@app.post("/load")
async def load() -> JSONResponse:
    if _pipeline is not None:
        return JSONResponse({"status": "ok", "model_loaded": True})
    if _loading:
        return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)
    import asyncio

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _load_pipeline)
    return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)


@app.post("/unload")
async def unload() -> JSONResponse:
    if _loading:
        raise HTTPException(409, detail="model load in progress; retry after it completes")
    if _pipeline is not None:
        _unload_pipeline()
    return JSONResponse({"status": "ok", "model_loaded": False})


def _generate(
    *,
    image_path: Path,
    audio_path: Path,
    prompt: str,
    infer_frames: int,
    output_path: Path,
) -> None:
    from wan.configs import MAX_AREA_CONFIGS  # noqa: PLC0415
    from wan.utils.utils import merge_video_audio, save_video  # noqa: PLC0415

    if _pipeline is None:
        raise RuntimeError("WanS2V pipeline is not loaded")

    with _infer_lock:
        kwargs: dict = {}
        if WAN_S2V_SAMPLE_STEPS:
            kwargs["sampling_steps"] = int(WAN_S2V_SAMPLE_STEPS)

        video = _pipeline.generate(
            input_prompt=prompt,
            ref_image_path=str(image_path),
            audio_path=str(audio_path),
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=None,
            pose_video=None,
            max_area=MAX_AREA_CONFIGS[WAN_S2V_SIZE],
            infer_frames=infer_frames,
            shift=WAN_S2V_SHIFT,
            offload_model=WAN_S2V_OFFLOAD_MODEL,
            init_first_frame=False,
            **kwargs,
        )
        import torch  # noqa: PLC0415

        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=WAN_S2V_FPS,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del video
        torch.cuda.synchronize()
        merge_video_audio(video_path=str(output_path), audio_path=str(audio_path))
        gc.collect()


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(...),
    duration_sec: float = Form(0.0),
    fps: int = Form(0),
    infer_frames: int = Form(0),
    shot_id: str = Form("shot"),
) -> Response:
    if _pipeline is None:
        raise HTTPException(503, detail="WanS2V model not loaded — call /load first")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        image_path = tmpdir_path / f"input{Path(image.filename or '.png').suffix or '.png'}"
        audio_path = tmpdir_path / f"input{Path(audio.filename or '.wav').suffix or '.wav'}"
        output_path = tmpdir_path / "output.mp4"

        image_path.write_bytes(await image.read())
        audio_path.write_bytes(await audio.read())
        effective_fps = fps if fps > 0 else WAN_S2V_FPS
        effective_infer_frames = (
            infer_frames
            if infer_frames > 0
            else _infer_frames_for_duration(duration_sec, effective_fps)
        )

        logger.info(
            "Running Wan S2V for shot %s (duration %.2fs, fps %s, infer_frames %s, size %s)",
            shot_id, duration_sec, effective_fps, effective_infer_frames, WAN_S2V_SIZE,
        )

        import asyncio

        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _generate(
                        image_path=image_path,
                        audio_path=audio_path,
                        prompt=prompt,
                        infer_frames=effective_infer_frames,
                        output_path=output_path,
                    ),
                ),
                timeout=WAN_S2V_TIMEOUT_SEC,
            )
        except Exception as exc:
            logger.error("Wan S2V generate failed for shot %s: %s", shot_id, exc, exc_info=True)
            raise HTTPException(500, detail=f"Wan S2V failed: {exc}") from exc

        if not output_path.exists():
            raise HTTPException(500, detail="Wan S2V produced no output video")

        return Response(content=output_path.read_bytes(), media_type="video/mp4")
