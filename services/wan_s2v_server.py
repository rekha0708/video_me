"""
Wan2.2 Speech-to-Video HTTP service wrapper.

Exposes the API contract expected by adapters/generate_video/wan_s2v_adapter.py:
  GET  /health   -> {"status": "ok"}
  POST /generate -> multipart/form-data:
                      image        (file, PNG/JPG)
                      audio        (file, WAV/MP3/etc.)
                      prompt       (str)
                      duration_sec (float, advisory)
                      shot_id      (str)
                    -> raw MP4 bytes

Environment variables:
  WAN_DIR             Path to the Wan2.2 repo containing generate.py
                      (default: /workspace/Wan2.2)
  WAN_S2V_MODEL_DIR   Path to Wan2.2-S2V-14B weights
                      (default: /workspace/Wan2.2-S2V-14B)
  WAN_S2V_SIZE        Area preset passed to --size (default: 1024*704)
  WAN_S2V_INFER_FRAMES Frames per generated S2V clip chunk (default: 80)
  WAN_S2V_FPS         FPS used to derive infer_frames when request omits it
                      (default: 16)
  WAN_S2V_TIMEOUT_SEC Subprocess timeout (default: 3600)
  WAN_S2V_OFFLOAD_MODEL
                      Pass-through for generate.py's --offload_model — shuttles
                      the model to CPU after every forward pass to cut GPU
                      memory use, at a real speed cost (default: false; this
                      wrapper is sized for a single high-VRAM GPU where the
                      14B model comfortably stays resident — set true on a
                      smaller/shared GPU that needs the memory back)
  WAN_S2V_SAMPLE_STEPS
                      Pass-through for generate.py's --sample_steps. Unset by
                      default, which falls through to Wan's own s2v_14B config
                      default (40) rather than us guessing a number. This is
                      the fallback speed lever if disabling --offload_model
                      alone isn't enough: no Lightning/distillation LoRA
                      exists for S2V-14B as of 2026-07 (only the MoE
                      T2V-A14B/I2V-A14B variants have one — S2V is a dense,
                      differently-shaped model), so cutting steps directly is
                      the only remaining knob, at a real quality/motion-
                      coherence cost Wan's own docs acknowledge. Revisit this
                      note if lightx2v or Wan-AI ever ship a native S2V
                      distillation LoRA.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

WAN_DIR = Path(os.getenv("WAN_DIR", "/workspace/Wan2.2"))
WAN_S2V_MODEL_DIR = Path(os.getenv("WAN_S2V_MODEL_DIR", "/workspace/Wan2.2-S2V-14B"))
WAN_S2V_SIZE = os.getenv("WAN_S2V_SIZE", "720*1280") # gram needs 9:16 aspect ratio, so 720 × 1280 is the largest 9:16 wan2.2 supports. 1024 × 704 is the largest 16:9 wan2.2 supports, but gram is 9:16, so we use 720 × 1280.
WAN_S2V_INFER_FRAMES = int(os.getenv("WAN_S2V_INFER_FRAMES", "80"))
WAN_S2V_FPS = int(os.getenv("WAN_S2V_FPS", "16"))
WAN_S2V_TIMEOUT_SEC = int(os.getenv("WAN_S2V_TIMEOUT_SEC", "3600"))
WAN_S2V_OFFLOAD_MODEL = os.getenv("WAN_S2V_OFFLOAD_MODEL", "false").strip().lower() in (
    "1", "true", "yes",
)
WAN_S2V_SAMPLE_STEPS = os.getenv("WAN_S2V_SAMPLE_STEPS", "").strip()


def _infer_frames_for_duration(duration_sec: float, fps: int) -> int:
    if duration_sec <= 0 or fps <= 0:
        return WAN_S2V_INFER_FRAMES
    return 4 * max(1, round(duration_sec * fps / 4)) + 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not WAN_DIR.exists():
        logger.error("WAN_DIR not found: %s — set WAN_DIR", WAN_DIR)
    elif not (WAN_DIR / "generate.py").exists():
        logger.error("Wan generate.py not found under WAN_DIR: %s", WAN_DIR)
    elif not WAN_S2V_MODEL_DIR.exists():
        logger.error("WAN_S2V_MODEL_DIR not found: %s", WAN_S2V_MODEL_DIR)
    else:
        logger.info("Wan S2V service ready (repo=%s, model=%s)", WAN_DIR, WAN_S2V_MODEL_DIR)
    yield


app = FastAPI(title="Wan2.2 Speech-to-Video", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not WAN_DIR.exists():
        return JSONResponse({"status": "down", "reason": "WAN_DIR missing"}, status_code=503)
    if not (WAN_DIR / "generate.py").exists():
        return JSONResponse({"status": "down", "reason": "generate.py missing"}, status_code=503)
    if not WAN_S2V_MODEL_DIR.exists():
        return JSONResponse({"status": "down", "reason": "WAN_S2V_MODEL_DIR missing"}, status_code=503)
    return JSONResponse({"status": "ok"})


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
    if not WAN_DIR.exists() or not (WAN_DIR / "generate.py").exists():
        raise HTTPException(503, detail="Wan2.2 repo not set up — check WAN_DIR")
    if not WAN_S2V_MODEL_DIR.exists():
        raise HTTPException(503, detail="Wan2.2-S2V weights missing — check WAN_S2V_MODEL_DIR")

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

        cmd = [
            sys.executable,
            str(WAN_DIR / "generate.py"),
            "--task",
            "s2v-14B",
            "--size",
            WAN_S2V_SIZE,
            "--ckpt_dir",
            str(WAN_S2V_MODEL_DIR),
            "--offload_model",
            str(WAN_S2V_OFFLOAD_MODEL),
            "--convert_model_dtype",
            "--prompt",
            prompt,
            "--image",
            str(image_path),
            "--audio",
            str(audio_path),
            "--infer_frames",
            str(effective_infer_frames),
            "--save_file",
            str(output_path),
        ]
        if WAN_S2V_SAMPLE_STEPS:
            cmd += ["--sample_steps", WAN_S2V_SAMPLE_STEPS]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(WAN_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        logger.info(
            "Running Wan S2V for shot %s (duration %.2fs, fps %s, infer_frames %s, size %s)",
            shot_id,
            duration_sec,
            effective_fps,
            effective_infer_frames,
            WAN_S2V_SIZE,
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(WAN_DIR),
            env=env,
            timeout=WAN_S2V_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            logger.error("Wan S2V stderr: %s", result.stderr[-2000:])
            raise HTTPException(500, detail=f"Wan S2V failed: {result.stderr[-500:]}")
        if not output_path.exists():
            logger.error("Wan S2V stdout: %s", result.stdout[-2000:])
            raise HTTPException(500, detail="Wan S2V produced no output video")

        return Response(content=output_path.read_bytes(), media_type="video/mp4")
