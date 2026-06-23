"""
Wan2.2 image-to-video HTTP service wrapper.

Exposes the API contract expected by adapters/generate_video/wan_adapter.py:
  GET  /health    → {"status": "ok"}
  POST /generate  → multipart/form-data:
                      image        (file, PNG),
                      prompt       (str),
                      duration_sec (float),
                      fps          (int)
                    → raw MP4 bytes

Environment variables:
  WAN_DIR        Path to the cloned Wan2.2 repo (default: /workspace/Wan2.2)
  WAN_MODEL_DIR  Path to the downloaded I2V model (default: /workspace/Wan2.2-I2V-A14B)

Run:
  uvicorn services.wan_server:app --host 0.0.0.0 --port 8030
"""

from __future__ import annotations

import glob
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
WAN_MODEL_DIR = Path(os.getenv("WAN_MODEL_DIR", "/workspace/Wan2.2-I2V-A14B"))

# Wan2.2 I2V supports 480p and 720p.  480p is faster and sufficient for shots.
_DEFAULT_SIZE = "832*480"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not WAN_DIR.exists():
        logger.error("WAN_DIR not found: %s — set WAN_DIR env var", WAN_DIR)
    if not WAN_MODEL_DIR.exists():
        logger.error("WAN_MODEL_DIR not found: %s — set WAN_MODEL_DIR env var", WAN_MODEL_DIR)
    logger.info("Wan2.2 service ready (model: %s)", WAN_MODEL_DIR)
    yield


app = FastAPI(title="Wan2.2 image-to-video", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not WAN_DIR.exists() or not WAN_MODEL_DIR.exists():
        return JSONResponse({"status": "down", "reason": "WAN_DIR or WAN_MODEL_DIR missing"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    duration_sec: float = Form(4.0),
    fps: int = Form(16),
) -> Response:
    if not WAN_DIR.exists() or not WAN_MODEL_DIR.exists():
        raise HTTPException(503, detail="Wan2.2 not set up — check WAN_DIR and WAN_MODEL_DIR")

    # Wan generates frames in multiples of 8; compute total frame count
    num_frames = max(8, round(duration_sec * fps / 8) * 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write input image
        img_path = tmpdir_path / "input.png"
        img_path.write_bytes(await image.read())

        output_path = tmpdir_path / "output.mp4"

        cmd = [
            sys.executable, str(WAN_DIR / "generate.py"),
            "--task", "i2v-A14B",
            "--size", _DEFAULT_SIZE,
            "--ckpt_dir", str(WAN_MODEL_DIR),
            "--offload_model", "True",
            "--convert_model_dtype",
            "--image", str(img_path),
            "--prompt", prompt,
            "--frame_num", str(num_frames),
            "--save_file", str(output_path),
        ]

        logger.info("Running Wan2.2: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(WAN_DIR),
            timeout=600,
        )

        if result.returncode != 0:
            logger.error("Wan2.2 stderr: %s", result.stderr[-2000:])
            raise HTTPException(500, detail=f"Wan2.2 generation failed: {result.stderr[-500:]}")

        # Locate output — try explicit path first, then glob for any MP4
        if not output_path.exists():
            mp4s = glob.glob(str(tmpdir_path / "**/*.mp4"), recursive=True)
            if not mp4s:
                raise HTTPException(500, detail="Wan2.2 produced no MP4 output")
            output_path = Path(mp4s[0])

        return Response(content=output_path.read_bytes(), media_type="video/mp4")
