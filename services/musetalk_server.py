"""
MuseTalk lip-sync HTTP service wrapper.

Exposes the API contract expected by adapters/lip_sync/lip_sync_adapter.py:
  GET  /health   → {"status": "ok"}
  POST /lipsync  → multipart/form-data:
                     video   (file, MP4),
                     audio   (file, WAV),
                     shot_id (str)
                   → raw synced MP4 bytes

Environment variables:
  MUSETALK_DIR  Path to the cloned MuseTalk repo (default: /workspace/MuseTalk)

Run from the video_me repo root using the dedicated musetalk venv:
  MUSETALK_DIR=/workspace/MuseTalk \
  /workspace/.venv_musetalk/bin/uvicorn services.musetalk_server:app \
    --host 0.0.0.0 --port 8040
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

MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", "/workspace/MuseTalk"))

# inference.py lives under scripts/, not the repo root
_INFERENCE_SCRIPT = "scripts/inference.py"
# v15 = MuseTalk v1.5 (unet.pth under models/musetalkV15/)
_MUSETALK_VERSION = "v15"
_UNET_CONFIG = "models/musetalkV15/musetalk.json"
_UNET_MODEL = "models/musetalkV15/unet.pth"
_WHISPER_DIR = "models/whisper"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MUSETALK_DIR.exists():
        logger.error("MUSETALK_DIR not found: %s — set MUSETALK_DIR env var", MUSETALK_DIR)
    else:
        logger.info("MuseTalk service ready (dir: %s, version: %s)", MUSETALK_DIR, _MUSETALK_VERSION)
    yield


app = FastAPI(title="MuseTalk lip-sync", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not MUSETALK_DIR.exists():
        return JSONResponse(
            {"status": "down", "reason": "MUSETALK_DIR missing"},
            status_code=503,
        )
    unet = MUSETALK_DIR / _UNET_MODEL
    if not unet.exists():
        return JSONResponse(
            {"status": "down", "reason": f"MuseTalk weights not found: {_UNET_MODEL}"},
            status_code=503,
        )
    return JSONResponse({"status": "ok"})


@app.post("/lipsync")
async def lipsync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    shot_id: str = Form(...),
) -> Response:
    if not MUSETALK_DIR.exists():
        raise HTTPException(503, detail="MuseTalk not set up — check MUSETALK_DIR")

    unet = MUSETALK_DIR / _UNET_MODEL
    if not unet.exists():
        raise HTTPException(503, detail=f"MuseTalk weights missing: {_UNET_MODEL}. Run download_weights.sh first.")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        video_path = tmpdir_path / "input.mp4"
        audio_path = tmpdir_path / "input.wav"
        result_dir = tmpdir_path / "output"
        result_dir.mkdir()

        video_path.write_bytes(await video.read())
        audio_path.write_bytes(await audio.read())

        # inference.py reads video/audio from a YAML config (not CLI args)
        cfg_path = tmpdir_path / "task.yaml"
        cfg_path.write_text(
            f"task_0:\n  video_path: {video_path}\n  audio_path: {audio_path}\n"
        )

        cmd = [
            sys.executable, str(MUSETALK_DIR / _INFERENCE_SCRIPT),
            "--version", _MUSETALK_VERSION,
            "--unet_config", _UNET_CONFIG,
            "--unet_model_path", _UNET_MODEL,
            "--whisper_dir", _WHISPER_DIR,
            "--inference_config", str(cfg_path),
            "--result_dir", str(result_dir),
            "--use_float16",
        ]

        # scripts/inference.py lives in scripts/ but imports musetalk from repo root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(MUSETALK_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        logger.info("Running MuseTalk for shot %s", shot_id)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(MUSETALK_DIR),
            env=env,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error("MuseTalk stderr: %s", result.stderr[-2000:])
            raise HTTPException(500, detail=f"MuseTalk failed: {result.stderr[-500:]}")

        # Output lands at result_dir/<version>/input.mp4 (named after input basename)
        mp4s = sorted(glob.glob(str(result_dir / "**/*.mp4"), recursive=True))
        if not mp4s:
            avis = glob.glob(str(result_dir / "**/*.avi"), recursive=True)
            if not avis:
                raise HTTPException(500, detail="MuseTalk produced no output video")
            output_path = _convert_to_mp4(Path(avis[0]), tmpdir_path / "synced.mp4")
        else:
            output_path = Path(mp4s[-1])

        return Response(content=output_path.read_bytes(), media_type="video/mp4")


def _convert_to_mp4(avi: Path, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264", "-c:a", "aac", str(out)],
        check=True,
        capture_output=True,
    )
    return out
