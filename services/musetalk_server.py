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

Run (from the video_me repo root):
  uvicorn services.musetalk_server:app --host 0.0.0.0 --port 8040

MuseTalk requires Python 3.10 + CUDA 11.7/11.8. If running the rest of the
pipeline in a different Python env, start this server from the MuseTalk conda env:
  conda activate MuseTalk
  uvicorn services.musetalk_server:app --host 0.0.0.0 --port 8040
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

# MuseTalk v1.5 inference script relative to MUSETALK_DIR
_INFERENCE_SCRIPT = "inference.py"
_MUSETALK_VERSION = "v1.5"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MUSETALK_DIR.exists():
        logger.error("MUSETALK_DIR not found: %s — set MUSETALK_DIR env var", MUSETALK_DIR)
    else:
        logger.info("MuseTalk service ready (dir: %s)", MUSETALK_DIR)
    yield


app = FastAPI(title="MuseTalk lip-sync", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not MUSETALK_DIR.exists():
        return JSONResponse(
            {"status": "down", "reason": "MUSETALK_DIR missing"},
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        video_path = tmpdir_path / "input.mp4"
        audio_path = tmpdir_path / "input.wav"
        output_dir = tmpdir_path / "output"
        output_dir.mkdir()

        video_path.write_bytes(await video.read())
        audio_path.write_bytes(await audio.read())

        # MuseTalk inference via its CLI
        cmd = [
            sys.executable, str(MUSETALK_DIR / _INFERENCE_SCRIPT),
            "--version", _MUSETALK_VERSION,
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--result_dir", str(output_dir),
            "--use_float16",
        ]

        logger.info("Running MuseTalk for shot %s", shot_id)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(MUSETALK_DIR),
            timeout=600,
        )

        if result.returncode != 0:
            logger.error("MuseTalk stderr: %s", result.stderr[-2000:])
            raise HTTPException(500, detail=f"MuseTalk failed: {result.stderr[-500:]}")

        # Find the output video
        mp4s = sorted(glob.glob(str(output_dir / "**/*.mp4"), recursive=True))
        if not mp4s:
            # MuseTalk may produce an .avi; ffmpeg-convert if needed
            avis = glob.glob(str(output_dir / "**/*.avi"), recursive=True)
            if not avis:
                raise HTTPException(500, detail="MuseTalk produced no output video")
            output_path = _convert_to_mp4(Path(avis[0]), tmpdir_path / "synced.mp4")
        else:
            output_path = Path(mp4s[-1])  # take the last (final) output

        return Response(content=output_path.read_bytes(), media_type="video/mp4")


def _convert_to_mp4(avi: Path, out: Path) -> Path:
    """Convert AVI to MP4 via ffmpeg if MuseTalk outputs AVI."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(avi), "-c:v", "libx264", "-c:a", "aac", str(out)],
        check=True,
        capture_output=True,
    )
    return out
