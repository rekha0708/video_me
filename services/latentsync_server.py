"""
LatentSync lip-sync HTTP service wrapper.

Exposes the API contract expected by adapters/lip_sync/latentsync_adapter.py:
  GET  /health   -> {"status": "ok"}
  POST /lipsync  -> multipart/form-data:
                     video           (file, MP4)
                     audio           (file, WAV)
                     shot_id         (str)
                     inference_steps (int, optional)
                     guidance_scale  (float, optional)
                   -> raw synced MP4 bytes

Environment variables:
  LATENTSYNC_DIR          Path to the LatentSync repo (default: /workspace/LatentSync)
  LATENTSYNC_UNET_CONFIG  Config path relative to repo
                          (default: configs/unet/stage2_512.yaml)
  LATENTSYNC_CKPT         Checkpoint path relative to repo
                          (default: checkpoints/latentsync_unet.pt)
  LATENTSYNC_TIMEOUT_SEC  Subprocess timeout (default: 1200)
  LATENTSYNC_DEEPCACHE    1 to pass --enable_deepcache (default: 1)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

LATENTSYNC_DIR = Path(os.getenv("LATENTSYNC_DIR", "/workspace/LatentSync"))
LATENTSYNC_UNET_CONFIG = os.getenv("LATENTSYNC_UNET_CONFIG", "configs/unet/stage2_512.yaml")
LATENTSYNC_CKPT = os.getenv("LATENTSYNC_CKPT", "checkpoints/latentsync_unet.pt")
LATENTSYNC_TIMEOUT_SEC = int(os.getenv("LATENTSYNC_TIMEOUT_SEC", "1200"))
LATENTSYNC_DEEPCACHE = os.getenv("LATENTSYNC_DEEPCACHE", "1") == "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LATENTSYNC_DIR.exists():
        logger.error("LATENTSYNC_DIR not found: %s — set LATENTSYNC_DIR", LATENTSYNC_DIR)
    else:
        logger.info("LatentSync service ready (dir=%s)", LATENTSYNC_DIR)
    yield


app = FastAPI(title="LatentSync lip-sync", lifespan=lifespan)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else LATENTSYNC_DIR / path


async def _run_subprocess(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> tuple[int, str, str]:
    """Run a command without blocking the event loop; kill it on timeout.

    A blocking subprocess.run here froze the whole uvicorn loop for the length
    of a repair (up to LATENTSYNC_TIMEOUT_SEC), so /health stopped answering
    mid-run — same fix as services/wan_s2v_server.py.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise HTTPException(
            504, detail=f"LatentSync subprocess timed out after {timeout_sec:.0f}s"
        )
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


@app.get("/health")
def health() -> JSONResponse:
    if not LATENTSYNC_DIR.exists():
        return JSONResponse({"status": "down", "reason": "LATENTSYNC_DIR missing"}, status_code=503)
    config_path = _resolve_repo_path(LATENTSYNC_UNET_CONFIG)
    ckpt_path = _resolve_repo_path(LATENTSYNC_CKPT)
    if not config_path.exists():
        return JSONResponse({"status": "down", "reason": f"config missing: {config_path}"}, status_code=503)
    if not ckpt_path.exists():
        return JSONResponse({"status": "down", "reason": f"checkpoint missing: {ckpt_path}"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/lipsync")
async def lipsync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    shot_id: str = Form(...),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.5),
) -> Response:
    if not LATENTSYNC_DIR.exists():
        raise HTTPException(503, detail="LatentSync not set up — check LATENTSYNC_DIR")
    config_path = _resolve_repo_path(LATENTSYNC_UNET_CONFIG)
    ckpt_path = _resolve_repo_path(LATENTSYNC_CKPT)
    if not config_path.exists():
        raise HTTPException(503, detail=f"LatentSync config missing: {config_path}")
    if not ckpt_path.exists():
        raise HTTPException(503, detail=f"LatentSync checkpoint missing: {ckpt_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        raw_video = tmpdir_path / "input_raw.mp4"
        raw_audio = tmpdir_path / "input_raw.wav"
        video_path = tmpdir_path / "input_25fps.mp4"
        audio_path = tmpdir_path / "input_16k.wav"
        output_path = tmpdir_path / "synced.mp4"

        raw_video.write_bytes(await video.read())
        raw_audio.write_bytes(await audio.read())

        await _normalize_inputs(raw_video, raw_audio, video_path, audio_path)

        cmd = [
            sys.executable,
            "-m",
            "scripts.inference",
            "--unet_config_path",
            str(config_path),
            "--inference_ckpt_path",
            str(ckpt_path),
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--video_path",
            str(video_path),
            "--audio_path",
            str(audio_path),
            "--video_out_path",
            str(output_path),
        ]
        if LATENTSYNC_DEEPCACHE:
            cmd.append("--enable_deepcache")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(LATENTSYNC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        logger.info(
            "Running LatentSync for shot %s (steps=%s, guidance=%.2f)",
            shot_id,
            inference_steps,
            guidance_scale,
        )
        returncode, stdout, stderr = await _run_subprocess(
            cmd,
            cwd=str(LATENTSYNC_DIR),
            env=env,
            timeout_sec=LATENTSYNC_TIMEOUT_SEC,
        )
        if returncode != 0:
            logger.error("LatentSync stderr: %s", stderr[-2000:])
            raise HTTPException(500, detail=f"LatentSync failed: {stderr[-500:]}")
        if not output_path.exists():
            logger.error("LatentSync stdout: %s", stdout[-2000:])
            raise HTTPException(500, detail="LatentSync produced no output video")

        return Response(content=output_path.read_bytes(), media_type="video/mp4")


async def _normalize_inputs(
    raw_video: Path, raw_audio: Path, video_path: Path, audio_path: Path
) -> None:
    """Match LatentSync's documented preprocessing assumptions: 25 FPS + 16 kHz mono audio."""
    for cmd in (
        [
            "ffmpeg", "-y",
            "-i", str(raw_video),
            "-vf", "fps=25",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ],
        [
            "ffmpeg", "-y",
            "-i", str(raw_audio),
            "-ac", "1",
            "-ar", "16000",
            str(audio_path),
        ],
    ):
        returncode, _, stderr = await _run_subprocess(cmd, timeout_sec=300)
        if returncode != 0:
            raise HTTPException(
                500, detail=f"ffmpeg input normalization failed: {stderr[-500:]}"
            )
