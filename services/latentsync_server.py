"""
LatentSync lip-sync HTTP service wrapper.

Exposes the API contract expected by adapters/lip_sync/latentsync_adapter.py:
  GET  /health   -> {"status": "ok"}
  POST /lipsync  -> multipart/form-data:
                     video           (file, MP4)
                     audio           (file, WAV)
                     shot_id         (str)
                     job_id          (opaque adapter-scoped str)
                     inference_steps (int, optional)
                     guidance_scale  (float, optional)
                   -> raw synced MP4 bytes
  POST /cancel   -> form data: job_id (str)

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
import signal
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

LATENTSYNC_DIR = Path(os.getenv("LATENTSYNC_DIR", "/workspace/LatentSync"))
LATENTSYNC_UNET_CONFIG = os.getenv("LATENTSYNC_UNET_CONFIG", "configs/unet/stage2_512.yaml")
LATENTSYNC_CKPT = os.getenv("LATENTSYNC_CKPT", "checkpoints/latentsync_unet.pt")
LATENTSYNC_TIMEOUT_SEC = int(os.getenv("LATENTSYNC_TIMEOUT_SEC", "1200"))
LATENTSYNC_DEEPCACHE = os.getenv("LATENTSYNC_DEEPCACHE", "1") == "1"
_PROCESS_TERMINATE_GRACE_SEC = float(
    os.getenv("LATENTSYNC_PROCESS_TERMINATE_GRACE_SEC", "5")
)
_CANCEL_MARKER_TTL_SEC = float(os.getenv("LATENTSYNC_CANCEL_MARKER_TTL_SEC", "60"))

# A single dashboard job can have preprocessing and inference subprocesses at
# different times (or several shots in flight).  Track every process under the
# adapter's opaque job token so a separate HTTP request can stop the complete
# job, including process descendants.
_ACTIVE_PROCESSES: dict[str, set[asyncio.subprocess.Process]] = {}
_CANCELLED_JOBS: set[str] = set()


class JobCancelledError(RuntimeError):
    """Raised inside a request after its job was cancelled via /cancel."""


def _validated_job_id(job_id: str) -> str:
    value = job_id.strip() if isinstance(job_id, str) else ""
    if not value or len(value) > 128:
        raise HTTPException(422, detail="job_id must be between 1 and 128 characters")
    return value


def _register_process(job_id: str | None, process: asyncio.subprocess.Process) -> None:
    if job_id is not None:
        _ACTIVE_PROCESSES.setdefault(job_id, set()).add(process)


def _unregister_process(job_id: str | None, process: asyncio.subprocess.Process) -> None:
    if job_id is None:
        return
    processes = _ACTIVE_PROCESSES.get(job_id)
    if processes is None:
        return
    processes.discard(process)
    if not processes:
        _ACTIVE_PROCESSES.pop(job_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LATENTSYNC_DIR.exists():
        logger.error("LATENTSYNC_DIR not found: %s — set LATENTSYNC_DIR", LATENTSYNC_DIR)
    else:
        logger.info("LatentSync service ready (dir=%s)", LATENTSYNC_DIR)
    try:
        yield
    finally:
        processes = [
            process
            for active in _ACTIVE_PROCESSES.values()
            for process in active
        ]
        if processes:
            await asyncio.gather(
                *(_terminate_process_group(process) for process in processes),
                return_exceptions=True,
            )
        _ACTIVE_PROCESSES.clear()
        _CANCELLED_JOBS.clear()


app = FastAPI(title="LatentSync lip-sync", lifespan=lifespan)


@app.exception_handler(JobCancelledError)
async def _job_cancelled_handler(_request: Request, exc: JobCancelledError) -> JSONResponse:
    return JSONResponse({"status": "cancelled", "detail": str(exc)}, status_code=409)


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    """Terminate a child process and all descendants in its process group."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        try:
            process.terminate()
        except ProcessLookupError:
            return
    try:
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_TERMINATE_GRACE_SEC)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                return
        await process.wait()


async def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float,
    job_id: str | None = None,
) -> tuple[int, bytes, bytes]:
    """Run a cancellable subprocess in its own process group."""
    if job_id is not None and job_id in _CANCELLED_JOBS:
        raise JobCancelledError(f"LatentSync job {job_id} was cancelled")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    _register_process(job_id, process)
    try:
        # /cancel may have arrived while create_subprocess_exec was awaiting.
        if job_id is not None and job_id in _CANCELLED_JOBS:
            await asyncio.shield(_terminate_process_group(process))
            raise JobCancelledError(f"LatentSync job {job_id} was cancelled")
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        # Shield cleanup so cancelling the HTTP handler cannot strand a GPU worker.
        await asyncio.shield(_terminate_process_group(process))
        raise
    except asyncio.TimeoutError:
        await asyncio.shield(_terminate_process_group(process))
        raise
    finally:
        _unregister_process(job_id, process)
    if job_id is not None and job_id in _CANCELLED_JOBS:
        raise JobCancelledError(f"LatentSync job {job_id} was cancelled")
    return process.returncode, stdout, stderr


async def _cancel_active_job(job_id: str) -> int:
    """Mark a job cancelled and terminate every currently registered child."""
    _CANCELLED_JOBS.add(job_id)
    processes = tuple(_ACTIVE_PROCESSES.get(job_id, ()))
    if processes:
        await asyncio.gather(
            *(_terminate_process_group(process) for process in processes),
            return_exceptions=True,
        )
    return len(processes)


@app.post("/cancel")
async def cancel_job(job_id: str = Form(...)) -> JSONResponse:
    job_key = _validated_job_id(job_id)
    cancelled_count = await _cancel_active_job(job_key)
    # Keep a short-lived marker to close the race where /cancel arrives while
    # create_subprocess_exec is being scheduled, without retaining job tokens
    # forever in this long-running service.
    asyncio.get_running_loop().call_later(
        _CANCEL_MARKER_TTL_SEC,
        _CANCELLED_JOBS.discard,
        job_key,
    )
    return JSONResponse(
        {
            "status": "cancelled",
            "job_id": job_key,
            "cancelled_processes": cancelled_count,
        }
    )


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else LATENTSYNC_DIR / path


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
    job_id: str = Form(...),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.5),
) -> Response:
    job_key = _validated_job_id(job_id)
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

        await _normalize_inputs(
            raw_video,
            raw_audio,
            video_path,
            audio_path,
            job_id=job_key,
        )

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
            cwd=LATENTSYNC_DIR,
            env=env,
            timeout=LATENTSYNC_TIMEOUT_SEC,
            job_id=job_key,
        )
        stderr_text = stderr.decode(errors="replace")
        stdout_text = stdout.decode(errors="replace")
        if returncode != 0:
            logger.error("LatentSync stderr: %s", stderr_text[-2000:])
            raise HTTPException(500, detail=f"LatentSync failed: {stderr_text[-500:]}")
        if not output_path.exists():
            logger.error("LatentSync stdout: %s", stdout_text[-2000:])
            raise HTTPException(500, detail="LatentSync produced no output video")

        return Response(content=output_path.read_bytes(), media_type="video/mp4")


async def _normalize_inputs(
    raw_video: Path,
    raw_audio: Path,
    video_path: Path,
    audio_path: Path,
    *,
    job_id: str,
) -> None:
    """Match LatentSync's documented preprocessing assumptions: 25 FPS + 16 kHz mono audio."""
    commands = (
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video),
            "-vf",
            "fps=25",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ],
    )
    for cmd in commands:
        returncode, stdout, stderr = await _run_subprocess(
            cmd,
            timeout=LATENTSYNC_TIMEOUT_SEC,
            job_id=job_id,
        )
        if returncode != 0:
            raise subprocess.CalledProcessError(
                returncode,
                cmd,
                output=stdout,
                stderr=stderr,
            )
