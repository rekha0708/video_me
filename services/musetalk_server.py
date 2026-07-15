"""
MuseTalk lip-sync HTTP service wrapper.

Exposes the API contract expected by adapters/lip_sync/lip_sync_adapter.py:
  GET  /health   → {"status": "ok"}
  POST /lipsync  → multipart/form-data:
                     video   (file, MP4),
                     audio   (file, WAV),
                     shot_id (str)
                     job_id  (opaque adapter-scoped str)
                   → raw synced MP4 bytes
  POST /cancel   → form data: job_id (str)

Environment variables:
  MUSETALK_DIR  Path to the cloned MuseTalk repo (default: /workspace/MuseTalk)

Run from the video_me repo root using the dedicated musetalk venv:
  MUSETALK_DIR=/workspace/MuseTalk \
  /workspace/.venv_musetalk/bin/uvicorn services.musetalk_server:app \
    --host 0.0.0.0 --port 8040
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import signal
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

MUSETALK_DIR = Path(os.getenv("MUSETALK_DIR", "/workspace/MuseTalk"))
# Directory holding sitecustomize.py — see musetalk_compat/sitecustomize.py
# for why this needs to be a subprocess-level PYTHONPATH shim rather than
# an in-process monkey-patch.
_COMPAT_DIR = Path(__file__).parent / "musetalk_compat"

# inference.py lives under scripts/, not the repo root
_INFERENCE_SCRIPT = "scripts/inference.py"
# v15 = MuseTalk v1.5 (unet.pth under models/musetalkV15/)
_MUSETALK_VERSION = "v15"
_UNET_CONFIG = "models/musetalkV15/musetalk.json"
_UNET_MODEL = "models/musetalkV15/unet.pth"
_WHISPER_DIR = "models/whisper"
MUSETALK_TIMEOUT_SEC = int(os.getenv("MUSETALK_TIMEOUT_SEC", "600"))
_PROCESS_TERMINATE_GRACE_SEC = float(
    os.getenv("MUSETALK_PROCESS_TERMINATE_GRACE_SEC", "5")
)
_CANCEL_MARKER_TTL_SEC = float(os.getenv("MUSETALK_CANCEL_MARKER_TTL_SEC", "60"))

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
    if not MUSETALK_DIR.exists():
        logger.error("MUSETALK_DIR not found: %s — set MUSETALK_DIR env var", MUSETALK_DIR)
    else:
        logger.info("MuseTalk service ready (dir: %s, version: %s)", MUSETALK_DIR, _MUSETALK_VERSION)
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


app = FastAPI(title="MuseTalk lip-sync", lifespan=lifespan)


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
        raise JobCancelledError(f"MuseTalk job {job_id} was cancelled")

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
        if job_id is not None and job_id in _CANCELLED_JOBS:
            await asyncio.shield(_terminate_process_group(process))
            raise JobCancelledError(f"MuseTalk job {job_id} was cancelled")
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
        raise JobCancelledError(f"MuseTalk job {job_id} was cancelled")
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
    job_id: str = Form(...),
) -> Response:
    job_key = _validated_job_id(job_id)
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

        # scripts/inference.py lives in scripts/ but imports musetalk from repo root.
        # _COMPAT_DIR first so its sitecustomize.py (torch.load weights_only patch,
        # required for mmengine checkpoint loading on PyTorch 2.6+) auto-applies
        # before mmengine/musetalk are imported in this subprocess.
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(_COMPAT_DIR) + os.pathsep + str(MUSETALK_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        )

        logger.info("Running MuseTalk for shot %s", shot_id)
        returncode, stdout, stderr = await _run_subprocess(
            cmd,
            cwd=MUSETALK_DIR,
            env=env,
            timeout=MUSETALK_TIMEOUT_SEC,
            job_id=job_key,
        )

        stderr_text = stderr.decode(errors="replace")
        if returncode != 0:
            logger.error("MuseTalk stderr: %s", stderr_text[-2000:])
            raise HTTPException(500, detail=f"MuseTalk failed: {stderr_text[-500:]}")

        # Output lands at result_dir/<version>/input.mp4 (named after input basename)
        mp4s = sorted(glob.glob(str(result_dir / "**/*.mp4"), recursive=True))
        if not mp4s:
            avis = glob.glob(str(result_dir / "**/*.avi"), recursive=True)
            if not avis:
                # Face detection likely failed (cartoon/stylised input). Return the
                # original Wan video unchanged so the pipeline can continue without
                # lip sync rather than crashing the whole run.
                logger.warning(
                    "MuseTalk produced no output for shot %s — face not detected in "
                    "cartoon-style frames. Returning original video as passthrough.",
                    shot_id,
                )
                return Response(
                    content=video_path.read_bytes(),
                    media_type="video/mp4",
                    headers={"X-Video-Me-Lipsync": "passthrough"},
                )
            output_path = await _convert_to_mp4(
                Path(avis[0]),
                tmpdir_path / "synced.mp4",
                job_id=job_key,
            )
        else:
            output_path = Path(mp4s[-1])

        return Response(
            content=output_path.read_bytes(),
            media_type="video/mp4",
            headers={"X-Video-Me-Lipsync": "applied"},
        )


async def _convert_to_mp4(avi: Path, out: Path, *, job_id: str) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(avi),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(out),
    ]
    returncode, _, stderr = await _run_subprocess(
        cmd,
        timeout=MUSETALK_TIMEOUT_SEC,
        job_id=job_id,
    )
    if returncode != 0:
        raise RuntimeError(
            "MuseTalk AVI conversion failed: " + stderr.decode(errors="replace")[-500:]
        )
    return out
