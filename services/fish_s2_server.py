"""
Fish Audio S2 TTS HTTP service wrapper.

Exposes the API contract expected by adapters/synthesize_voice/fish_s2_adapter.py:
  GET  /health      → {"status": "ok"|"loading", "model_loaded": bool, "loading": bool}
  POST /load        → (re)load the model. 200 if already loaded, 202 if load started.
  POST /unload       → drop the model, gc + empty CUDA cache. Idempotent 200.
                       409 if a load is in progress.
  POST /synthesize  → multipart/form-data: text, reference_audio (file),
                       language, format → raw WAV bytes
                       Safety net: if the model is unloaded, /synthesize blocks on a
                       load first — standalone use keeps working.

Environment variables (resolved at startup):
  FISH_SPEECH_DIR   Path to the cloned fish-speech repo (default: /workspace/fish-speech)
  FISH_LLAMA_CKPT   LLaMA checkpoint dir  (default: $FISH_SPEECH_DIR/checkpoints/fish-speech-1.5)
  FISH_DECODER_CKPT Decoder .pth path     (default: $FISH_LLAMA_CKPT/firefly-gan-vq-fsq-8x1024-21hz-generator.pth)
  FISH_DECODER_CFG  Hydra config name     (default: modded_dac_vq)
  FISH_DEVICE       cuda | cpu            (default: cuda if available)
  FISH_HALF         1 = float16           (default: 1)
  FISH_COMPILE      1 = torch.compile     (default: 0)

Run:
  FISH_SPEECH_DIR=/workspace/fish-speech \
  uvicorn services.fish_s2_server:app --host 0.0.0.0 --port 8025
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

_engine = None
_sample_rate: int = 44100
_load_error: str | None = None
_load_lock = threading.Lock()   # serializes load attempts / guards _loading
_loading = False


def _resolve_env() -> dict:
    fish_dir = os.environ.get("FISH_SPEECH_DIR", "/workspace/fish-speech")
    llama_ckpt = os.environ.get(
        "FISH_LLAMA_CKPT",
        str(Path(fish_dir) / "checkpoints" / "s2-pro"),
    )
    decoder_ckpt = os.environ.get(
        "FISH_DECODER_CKPT",
        str(Path(llama_ckpt) / "codec.pth"),
    )
    return {
        "fish_dir": fish_dir,
        "llama_ckpt": llama_ckpt,
        "decoder_ckpt": decoder_ckpt,
        "decoder_cfg": os.environ.get("FISH_DECODER_CFG", "modded_dac_vq"),
        "device": os.environ.get("FISH_DEVICE", ""),
        "half": os.environ.get("FISH_HALF", "1") == "1",
        "compile": os.environ.get("FISH_COMPILE", "0") == "1",
    }


def _load_engine() -> None:
    """Load the Fish Speech engine into memory. Runs in a thread executor, never on the event loop."""
    global _engine, _sample_rate, _load_error, _loading
    with _load_lock:
        if _engine is not None:
            return
        if _loading:
            return
        _loading = True
        _load_error = None
    try:
        cfg = _resolve_env()

        # fish-speech uses pyrootutils + hydra with paths relative to its repo root.
        # Ensure the repo is on sys.path so `tools.*` and `fish_speech.*` are importable.
        fish_dir = cfg["fish_dir"]
        if fish_dir not in sys.path:
            sys.path.insert(0, fish_dir)

        # Change working directory so hydra's relative config path resolves correctly.
        os.chdir(fish_dir)

        import torch
        from tools.server.model_manager import ModelManager

        device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Fish Speech models on %s ...", device)
        logger.info("  LLaMA checkpoint : %s", cfg["llama_ckpt"])
        logger.info("  Decoder checkpoint: %s", cfg["decoder_ckpt"])

        manager = ModelManager(
            mode="tts",
            device=device,
            half=cfg["half"],
            compile=cfg["compile"],
            llama_checkpoint_path=cfg["llama_ckpt"],
            decoder_checkpoint_path=cfg["decoder_ckpt"],
            decoder_config_name=cfg["decoder_cfg"],
        )
        _engine = manager.tts_inference_engine
        _sample_rate = _engine.decoder_model.sample_rate
        logger.info("Fish Speech ready — sample rate %d Hz", _sample_rate)
    except Exception as exc:
        _load_error = str(exc)
        logger.error("Fish Speech failed to load: %s", exc, exc_info=True)
    finally:
        with _load_lock:
            _loading = False


def _unload_engine() -> None:
    """Drop the engine and release CUDA memory. Idempotent."""
    global _engine, _load_error
    _engine = None
    _load_error = None
    import gc  # noqa: PLC0415
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("Fish Speech unloaded — VRAM released")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded eagerly at startup (unlike Wan, which defers to POST /load) so that
    # standalone use of this server keeps working exactly as before. The GPU
    # lifecycle manager (core/gpu_sequencer.py) explicitly unloads/reloads this
    # around the render/video phases via POST /unload and /load.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_engine)
    yield
    _unload_engine()


app = FastAPI(title="Fish Audio S2 TTS", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    # Always 200 — the process being up and the model being loaded are two
    # different signals (mirrors services/wan_server.py's /health contract).
    # A deliberate POST /unload during a phase transition must not read as
    # "service down" to callers like scripts/check_runtime_readiness.py.
    return JSONResponse({
        "status": "ok",
        "model_loaded": _engine is not None,
        "loading": _loading,
        "error": _load_error,
    })


@app.post("/load")
async def load() -> JSONResponse:
    if _engine is not None:
        return JSONResponse({"status": "ok", "model_loaded": True})
    if _loading:
        return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _load_engine)  # fire and forget; poll /health
    return JSONResponse({"status": "loading", "model_loaded": False}, status_code=202)


@app.post("/unload")
async def unload() -> JSONResponse:
    if _loading:
        # Unloading now would race the loader thread, which could repopulate
        # _engine after we return — leaving the model resident unexpectedly.
        raise HTTPException(409, detail="model load in progress; retry after it completes")
    if _engine is not None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _unload_engine)
    return JSONResponse({"status": "ok", "model_loaded": False})


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    language: str = Form("en"),
    format: str = Form("wav"),
) -> Response:
    if _engine is None:
        # Safety net for standalone use: block on a load instead of failing.
        # The orchestrated path always POSTs /load and polls /health first.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _load_engine)
        while _loading:  # another caller may hold the load; wait it out
            await asyncio.sleep(2)
        if _engine is None:
            detail = "Fish Speech not ready" + (f": {_load_error}" if _load_error else "")
            raise HTTPException(503, detail=detail)

    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
    from tools.server.inference import inference_wrapper as inference

    ref_bytes = await reference_audio.read()

    req = ServeTTSRequest(
        text=text,
        references=[ServeReferenceAudio(audio=ref_bytes, text="")],
        format=format,
        streaming=False,
    )

    try:
        import soundfile as sf

        fake_audio = next(inference(req, _engine))
        buf = io.BytesIO()
        sf.write(buf, fake_audio, _sample_rate, format=format)
        buf.seek(0)
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as exc:
        logger.exception("Fish Speech inference error")
        raise HTTPException(500, detail=str(exc)) from exc
