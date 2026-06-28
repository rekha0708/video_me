"""
Fish Audio S2 TTS HTTP service wrapper.

Exposes the API contract expected by adapters/synthesize_voice/fish_s2_adapter.py:
  GET  /health      → {"status": "ok"}
  POST /synthesize  → multipart/form-data: text, reference_audio (file),
                       language, format → raw WAV bytes

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

import io
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_engine = None
_sample_rate: int = 44100


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _sample_rate

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
    yield
    _engine = None


app = FastAPI(title="Fish Audio S2 TTS", lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if _engine is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    language: str = Form("en"),
    format: str = Form("wav"),
) -> Response:
    if _engine is None:
        raise HTTPException(503, detail="Model not loaded yet")

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
